"""Ablations of the ModelBlaster <-> KernelBlaster optimisation flow.

One spec file names a base configuration and a set of arms (each a diff of
flags, environment and knowledge set-up); ``run`` executes every arm x repeat
in its own isolated run directory, interleaved, under a concurrency limit and
an LLM budget; ``report`` turns the runs into per-problem / per-arm tables and
paired comparisons (geo-mean ratio, bootstrap CI, permutation and Wilcoxon
p-values). See ``docs/ablations.md``.

    python -m modelblaster.pipeline.kb_ablate run SPEC [--root DIR] [--dry-run] [--max-parallel N]
                                                       [--budget-usd X] [--who NAME] [--force]
    python -m modelblaster.pipeline.kb_ablate status ROOT
    python -m modelblaster.pipeline.kb_ablate report ROOT [--force]

Entry points (driven as subprocesses, so new flags need no change here):
``multislot`` = KernelBlaster ``scripts/run_multislot.py`` on exported problem
dirs; ``graph_opt`` = ``python -m modelblaster.pipeline.kb_graph_opt`` on IR
dirs. ``--resume`` / ``--trace`` are used when the target's ``--help`` lists
them.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import fcntl
import fnmatch
import glob
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time
from pathlib import Path

MB_ROOT = Path(__file__).resolve().parents[1]
STATE_FORMAT = "mb-kb-ablate/1"
ENTRIES = ("multislot", "graph_opt")
INTERLEAVE = ("counterbalance", "abab", "random", "sequential")

# flags the runner owns (it sets them per run); an arm or the base may not
RESERVED = {"multislot": {"--out", "--knowledge-dir", "--seed", "--resume", "--force-restart", "--gpu-model"},
            "graph_opt": {"--work", "--kb", "--python", "--seed", "--gpu-model"}}
MODEL_FLAGS = ("--model", "--model-simple", "--model-hard", "--fix-model", "--reflect-model", "--upkeep-model")
DEFAULT_STEPS = {"multislot": {"--rollouts": 3, "--steps": 5, "--rounds": 1},
                 "graph_opt": {"--rollouts": 2, "--steps": 4, "--rounds": 4}}

# inherited variables that would change a run's result or point it at shared state; dropped from
# the parent environment (a spec's base / arm env may set them again, explicitly)
DROP_ENV = ("PYTHONPATH", "MB_KB_ROOT", "MB_KB_PYTHON", "KB_KNOWLEDGE", "KB_KNOWLEDGE_DIR",
            "KB_KNOWLEDGE_SEED_DIR", "KB_GRAPH_HINTS", "KB_REFLECT_MODEL", "KB_REFLECT_EVERY",
            "KB_UPKEEP_MODEL", "KB_HETERO_POLICY", "KB_OP_CLASS", "KB_TELEMETRY_FILE", "KB_TELEMETRY_RUN_ID",
            "KB_TRACE", "MODEL_CODEGEN_SIMPLE", "MODEL_CODEGEN_HARD", "MODEL_FIX",
            "KERNELBLASTER_MULTISLOT_REWARD", "KERNELBLASTER_MULTISLOT_RUNTIME",
            "KERNELBLASTER_MULTISLOT_E2E_ACCEPT_PCT", "KERNELBLASTER_BUILD_CACHE_DIR",
            "KERNELBLASTER_PERF_LOG_DB", "KERNELBLASTER_PERF_LOG_RUN_ID", "KERNELBLASTER_SLOT_UNITS",
            "KERNELBLASTER_FAST_COMPILE", "KERNELBLASTER_NCU_MIN_IMPROVEMENTS", "KERNELBLASTER_OPT_RL_NCU",
            "MODELBLASTER_MAX_USD", "TMPDIR", "TMP", "TEMP")
_SECRET = re.compile(r"(^|_)(TOKEN|SECRET|PASSWORD|PASSWD|BEARER|CREDENTIALS?|API_?KEY|ACCESS_KEY|PRIVATE_KEY|"
                     r"SESSION_KEY|KEY)($|_)", re.I)
_SECRET_ARG = re.compile(r"(token|secret|password|api-key|apikey|bearer)", re.I)

# $ per million tokens (in, out): only used when a run's telemetry has no cost / estimate.
# Bedrock reports tokens only; these are the ledger README's conservative prices.
DEFAULT_PRICES = {"haiku": [1.0, 5.0], "sonnet": [3.0, 15.0], "opus": [5.0, 25.0]}


class AblateError(RuntimeError):
    pass


# ---------------------------------------------------------------- small helpers

def _sha(data) -> str:
    if not isinstance(data, (bytes, bytearray)):
        data = json.dumps(data, sort_keys=True, default=str).encode()
    return hashlib.sha1(data).hexdigest()[:16]


def tree_hash(path: Path | None) -> str | None:
    """Content hash of a file or directory tree (relative paths + bytes)."""
    if path is None or not Path(path).exists():
        return None
    p = Path(path)
    h = hashlib.sha1()
    files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file() and "__pycache__" not in f.parts)
    for f in files:
        h.update(str(f.relative_to(p) if f != p else f.name).encode() + b"\0")
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()[:16]


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def _read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def scrub_env(env: dict) -> dict:
    """Values of secret-looking variables replaced (presence kept)."""
    return {k: ("<redacted>" if is_secret(k) else v) for k, v in env.items()}


def is_secret(name: str) -> bool:
    return bool(_SECRET.search(name)) and not name.upper().endswith(("_FILE", "_PATH", "_DIR"))


def scrub_argv(argv: list) -> list:
    out, hide = [], False
    for a in argv:
        s = str(a)
        if hide:
            out.append("<redacted>")
            hide = False
            continue
        if s.startswith("--") and _SECRET_ARG.search(s) and not s.endswith(("-file", "-path")):
            if "=" in s:
                out.append(s.split("=", 1)[0] + "=<redacted>")
            else:
                out.append(s)
                hide = True
            continue
        out.append(s)
    return out


def _run_quiet(argv: list, env: dict | None = None, timeout: float = 60, cwd=None) -> str | None:
    try:
        cp = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env, cwd=cwd)
        return (cp.stdout + cp.stderr).strip() if cp.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


# ---------------------------------------------------------------- code identity

def git_identity(path: Path) -> dict:
    """commit, branch, dirty flag and a hash of the uncommitted diff of a git checkout."""
    path = Path(path)
    marker = path / ".ablate_code.json"
    if marker.exists():                              # a frozen copy made by freeze_code
        return _read_json(marker, {})
    head = _run_quiet(["git", "-C", str(path), "rev-parse", "HEAD"])
    if not head:
        return {"path": str(path), "commit": None, "dirty": None, "git": False, "tree": tree_hash(path / "src")
                if (path / "src").exists() else None}
    st = _run_quiet(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"]) or ""
    diff = subprocess.run(["git", "-C", str(path), "diff", "HEAD"], capture_output=True).stdout
    return {"path": str(path), "commit": head.split()[0], "dirty": bool(st.strip()),
            "diff_sha": _sha(diff) if st.strip() else None,
            "branch": (_run_quiet(["git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"]) or "").strip()}


def same_code(a: dict | None, b: dict | None) -> bool:
    a, b = a or {}, b or {}
    return (a.get("commit"), a.get("diff_sha"), a.get("tree")) == (b.get("commit"), b.get("diff_sha"), b.get("tree"))


def freeze_code(src: Path, dst: Path) -> dict:
    """A copy of ``src``'s tracked files as they are now (HEAD plus uncommitted changes to tracked
    files, via ``git stash create``) at ``dst``; untracked files are not copied. -> identity."""
    ident = git_identity(src)
    if not ident.get("commit"):
        raise AblateError(f"freeze_code: {src} is not a git checkout (set freeze_code: false)")
    if (dst / ".ablate_code.json").exists():
        return _read_json(dst / ".ablate_code.json")
    tree = (_run_quiet(["git", "-C", str(src), "stash", "create"]) or "").strip() or ident["commit"]
    data = subprocess.run(["git", "-C", str(src), "archive", "--format=tar", tree], capture_output=True, check=True).stdout
    tmp = dst.with_name(dst.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        tf.extractall(tmp, filter="data") if sys.version_info >= (3, 12) else tf.extractall(tmp)  # noqa: S202
    ident = {**ident, "frozen_from": str(src), "frozen_tree": tree, "path": str(dst)}
    (tmp / ".ablate_code.json").write_text(json.dumps(ident, indent=1))
    if dst.exists():
        shutil.rmtree(dst)
    os.replace(tmp, dst)
    return ident


# ---------------------------------------------------------------- spec

def load_spec_file(path: Path) -> dict:
    text = Path(path).read_text()
    if Path(path).suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise AblateError(f"{path}: YAML spec needs PyYAML (pip install pyyaml), or write it as JSON") from e
        return yaml.safe_load(text)
    return json.loads(text)


def _strip_comments(x):
    if isinstance(x, dict):
        return {k: _strip_comments(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, list):
        return [_strip_comments(v) for v in x]
    return x


def _flags(d) -> dict:
    """``flags`` as an ordered dict flag -> value (True: bare flag; list: repeated; None/False: unset)."""
    if d is None:
        return {}
    if not isinstance(d, dict):
        raise AblateError(f"flags must be a mapping of --flag: value, got {type(d).__name__}")
    out = {}
    for k, v in d.items():
        k = k if k.startswith("-") else "--" + k
        out[k] = v
    return out


def _resolve(base: Path, p):
    if p is None:
        return None
    p = os.path.expandvars(os.path.expanduser(str(p)))
    if re.search(r"\$\{?[A-Za-z_]", p):
        raise AblateError(f"{p}: environment variable not set")
    q = Path(p)
    return str(q if q.is_absolute() else (base / q).resolve())


def _knowledge_cfg(k: dict | None, base: Path) -> dict:
    k = dict(k or {})
    bad = set(k) - {"seed_dir", "mined", "snapshot", "prebuild"}
    if bad:
        raise AblateError(f"knowledge: unknown keys {sorted(bad)}")
    for f in ("seed_dir", "snapshot"):
        if k.get(f):
            k[f] = _resolve(base, k[f])
    return k


def _cache_cfg(c: dict | None, base: Path) -> dict | None:
    if not c:
        return None
    if set(c) - {"snapshot", "shared"} or len(c) != 1:
        raise AblateError("mb_cache: give exactly one of {snapshot: DIR} (copied per run) or {shared: DIR}")
    k, v = next(iter(c.items()))
    return {k: _resolve(base, v)}


def normalize_spec(raw: dict, spec_dir: Path) -> dict:
    """Validated, path-resolved spec with every default filled in."""
    raw = _strip_comments(copy.deepcopy(raw))
    known = {"name", "description", "entry", "mb", "kb", "python", "script", "inputs", "per_input", "flags",
             "args", "env", "drop_env", "path_exclude", "path_prepend", "gpu_model", "knowledge", "mb_cache",
             "repeats", "seeds", "interleave", "order_seed", "max_parallel", "budget", "freeze_code", "trace",
             "report", "arms", "torch_baseline", "timeout_s", "stop_grace_s"}
    bad = set(raw) - known
    if bad:
        raise AblateError(f"spec: unknown keys {sorted(bad)} (known: {sorted(known)})")
    entry = raw.get("entry", "multislot")
    if entry not in ENTRIES:
        raise AblateError(f"entry must be one of {ENTRIES}, got {entry!r}")
    kb = raw.get("kb") or os.environ.get("MB_KB_ROOT")
    if not kb:
        raise AblateError("spec: 'kb' (KernelBlaster checkout) is required (or set MB_KB_ROOT)")
    inputs = []
    for p in raw.get("inputs") or []:
        r = _resolve(spec_dir, p)
        hits = sorted(glob.glob(r)) if any(ch in r for ch in "*?[") else [r]
        if not hits:
            raise AblateError(f"inputs: {p} matches nothing")
        inputs += hits
    if not inputs:
        raise AblateError("spec: 'inputs' (problem dirs / IR dirs) is required")
    names = [Path(p).name for p in inputs]
    if len(set(names)) != len(names):
        raise AblateError(f"inputs: names collide: {sorted(n for n in set(names) if names.count(n) > 1)}")
    arms_raw = raw.get("arms") or {}
    if isinstance(arms_raw, list):
        arms_raw = {a["name"]: {k: v for k, v in a.items() if k != "name"} for a in arms_raw}
    if len(arms_raw) < 1:
        raise AblateError("spec: at least one arm is required")
    reps = raw.get("seeds")
    seeds = [int(s) for s in reps] if reps is not None else list(range(int(raw.get("repeats", 2))))
    budget = dict(raw.get("budget") or {})
    bad = set(budget) - {"usd", "est_usd_per_run", "ledger", "who", "prices"}
    if bad:
        raise AblateError(f"budget: unknown keys {sorted(bad)}")
    if budget.get("ledger"):
        budget["ledger"] = _resolve(spec_dir, budget["ledger"])
    spec = {
        "name": raw.get("name") or "ablation",
        "description": raw.get("description", ""),
        "entry": entry,
        "mb": _resolve(spec_dir, raw.get("mb") or MB_ROOT),
        "kb": _resolve(spec_dir, kb),
        "python": (_resolve(spec_dir, raw["python"]) if raw.get("python") and "/" in str(raw["python"])
                   else raw.get("python") or os.environ.get("MB_KB_PYTHON") or sys.executable),
        "script": _resolve(spec_dir, raw.get("script")) if raw.get("script") else None,
        "inputs": inputs,
        "per_input": bool(raw.get("per_input", False)),
        "flags": _flags(raw.get("flags")),
        "args": [str(a) for a in raw.get("args") or []],
        "env": {k: str(v) for k, v in (raw.get("env") or {}).items()},
        "drop_env": list(raw.get("drop_env") or []),
        "path_exclude": raw.get("path_exclude", os.environ.get("MB_PATH_EXCLUDE")),
        "path_prepend": [str(p) for p in raw.get("path_prepend") or []],
        "gpu_model": raw.get("gpu_model"),
        "knowledge": _knowledge_cfg(raw.get("knowledge"), spec_dir),
        "mb_cache": _cache_cfg(raw.get("mb_cache"), spec_dir),
        "seeds": seeds,
        "interleave": raw.get("interleave", "counterbalance"),
        "order_seed": int(raw.get("order_seed", 0)),
        "max_parallel": int(raw.get("max_parallel", 1)),
        "budget": budget,
        "freeze_code": bool(raw.get("freeze_code", True)),
        "trace": raw.get("trace", "meta"),
        "timeout_s": raw.get("timeout_s"),
        "stop_grace_s": float(raw.get("stop_grace_s", 150)),
        "torch_baseline": _resolve(spec_dir, raw["torch_baseline"]) if raw.get("torch_baseline") else None,
        "report": {"baseline": None, "pairs": "baseline", "bootstrap": 10000, "permutations": 20000,
                   **(raw.get("report") or {})},
        "arms": {},
    }
    if spec["interleave"] not in INTERLEAVE:
        raise AblateError(f"interleave must be one of {INTERLEAVE}")
    for k in ("env",):
        for n in spec[k]:
            if is_secret(n):
                raise AblateError(f"env: {n} looks like a secret; pass it through the calling environment, "
                                  f"not the spec (specs and manifests are stored)")
    for name, a in arms_raw.items():
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(name)):
            raise AblateError(f"arm name {name!r}: use letters, digits, _ . -")
        a = dict(a or {})
        bad = set(a) - {"flags", "args", "env", "knowledge", "mb_cache", "description"}
        if bad:
            raise AblateError(f"arm {name}: unknown keys {sorted(bad)} (inputs, seeds and the GPU model are "
                              f"shared by every arm)")
        for n in a.get("env") or {}:
            if is_secret(n):
                raise AblateError(f"arm {name} env: {n} looks like a secret; pass it through the environment")
        k = {**spec["knowledge"], **_knowledge_cfg(a.get("knowledge"), spec_dir)}
        flags = {**spec["flags"], **_flags(a.get("flags"))}
        flags = {f: v for f, v in flags.items() if v is not None and v is not False}
        own = RESERVED[entry] & set(flags) | {x for x in spec["args"] + [str(s) for s in a.get("args") or []]
                                               if x.split("=")[0] in RESERVED[entry]}
        if own:
            raise AblateError(f"arm {name}: {sorted(own)} are set by the runner (seeds: 'seeds'/'repeats'; GPU "
                              f"model: 'gpu_model')")
        arm = {"description": a.get("description", ""), "flags": flags,
               "args": spec["args"] + [str(s) for s in a.get("args") or []],
               "env": {**spec["env"], **{k2: str(v) for k2, v in (a.get("env") or {}).items()}},
               "knowledge": k,
               "mb_cache": _cache_cfg(a.get("mb_cache"), spec_dir) if "mb_cache" in a else spec["mb_cache"]}
        if entry == "multislot" and arm["mb_cache"]:
            raise AblateError(f"arm {name}: mb_cache applies to graph_opt only")
        if not any(f in flags for f in ("--model",)) and not any(x == "--model" or x.startswith("--model=")
                                                                  for x in arm["args"]):
            raise AblateError(f"arm {name}: set --model explicitly (the entry points default to a paid model)")
        spec["arms"][str(name)] = arm
    if spec["report"]["baseline"] is None:
        spec["report"]["baseline"] = next(iter(spec["arms"]))
    if spec["report"]["baseline"] not in spec["arms"]:
        raise AblateError(f"report.baseline {spec['report']['baseline']!r} is not an arm")
    return spec


# what changes a result: everything but bookkeeping (budget, parallelism, report, names)
_BOOKKEEPING = ("name", "description", "budget", "max_parallel", "report", "arms", "interleave", "order_seed",
                "seeds", "timeout_s", "stop_grace_s")


def base_hash(spec: dict) -> str:
    return _sha({k: v for k, v in spec.items() if k not in _BOOKKEEPING})


def arm_hash(spec: dict, arm: str) -> str:
    a = {k: v for k, v in spec["arms"][arm].items() if k != "description"}
    return _sha({"base": base_hash(spec), "arm": a})


# ---------------------------------------------------------------- plan

def plan_runs(spec: dict) -> list[dict]:
    """Every arm x repeat (x input with per_input), in launch order."""
    arms = list(spec["arms"])
    groups = [[p] for p in spec["inputs"]] if spec["per_input"] else [spec["inputs"]]
    rng = random.Random(spec["order_seed"])
    order: list[tuple[int, str]] = []
    if spec["interleave"] == "sequential":
        order = [(r, a) for a in arms for r in range(len(spec["seeds"]))]
    else:
        for r in range(len(spec["seeds"])):
            blk = list(arms)
            if spec["interleave"] == "counterbalance" and r % 2 == 1:
                blk.reverse()
            elif spec["interleave"] == "random":
                rng.shuffle(blk)
            order += [(r, a) for a in blk]
    runs = []
    for r, a in order:
        for g in groups:
            rid = f"{a}__r{r:02d}" + (f"__{Path(g[0]).name}" if spec["per_input"] else "")
            runs.append({"id": rid, "arm": a, "repeat": r, "seed": spec["seeds"][r], "inputs": g})
    return runs


def models_of(arm: dict) -> list[str]:
    ms = [str(arm["flags"][f]) for f in MODEL_FLAGS if f in arm["flags"] and arm["flags"][f] not in (True,)]
    a = arm["args"]
    for i, x in enumerate(a):
        if x in MODEL_FLAGS and i + 1 < len(a):
            ms.append(a[i + 1])
    return ms


def is_stub(arm: dict) -> bool:
    ms = models_of(arm)
    return bool(ms) and all(m.startswith("stub:") for m in ms)


def estimate_run_usd(spec: dict, arm: str, n_inputs: int, done_costs: list[float] | None = None) -> float:
    """Expected $ of one run: completed runs of the arm if any, else budget.est_usd_per_run, else a
    rough per-call estimate (4k in / 3k out tokens per call; inputs x rounds x rollouts x steps x 1.5
    calls -- fixes, reflection -- and x 3 for graph_opt's variants)."""
    a = spec["arms"][arm]
    if is_stub(a):
        return 0.0
    if done_costs:
        return sum(done_costs) / len(done_costs)
    if spec["budget"].get("est_usd_per_run") is not None:
        return float(spec["budget"]["est_usd_per_run"])
    d = DEFAULT_STEPS[spec["entry"]]
    get = lambda f: float(a["flags"].get(f, d.get(f, 1)))  # noqa: E731
    calls = n_inputs * get("--rollouts") * get("--steps") * 1.5 * (get("--rounds") * 3 if spec["entry"] ==
                                                                   "graph_opt" else 1)
    p_in, p_out = price_for((a["flags"].get("--model") or ""), spec["budget"].get("prices"))
    return round(calls * (4000 * p_in + 3000 * p_out) / 1e6, 4)


def price_for(model: str, prices: dict | None = None) -> tuple[float, float]:
    if not model or str(model).startswith("stub:"):
        return 0.0, 0.0
    table = {**DEFAULT_PRICES, **(prices or {})}
    m = str(model).lower()
    for k in sorted(table, key=len, reverse=True):
        if k.lower() in m:
            return float(table[k][0]), float(table[k][1])
    return DEFAULT_PRICES["sonnet"][0], DEFAULT_PRICES["sonnet"][1]    # unknown: conservative


# ---------------------------------------------------------------- entry points

def entry_command(spec: dict, code: dict) -> tuple[list[str], dict]:
    """argv prefix and extra env to start the entry point (frozen code when frozen)."""
    py = spec["python"]
    env = {}
    if spec["entry"] == "multislot":
        script = spec["script"] or str(Path(code["kb_path"]) / "scripts" / "run_multislot.py")
        return [py, script], env
    env["PYTHONPATH"] = str(Path(code["mb_path"]).parent)
    env["MB_KB_ROOT"] = code["kb_path"]
    if spec["script"]:
        return [py, spec["script"]], env
    if Path(code["mb_path"]).name != "modelblaster":
        raise AblateError(f"graph_opt needs the ModelBlaster checkout in a dir named 'modelblaster' "
                          f"({code['mb_path']})")
    return [py, "-m", "modelblaster.pipeline.kb_graph_opt"], env


def detect_caps(prefix: list[str], env: dict) -> dict:
    """What the target supports, from its ``--help``."""
    txt = _run_quiet(prefix + ["--help"], env=env, timeout=300) or ""
    return {"resume": bool(re.search(r"(^|\s)--resume\b", txt)),
            "resume_implicit": "--more-rounds" in txt or "kb_resume" in txt,
            "trace": bool(re.search(r"(^|\s)--trace\b", txt)), "help_ok": bool(txt)}


def flags_argv(flags: dict) -> list[str]:
    out = []
    for k, v in flags.items():
        if v is True:
            out.append(k)
        elif isinstance(v, (list, tuple)):
            for e in v:
                out += [k, str(e)]
        else:
            out += [k, str(v)]
    return out


def _subst(s: str, ctx: dict) -> str:
    for k, v in ctx.items():
        s = s.replace("{" + k + "}", str(v))
    return os.path.expandvars(s)


def run_command(spec: dict, run: dict, rdir: Path, code: dict, caps: dict, *, resume: bool) -> list[str]:
    prefix, _ = entry_command(spec, code)
    arm = spec["arms"][run["arm"]]
    ctx = {"run_dir": rdir, "root": rdir.parent.parent, "arm": run["arm"], "seed": run["seed"],
           "repeat": run["repeat"]}
    flags = {k: ([_subst(str(e), ctx) for e in v] if isinstance(v, list) else
                 (_subst(v, ctx) if isinstance(v, str) else v)) for k, v in arm["flags"].items()}
    cmd = prefix + list(run["inputs"])
    if spec["entry"] == "multislot":
        cmd += ["--out", str(rdir / "out"), "--knowledge-dir", str(rdir / "knowledge")]
        if resume and caps.get("resume"):
            cmd.append("--resume")
    else:
        cmd += ["--work", str(rdir / "work"), "--kb", code["kb_path"], "--python", spec["python"]]
        if resume and caps.get("resume"):
            cmd.append("--resume")
    cmd += ["--seed", str(run["seed"])]
    if spec["gpu_model"]:
        cmd += ["--gpu-model", spec["gpu_model"]]
    if spec["trace"] and caps.get("trace") and "--trace" not in flags:
        cmd += ["--trace", str(spec["trace"])]
    cmd += flags_argv(flags) + [_subst(a, ctx) for a in arm["args"]]
    return cmd


def run_env(spec: dict, run: dict, rdir: Path, code: dict, parent: dict | None = None) -> tuple[dict, dict]:
    """The run's environment and its record (what was dropped / set, scrubbed)."""
    parent = dict(os.environ if parent is None else parent)
    arm = spec["arms"][run["arm"]]
    drop = set(DROP_ENV) | set(spec["drop_env"])
    env = {k: v for k, v in parent.items() if k not in drop}
    dropped = sorted(k for k in parent if k in drop)
    path = env.get("PATH", "")
    if spec["path_exclude"]:
        path = ":".join(p for p in path.split(":") if p and not re.search(spec["path_exclude"], p))
    if spec["path_prepend"]:
        path = ":".join(spec["path_prepend"] + [path])
    env["PATH"] = path
    _, extra = entry_command(spec, code)
    env.update(extra)
    env["TMPDIR"] = str(rdir / "tmp")
    env["KB_KNOWLEDGE_SEED_DIR"] = str(rdir / "seed")
    ctx = {"run_dir": rdir, "root": rdir.parent.parent, "arm": run["arm"], "seed": run["seed"],
           "repeat": run["repeat"]}
    for k, v in arm["env"].items():
        env[k] = _subst(v, ctx)
    changed = {k: v for k, v in env.items() if parent.get(k) != v}
    rec = {"set": scrub_env(changed), "dropped": dropped,
           "removed": sorted(k for k in parent if k not in env and k not in dropped)}
    return env, rec


# ---------------------------------------------------------------- isolation (seed, knowledge, cache)

def build_seed(root: Path, kcfg: dict, code: dict) -> tuple[Path, str]:
    """ROOT/seeds/<hash>: the knowledge seed dir (with or without mined evidence), built once."""
    src = Path(kcfg.get("seed_dir") or Path(code["kb_path"]) / "data" / "kernelblaster" / "knowledge")
    if not src.is_dir():
        raise AblateError(f"knowledge seed dir {src} does not exist")
    mined = kcfg.get("mined", True)
    key = _sha({"src": tree_hash(src), "mined": bool(mined)})
    dst = root / "seeds" / key
    if not dst.exists():
        tmp = dst.with_name(key + ".partial")
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(src, tmp)
        if not mined:
            (tmp / "mined.json").unlink(missing_ok=True)
        os.replace(tmp, dst)
    return dst, tree_hash(dst)


def build_snapshot(root: Path, spec: dict, kcfg: dict, seed_dir: Path, code: dict) -> Path | None:
    """The knowledge dir a run starts from: an explicit snapshot, or (prebuild: v2) a database built
    once from the seed with KB's ``scripts/seed_knowledge.py``; None = start empty."""
    if kcfg.get("snapshot"):
        p = Path(kcfg["snapshot"])
        if not p.is_dir():
            raise AblateError(f"knowledge snapshot {p} does not exist")
        return p
    if not kcfg.get("prebuild"):
        return None
    key = _sha({"seed": tree_hash(seed_dir), "kb": code.get("kb", {}).get("commit"), "prebuild": "v2"})
    dst = root / "snapshots" / key
    if not dst.exists():
        tmp = dst.with_name(key + ".partial")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        cmd = [spec["python"], str(Path(code["kb_path"]) / "scripts" / "seed_knowledge.py"), str(tmp),
               "--seed-dir", str(seed_dir)] + ([] if kcfg.get("mined", True) else ["--no-mined"])
        env = {k: v for k, v in os.environ.items() if k not in DROP_ENV}
        env["TMPDIR"] = str(root / "tmp")
        (root / "tmp").mkdir(parents=True, exist_ok=True)
        cp = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=code["kb_path"], timeout=900)
        if cp.returncode != 0:
            raise AblateError(f"seed_knowledge.py failed: {cp.stderr[-1500:]}")
        (tmp / "seed_report.json").write_text(cp.stdout)
        os.replace(tmp, dst)
    return dst


def prepare_run_dir(root: Path, spec: dict, run: dict, code: dict) -> dict:
    """A fresh run dir: seed copy, knowledge copy, MB cache (graph_opt), TMPDIR. -> isolation record."""
    rdir = root / "runs" / run["id"]
    if rdir.exists():
        shutil.rmtree(rdir)
    rdir.mkdir(parents=True)
    (rdir / "tmp").mkdir()
    arm = spec["arms"][run["arm"]]
    seed_src, seed_hash = build_seed(root, arm["knowledge"], code)
    shutil.copytree(seed_src, rdir / "seed")
    snap = build_snapshot(root, spec, arm["knowledge"], seed_src, code)
    kdir = rdir / ("knowledge" if spec["entry"] == "multislot" else "work/knowledge")
    if snap is not None:
        shutil.copytree(snap, kdir, ignore=shutil.ignore_patterns("seed_report.json"))
    iso = {"seed_dir": str(seed_src), "seed_hash": seed_hash, "mined": bool(arm["knowledge"].get("mined", True)),
           "knowledge_snapshot": str(snap) if snap else None, "knowledge_hash": tree_hash(snap) if snap else None,
           "knowledge_dir": str(kdir), "mb_cache": None, "cache_isolated": True}
    c = arm.get("mb_cache")
    if c:
        cdir = rdir / "work" / "cache"
        cdir.parent.mkdir(parents=True, exist_ok=True)
        if "snapshot" in c:
            shutil.copytree(c["snapshot"], cdir)
            iso["mb_cache"] = {"snapshot": c["snapshot"], "hash": tree_hash(Path(c["snapshot"]))}
        else:
            Path(c["shared"]).mkdir(parents=True, exist_ok=True)
            cdir.symlink_to(Path(c["shared"]).resolve(), target_is_directory=True)
            iso["mb_cache"] = {"shared": c["shared"]}
            iso["cache_isolated"] = False
    return iso


def input_fingerprints(inputs: list[str]) -> dict:
    out = {}
    for p in inputs:
        q = Path(p)
        f = next((q / n for n in ("manifest.json", "graph.json") if (q / n).exists()), None)
        out[q.name] = {"path": str(q), "file": f.name if f else None, "sha": tree_hash(f) if f else None}
    return out


_ENV_CACHE: dict = {}


def tool_versions(spec: dict, env: dict) -> dict:
    key = (spec["python"], env.get("PATH"))
    if key not in _ENV_CACHE:
        py = _run_quiet([spec["python"], "-c", "import sys; print(sys.version.split()[0])"], env=env)
        torch = _run_quiet([spec["python"], "-c", "import torch; print(torch.__version__, torch.version.cuda)"],
                           env=env, timeout=120)
        nvcc = _run_quiet(["nvcc", "--version"], env=env)
        smi = _run_quiet(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version", "--format=csv,noheader"])
        _ENV_CACHE[key] = {"python": py, "torch": torch, "nvcc": (nvcc or "").splitlines()[-1] if nvcc else None,
                           "host": socket.gethostname(),
                           "local_gpus": [dict(zip(("index", "name", "uuid", "driver"), [x.strip() for x in
                                                                                         l.split(",")]))
                                          for l in (smi or "").splitlines() if l.strip()]}
    return dict(_ENV_CACHE[key])


# ---------------------------------------------------------------- reading a run's outputs

def _num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def scan_telemetry(rdir: Path, prices: dict | None = None) -> dict | None:
    """LLM calls, tokens, $, truncations and wins from every telemetry.jsonl under the run dir (each
    LLM call once, replayed journal responses not counted). Per problem name as well."""
    files = [f for f in rdir.rglob("telemetry.jsonl") if "tmp" not in f.relative_to(rdir).parts[:1]]
    if not files:
        return None
    seen = set()
    tot = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0, "truncated": 0, "wins": 0,
           "errors": 0}
    per: dict = {}
    by_model: dict = {}
    for f in files:
        with open(f, errors="replace") as fh:
            for line in fh:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue                          # a torn last line
                t = e.get("t")
                if t not in ("llm", "count"):
                    continue
                key = (t, e.get("run"), e.get("span"), e.get("ts"), e.get("name"))
                if key in seen:
                    continue
                seen.add(key)
                prob = e.get("problem") or (e.get("attrs") or {}).get("problem")
                tgt = [tot] + ([per.setdefault(prob, {k: 0 for k in tot})] if prob else [])
                if t == "count":
                    if e.get("name") == "win":
                        for d in tgt:
                            d["wins"] += int(e.get("n", 1) or 1)
                    continue
                n = int(e.get("calls", 1) or 1)
                cost = _num(e.get("cost_usd")) or 0.0
                if not cost:
                    cost = _num(e.get("est_usd"))
                    if cost is None:
                        pi, po = price_for(e.get("model") or "", prices)
                        cost = ((e.get("input_tokens") or 0) * pi + (e.get("output_tokens") or 0) * po) / 1e6
                for d in tgt:
                    d["calls"] += n
                    d["input_tokens"] += int(e.get("input_tokens") or 0)
                    d["output_tokens"] += int(e.get("output_tokens") or 0)
                    d["usd"] += cost
                    d["truncated"] += int(e.get("truncated") or 0)
                    d["errors"] += int(e.get("errors") or 0)
                m = by_model.setdefault(e.get("model") or "?", {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                                                               "usd": 0.0})
                m["calls"] += n
                m["input_tokens"] += int(e.get("input_tokens") or 0)
                m["output_tokens"] += int(e.get("output_tokens") or 0)
                m["usd"] += cost
    tot["per_problem"] = per
    tot["by_model"] = by_model
    return tot


def _usage_usd(r: dict, prices: dict | None) -> float:
    by = r.get("llm_usage_by_model") or []
    if by:
        s = 0.0
        for m in by:
            c = _num(m.get("cost_usd")) or _num(m.get("est_usd"))
            if c is None:
                pi, po = price_for(m.get("model") or "", prices)
                c = ((m.get("prompt_tokens") or 0) * pi + (m.get("completion_tokens") or 0) * po) / 1e6
            s += c
        return s
    u = r.get("llm_usage") or {}
    if _num(u.get("cost_usd")):
        return float(u["cost_usd"])
    pi, po = price_for(r.get("model") or "", prices)
    return ((u.get("prompt_tokens") or 0) * pi + (u.get("completion_tokens") or 0) * po) / 1e6


def _gpu_name(g) -> str | None:
    if isinstance(g, dict):
        return g.get("name")
    return g if isinstance(g, str) else None


def _torch_keys(r: dict) -> float | None:
    for k in ("torch_us", "torch_e2e_us", "pytorch_us", "eager_torch_us"):
        if _num(r.get(k)):
            return float(r[k])
    return None


def collect_multislot(rdir: Path, prices: dict | None) -> dict:
    out = rdir / "out"
    summ = _read_json(out / "summary.json")
    rows: dict = {}
    entries = [e for e in summ if isinstance(e, dict) and "problem" in e] if isinstance(summ, list) else []
    results = {}
    if out.is_dir():
        for f in sorted(out.glob("*/result.json")):
            r = _read_json(f)
            if isinstance(r, dict) and r.get("problem"):
                results[r["problem"]] = r
    for e in entries:
        results.setdefault(e["problem"], e)
    gpus: dict = {}
    for name, r in results.items():
        tel = r.get("telemetry") or {}
        der = tel.get("derived") or tel
        e2e = _num(r.get("baseline_e2e_us")) and _num(r.get("final_e2e_us"))
        base = _num(r.get("baseline_e2e_us")) if e2e else _num(r.get("baseline_partition_us"))
        fin = _num(r.get("final_e2e_us")) if e2e else _num(r.get("final_partition_us"))
        u = r.get("llm_usage") or {}
        g = r.get("gpu")
        if _gpu_name(g):
            gpus.setdefault(_gpu_name(g), set()).add((g or {}).get("uuid") if isinstance(g, dict) else None)
        llm_t = (tel.get("llm") or {}).get("total") or (tel.get("llm") if "calls" in (tel.get("llm") or {}) else {})
        rows[name] = {"ok": bool(r.get("final_ok")) and fin is not None and "error" not in r,
                      "error": r.get("error"), "basis": "e2e" if e2e else "partition",
                      "base_us": base, "final_us": fin, "speedup": (base / fin) if base and fin else None,
                      "partition_speedup": _num(r.get("speedup")),
                      "torch_us": _torch_keys(r), "llm_calls": r.get("llm_calls"),
                      "input_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens"),
                      "usd": _usage_usd(r, prices), "wins": der.get("accepted_wins",
                                                                     (tel.get("counters") or {}).get("win")),
                      "truncated": llm_t.get("truncated"), "seconds": _num(r.get("seconds")),
                      "gpu": _gpu_name(g)}
    extra = {}
    if isinstance(summ, list):
        for e in summ:
            if isinstance(e, dict) and "reflection_usage" in e:
                extra["reflection_usage"] = e["reflection_usage"]
    return {"problems": rows, "complete": isinstance(summ, list),
            "gpu": {"models": sorted(gpus), "uuids": sorted({u for s in gpus.values() for u in s if u})},
            **extra}


def collect_graph_opt(rdir: Path, prices: dict | None, inputs: list | None = None) -> dict:
    w = rdir / "work"
    s = _read_json(w / "summary.json")
    # summary.json keys an IR by its graph name, or by its dir name when the final verification had
    # nothing to verify (no cache entries): report every IR under its graph name
    alias = {}
    for d in inputs or []:
        g = _read_json(Path(d) / "graph.json") or {}
        if g.get("name"):
            alias[Path(d).name] = g["name"]
    rows: dict = {}
    gpu = {"models": [], "uuids": []}
    if isinstance(s, dict):
        g = s.get("gpu") or {}
        models = set(g.get("models") or ([g["gpu_model"]] if g.get("gpu_model") else []))
        uuids = {st.get("uuid") for v in (g.get("steps") or {}).values() for st in (v or []) if isinstance(st, dict)
                 and st.get("uuid")}
        for ir, v in (s.get("irs") or {}).items():
            ir = alias.get(ir, ir)
            if v.get("model_gpu"):
                models.add(v["model_gpu"])
            model = _num(v.get("model_e2e_us")) and _num(v.get("seed_e2e_us"))
            base = _num(v.get("seed_e2e_us")) if model else _num(v.get("first_us"))
            fin = _num(v.get("model_e2e_us")) if model else _num(v.get("final_us"))
            rows[ir] = {"ok": fin is not None and bool(v.get("model_passed") if model else v.get("passed")),
                        "error": v.get("model_error"), "basis": "model" if model else "partition",
                        "base_us": base, "final_us": fin, "speedup": (base / fin) if base and fin else None,
                        "partition_speedup": _num(v.get("speedup")), "model_speedup": _num(v.get("model_speedup")),
                        "torch_us": _torch_keys(v), "hints_accepted": len(v.get("hints") or []),
                        "gpu": v.get("model_gpu") or v.get("verify_gpu")}
        gpu = {"models": sorted(m for m in models if m), "uuids": sorted(uuids)}
    # LLM usage per IR from the KernelBlaster runs' result.json files (problem names start with the IR's)
    for f in w.rglob("result.json") if w.is_dir() else []:
        r = _read_json(f)
        if not isinstance(r, dict) or not r.get("problem"):
            continue
        ir = next((k for k in sorted(rows, key=len, reverse=True)
                   if r["problem"] == k or r["problem"].startswith(k + "__")), None)
        if ir is None:
            continue
        d = rows[ir]
        u = r.get("llm_usage") or {}
        d["llm_calls"] = (d.get("llm_calls") or 0) + (r.get("llm_calls") or 0)
        d["input_tokens"] = (d.get("input_tokens") or 0) + (u.get("prompt_tokens") or 0)
        d["output_tokens"] = (d.get("output_tokens") or 0) + (u.get("completion_tokens") or 0)
        d["usd"] = (d.get("usd") or 0.0) + _usage_usd(r, prices)
    return {"problems": rows, "complete": isinstance(s, dict), "gpu": gpu,
            "rounds": (s or {}).get("rounds") if isinstance(s, dict) else None,
            "accepted": (s or {}).get("accepted") if isinstance(s, dict) else None}


def collect(rdir: Path, entry: str, prices: dict | None = None, inputs: list | None = None) -> dict:
    m = collect_multislot(rdir, prices) if entry == "multislot" else collect_graph_opt(rdir, prices, inputs)
    tel = scan_telemetry(rdir, prices)
    rows = m["problems"]
    if tel:
        tot = {k: tel[k] for k in ("calls", "input_tokens", "output_tokens", "usd", "truncated", "wins", "errors")}
        tot["source"] = "telemetry"
        tot["by_model"] = tel["by_model"]
        for name, d in rows.items():           # per-problem wins / truncations where the result lacks them
            pp = [v for k, v in tel["per_problem"].items() if k == name or k.startswith(name + "__")]
            if pp:
                if d.get("wins") is None:
                    d["wins"] = sum(v["wins"] for v in pp)
                if d.get("truncated") is None:
                    d["truncated"] = sum(v["truncated"] for v in pp)
                if d.get("llm_calls") is None:
                    d["llm_calls"] = sum(v["calls"] for v in pp)
    else:
        tot = {"calls": sum(d.get("llm_calls") or 0 for d in rows.values()),
               "input_tokens": sum(d.get("input_tokens") or 0 for d in rows.values()),
               "output_tokens": sum(d.get("output_tokens") or 0 for d in rows.values()),
               "usd": sum(d.get("usd") or 0.0 for d in rows.values()),
               "truncated": sum(d.get("truncated") or 0 for d in rows.values()),
               "wins": sum(d.get("wins") or 0 for d in rows.values()), "source": "results"}
        ru = m.get("reflection_usage") or {}
        if ru:
            tot["input_tokens"] += ru.get("input_tokens", 0)
            tot["output_tokens"] += ru.get("output_tokens", 0)
            tot["usd"] += _num(ru.get("cost_usd")) or 0.0
    m["totals"] = tot
    return m


# ---------------------------------------------------------------- state

class Ablation:
    """An ablation root: spec, frozen code, state of every run."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.state_path = self.root / "state.json"
        self._lock = None

    # -- persistence
    def load(self) -> dict:
        st = _read_json(self.state_path)
        if not st:
            raise AblateError(f"{self.root} is not an ablation root (no state.json)")
        return st

    def save(self, st: dict) -> None:
        _write_json(self.state_path, st)

    def lock(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = open(self.root / ".lock", "w")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            raise AblateError(f"another kb_ablate process runs in {self.root}") from e

    def spec(self) -> dict:
        return _read_json(self.root / "spec.json")


def _proc_start(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def _alive(pid: int | None, start: str | None) -> bool:
    return bool(pid) and _proc_start(pid) is not None and _proc_start(pid) == start


def init_or_resume(root: Path, spec: dict, spec_src: Path | None, *, force: bool) -> tuple[Ablation, dict]:
    ab = Ablation(root)
    ab.lock()
    st = _read_json(ab.state_path)
    if st:
        old = ab.spec()
        if base_hash(old) != base_hash(spec) and not force:
            raise AblateError(f"{root}: the spec's base configuration changed since the ablation started "
                              f"(base hash {base_hash(old)} -> {base_hash(spec)}); use a new --root, or --force "
                              f"(the report then refuses to compare runs of different configurations)")
        for a in spec["arms"]:
            if a in old["arms"] and arm_hash(old, a) != arm_hash(spec, a) and not force:
                raise AblateError(f"arm {a} changed since the ablation started; rename it or use --force")
        st["spec_history"] = st.get("spec_history", []) + ([base_hash(spec)] if base_hash(old) != base_hash(spec)
                                                            else [])
    else:
        st = {"format": STATE_FORMAT, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "runs": {}, "order": [],
              "spent_usd": 0.0}
    _write_json(ab.root / "spec.json", spec)
    if spec_src is not None:
        shutil.copyfile(spec_src, ab.root / ("spec.src" + Path(spec_src).suffix))
    # code: frozen once per ablation (later invocations reuse the copy), or the live checkouts
    if "code" not in st:
        if spec["freeze_code"]:
            kb = freeze_code(Path(spec["kb"]), ab.root / "code" / "kb")
            mb = (freeze_code(Path(spec["mb"]), ab.root / "code" / "mb" / "modelblaster")
                  if spec["entry"] == "graph_opt" else git_identity(Path(spec["mb"])))
            st["code"] = {"frozen": True, "kb": kb, "mb": mb, "kb_path": str(ab.root / "code" / "kb"),
                          "mb_path": str(ab.root / "code" / "mb" / "modelblaster") if spec["entry"] == "graph_opt"
                          else spec["mb"]}
        else:
            st["code"] = {"frozen": False, "kb": git_identity(Path(spec["kb"])), "mb": git_identity(Path(spec["mb"])),
                          "kb_path": spec["kb"], "mb_path": spec["mb"]}
    plan = plan_runs(spec)
    for r in plan:
        if r["id"] not in st["runs"]:
            st["runs"][r["id"]] = {**r, "status": "pending", "attempts": 0, "config_hash": arm_hash(spec, r["arm"])}
    st["order"] = [r["id"] for r in plan] + [i for i in st["order"] if i not in {r["id"] for r in plan}]
    st["base_hash"] = base_hash(spec)
    ab.save(st)
    return ab, st


def check_code(st: dict, *, force: bool) -> None:
    """Live (not frozen) checkouts must not have moved since the ablation started."""
    if st["code"]["frozen"]:
        return
    for k in ("kb", "mb"):
        now = git_identity(Path(st["code"][f"{k}_path"]))
        if not same_code(now, st["code"][k]):
            msg = (f"{k.upper()} checkout {st['code'][f'{k}_path']} changed since the ablation started "
                   f"({st['code'][k].get('commit')}/{st['code'][k].get('diff_sha')} -> "
                   f"{now.get('commit')}/{now.get('diff_sha')})")
            if not force:
                raise AblateError(msg + "; use --force to run on the new code (the report will refuse to compare)")
            print("warning: " + msg, file=sys.stderr)


# ---------------------------------------------------------------- run

def run_spend(st: dict) -> float:
    """$ of every run, including attempts that were restarted from scratch."""
    return sum(float(r.get("usd") or 0.0) + float(r.get("usd_lost") or 0.0) for r in st["runs"].values())


class Runner:
    def __init__(self, ab: Ablation, st: dict, spec: dict, *, max_parallel: int, budget_usd: float | None,
                 who: str, ledger: str | None, force: bool, poll_s: float | None = None):
        self.ab, self.st, self.spec = ab, st, spec
        self.max_parallel = max(1, max_parallel)
        self.budget = budget_usd
        self.who, self.ledger, self.force = who, ledger, force
        self.poll_s = poll_s if poll_s is not None else float(os.environ.get("MB_ABLATE_POLL_S", "0.5"))
        self.procs: dict = {}          # run id -> (Popen | None for adopted, log file)
        self.stopping = False
        self.caps = None

    def _caps(self) -> dict:
        if self.caps is None:
            key = self.spec["entry"] + ":" + (self.spec["script"] or "")
            cached = self.st.setdefault("caps", {})
            if key not in cached:
                prefix, extra = entry_command(self.spec, self.st["code"])
                env = {**{k: v for k, v in os.environ.items() if k not in DROP_ENV}, **extra}
                cached[key] = detect_caps(prefix, env)
                self.ab.save(self.st)
            self.caps = cached[key]
        return self.caps

    # -- budget
    def spent(self) -> float:
        return run_spend(self.st)

    def _estimate(self, run: dict) -> float:
        costs = [float(r["usd"]) for r in self.st["runs"].values() if r["arm"] == run["arm"]
                 and r["status"] == "done" and r.get("usd") is not None]
        return estimate_run_usd(self.spec, run["arm"], len(run["inputs"]), costs)

    def _would_exceed(self, run: dict) -> bool:
        if self.budget is None:
            return False
        inflight = sum(self._estimate(self.st["runs"][i]) for i in self.procs)
        return self.spent() + inflight + self._estimate(run) > self.budget + 1e-9

    # -- one run
    def launch(self, rid: str) -> None:
        run = self.st["runs"][rid]
        code = self.st["code"]
        caps = self._caps()
        rdir = self.ab.root / "runs" / rid
        resume = run["status"] in ("interrupted", "running") and rdir.exists()
        mode = "fresh"
        if resume and caps.get("resume"):
            mode = "resume-flag"
        elif resume and caps.get("resume_implicit"):
            mode = "resume-implicit"
        if mode == "fresh":
            if run["attempts"]:                       # an earlier attempt's spend stays counted
                run["usd_lost"] = round(run.get("usd_lost", 0.0) + float(run.get("usd") or 0.0), 6)
            run.update(usd=0.0, usd_ledgered=0.0, ledger_tokens=[0, 0, 0], wall_s_prev=0.0)
            iso = prepare_run_dir(self.ab.root, self.spec, run, code)
        else:
            iso = (_read_json(rdir / "manifest.json") or {}).get("isolation")
            (rdir / "tmp").mkdir(exist_ok=True)
        cmd = run_command(self.spec, run, rdir, code, caps, resume=(mode == "resume-flag"))
        env, envrec = run_env(self.spec, run, rdir, code)
        man = {"format": STATE_FORMAT, "ablation": self.spec["name"], "run": rid, "arm": run["arm"],
               "repeat": run["repeat"], "seed": run["seed"], "entry": self.spec["entry"],
               "command": scrub_argv(cmd), "cwd": str(rdir), "env": envrec,
               "code": ({"frozen": True, "mb": code["mb"], "kb": code["kb"]} if code["frozen"] else
                        {"frozen": False, "mb": git_identity(Path(code["mb_path"])),     # what runs now
                         "kb": git_identity(Path(code["kb_path"]))}),
               "versions": tool_versions(self.spec, env), "isolation": iso,
               "inputs": input_fingerprints(run["inputs"]), "spec_hash": self.st["base_hash"],
               "config_hash": run["config_hash"], "caps": caps, "mode": mode,
               "attempt": run["attempts"] + 1, "gpu_model_requested": self.spec["gpu_model"],
               "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
        prev = _read_json(rdir / "manifest.json")
        if prev and mode != "fresh":
            man["segments"] = prev.get("segments", []) + [{k: prev.get(k) for k in ("started", "finished", "rc",
                                                                                     "mode", "attempt")}]
        _write_json(rdir / "manifest.json", man)
        log = open(rdir / "run.log", "a")
        log.write(f"\n=== kb_ablate {mode} attempt {run['attempts'] + 1} {man['started']}\n")
        log.flush()
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(rdir),
                             start_new_session=True)
        run.update(status="running", attempts=run["attempts"] + 1, pid=p.pid, pid_start=_proc_start(p.pid),
                   t0=time.time(), mode=mode)
        run.pop("reason", None)
        self.procs[rid] = (p, log)
        self.ab.save(self.st)
        print(f"[{time.strftime('%H:%M:%S')}] start {rid} ({mode}, seed {run['seed']})", flush=True)

    def finish(self, rid: str, rc: int | None, *, interrupted: bool = False) -> None:
        run = self.st["runs"][rid]
        rdir = self.ab.root / "runs" / rid
        prices = self.spec["budget"].get("prices")
        m = collect(rdir, self.spec["entry"], prices, run["inputs"])
        _write_json(rdir / "metrics.json", m)
        wall = round(time.time() - run.get("t0", time.time()), 1)
        run["wall_s"] = round(run.get("wall_s_prev", 0.0) + wall, 1)
        run["wall_s_prev"] = run["wall_s"]
        prev_usd = run.get("usd_ledgered", 0.0)
        run["usd"] = round(m["totals"]["usd"], 6)
        run["rc"] = rc
        if interrupted or rc == 75 or (self.stopping and rc not in (0, None)):
            run["status"] = "interrupted"
        elif rc == 0 or (rc is None and m["complete"]):
            run["status"] = "done" if m["complete"] else "failed"
        else:
            run["status"] = "failed"
        man = _read_json(rdir / "manifest.json") or {}
        man.update(finished=time.strftime("%Y-%m-%dT%H:%M:%S"), rc=rc, status=run["status"], wall_s=run["wall_s"],
                   gpu_recorded=m["gpu"], usd=run["usd"])
        _write_json(rdir / "manifest.json", man)
        shutil.rmtree(rdir / "tmp", ignore_errors=True)
        self._ledger(run, m, prev_usd)
        self.ab.save(self.st)
        print(f"[{time.strftime('%H:%M:%S')}] {run['status']} {rid} rc={rc} ${run['usd']:.4f} {run['wall_s']}s "
              f"(spent ${self.spent():.4f}{'' if self.budget is None else f' of ${self.budget:.2f}'})", flush=True)

    def _ledger(self, run: dict, m: dict, prev_usd: float) -> None:
        """One ledger row per run segment with spend (the part not ledgered before)."""
        if not self.ledger:
            return
        t = m["totals"]
        usd = t["usd"] - prev_usd
        if usd <= 0 and (is_stub(self.spec["arms"][run["arm"]]) or not t["calls"]):
            return
        prev = run.get("ledger_tokens", [0, 0, 0])
        row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "who": self.who,
               "what": f"kb_ablate {self.spec['name']} {run['id']} (seed {run['seed']})",
               "model": ",".join(sorted(set(models_of(self.spec["arms"][run["arm"]])))),
               "calls": t["calls"] - prev[0], "prompt_tokens": t["input_tokens"] - prev[1],
               "completion_tokens": t["output_tokens"] - prev[2], "est_usd": round(usd, 4)}
        Path(self.ledger).parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(row) + "\n")
        run["usd_ledgered"] = t["usd"]
        run["ledger_tokens"] = [t["calls"], t["input_tokens"], t["output_tokens"]]

    # -- loop
    def _signal(self, signum, frame):  # noqa: ARG002
        if self.stopping:
            return
        self.stopping = True
        print(f"kb_ablate: signal {signum}: stopping the running runs (they resume on the next 'run')",
              file=sys.stderr, flush=True)
        for rid, (p, _) in self.procs.items():
            pid = p.pid if p else self.st["runs"][rid].get("pid")
            with contextlib.suppress(OSError):
                os.killpg(pid, signal.SIGTERM)

    def _poll(self) -> None:
        for rid in list(self.procs):
            p, log = self.procs[rid]
            run = self.st["runs"][rid]
            if p is not None:
                rc = p.poll()
                if rc is None and self.spec.get("timeout_s") and time.time() - run["t0"] > float(self.spec["timeout_s"]):
                    print(f"{rid}: timeout ({self.spec['timeout_s']} s), stopping it", file=sys.stderr)
                    with contextlib.suppress(OSError):
                        os.killpg(p.pid, signal.SIGTERM)
                    run["reason"] = "timeout"
                if rc is None:
                    continue
            else:
                if _alive(run.get("pid"), run.get("pid_start")):
                    continue
                rc = None
            if log:
                log.close()
            del self.procs[rid]
            self.finish(rid, rc)

    def run(self) -> int:
        order = self.st["order"]
        old = {s: signal.signal(s, self._signal) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
        try:
            # runs a killed runner left behind: still alive -> wait for them; dead -> resume / restart
            for rid in order:
                r = self.st["runs"][rid]
                if r["status"] == "running":
                    if _alive(r.get("pid"), r.get("pid_start")):
                        print(f"{rid}: still running (pid {r['pid']}) from an earlier kb_ablate; waiting for it")
                        self.procs[rid] = (None, None)
                    else:                         # killed with the runner: account, then resume it
                        self.finish(rid, None, interrupted=True)
            self.ab.save(self.st)
            budget_hit = False
            while True:
                self._poll()
                if self.stopping:
                    if not self.procs:
                        break
                    time.sleep(self.poll_s)
                    continue
                todo = [i for i in order if self.st["runs"][i]["status"] in ("pending", "interrupted")
                        and i not in self.procs]
                if not todo and not self.procs:
                    break
                if todo and len(self.procs) < self.max_parallel and not budget_hit:
                    nxt = todo[0]
                    if self._would_exceed(self.st["runs"][nxt]):
                        budget_hit = True
                        for i in todo:
                            self.st["runs"][i]["reason"] = "budget"
                        self.ab.save(self.st)
                        print(f"budget: ${self.spent():.4f} spent, next run est. "
                              f"${self._estimate(self.st['runs'][nxt]):.4f} would pass the cap ${self.budget:.2f}; "
                              f"not launching {len(todo)} run(s)", flush=True)
                        continue
                    check_code(self.st, force=self.force)
                    self.launch(nxt)
                    continue
                if budget_hit and not self.procs:
                    break
                time.sleep(self.poll_s)
            if self.stopping:
                grace = time.time() + self.spec["stop_grace_s"]
                while self.procs and time.time() < grace:
                    self._poll()
                    time.sleep(self.poll_s)
                for rid, (p, _) in list(self.procs.items()):
                    with contextlib.suppress(OSError):
                        os.killpg(p.pid if p else self.st["runs"][rid]["pid"], signal.SIGKILL)
                self._poll()
                return 75
            return 4 if budget_hit else (0 if all(self.st["runs"][i]["status"] == "done" for i in order) else 1)
        finally:
            for s, h in old.items():
                signal.signal(s, h)
            self.st["spent_usd"] = self.spent()
            self.ab.save(self.st)


def dry_run(spec: dict, budget_usd: float | None) -> dict:
    plan = plan_runs(spec)
    code = {"kb_path": spec["kb"], "mb_path": spec["mb"]}
    caps = {"resume": False, "trace": False}
    with contextlib.suppress(Exception):
        prefix, extra = entry_command(spec, code)
        caps = detect_caps(prefix, {**{k: v for k, v in os.environ.items() if k not in DROP_ENV}, **extra})
    rows, total = [], 0.0
    for r in plan:
        est = estimate_run_usd(spec, r["arm"], len(r["inputs"]))
        total += est
        rows.append({**r, "est_usd": est})
    print(f"ablation {spec['name']}: entry {spec['entry']}, {len(spec['arms'])} arm(s) x {len(spec['seeds'])} "
          f"repeat(s){' x ' + str(len(spec['inputs'])) + ' input(s)' if spec['per_input'] else ''} = {len(plan)} "
          f"run(s), interleave {spec['interleave']}, max parallel {spec['max_parallel']}")
    print(f"target supports: --resume {caps.get('resume')}, --trace {caps.get('trace')}")
    for a, arm in spec["arms"].items():
        k = arm["knowledge"]
        print(f"  arm {a}: models {models_of(arm)}, knowledge seed {'with' if k.get('mined', True) else 'without'} "
              f"mined{', snapshot ' + k['snapshot'] if k.get('snapshot') else ''}"
              f"{', prebuilt v2 DB' if k.get('prebuild') else ''}"
              f"{', MB cache ' + json.dumps(arm['mb_cache']) if arm['mb_cache'] else ''}")
        r0 = next(r for r in plan if r["arm"] == a)
        cmd = run_command(spec, r0, Path("<root>/runs") / r0["id"], code, caps, resume=False)
        print("    " + " ".join(scrub_argv(cmd)))
        if arm["env"]:
            print("    env: " + " ".join(f"{k}={v}" for k, v in scrub_env(arm["env"]).items()))
    print("order: " + " ".join(r["id"] for r in plan))
    print(f"estimated cost: ${total:.2f}" + (f" (cap ${budget_usd:.2f})" if budget_usd is not None else "")
          + ("  [rough: per-call estimate; set budget.est_usd_per_run]" if total and
             spec["budget"].get("est_usd_per_run") is None else ""))
    if budget_usd is not None and total > budget_usd:
        print("warning: the estimate exceeds the cap; the runner stops launching when the cap would be passed")
    return {"plan": rows, "est_usd": total, "caps": caps}


# ---------------------------------------------------------------- statistics

def _mean(x):
    return sum(x) / len(x) if x else float("nan")


def geomean(x):
    x = [v for v in x if v and v > 0]
    return math.exp(_mean([math.log(v) for v in x])) if x else None


def bootstrap_ci(logs: list[float], n: int = 10000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile CI of exp(mean log ratio), resampling the pairs."""
    if len(logs) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    k = len(logs)
    ms = sorted(sum(logs[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    lo, hi = ms[int(alpha / 2 * n)], ms[min(n - 1, int((1 - alpha / 2) * n))]
    return math.exp(lo), math.exp(hi)


def sign_flip_p(logs: list[float], n: int = 20000, seed: int = 0) -> float:
    """Two-sided paired permutation test of mean(log ratio) = 0 (sign flips; exact up to 2^16)."""
    k = len(logs)
    if k == 0:
        return float("nan")
    obs = abs(sum(logs))
    eps = 1e-12 * max(1.0, obs)
    if k <= 16:
        hit = 0
        for mask in range(1 << k):
            s = sum(-v if mask >> i & 1 else v for i, v in enumerate(logs))
            hit += abs(s) >= obs - eps
        return hit / (1 << k)
    rng = random.Random(seed)
    hit = sum(abs(sum(v if rng.random() < 0.5 else -v for v in logs)) >= obs - eps for _ in range(n))
    return (hit + 1) / (n + 1)


def wilcoxon_p(d: list[float]) -> float:
    """Two-sided Wilcoxon signed-rank p (zeros dropped; average ranks for ties; exact null for n <= 25,
    else the normal approximation with tie correction)."""
    d = [v for v in d if abs(v) > 1e-12]
    n = len(d)
    if n == 0:
        return 1.0
    a = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    ties = []
    while i < n:
        j = i
        while j + 1 < n and abs(abs(d[a[j + 1]]) - abs(d[a[i]])) < 1e-12:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[a[k]] = r
        ties.append(j - i + 1)
        i = j + 1
    wp = sum(r for r, v in zip(ranks, d) if v > 0)
    if n <= 25:
        r2 = [int(round(2 * r)) for r in ranks]
        tot = sum(r2)
        dist = [0] * (tot + 1)
        dist[0] = 1
        for r in r2:
            for s in range(tot, r - 1, -1):
                dist[s] += dist[s - r]
        m = 1 << n
        w2 = int(round(2 * wp))
        lo = sum(dist[: w2 + 1]) / m
        hi = sum(dist[w2:]) / m
        return min(1.0, 2 * min(lo, hi))
    mu = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24 - sum(t ** 3 - t for t in ties) / 48
    z = (abs(wp - mu) - 0.5) / math.sqrt(var) if var > 0 else 0.0
    return min(1.0, math.erfc(z / math.sqrt(2)))


def paired(rows_a: dict, rows_b: dict, key: str, *, nboot: int, nperm: int) -> dict:
    """B vs A on the (problem, repeat) pairs both completed: ratio = B / A of ``key``."""
    keys = sorted(set(rows_a) & set(rows_b))
    logs, used, dropped = [], [], []
    for k in keys:
        a, b = rows_a[k], rows_b[k]
        va, vb = a.get(key), b.get(key)
        if a.get("ok") and b.get("ok") and va and vb and va > 0 and vb > 0:
            logs.append(math.log(vb / va))
            used.append(k)
        else:
            dropped.append(k)
    unmatched = sorted(set(rows_a) ^ set(rows_b))
    if not logs:
        return {"n": 0, "dropped": len(dropped), "unmatched": len(unmatched)}
    lo, hi = bootstrap_ci(logs, nboot)
    return {"n": len(logs), "ratio": math.exp(_mean(logs)), "ci95": [lo, hi], "p_perm": sign_flip_p(logs, nperm),
            "p_wilcoxon": wilcoxon_p(logs), "b_better": sum(v < 0 for v in logs) if key == "final_us"
            else sum(v > 0 for v in logs), "dropped": len(dropped), "unmatched": len(unmatched),
            "problems": len({k[0] for k in used})}


# ---------------------------------------------------------------- report

def comparability(runs: list[dict], manifests: dict) -> list[str]:
    """Why runs may not be compared (empty: they may)."""
    issues = []
    mans = [manifests[r["id"]] for r in runs if manifests.get(r["id"])]
    if not mans:
        return issues
    for k in ("kb", "mb"):
        ids = {(m["code"][k].get("commit"), m["code"][k].get("diff_sha"), m["code"][k].get("tree")) for m in mans}
        if len(ids) > 1:
            issues.append(f"runs used different {k.upper()} code: {sorted(map(str, ids))}")
    if len({m.get("spec_hash") for m in mans}) > 1:
        issues.append(f"runs used different base configurations (spec hashes {sorted({m.get('spec_hash') for m in mans})})")
    by_arm: dict = {}
    for r in runs:
        m = manifests.get(r["id"])
        if m:
            by_arm.setdefault(r["arm"], set()).add(m.get("config_hash"))
    for a, hs in by_arm.items():
        if len(hs) > 1:
            issues.append(f"arm {a}: runs with different arm configurations ({sorted(hs)})")
    models = {g for m in mans for g in ((m.get("gpu_recorded") or {}).get("models") or [])}
    if len(models) > 1:
        issues.append(f"runs were timed on different GPU models: {sorted(models)}")
    return issues


def build_report(root: Path, *, force: bool = False) -> dict:
    ab = Ablation(root)
    st = ab.load()
    spec = ab.spec()
    prices = spec["budget"].get("prices")
    torch_base = _read_json(Path(spec["torch_baseline"])) if spec.get("torch_baseline") else None
    runs, manifests, rows = [], {}, []
    for rid in st["order"]:
        r = st["runs"][rid]
        rdir = ab.root / "runs" / rid
        if r["status"] not in ("done", "failed") or not rdir.exists():
            continue
        m = collect(rdir, spec["entry"], prices, r["inputs"])
        manifests[rid] = _read_json(rdir / "manifest.json") or {}
        manifests[rid].setdefault("gpu_recorded", m["gpu"])
        runs.append({**r, "metrics": m})
        for p, d in m["problems"].items():
            t = d.get("torch_us") or (torch_base or {}).get(p)
            rows.append({"arm": r["arm"], "repeat": r["repeat"], "seed": r["seed"], "run": rid, "problem": p,
                         "run_status": r["status"], **{k: v for k, v in d.items()},
                         "vs_torch": (t / d["final_us"]) if t and d.get("final_us") else None})
    arms = list(spec["arms"])
    issues = comparability(runs, manifests)
    if not any((manifests.get(r["id"]) or {}).get("gpu_recorded", {}).get("models") for r in runs):
        issues_soft = ["no run recorded the GPU it was timed on"]
    else:
        issues_soft = []
    per_arm = {}
    for a in arms:
        ar = [r for r in runs if r["arm"] == a]
        pr = [x for x in rows if x["arm"] == a]
        ok = [x for x in pr if x.get("ok")]
        t = [r["metrics"]["totals"] for r in ar]
        usd = sum(x["usd"] for x in t)
        wins = sum(x["wins"] or 0 for x in t)
        calls = sum(x["calls"] for x in t)
        trunc = sum(x["truncated"] or 0 for x in t)
        all_r = [st["runs"][i] for i in st["order"] if st["runs"][i]["arm"] == a]
        per_arm[a] = {"runs": len(ar), "runs_planned": len(all_r),
                      "runs_failed": sum(r["status"] == "failed" for r in ar),
                      "runs_pending": sum(r["status"] not in ("done", "failed") for r in all_r),
                      "results": len(pr), "results_ok": len(ok), "results_failed": len(pr) - len(ok),
                      "geomean_speedup": geomean([x["speedup"] for x in ok]),
                      "geomean_final_us": geomean([x["final_us"] for x in ok]),
                      "geomean_vs_torch": geomean([x["vs_torch"] for x in ok if x.get("vs_torch")]),
                      "improved": sum(1 for x in ok if x["speedup"] and x["speedup"] > 1.01),
                      "wins": wins, "llm_calls": calls,
                      "input_tokens": sum(x["input_tokens"] for x in t),
                      "output_tokens": sum(x["output_tokens"] for x in t), "usd": usd,
                      "usd_per_win": (usd / wins) if wins else None,
                      "wall_s_mean": _mean([r.get("wall_s") or 0 for r in ar]) if ar else None,
                      "truncation_rate": (trunc / calls) if calls else None,
                      "gpus": sorted({g for r in ar for g in r["metrics"]["gpu"]["models"]})}
    problems = sorted({x["problem"] for x in rows})
    per_problem = {}
    for p in problems:
        per_problem[p] = {}
        for a in arms:
            xs = sorted((x for x in rows if x["problem"] == p and x["arm"] == a), key=lambda x: x["repeat"])
            ok = [x for x in xs if x.get("ok")]
            per_problem[p][a] = {"n": len(xs), "ok": len(ok), "geomean_final_us": geomean([x["final_us"] for x in ok]),
                                 "geomean_speedup": geomean([x["speedup"] for x in ok]),
                                 "final_us": [x["final_us"] if x.get("ok") else None for x in xs],
                                 "speedup": [x["speedup"] if x.get("ok") else None for x in xs],
                                 "wins": sum(x.get("wins") or 0 for x in xs)}
    comps = []
    refused = bool(issues) and not force
    if not refused:
        base = spec["report"]["baseline"]
        pairs = ([(base, b) for b in arms if b != base] if spec["report"].get("pairs", "baseline") == "baseline"
                 else [(arms[i], arms[j]) for i in range(len(arms)) for j in range(i + 1, len(arms))])
        idx = lambda a: {(x["problem"], x["repeat"]): x for x in rows if x["arm"] == a}  # noqa: E731
        for a, b in pairs:
            ra, rb = idx(a), idx(b)
            comps.append({"a": a, "b": b,
                          "final_us": paired(ra, rb, "final_us", nboot=int(spec["report"]["bootstrap"]),
                                             nperm=int(spec["report"]["permutations"])),
                          "speedup": paired(ra, rb, "speedup", nboot=int(spec["report"]["bootstrap"]),
                                            nperm=int(spec["report"]["permutations"]))})
    rep = {"ablation": spec["name"], "description": spec.get("description", ""), "root": str(ab.root),
           "entry": spec["entry"], "arms": {a: {"description": spec["arms"][a].get("description", ""),
                                                "flags": spec["arms"][a]["flags"],
                                                "env": scrub_env(spec["arms"][a]["env"]),
                                                "knowledge": spec["arms"][a]["knowledge"],
                                                "mb_cache": spec["arms"][a].get("mb_cache")} for a in arms},
           "code": {k: st["code"][k] for k in ("frozen", "kb", "mb")}, "gpu_model": spec.get("gpu_model"),
           "issues": issues, "warnings": issues_soft, "refused": refused, "forced": bool(issues) and force,
           "per_arm": per_arm, "per_problem": per_problem, "comparisons": comps, "rows": rows,
           "spent_usd": run_spend(st),
           "status": {s: sum(r["status"] == s for r in st["runs"].values())
                      for s in ("done", "failed", "running", "interrupted", "pending")}}
    return rep


def _f(x, nd=3, pct=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    if pct:
        return f"{100 * x:.1f}%"
    if isinstance(x, float):
        return f"{x:.{nd}f}" if abs(x) < 1e5 else f"{x:.3g}"
    return str(x)


def _p(x):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else (f"{x:.2g}" if x < 0.01 else f"{x:.3f}")


def render_markdown(rep: dict) -> str:
    L = [f"# Ablation `{rep['ablation']}`", ""]
    if rep["description"]:
        L += [rep["description"], ""]
    c = rep["code"]
    L += [f"Entry `{rep['entry']}`; root `{rep['root']}`; KB {c['kb'].get('commit', '?')[:10]}"
          f"{' (dirty ' + str(c['kb'].get('diff_sha')) + ')' if c['kb'].get('dirty') else ''}, "
          f"MB {str(c['mb'].get('commit'))[:10]}{' (dirty ' + str(c['mb'].get('diff_sha')) + ')' if c['mb'].get('dirty') else ''}"
          f"{', frozen copies' if c['frozen'] else ''}; GPU model requested: {rep['gpu_model'] or '(not pinned)'}; "
          f"runs: " + ", ".join(f"{k} {v}" for k, v in rep["status"].items() if v) +
          f"; spent ${rep['spent_usd']:.4f}.", ""]
    if rep["issues"]:
        L += ["**Comparability problems" + (" (comparisons forced)" if rep["forced"] else
                                            " -- paired comparisons refused (`report --force` to override)") + ":**", ""]
        L += [f"- {i}" for i in rep["issues"]] + [""]
    for w in rep["warnings"]:
        L += [f"Note: {w}.", ""]
    L += ["## Arms", "", "| arm | flags | env | knowledge seed | snapshot | MB cache |", "|---|---|---|---|---|---|"]
    for a, d in rep["arms"].items():
        k = d["knowledge"]
        L.append(f"| {a} | `{' '.join(flags_argv(d['flags']))}` | {' '.join(f'{x}={y}' for x, y in d['env'].items()) or '-'} "
                 f"| {'with' if k.get('mined', True) else 'without'} mined{' (' + k['seed_dir'] + ')' if k.get('seed_dir') else ''} "
                 f"| {k.get('snapshot') or ('prebuilt v2' if k.get('prebuild') else 'empty')} | "
                 f"{json.dumps(d['mb_cache']) if d['mb_cache'] else 'fresh per run'} |")
    L += ["", "## Per arm", "",
          "| arm | runs (failed) | results ok/all | geo-mean speedup | geo-mean final us | vs torch | improved | wins "
          "| LLM calls | tokens in / out | $ | $/win | wall s/run | truncation | GPUs |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a, d in rep["per_arm"].items():
        L.append(f"| {a} | {d['runs']}/{d['runs_planned']} ({d['runs_failed']}) | {d['results_ok']}/{d['results']} | "
                 f"{_f(d['geomean_speedup'])} | {_f(d['geomean_final_us'], 1)} | {_f(d['geomean_vs_torch'])} | "
                 f"{d['improved']} | {d['wins']} | {d['llm_calls']} | {d['input_tokens']} / {d['output_tokens']} | "
                 f"{d['usd']:.4f} | {_f(d['usd_per_win'], 4)} | {_f(d['wall_s_mean'], 0)} | "
                 f"{_f(d['truncation_rate'], pct=True)} | {', '.join(d['gpus']) or '-'} |")
    arms = list(rep["arms"])
    L += ["", "## Per problem (geo-mean over repeats [per repeat])", "",
          "Final time, us (speedup vs the run's seed):", "",
          "| problem | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for p, d in rep["per_problem"].items():
        cells = []
        for a in arms:
            x = d.get(a) or {}
            per = ", ".join(_f(v, 1) for v in x.get("final_us", []))
            cells.append(f"{_f(x.get('geomean_final_us'), 1)} ({_f(x.get('geomean_speedup'))}x) [{per}]"
                         if x.get("n") else "-")
        L.append(f"| {p} | " + " | ".join(cells) + " |")
    L += ["", "## Paired comparisons", ""]
    if rep["refused"]:
        L += ["Refused: see the comparability problems above.", ""]
    elif not rep["comparisons"]:
        L += ["(one arm: nothing to compare)", ""]
    else:
        L += ["Pairs: the same problem and repeat (seed) in both arms, both completed. Final-time ratio B/A < 1 "
              "and speedup ratio B/A > 1 favour B. 95 % CI: bootstrap over the pairs; p: two-sided paired "
              "sign-flip permutation test on the log ratios, and the Wilcoxon signed-rank test.", "",
              "| B vs A | metric | n pairs (problems) | geo-mean ratio B/A | 95 % CI | p (perm) | p (Wilcoxon) | "
              "B better | dropped / unmatched |", "|---|---|---|---|---|---|---|---|---|"]
        for c in rep["comparisons"]:
            for k, lab in (("final_us", "final time"), ("speedup", "speedup")):
                s = c[k]
                if not s.get("n"):
                    L.append(f"| {c['b']} vs {c['a']} | {lab} | 0 | - | - | - | - | - | {s.get('dropped')} / {s.get('unmatched')} |")
                    continue
                L.append(f"| {c['b']} vs {c['a']} | {lab} | {s['n']} ({s['problems']}) | {s['ratio']:.3f} | "
                         f"{_f(s['ci95'][0])}-{_f(s['ci95'][1])} | {_p(s['p_perm'])} | {_p(s['p_wilcoxon'])} | "
                         f"{s['b_better']}/{s['n']} | {s['dropped']} / {s['unmatched']} |")
        small = [c for c in rep["comparisons"] if 0 < c["final_us"].get("n", 0) < 6]
        if small:
            L += ["", "n < 6 pairs: the smallest attainable two-sided p is 2/2^n; treat these as pilot numbers."]
    L.append("")
    return "\n".join(L)


ROW_FIELDS = ("arm", "repeat", "seed", "problem", "run", "run_status", "ok", "basis", "base_us", "final_us",
              "speedup", "partition_speedup", "model_speedup", "vs_torch", "wins", "llm_calls", "input_tokens",
              "output_tokens", "usd", "truncated", "seconds", "gpu", "error")


def write_report(root: Path, *, force: bool = False) -> dict:
    rep = build_report(root, force=force)
    root = Path(root)
    _write_json(root / "report.json", rep)
    (root / "report.md").write_text(render_markdown(rep))
    with open(root / "report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rep["rows"]:
            w.writerow(r)
    return rep


def status(root: Path) -> str:
    ab = Ablation(root)
    st = ab.load()
    spec = ab.spec()
    L = [f"ablation {spec['name']} ({spec['entry']}) at {ab.root}",
         f"{'run':40s} {'status':12s} {'att':>3s} {'rc':>4s} {'$':>9s} {'wall s':>8s}  note"]
    for rid in st["order"]:
        r = st["runs"][rid]
        alive = r["status"] == "running" and _alive(r.get("pid"), r.get("pid_start"))
        stt = r["status"] + ("" if r["status"] != "running" else (" (alive)" if alive else " (dead)"))
        L.append(f"{rid:40s} {stt:12s} {r['attempts']:3d} {str(r.get('rc', '-')):>4s} "
                 f"{(r.get('usd') or 0):9.4f} {str(r.get('wall_s', '-')):>8s}  {r.get('reason', '')}")
    spent = run_spend(st)
    L.append(f"spent ${spent:.4f}" + (f" of ${spec['budget']['usd']:.2f}" if spec["budget"].get("usd") else ""))
    return "\n".join(L)


# ---------------------------------------------------------------- CLI

def default_root(spec: dict) -> Path:
    base = Path(os.environ.get("MB_ABLATE_ROOT") or Path.cwd() / "ablations")
    return base / spec["name"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run (or continue) an ablation")
    r.add_argument("spec")
    r.add_argument("--root", help="ablation root (default $MB_ABLATE_ROOT/<name>, else ./ablations/<name>)")
    r.add_argument("--dry-run", action="store_true", help="print the plan, commands and estimated cost only")
    r.add_argument("--max-parallel", type=int, help="runs at a time (default: the spec's max_parallel)")
    r.add_argument("--budget-usd", type=float, help="cap for this ablation (default: the spec's budget.usd)")
    r.add_argument("--who", help="ledger 'who' (default: the spec's budget.who, else 'ablate')")
    r.add_argument("--ledger", help="spend ledger JSONL to append to (default: the spec's budget.ledger)")
    r.add_argument("--force", action="store_true", help="continue despite a changed spec / moved checkout")
    r.add_argument("--retry-failed", action="store_true", help="run failed runs again (from scratch)")
    r.add_argument("--no-report", action="store_true")
    s = sub.add_parser("status", help="state of every run")
    s.add_argument("root")
    p = sub.add_parser("report", help="(re-)render report.md / report.json / report.csv from the runs")
    p.add_argument("root")
    p.add_argument("--force", action="store_true", help="compare despite different code / configs / GPU models")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "status":
            print(status(Path(a.root)))
            return 0
        if a.cmd == "report":
            rep = write_report(Path(a.root), force=a.force)
            print(render_markdown(rep))
            return 3 if rep["refused"] else 0
        spec_path = Path(a.spec).resolve()
        spec = normalize_spec(load_spec_file(spec_path), spec_path.parent)
        budget = a.budget_usd if a.budget_usd is not None else spec["budget"].get("usd")
        if a.max_parallel:
            spec["max_parallel"] = a.max_parallel
        if a.dry_run:
            dry_run(spec, budget)
            return 0
        root = Path(a.root) if a.root else default_root(spec)
        ab, st = init_or_resume(root, spec, spec_path, force=a.force)
        if a.retry_failed:
            for x in st["runs"].values():
                if x["status"] == "failed":
                    x["status"] = "pending"
            ab.save(st)
        runner = Runner(ab, st, spec, max_parallel=spec["max_parallel"], budget_usd=budget,
                        who=a.who or spec["budget"].get("who") or "ablate",
                        ledger=a.ledger or spec["budget"].get("ledger"), force=a.force)
        rc = runner.run()
        if not a.no_report:
            rep = write_report(ab.root)
            print(f"report: {ab.root / 'report.md'}" + (" (comparisons refused: see the report)" if rep["refused"]
                                                       else ""))
        return rc
    except AblateError as e:
        print(f"kb_ablate: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
