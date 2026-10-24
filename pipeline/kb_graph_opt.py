"""ModelBlaster as the graph optimiser: KernelBlaster rounds to a fixpoint.

One round, for a set of IRs sharing a work dir (hints, kernel cache and
KernelBlaster knowledge carry over between rounds and runs):

1. **profile** every IR in its current state -- accepted graph hints applied,
   cached winners in place (``kb_plan.profile_ir``);
2. **MB proposals** from the whole-model profiles
   (``graph_hints.propose_from_profile``: elementwise consumers worth fusing,
   including consumers of already-fused ops, so chains grow);
3. **plan** -- rank op instances by (entry, signature, shape) across the IRs,
   select under the budget, skipping items sent in earlier rounds
   (``kb_plan``); export the problems seeded from the cache, run
   KernelBlaster on them, ingest its winners; its ``proposals.json`` (fuse,
   split) become hints;
4. **decide** -- for the undecided hints with the largest estimated gain
   (at most ``max_variants``): export the affected ops as an *original*
   problem and, with the hint applied, as a *variant* problem, both seeded
   with the same current-best kernels; KernelBlaster optimises both with the
   same budget; both composite bests are rebuilt and re-timed back to back
   on one GPU (``kb_proposals.remeasure``); the hint is accepted if the
   variant is ``accept_gain`` faster. Accepted hints apply from the next
   round on, and the variant's new slots (the fused kernel, the split
   variants) go into the cache. Hints that cannot affect each other
   (``footprint`` / ``conflicts``) are decided together in *waves*
   (``schedule``): one KernelBlaster run for all their problems, re-timing
   as soon as a hint's two problems finish; conflicting hints wait for the
   wave before them, so the outcome is that of deciding them one by one in
   rank order (``--parallel-hints 1``);
5. stop when a round accepts nothing and has nothing new to send (the
   fixpoint), or after ``rounds``.

Finally every IR is verified at model level with everything applied
(``kb_ingest.verify_ir``), the model's result is timed through KernelBlaster's
evaluator on the runs' pool and GPU model (``kb_model_run``), and
``summary.json`` lists first vs final times.

    python -m modelblaster.pipeline.kb_graph_opt IR_DIR [...] --work W --kb KB_CHECKOUT \\
        [--model claude-code:sonnet] [--plugin stubs.py] [--rounds 4] \\
        [--coverage 0.8 --top-k 6] [--rollouts 2 --steps 4 --variant-steps 3] \\
        [--max-variants 3 --remeasure 5 --accept-gain 0.03] [--arch 75] \\
        [--parallel-hints auto|N] [--gpus auto|all|0,2] \\
        [--ssh-host H [--ssh-host H2] --cuda-visible-devices auto|ids [--ssh-key K]] \\
        [--runtime auto|host|process] [--gpu-model M] [--allow-gpu-mismatch]

Work dir: ``graph_hints.json`` (modelblaster.graph_hints/v1), ``cache/``
(mb-cuda-cache/1), ``knowledge/`` (KernelBlaster), ``state.json``,
``rNN/{profile,plan.json,problems,run,winners,variants,round.json}``,
``summary.json``. One process at a time owns a work dir (``.lock``).
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import graph_hints as gh
from . import kb_telemetry as kbt
from .kb_export import base_slot_id, export_problem
from .kb_ingest import CudaCache, cache_plan, ingest, verify_ir
from .kb_plan import export_plan, make_plan, profile_ir
from .kb_proposals import (ACCEPT_GAIN, _all_specs, _median, best_sources, find_chains, remeasure,
                           with_fused_specs, write_variant_ir)
from .kernel_abi import parse_signature


@dataclass
class Config:
    kb: Path
    model: str = "claude-code:sonnet"
    reflect_model: str | None = None
    plugins: list = field(default_factory=list)
    python: str = sys.executable
    arch: str = "75"
    rounds: int = 4
    rollouts: int = 2
    steps: int = 4
    variant_steps: int | None = None
    max_fix: int = 2
    coverage: float = 0.8
    top_k: int | None = 6
    min_share: float = 0.01
    min_gain_pct: float = 2.0
    max_variants: int = 3
    remeasure: int = 5
    accept_gain: float = ACCEPT_GAIN
    profile_runs: int = 3
    seed: int = 0
    # hints decided at once (0 = auto: max(4, 2 x GPUs in the pool)); 1 = one by one
    parallel_hints: int = 0
    # KernelBlaster's GPU pool (run_multislot --gpus / --ssh-host / --cuda-visible-devices)
    gpus: str = "auto"
    ssh_hosts: list = field(default_factory=list)
    cuda_visible_devices: list = field(default_factory=list)
    ssh_key: str | None = None
    remote_dir: str | None = None
    # KernelBlaster's runtime (run_multislot / time_problems --runtime): auto | host | process
    runtime: str = "auto"
    # the GPU model every timing must run on (default: the model the KernelBlaster runs recorded);
    # allow_gpu_mismatch: time anyway where it is not available (loud warnings)
    gpu_model: str | None = None
    allow_gpu_mismatch: bool = False


_LIST_FIELDS = ("plugins", "ssh_hosts", "cuda_visible_devices")
_MANUAL_FIELDS = ("kb", "model", "reflect_model", "python", "ssh_key", "remote_dir", "gpus", "parallel_hints",
                  "runtime", "gpu_model", "allow_gpu_mismatch", *_LIST_FIELDS)


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:80]


# ---------------------------------------------------------------- locks

def _flock_off() -> bool:
    return os.getenv("KERNELBLASTER_GPU_FLOCK", "1").lower() in ("0", "false", "no", "off")


def gpu_lock_path(ident: str) -> Path:
    """KernelBlaster's per-GPU lock file (``resources.gpu_lock_wrap``): every
    KernelBlaster timed run holds an flock on it, so MB's own timed runs
    holding it too take turns with KernelBlaster runs of any process."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", ident)
    return Path(os.getenv("KERNELBLASTER_GPU_LOCK_DIR", "/dev/shm")) / f"kernelblaster_gpu_{safe}.lock"


@contextlib.contextmanager
def _flock(path: Path, *, blocking: bool = True, timeout: float = 3600.0):
    """flock(2) on ``path`` (created if missing). Blocking waits up to
    ``timeout`` and then raises TimeoutError; non-blocking raises
    BlockingIOError when the lock is held."""
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o666)
    try:
        if not blocking:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            t_end = time.monotonic() + timeout
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() > t_end:
                        raise TimeoutError(f"lock {path} held for more than {timeout:.0f} s")
                    time.sleep(0.05)
        yield fd
    finally:
        os.close(fd)                                  # closing releases the lock


@contextlib.contextmanager
def work_lock(work: Path):
    """One kb_graph_opt process per work dir: it is the only writer of the
    hints file, the cache index, state.json and (through the one
    KernelBlaster process it runs at a time) the knowledge dir."""
    work.mkdir(parents=True, exist_ok=True)
    try:
        with _flock(work / ".lock", blocking=False):
            yield
    except BlockingIOError:
        raise RuntimeError(f"{work} is in use by another kb_graph_opt run (lock {work / '.lock'})") from None


class GpuLeases:
    """Exclusive leases on this host's GPUs for ModelBlaster's own timed runs
    (the re-measurement that decides a hint): one lease per device in this
    process, least recently used first, and while leased the device's
    KernelBlaster lock file is held, so KernelBlaster runs (this flow's or
    another process's) wait instead of timing on it at the same time.

    ``devices``: [{"ident": UUID or index, "name": ...}]; empty = one
    unpinned device (the environment's default GPU, no lock file). A device
    with a ``host`` (``--ssh-host``) is leased in this process only: the
    KernelBlaster evaluator that times on it takes that host's GPU lock per
    run (``lease_device``)."""

    def __init__(self, devices: list[dict] | None = None, *, lock_timeout: float = 3600.0):
        self.devices = list(devices) or [{"ident": None, "name": "default"}]
        self.lock_timeout = lock_timeout
        self._cv = threading.Condition()
        self._busy: set[int] = set()
        self._last: dict[int, float] = {}
        self.log: list[dict] = []                     # (device, t0, t1) per lease, for tests and stats

    @contextlib.contextmanager
    def lease(self):
        """-> the environment of the leased local GPU (``CUDA_VISIBLE_DEVICES``)."""
        with self.lease_device() as dev:
            ident = dev.get("ident")
            yield {"CUDA_VISIBLE_DEVICES": str(ident)} if ident is not None else {}

    @contextlib.contextmanager
    def lease_device(self):
        """-> the leased device dict (local: its KernelBlaster lock held)."""
        with self._cv:
            while len(self._busy) == len(self.devices):
                self._cv.wait()
            i = min((k for k in range(len(self.devices)) if k not in self._busy),
                    key=lambda k: (self._last.get(k, 0.0), k))
            self._busy.add(i)
        dev = self.devices[i]
        ident = dev.get("ident")
        try:
            with contextlib.ExitStack() as stack:
                if ident is not None and not dev.get("host") and not _flock_off():
                    try:
                        stack.enter_context(_flock(gpu_lock_path(str(ident)), timeout=self.lock_timeout))
                    except PermissionError:            # another user's lock file: run unlocked, as KB does
                        pass
                t0 = time.time()
                try:
                    yield dev
                finally:
                    self.log.append({"device": i, "t0": t0, "t1": time.time()})
        finally:
            with self._cv:
                self._busy.discard(i)
                self._last[i] = time.monotonic()
                self._cv.notify_all()


# ---------------------------------------------------------------- GPUs

_DETECT = r'''
import asyncio, json, sys
sys.path.insert(0, sys.argv[1] + "/src")
from kernelblaster.servers.utils import resources as r
a = json.loads(sys.argv[2])
out = {"local": [], "remote": []}
if not a["ssh_hosts"]:
    inv = r.gpu_inventory()
    chosen, why = r.select_gpus(a["gpus"], inv)
    out["local"] = [{"ident": g.uuid or str(g.index), "index": g.index, "name": g.name} for g in chosen]
    out["why"] = why
else:
    cvd = a["cvd"] or [None]
    for k, h in enumerate(a["ssh_hosts"]):
        spec = cvd[k] if len(cvd) == len(a["ssh_hosts"]) else cvd[0]
        devs = asyncio.run(r.remote_devices(h, spec, ssh_key=a["ssh_key"]))
        out["remote"] += [{"host": h, "ident": d.ident, "index": d.index, "name": d.name} for d in devs]
print(json.dumps(out))
'''


def detect_gpus(cfg: Config) -> dict:
    """KernelBlaster's own GPU selection (``resources.select_gpus``: auto =
    the idle GPUs, else the least busy one) for the pool the KernelBlaster
    runs will use -> {"local": [...], "remote": [...]}; on failure, one
    unpinned local device."""
    args = {"gpus": cfg.gpus, "ssh_hosts": list(cfg.ssh_hosts), "cvd": list(cfg.cuda_visible_devices),
            "ssh_key": cfg.ssh_key}
    try:
        cp = subprocess.run([cfg.python, "-c", _DETECT, str(Path(cfg.kb).resolve()), json.dumps(args)],
                            capture_output=True, text=True, timeout=120)
        return json.loads(cp.stdout.strip().splitlines()[-1])
    except Exception as e:  # noqa: BLE001
        return {"local": [], "remote": [], "why": f"detection failed: {e!r}"}


def auto_parallel_hints(n_gpus: int) -> int:
    """Hints per wave by default: max(4, 2 x GPUs). The wave width does not
    decide how many runs share a GPU -- KernelBlaster's pool gives each timed
    run a GPU of its own and its build limiter bounds the compiles -- it
    decides how much LLM latency and compile time overlaps other hints'
    runs. Measured (docs/graph_optimizer.md): 4 hints per wave beat 2 on one
    TITAN RTX and on two L40S, most with LLM latency."""
    return max(4, 2 * int(n_gpus))


# ---------------------------------------------------------------- KernelBlaster

class KBRunner:
    """KernelBlaster's ``scripts/run_multislot.py`` as a subprocess."""

    def __init__(self, cfg: Config, knowledge: Path):
        self.cfg, self.knowledge = cfg, knowledge
        self.calls = 0
        self.telemetry: list = []                     # each run's run-level telemetry (kb_telemetry)
        self.gpus: list = []                          # the GPU each problem's run recorded (kb_eval.recorded_gpus)

    def command(self, problems: list[Path], out: Path, *, steps: int | None = None,
                winners: Path | None = None, parallel_problems: int | None = None) -> list[str]:
        c = self.cfg
        cmd = [c.python, str(Path(c.kb) / "scripts" / "run_multislot.py"), *map(str, problems), "--out", str(out),
               "--knowledge-dir", str(self.knowledge), "--model", c.model, "--arch", c.arch,
               "--rollouts", str(c.rollouts), "--steps", str(steps if steps is not None else c.steps),
               "--max-fix", str(c.max_fix), "--seed", str(c.seed)]
        if c.reflect_model:
            cmd += ["--reflect-model", c.reflect_model]
        if winners is not None:
            cmd += ["--export-winners", str(winners)]
        if parallel_problems:
            cmd += ["--parallel-problems", str(parallel_problems)]
        if c.ssh_hosts:
            cmd += ["--evaluator", "ssh"]
            for h in c.ssh_hosts:
                cmd += ["--ssh-host", h]
            for v in c.cuda_visible_devices:
                cmd += ["--cuda-visible-devices", v]
            if c.ssh_key:
                cmd += ["--ssh-key", c.ssh_key]
            if c.remote_dir:
                cmd += ["--remote-dir", c.remote_dir]
        elif c.gpus and c.gpus != "auto":
            cmd += ["--gpus", c.gpus]
        if c.runtime and c.runtime != "auto":
            cmd += ["--runtime", c.runtime]
        if c.gpu_model:
            cmd += ["--gpu-model", c.gpu_model]
        for p in c.plugins:
            cmd += ["--plugin", str(p)]
        return cmd

    def start(self, problems: list[Path], out: Path, **kw) -> subprocess.Popen:
        out.mkdir(parents=True, exist_ok=True)
        log = open(out.with_suffix(".log"), "w")
        try:
            p = subprocess.Popen(self.command(problems, out, **kw), stdout=log, stderr=subprocess.STDOUT,
                                 cwd=str(self.cfg.kb))
        finally:
            log.close()                               # the child holds its own descriptor
        return p

    def collect(self, problems: list[Path], out: Path, proc: subprocess.Popen) -> list[dict]:
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"KernelBlaster failed ({rc}); see {out.with_suffix('.log')}")
        res = []
        for d in problems:
            r = read_result(out, Path(d)) or {"problem": problem_name(Path(d)), "final_ok": False}
            r["_dir"] = str(out / problem_name(Path(d)))
            self.calls += r.get("llm_calls", 0) or 0
            if r.get("final_ok"):
                self.gpus.append({"run": r["_dir"], "gpu": r.get("gpu"), "arch": r.get("arch")})
            res.append(r)
        self.telemetry.append(kbt.kb_run_summary(out))
        return res

    def run(self, problems: list[Path], out: Path, *, steps: int | None = None,
            winners: Path | None = None, parallel_problems: int | None = None) -> list[dict]:
        p = self.start(problems, out, steps=steps, winners=winners, parallel_problems=parallel_problems)
        return self.collect(problems, out, p)


def problem_name(d: Path) -> str:
    return json.loads((Path(d) / "manifest.json").read_text())["name"]


def read_result(out: Path, problem: Path) -> dict | None:
    """A problem's result.json in a (possibly still running) KernelBlaster
    run; None until it is complete. KernelBlaster writes the best slot
    sources before result.json, so once it parses the bests are final."""
    f = out / problem_name(problem) / "result.json"
    try:
        return json.loads(f.read_text())
    except (OSError, ValueError):
        return None


def _usage(results: list[dict]) -> dict:
    tot = {"llm_calls": 0, "cost_usd": 0.0}
    for r in results:
        tot["llm_calls"] += r.get("llm_calls", 0) or 0
        tot["cost_usd"] += (r.get("llm_usage") or {}).get("cost_usd", 0.0) or 0.0
    return tot


def _est_gain(h: dict) -> float:
    return max((s.get("est_gain_pct") or 0.0) for s in h.get("source", [{}]) or [{}])


def _hint_ir(h: dict, profiles: list[dict]) -> dict | None:
    irs = [s.get("ir") for s in h.get("source", []) if s.get("ir")]
    return next((p for ir in irs for p in profiles if p["ir"] == ir), None)


def _scope_ops(h: dict, graph: dict, specs: dict) -> list[str]:
    """Current op names a hint touches in ``graph``."""
    if h["kind"] == "fuse":
        chains = h.get("chains") or find_chains(graph, h["fused_kind"], specs)
        names = []
        for ch in chains:
            try:
                names += gh._locate(graph, ch)
            except ValueError:
                continue
        return list(dict.fromkeys(names))
    if h["kind"] == "split_slot":
        return [o["name"] for o in graph["ops"] if o["op"] in specs and base_slot_id(o["op"], specs) == h["slot"]]
    return []


# ---------------------------------------------------------------- conflicts

def _entry(kind: str, specs: dict) -> str:
    """The C entry a kind's kernels are cached under (kinds can share one)."""
    try:
        spec = specs.get(kind)
        if spec is None and kind.startswith("fused__"):
            from .kb_proposals import spec_from_kind
            spec = spec_from_kind(kind, specs)
        return parse_signature(spec.signature)[0] if spec is not None else kind
    except Exception:  # noqa: BLE001
        return kind


def footprint(h: dict, ir: str, graph: dict, specs: dict) -> dict:
    """What deciding ``h`` on ``graph`` (its IR's current state) reads and
    writes of the shared state.

    - ``ops``: (IR, original op name) of every op the hint's problems hold;
    - ``reads``: cache entries its problems are seeded from -- the entries of
      those ops and of their stages (fused ops), plus the variant's own new
      slot (a cached fused kernel of that kind, a split variant's base slot);
    - ``writes``: cache entries an acceptance ingests -- the fused kind
      (``fuse``), or the base slot's entry, guarded on its shapes (``split``).

    Entries, not (entry, shape): a slot's cache lookup and coverage split look
    at every instance of the slot in the IR, not only the touched ones."""
    specs = with_fused_specs(graph, specs)
    names = _scope_ops(h, graph, specs)
    by_name = {o["name"]: o for o in graph["ops"]}
    ops, reads = set(), set()
    for n in names:
        ops.update((ir, a) for a in gh.atoms(n))
        o = by_name.get(n)
        if o is None:
            continue
        reads.add(_entry(o["op"], specs))
        reads.update(_entry(s["op"], specs) for s in o.get("sub_ops", []))
    if h["kind"] == "fuse":
        writes = {_entry(h["fused_kind"], specs)}
    elif h["kind"] == "split_slot":
        writes = {_entry(by_name[names[0]]["op"], specs)} if names else {h["slot"]}
    else:
        writes = set()
    return {"ops": ops, "reads": reads | writes, "writes": writes}


def conflicts(a: dict, b: dict) -> bool:
    """Two hints conflict when deciding one can change the other's inputs or
    outcome: they share an op (one's rewrite changes the other's graph;
    a chain and its prefix fusion), or one's acceptance writes a cache entry
    the other's problems are seeded from."""
    return bool(a["ops"] & b["ops"] or a["writes"] & b["reads"] or b["writes"] & a["reads"])


def schedule(fps: list[dict], width: int) -> list[list[int]]:
    """Waves of hint indices (``fps`` in rank order): hint i goes to the
    first wave after every earlier hint it conflicts with, and that has
    room (``width``). Every conflicting pair is thus decided in rank order,
    so the result equals deciding the hints one by one in rank order;
    independent hints share a wave. Deterministic: depends only on the
    order and the footprints."""
    width = max(1, int(width))
    wave_of: list[int] = []
    waves: list[list[int]] = []
    for i, f in enumerate(fps):
        w = 1 + max((wave_of[j] for j in range(i) if conflicts(fps[j], f)), default=-1)
        while w < len(waves) and len(waves[w]) >= width:
            w += 1
        while len(waves) <= w:
            waves.append([])
        waves[w].append(i)
        wave_of.append(w)
    return waves


# ---------------------------------------------------------------- decisions

def _ingest_slots(cache: CudaCache, problem: Path, result: dict, slots: list[str], specs: dict, tag: str) -> list:
    """Put a run's best sources for ``slots`` into the cache (guarded on the
    problem's instance shapes), whatever their speedup over their seed: the
    hint's acceptance already measured them against the unchanged graph."""
    man = json.loads((Path(problem) / "manifest.json").read_text())
    best = best_sources(Path(result["_dir"]))
    out = []
    for s in man["slots"]:
        if s["id"] not in slots or s["id"] not in best:
            continue
        spec = specs.get(s["op"])
        if spec is None:
            from .kb_proposals import spec_from_kind
            spec = spec_from_kind(s["op"], specs)
        entry, _ = parse_signature(spec.signature)
        sr = result.get("slots", {}).get(s["id"], {})
        e, st = cache.add(op=s["op"], entry=entry, signature=spec.signature, arch=f"sm_{man_arch(result)}",
                          shapes=[i["shape"] for i in s["instances"]], code=best[s["id"]],
                          slot_us=sr.get("best_us") or sr.get("final_us") or 0.0,
                          speedup=(sr["baseline_us"] / sr["best_us"]) if sr.get("baseline_us") and sr.get("best_us") else None,
                          impl_id=f"{result.get('run_id', 'run')}:{s['id']}", source={"hint": tag})
        out.append({"slot": s["id"], "status": st})
    cache.save()
    return out


def man_arch(result: dict) -> str:
    return str(result.get("arch", "sm_75")).removeprefix("sm_")


def prepare_hint(h: dict, prof: dict, out: Path, *, cache: CudaCache, cfg: Config, specs: dict) -> dict:
    """Export a hint's original and variant problems from the current state
    (reads the cache, writes only under ``out``). -> a job; ``job["early"]``
    is set when there is nothing to run (the hint does not occur / apply)."""
    base_graph = json.loads((Path(prof["ir_dir"]) / "graph.json").read_text())
    specs = with_fused_specs(base_graph, specs)
    base = cache_plan(base_graph, cache, f"sm_{cfg.arch}", specs)
    g0 = base["graph"]
    names = _scope_ops(h, g0, specs)
    job = {"hint": h, "specs": specs, "names": names,
           "ev": {"problem": None, "ir": prof["ir"], "ts": gh._now()}}
    if not names:
        job["early"] = "hint does not occur in this IR"
        return job
    hh = dict(h)
    if h["kind"] == "fuse":                             # scope the variant to this IR's instances
        hh["chains"] = [c for c in (h.get("chains") or [])
                        if all(a in {x for n in names for x in gh.atoms(n)} for a in c)] or None
    vg, _, prov = gh.apply_hints(g0, [hh], specs, statuses=None)
    if not prov["applied"]:
        job["early"] = "; ".join(s["why"] for s in prov["skipped"])[:500]
        return job
    vplan = cache_plan(vg, cache, f"sm_{cfg.arch}", specs, split=False)
    out.mkdir(parents=True, exist_ok=True)
    src_ir = Path(prof["source_ir"])
    oir = write_variant_ir(src_ir, g0, out / "orig_ir")
    vir = write_variant_ir(src_ir, vplan["graph"], out / "variant_ir")
    stem = _safe(f"{prof['ir']}__{h['id']}")
    oidx = [i for i, o in enumerate(g0["ops"]) if o["name"] in set(names)]
    vnames = {prov["renamed"].get(n, n) for n in names}
    vidx = [i for i, o in enumerate(vplan["graph"]["ops"]) if o["name"] in vnames]
    seeds = {**base["sources"], **vplan["sources"]}
    tags = {**base["tags"], **vplan["tags"]}
    job["orig"] = export_problem(oir, out / "orig", op_indices=oidx, specs=specs, name=stem + "__orig",
                                 strategy="hint-original", slot_sources=base["sources"],
                                 slot_source_tags=base["tags"], sub_sources=base["sub_sources"])
    job["variant"] = export_problem(vir, out / "variant", op_indices=vidx, specs=specs, name=stem + "__variant",
                                    strategy="hint-variant", slot_sources=seeds, slot_source_tags=tags,
                                    sub_sources=vplan["sub_sources"])
    job["stem"], job["prov"] = stem, prov
    return job


def kb_eval_for(cfg: Config):
    """KernelBlaster's evaluator for ModelBlaster's own timings (the model run; re-measurement
    over ssh) on the KernelBlaster runs' pool: ssh with ``ssh_hosts``, else local ``gpus``.
    None when the checkout has no ``scripts/time_problems.py`` (an older pin)."""
    from .kb_eval import KBEval

    ev = KBEval(evaluator="ssh" if cfg.ssh_hosts else "local", kb=Path(cfg.kb), python=cfg.python, gpus=cfg.gpus,
                ssh_hosts=list(cfg.ssh_hosts), cuda_visible_devices=list(cfg.cuda_visible_devices),
                ssh_key=cfg.ssh_key, remote_dir=cfg.remote_dir, runtime=cfg.runtime)
    return ev if ev.available() else None


def measure_hint(job: dict, ro: dict, rv: dict, cfg: Config, leases: GpuLeases | None) -> list | None:
    """Re-time both composite bests interleaved on one leased GPU ->
    [original times, variant times], or None (a side failed / no
    re-measurement). Thread-safe: writes only under the job's problem dirs.

    With ``ssh_hosts`` the leased GPU is a remote one (the KernelBlaster
    runs' pool) and the runs go through KernelBlaster's ssh evaluator pinned
    to it, on the GPU model the two runs recorded; locally, ModelBlaster's
    own runs on the leased local GPU (its lock held)."""
    if not (ro.get("final_ok") and rv.get("final_ok")) or not cfg.remeasure:
        return None
    pair = [(job["orig"], best_sources(Path(ro["_dir"]))), (job["variant"], best_sources(Path(rv["_dir"])))]
    ev = kb_eval_for(cfg) if cfg.ssh_hosts else None
    if ev is None:
        lease = leases.lease() if leases is not None else None
        return remeasure(pair, arch=cfg.arch, reps=cfg.remeasure, lease=lease)
    from .kb_eval import recorded_gpus, required_hardware

    hw = required_hardware(recorded_gpus([Path(ro["_dir"]), Path(rv["_dir"])]), gpu_model=cfg.gpu_model,
                           arch=cfg.arch, allow_mismatch=cfg.allow_gpu_mismatch)
    info: dict = {}
    with (leases.lease_device() if leases is not None else contextlib.nullcontext(None)) as dev:
        times = remeasure(pair, arch=cfg.arch, reps=cfg.remeasure, kbeval=ev.for_device(dev),
                          gpu_model=hw["gpu_model"], info=info, work=Path(job["variant"]).parent / "remeasure")
    job["remeasure_device"] = info.get("device")
    return times


def finish_hint(job: dict, ro: dict | None, rv: dict | None, times: list | None, *, cache: CudaCache, doc: dict,
                cfg: Config) -> dict:
    """Decide from the measurements, record the evidence, ingest an accepted
    variant's new slots. Writes the hints doc and the cache: call from one
    thread, in rank order."""
    h, ev, names = job["hint"], job["ev"], job["names"]
    if "early" in job:
        ev.update(decision="rejected", why=job["early"])
        gh.record_decision(doc, h, ev)
        return {"hint": h["id"], "status": "rejected", "why": ev["why"]}
    ev.update(problem=job["stem"], original_us=ro.get("final_partition_us"), variant_us=rv.get("final_partition_us"),
              kb_original_us=ro.get("final_partition_us"), kb_variant_us=rv.get("final_partition_us"),
              llm_calls=(ro.get("llm_calls") or 0) + (rv.get("llm_calls") or 0))
    ok = bool(ro.get("final_ok") and rv.get("final_ok"))
    if ok and cfg.remeasure:
        ok = bool(times and times[0] and times[1])
        if ok:
            ev.update(original_us=_median(times[0]), variant_us=_median(times[1]),
                      remeasured={"original_us": times[0], "variant_us": times[1],
                                  **({"device": job["remeasure_device"]} if job.get("remeasure_device") else {})})
    if not ok:
        ev["why"] = ("KernelBlaster: " + ", ".join(f"{side} not final_ok ({r.get('error') or 'see result.json'})"
                                                   for side, r in (("original", ro), ("variant", rv))
                                                   if not r.get("final_ok"))
                     if not (ro.get("final_ok") and rv.get("final_ok")) else
                     f"re-measurement: no passing run on a side ({[len(x) for x in times or []]} of {cfg.remeasure})")
    gain = (ev["original_us"] - ev["variant_us"]) / ev["original_us"] if ok else None
    status = "accepted" if ok and gain >= cfg.accept_gain else "rejected"
    ev.update(gain=gain, decision=status, applied=job["prov"]["applied"])
    rec = gh.record_decision(doc, h, ev)
    res = {"hint": h["id"], "status": status, "hint_status": rec["status"], "gain": gain,
           "original_us": ev["original_us"], "variant_us": ev["variant_us"], "ops": names}
    if rec["status"] == "accepted":
        man = json.loads((Path(job["variant"]) / "manifest.json").read_text())
        changed = [s["id"] for s in man["slots"]
                   if s["op"].startswith("fused__") or s.get("variant_of")]
        res["cached"] = _ingest_slots(cache, job["variant"], rv, changed, job["specs"], h["id"])
    return res


def evaluate_hints(items: list[tuple[dict, dict, Path]], run_dir: Path, *, kb: KBRunner, cache: CudaCache,
                   doc: dict, cfg: Config, specs: dict, leases: GpuLeases | None = None,
                   poll_s: float = 1.0) -> list[dict]:
    """Decide a wave of independent hints ((hint, profile, out dir) each).

    Every hint's original and variant problems go to one KernelBlaster run
    (``--parallel-problems`` = all of them: KernelBlaster's GPU pool gives each
    timed run one GPU, its build limiter bounds the compiles, and its store /
    outcome log / claims stay single-writer in that one process); a hint is
    re-timed as soon as both its results are written, on one leased GPU, while
    KernelBlaster works on the others. Decisions and cache ingests are then
    made in the given order, in this thread."""
    T = kbt.get()
    with T.span("mb.prepare", hints=len(items)):
        jobs = [prepare_hint(h, prof, out, cache=cache, cfg=cfg, specs=specs) for h, prof, out in items]
    live = [k for k, j in enumerate(jobs) if "early" not in j]
    results: dict[int, tuple] = {}
    if live:
        problems = [p for k in live for p in (jobs[k]["orig"], jobs[k]["variant"])]
        parent = T.current()
        wait = kbt.section("mb.kb_wait", kind="hints", problems=len(problems))
        proc = kb.start(problems, run_dir, steps=cfg.variant_steps, parallel_problems=len(problems))
        futs = {}
        # re-timing workers: two per GPU (one builds while the other times), at most one per hint
        n_dev = len(leases.devices) if leases is not None else 1
        with ThreadPoolExecutor(max_workers=max(1, min(len(live), 2 * n_dev))) as ex:
            try:
                while True:
                    done = proc.poll() is not None
                    for k in live:
                        if k in futs:
                            continue
                        ro, rv = read_result(run_dir, jobs[k]["orig"]), read_result(run_dir, jobs[k]["variant"])
                        if (ro is not None and rv is not None) or done:
                            ro = dict(ro or {"final_ok": False}, _dir=str(run_dir / problem_name(jobs[k]["orig"])))
                            rv = dict(rv or {"final_ok": False},
                                      _dir=str(run_dir / problem_name(jobs[k]["variant"])))
                            futs[k] = ex.submit(kbt.run_within, parent, "mb.remeasure", measure_hint, jobs[k],
                                                ro, rv, cfg, leases)
                    if done:
                        break
                    time.sleep(poll_s)
            except BaseException:
                proc.kill()
                raise
            res = kb.collect(problems, run_dir, proc)          # final result.json (llm calls, proposals)
            wait.end()
            times = {k: f.result() for k, f in futs.items()}
        for n, k in enumerate(live):
            results[k] = (res[2 * n], res[2 * n + 1], times.get(k))
    with T.span("mb.finish"):
        return [finish_hint(j, *results.get(k, (None, None, None)), cache=cache, doc=doc, cfg=cfg)
                for k, j in enumerate(jobs)]


def evaluate_hint(h: dict, prof: dict, out: Path, *, kb: KBRunner, cache: CudaCache, doc: dict,
                  cfg: Config, specs: dict, leases: GpuLeases | None = None) -> dict:
    """Build original + variant problems for one hint on one IR, optimise
    both with the same budget, re-measure, decide, record."""
    return evaluate_hints([(h, prof, out)], out / "run", kb=kb, cache=cache, doc=doc, cfg=cfg, specs=specs,
                          leases=leases)[0]


def decide_hints(cands: list[tuple[dict, dict]], graphs: dict, out: Path, width: int, *, kb: KBRunner,
                 cache: CudaCache, doc: dict, cfg: Config, specs: dict, leases: GpuLeases | None = None,
                 hints_path: Path | None = None) -> tuple[list[list[str]], list[dict]]:
    """A round's decisions: ``cands`` = (hint, profile) in rank order,
    ``graphs`` = {IR: its current graph}. Schedules them into waves of at
    most ``width`` independent hints, evaluates wave after wave (the hints
    file is saved after each) -> (waves as hint ids, decisions in rank order)."""
    fps = [footprint(h, p["ir"], graphs[p["ir"]], specs) for h, p in cands]
    waves = schedule(fps, width)
    decided: dict[int, dict] = {}
    for k, w in enumerate(waves):
        items = [(cands[i][0], cands[i][1], out / _safe(cands[i][0]["id"])) for i in w]
        run_dir = items[0][2] / "run" if len(w) == 1 else out / f"wave{k:02d}"
        for i, d in zip(w, evaluate_hints(items, run_dir, kb=kb, cache=cache, doc=doc, cfg=cfg, specs=specs,
                                          leases=leases)):
            decided[i] = d
        if hints_path is not None:
            gh.save(doc, hints_path)
    return [[cands[i][0]["id"] for i in w] for w in waves], [decided[i] for i in sorted(decided)]


# ---------------------------------------------------------------- rounds

def run(irs: list[Path], work: Path, cfg: Config) -> dict:
    work = Path(work).resolve()
    with work_lock(work):
        tel = kbt.load(cfg.kb)                    # telemetry (kb_telemetry.py); a no-op without it
        tel.configure(work / "telemetry.jsonl")
        try:
            with tel.span("mb.run", collect=True):
                return _run(irs, work, cfg)
        finally:
            tel.close()


def _run(irs: list[Path], work: Path, cfg: Config) -> dict:
    hints_path, state_path = work / "graph_hints.json", work / "state.json"
    doc, cache = gh.load(hints_path), CudaCache(work / "cache")
    state = json.loads(state_path.read_text()) if state_path.exists() else {"sent": [], "rounds": []}
    kb = KBRunner(cfg, work / "knowledge")
    specs = dict(_all_specs())
    gpus = detect_gpus(cfg)
    n_gpus = len(gpus["remote"]) if cfg.ssh_hosts else len(gpus["local"])
    width = cfg.parallel_hints or auto_parallel_hints(n_gpus)
    # MB's own timed runs (re-measurement) lease from the KernelBlaster runs' pool: remote with --ssh-host
    leases = GpuLeases(gpus["remote"] if cfg.ssh_hosts else gpus["local"])
    state["gpus"] = {"pool": gpus, "parallel_hints": width}
    first = None
    T = kbt.get()
    run_span = T.current()
    for r in range(len(state["rounds"]), len(state["rounds"]) + cfg.rounds):
        R = work / f"r{r:02d}"
        t0 = time.time()
        rec = {"round": r, "kb_runs": [], "decisions": []}
        rsec, n_kb = kbt.section("mb.round", collect=True, laps=True, round=r), len(kb.telemetry)
        T.lap("mb.profile")
        profiles = [profile_ir(Path(d), R / "profile" / Path(d).name, hints_doc=doc, cache=cache, arch=cfg.arch,
                               specs=specs, runs=cfg.profile_runs) for d in irs]
        bad = [p["ir"] for p in profiles if not p["passed"]]
        if bad:
            raise RuntimeError(f"round {r}: profile failed for {bad}; see {R / 'profile'}")
        first = first or {p["ir"]: p["partition_us"] for p in profiles}
        rec["partition_us"] = {p["ir"]: p["partition_us"] for p in profiles}
        T.lap("mb.propose")
        graphs = {}
        for p in profiles:                                  # MB's own proposals
            g = graphs[p["ir"]] = json.loads((Path(p["ir_dir"]) / "graph.json").read_text())
            gh.merge(doc, gh.propose_from_profile(g, {o["name"]: o["us"] or 0.0 for o in p["ops"]}, specs,
                                                   doc=doc, min_gain_pct=cfg.min_gain_pct, ir=p["ir"],
                                                   launch_us=p.get("launch_us") or 0.0, total_us=p.get("e2e_us")))
        T.lap("mb.plan")
        plan = make_plan(profiles, coverage=cfg.coverage, top_k=cfg.top_k, min_share=cfg.min_share,
                         exclude=set(state["sent"]))
        (R / "plan.json").write_text(json.dumps(plan, indent=1))
        rec["plan"] = {"selected_share": plan["selected_share"],
                       "problems": [(p["ir"], len(p["op_names"])) for p in plan["problems"]]}
        if plan["problems"]:
            T.lap("mb.export")
            dirs = export_plan(plan, R / "problems", cache=cache, arch=cfg.arch, specs=specs,
                               names={p["ir"]: f"{p['ir']}__r{r:02d}" for p in plan["problems"]})
            T.lap("mb.kb_wait", kind="plan")
            results = kb.run(dirs, R / "run", winners=R / "winners")
            T.lap("mb.ingest")
            rec["kb_runs"] = [{k: x.get(k) for k in ("problem", "baseline_partition_us", "final_partition_us",
                                                      "speedup", "final_ok", "llm_calls")} for x in results]
            if (R / "winners" / "winners.json").exists():
                rec["ingest"] = ingest(R / "winners" / "winners.json", cache, specs)
            state["sent"] += [it["key"] for it in plan["items"] if it.get("selected")]
            for x in results:
                pf = Path(x["_dir"]) / "proposals.json"
                if pf.exists():
                    props = json.loads(pf.read_text())
                    for h in gh.from_kb_proposals(props, specs=specs):
                        for s in h["source"]:
                            s["ir"] = json.loads((Path(props["problem_dir"]) / "manifest.json").read_text())["ir_name"]
                        gh.merge(doc, [h])
        gh.save(doc, hints_path)
        T.lap("mb.decide")
        todo = [h for h in doc["hints"] if h.get("status") == "proposed" and h.get("target", "cuda") == "cuda"
                and h["kind"] in ("fuse", "split_slot") and _est_gain(h) >= cfg.min_gain_pct]
        todo.sort(key=lambda h: -_est_gain(h))              # stable: ties keep the hints file's order
        cands = [(h, p) for h in todo[:cfg.max_variants] if (p := _hint_ir(h, profiles)) is not None]
        t_dec = time.time()
        waves, rec["decisions"] = decide_hints(cands, graphs, R / "variants", width, kb=kb, cache=cache, doc=doc,
                                               cfg=cfg, specs=specs, leases=leases, hints_path=hints_path)
        rec["parallel_hints"], rec["waves"] = width, waves
        rec["decide_seconds"] = round(time.time() - t_dec, 1)
        rec["seconds"] = round(time.time() - t0, 1)
        rec["llm_calls_total"] = kb.calls
        T.end_laps(rsec.span)
        rsec.end()
        tr = {"mb": kbt.compact(T.full_summary(rsec.span)), "kb": kbt.kb_rollup(kb.telemetry[n_kb:])}
        if tr["mb"] or tr["kb"]:
            rec["telemetry"] = tr
        state["rounds"].append(rec)
        state_path.write_text(json.dumps(state, indent=1))
        (R / "round.json").write_text(json.dumps(rec, indent=1))
        accepted = [d for d in rec["decisions"] if d.get("hint_status") == "accepted"]
        print(json.dumps({"round": r, "partition_us": rec["partition_us"], "plan": rec["plan"]["problems"],
                          "waves": rec["waves"],
                          "decisions": [(d["hint"], d["status"], d.get("gain")) for d in rec["decisions"]]}),
              flush=True)
        if not accepted and not plan["problems"] and len(todo) <= cfg.max_variants:
            state["fixpoint"] = r
            break
    final = {}
    with T.span("mb.verify"):
        for d in irs:
            final[Path(d).name] = verify_ir(Path(d), cache, arch=cfg.arch, specs=specs, hints=hints_path,
                                            work=work / "final" / Path(d).name)
    summary = {"irs": {}, "rounds": len(state["rounds"]), "fixpoint": state.get("fixpoint"),
               "accepted": [h["id"] for h in doc["hints"] if h.get("status") == "accepted"],
               "rejected": [h["id"] for h in doc["hints"] if h.get("status") == "rejected"],
               "llm_calls": kb.calls, "parallel_hints": width}
    first0 = state["rounds"][0]["partition_us"] if state["rounds"] else (first or {})
    # the model's result: the whole model on the final cache and hints, as deployed (CUDA graph
    # replay in the persistent runtime), against the hand-seeded original timed the same way
    # -- through KernelBlaster's evaluator on the KernelBlaster runs' pool (local or ssh), on the GPU
    # model those runs recorded (kb_eval), so the model and its components share the hardware
    from .kb_model_run import model_runtime
    models = {}
    kbeval = kb_eval_for(cfg)
    with T.span("mb.model_run", evaluator=kbeval.evaluator if kbeval else "builtin"):
        for d in irs:
            try:
                models[Path(d).name] = model_runtime(Path(d), work=work / "final_model" / Path(d).name, cache=cache,
                                                     hints=hints_path, arch=cfg.arch, specs=specs, kbeval=kbeval,
                                                     gpu_model=cfg.gpu_model, recorded=kb.gpus,
                                                     allow_gpu_mismatch=cfg.allow_gpu_mismatch)
            except Exception as e:                   # the verification above stands; report why
                models[Path(d).name] = {"error": repr(e)}
    for name, v in final.items():
        ir = v.get("ir", name)
        f0 = first0.get(ir)
        m = models.get(name, {})
        summary["irs"][ir] = {"first_us": f0, "final_us": v.get("partition_us"), "passed": v.get("passed"),
                              "speedup": (f0 / v["partition_us"]) if f0 and v.get("partition_us") else None,
                              "hints": v.get("hints"), "fused": v.get("fused"),
                              "cache_split": v.get("cache_split"),
                              "model_e2e_us": m.get("model_e2e_us"), "model_basis": m.get("model_basis"),
                              "seed_e2e_us": m.get("seed_e2e_us"), "model_speedup": m.get("speedup"),
                              "model_passed": m.get("passed"), "model_gpu": m.get("gpu_name"),
                              "model_device": (m.get("device") or {}).get("label"),
                              "model_hardware_match": (m.get("hardware") or {}).get("match"),
                              **({"model_error": m["error"]} if "error" in m else {})}
    (work / "model_runtime.json").write_text(json.dumps(models, indent=1, default=str))
    tr = {"mb": kbt.compact(T.full_summary(run_span)), "kb": kbt.kb_rollup(kb.telemetry)}
    if tr["mb"] or tr["kb"]:
        summary["telemetry"] = tr
    (work / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def _parallel_arg(s: str) -> int:
    return 0 if s == "auto" else max(1, int(s))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("irs", nargs="+")
    ap.add_argument("--work", required=True)
    ap.add_argument("--kb", required=True, help="KernelBlaster checkout")
    ap.add_argument("--model", default="claude-code:sonnet")
    ap.add_argument("--reflect-model")
    ap.add_argument("--plugin", action="append", default=[], help="passed to run_multislot.py (stub: models)")
    ap.add_argument("--python", default=sys.executable, help="python for KernelBlaster")
    ap.add_argument("--parallel-hints", type=_parallel_arg, default=0,
                    help="hints decided at once: auto (default: max(4, 2 x GPUs in the pool)) or N (1 = one by one)")
    ap.add_argument("--gpus", default="auto", help="local GPUs for KernelBlaster and re-timing: auto|all|0,2")
    ap.add_argument("--ssh-host", action="append", default=[], help="KernelBlaster runs on these hosts (repeatable)")
    ap.add_argument("--cuda-visible-devices", action="append", default=[],
                    help="with --ssh-host: auto|ids, once for all hosts or once per host")
    ap.add_argument("--ssh-key")
    ap.add_argument("--remote-dir")
    ap.add_argument("--runtime", choices=["auto", "host", "process"], default="auto",
                    help="KernelBlaster runtime for its runs and ModelBlaster's timings through it")
    ap.add_argument("--gpu-model", help="the GPU model every timing must use (default: what the KernelBlaster runs "
                    "recorded), e.g. 'NVIDIA L40S'")
    ap.add_argument("--allow-gpu-mismatch", action="store_true",
                    help="time the model / re-measure even where no GPU of that model is available (warnings)")
    for k, v in asdict(Config(kb=Path("."))).items():
        if k in _MANUAL_FIELDS:
            continue
        ap.add_argument("--" + k.replace("_", "-"), type=type(v) if v is not None else
                        (int if k in ("top_k", "variant_steps") else str), default=v)
    a = ap.parse_args()
    cfg = Config(kb=Path(a.kb).resolve(), model=a.model, reflect_model=a.reflect_model,
                 plugins=[str(Path(p).resolve()) for p in a.plugin], python=a.python,
                 parallel_hints=a.parallel_hints, gpus=a.gpus, ssh_hosts=a.ssh_host,
                 cuda_visible_devices=a.cuda_visible_devices, ssh_key=a.ssh_key, remote_dir=a.remote_dir,
                 runtime=a.runtime, gpu_model=a.gpu_model, allow_gpu_mismatch=a.allow_gpu_mismatch,
                 **{k: getattr(a, k) for k in asdict(Config(kb=Path("."))) if k not in _MANUAL_FIELDS})
    print(json.dumps(run([Path(d) for d in a.irs], Path(a.work), cfg), indent=1))


if __name__ == "__main__":
    main()
