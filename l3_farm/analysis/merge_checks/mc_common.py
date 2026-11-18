#!/usr/bin/env python3
"""Shared helpers of the merge-check harness (X-135 / X-136 / X-137). Local (Mac) side, stdlib only.

Records for every check: tree stamp + sha256 of the frozen tarball, flags, node job id, host, memory clock.
Remote work goes only through farm.ssh / farm.put (the `computelab` alias, key auth) and only on jobs listed in
my_jobs.json. Node files live under /tmp/krisd/mc only. The API key is never handled here: the nodes use the relay
the farm brought up (/tmp/krisd/farm/relay.env, key delivered by farm.py over ssh stdin).
"""
from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
L3 = HERE.parents[1]                      # l3_farm
RES = L3 / "results"
MINE = HERE / "my_jobs.json"
RUNS = HERE / "runs"                      # collected run packs: runs/<tree label>/<arm>/<run name>/
FROZEN = HERE / "frozen"
NODE_ROOT = "/tmp/krisd/mc"
CUDA_N = "/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04"
MODEL = "azure/anthropic/claude-opus-5-5"
MEM_CLOCK_OK = 9001                        # L40S dies with clocks.max.mem 9001 MHz (8001 = slow die, ~11% slower DRAM)

sys.path.insert(0, str(L3))


def farm():
    import farm as f                      # noqa: WPS433  (imports config/state paths only; never writes farm.json)
    return f


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def tree_info(tarball) -> dict:
    """Frozen tree identity: stamp (file name `mb_<label>_<stamp>.tar.gz` -> label/stamp), sha256, size."""
    p = Path(tarball).resolve()
    m = re.match(r"mb_(?P<label>[a-z0-9]+)_(?P<stamp>[0-9-]+)\.tar\.gz$", p.name)
    return {"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size,
            "label": m.group("label") if m else p.stem, "stamp": m.group("stamp") if m else ""}


def load_json(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default


def write_json(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1, default=str))


def md_table(rows: list[list], header: list[str]) -> str:
    def c(x):
        return "" if x is None else str(x).replace("|", "\\|")
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(c(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


# ----------------------------------------------------------------------------------------- nodes (my jobs only)

def mine() -> dict:
    return load_json(MINE, {})


def save_mine(d):
    write_json(MINE, d)


def check_mine(job):
    if str(job) not in mine():
        raise SystemExit(f"job {job} is not one of mine ({MINE})")


def rsh(job, script: str | bytes, args=(), timeout=600):
    """Run a script (text, fed on stdin) on my job's node: ssh computelab -> srun --jobid=J --overlap bash -s."""
    check_mine(job)
    body = script if isinstance(script, bytes) else script.encode()
    cmd = f"srun --jobid={job} --overlap bash -s -- " + " ".join(shlex.quote(str(a)) for a in args)
    rc, out, err = farm().ssh(cmd, body + b"\n", timeout)
    return rc, out.decode(errors="replace"), err.decode(errors="replace")


def rfile(job, name: str, args=(), timeout=600):
    return rsh(job, (HERE / "remote" / name).read_bytes(), args, timeout)


def put(job, local, remote):
    check_mine(job)
    return farm().put(job, local, remote)


def squeue() -> dict:
    rc, out, err = farm().ssh("squeue --me -h -o '%i|%T|%N|%L|%j'")
    res = {}
    for ln in out.decode().splitlines():
        p = ln.split("|")
        if len(p) == 5:
            res[int(p[0])] = dict(state=p[1], host=p[2], left=p[3], name=p[4])
    return res


# ----------------------------------------------------------------------------------------- result readers

def read_run(d) -> dict:
    """Condense a collected run pack (see remote/mc_pack.sh) into the fields the checks compare."""
    d = Path(d)
    w = d / "work"
    st = load_json(w / "state.json", {})
    sm = load_json(w / "summary.json", {})
    mr = load_json(d / "remeasure.json", {}) or {}
    mrt = load_json(w / "model_runtime.json", {}) or {}
    irs = sm.get("irs") or {}
    ir0 = next(iter(irs.values()), {}) if irs else {}
    meta = load_json(d / "meta.json", {})
    rounds = []
    for rj in sorted(w.glob("r[0-9][0-9]/round.json")):
        r = load_json(rj, {})
        kd = key_dict(rj.parent)
        ing = r.get("ingest") or []
        rounds.append({
            "round": r.get("round"), "key": kd["hash"] if kd else r.get("key"), "raw_key": r.get("key"), "key_dict": kd["dict"] if kd else None,
            "partition_us": r.get("partition_us"),
            "llm_calls": r.get("llm_calls_total"), "seconds": r.get("seconds"),
            "decisions": [(x.get("hint"), x.get("status")) for x in (r.get("decisions") or [])],
            "ingest_status": _count([x.get("status") for x in ing]),
            "n_plan_items": len((r.get("plan") or {}).get("problems") or []),
        })
    mu = next(iter(mrt.values()), {}) if mrt else {}
    return {"dir": str(d), "meta": meta, "flow_status": (d / "flow_status").read_text().strip() if (d / "flow_status").exists() else None,
            "keys": [r["key"] for r in rounds], "rounds": rounds,
            "model_e2e_us": ir0.get("model_e2e_us") or mu.get("model_e2e_us"),
            "seed_e2e_us": ir0.get("seed_e2e_us") or mu.get("seed_e2e_us"),
            "torch_us": ir0.get("torch_us") or mu.get("torch_us"),
            "quiet_e2e_us": (mr.get("model_e2e_us") if isinstance(mr, dict) else None),
            "hints": ir0.get("hints"), "fixpoint": sm.get("fixpoint"), "llm_calls": sm.get("llm_calls"),
            "state_keys": [r.get("key") for r in st.get("rounds", [])] if st else []}


def key_dict(rdir):
    """The round key as the flow built it (rNN/progress.json `key`), with the IR path reduced to its basename: the flow's own key hash
    includes the absolute IR path (so it differs between run dirs); this one is comparable across runs and trees."""
    p = load_json(Path(rdir) / "progress.json", {}) or {}
    k = p.get("key")
    if not isinstance(k, dict):
        return None
    k = dict(k)
    if "irs" in k:
        k["irs"] = sorted(Path(x).name for x in k["irs"])
    return {"dict": k, "hash": hashlib.sha1(json.dumps(k, sort_keys=True, default=str).encode()).hexdigest()[:16]}


def _count(xs):
    c = {}
    for x in xs:
        c[x] = c.get(x, 0) + 1
    return c


def cost_usd(ledger_path) -> tuple[int, float]:
    """Calls and $ of a relay-ledger slice at the Opus prices E0 used ($5 in, $25 out, $6.25 cache write, $0.50 read per M)."""
    n = 0
    u = 0.0
    p = Path(ledger_path)
    if not p.exists():
        return 0, 0.0
    for ln in p.read_text().splitlines():
        try:
            d = json.loads(ln)
        except ValueError:
            continue
        cr, cw, pi, o = (d.get(k) or 0 for k in ("cache_read_input_tokens", "cache_write_input_tokens",
                                                   "prompt_tokens", "completion_tokens"))
        n += d.get("calls", 1) or 1
        u += (max(0, pi - cr - cw) * 5 + cr * 0.5 + cw * 6.25 + o * 25) / 1e6
    return n, round(u, 2)


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def md_head(title: str, meta: dict) -> str:
    keys = ["generated", "tree", "flags", "job", "host", "mem_clock_mhz"]
    lines = [f"# {title}", ""] + [f"- **{k}**: {meta[k]}" for k in keys if k in meta]
    return "\n".join(lines) + "\n"
