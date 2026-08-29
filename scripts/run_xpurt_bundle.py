"""Drive a candidate bundle (Contract 1) end-to-end on ModelBlaster.

Reads an XPU-RT `xpurt.candidate_bundle/v1` JSON (typically at
`$XPURT_ROOT/artifacts/iterate/firesim_batch.json`), and
for each selected candidate:

  - resolves the matching XPU-RT schedule fixture
    (`scheduled__iter_<label>_<hw>_<solver_tag>_profiled.json`),
  - maps the candidate's `profile_hw` to ModelBlaster CPU_P_KIND /
    CPU_E_KIND on the hetero bitstream,
  - invokes `examples/xpurt_demo/run.sh` with RUNNER=firesim and
    XPURT_TRACE=1 so the resulting ELF emits a per-dispatch trace,
  - copies the FireSim uartlog + extracted `xpurt_trace.csv` into
    `artifacts/bundle/<id>/`.

The whole thing is one Python script so the loop can be driven by an
agent (the `/realize-and-run` skill) without bash babysitting. Each
candidate runs sequentially through the shared FIRESIM_QUEUE; the
infrasetup cost is amortized across the bundle.

Usage:

    python3 scripts/run_xpurt_bundle.py \\
        --batch $XPURT_ROOT/artifacts/iterate/firesim_batch.json \\
        --out-dir artifacts/bundle \\
        --include baseline,A2          # optional id allow-list
        --runner firesim               # spike|firesim (default firesim)

A `manifest.json` is written next to the per-candidate dirs:

    {"candidates": [
        {"id": "baseline", "status": "ok",
         "elf": "artifacts/bundle/baseline/zephyr.elf",
         "uartlog": "...", "trace_csv": "...", "wall_s": 1812.4,
         "predicted_makespan_us": 65.4, "measured_makespan_us": 71.2},
        {"id": "A2", "status": "ok", ...},
        {"id": "C1", "status": "skipped",
         "reason": "fusion codegen for __fused__ ops not implemented"}
    ]}

so `/close-loop` (XPU-RT skill) can pick up where this leaves off.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_xpurt_root() -> Path:
    """Locate the XPU-RT checkout; see the same helper in
    `scripts/decision_loop.py` and `scripts/close_xpurt_loop.py`. Identified
    by `schedules/`, which is the directory this script resolves fixtures
    against. Replaces a hardcoded /scratch2/<user>/XPU-RT.
    """
    env = os.environ.get("XPURT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    for cand in (REPO_ROOT.parent, REPO_ROOT.parent / "XPU-RT"):
        if (cand / "schedules").is_dir() and (cand / "xpu-rt").is_dir():
            return cand.resolve()
    return REPO_ROOT.parent.resolve()


XPURT_ROOT = _resolve_xpurt_root()
GREEDY_FAMILY = {"greedy", "greedy_periodic", "decomposed"}


def _solver_tag(solver: str, scheduler: str | None) -> str:
    """Mirror XPU-RT's _sched_eval._solver_tag (so we resolve the same path)."""
    if solver in GREEDY_FAMILY:
        return f"_{solver}"
    if scheduler:
        return f"_{scheduler}"
    return ""


def _profile_hw_to_kinds(profile_hw: dict[str, str]) -> tuple[str, str]:
    """Map XPU-RT profile_hw labels to ModelBlaster CPU_P_KIND / CPU_E_KIND.

    XPU-RT uses profile-DB labels (`gemmini_q31`, `V256D128_rvv`) that
    aren't the same as the registry `kind` strings. The hetero
    bitstream has two kinds: `gemmini` (the Gemmini hart) and
    `rvv_opu` (the V+Saturn-OPU hart).
    """
    mapping = {
        "gemmini_q31": "gemmini",
        "gemmini": "gemmini",
        "V256D128_rvv": "rvv_opu",
        "rvv_opu": "rvv_opu",
        "rvv": "rvv_opu",
    }
    try:
        cpu_p = mapping[profile_hw["cpu_p"]]
        cpu_e = mapping[profile_hw["cpu_e"]]
    except KeyError as e:
        raise SystemExit(
            f"unknown profile_hw label {e.args[0]!r}; this driver only "
            f"supports the hetero bitstream (gemmini + rvv_opu)")
    return cpu_p, cpu_e


def _baseline_candidate(bundle: dict[str, Any], batch_path: Path) -> dict[str, Any]:
    """Synthesize a candidate dict for the baseline so it runs through
    the same code path as the rest.

    The XPU-RT bundle doesn't currently emit a `baseline` field — it
    lives in the sibling `iteration_result.json` under `runs[0]`. Read
    it from there if present; otherwise infer the standard
    decomposed-on-hetero default (matches `iterate_firesim.py`'s
    `--baseline-solver decomposed`).
    """
    base_solver = "decomposed"
    base_scheduler = None
    base_profile_hw = None
    iter_path = batch_path.parent / "iteration_result.json"
    if iter_path.is_file():
        try:
            ir = json.loads(iter_path.read_text())
            for run in ir.get("runs", []):
                if run.get("id") == "baseline":
                    base_profile_hw = run.get("profile_hw")
                    label = run.get("label") or ""
                    if "/" in label:
                        # e.g. "milp/heft" — solver is before the slash
                        base_solver, base_scheduler = label.split("/", 1)
                    else:
                        base_solver = label or base_solver
                    break
        except (OSError, json.JSONDecodeError):
            pass
    if base_profile_hw is None:
        # Fall back to the first candidate's profile_hw — the baseline
        # always shares it with axis-A candidates.
        for c in bundle.get("candidates", []):
            if c.get("axis") == "scheduler":
                base_profile_hw = c["profile_hw"]
                break
    if base_profile_hw is None:
        raise SystemExit("could not infer baseline profile_hw from bundle")
    return {
        "id": "baseline",
        "axis": "baseline",
        "realizable_by": "xpurt",
        "solver": base_solver,
        "scheduler": base_scheduler,
        "profile_hw": base_profile_hw,
        "rationale": "baseline reference (XPU-RT iterate_firesim default)",
    }


def _fixture_path(candidate: dict[str, Any]) -> Path:
    """Resolve the XPU-RT schedule fixture for one candidate.

    Mirrors `_sched_eval.run_candidate`'s stem rule:
      label = solver (if in GREEDY_FAMILY) else (scheduler or solver)
      hw    = "-".join(sorted(set(profile_hw.values())))
      stem  = "_iter_baseline" for the baseline; "_iter_<label>_<hw>"
              otherwise.
    """
    if candidate["id"] == "baseline":
        stem = "_iter_baseline"
    else:
        solver = candidate["solver"]
        scheduler = candidate.get("scheduler")
        label = solver if solver in GREEDY_FAMILY else (scheduler or solver)
        hw = "-".join(sorted(set(candidate["profile_hw"].values())))
        stem = f"_iter_{label}_{hw}"
    tag = _solver_tag(candidate["solver"], candidate.get("scheduler"))
    return XPURT_ROOT / "schedules" / f"scheduled_{stem}{tag}_profiled.json"


def _models_quants_for_networks_json(spec_path: Path) -> tuple[str, str]:
    """Resolve MODELS / QUANTS from the XPU-RT workload spec.

    Returns UNIQUE network names (one per `name`), not per-instance —
    the harness builds one model library per name, and the schedule
    references them by instance suffix (e.g. `mlp_control0`,
    `mlp_control1` both link the single `mlp_control` model lib).
    Emitting duplicates triggers CMake "target already exists" errors.

    On the hetero (Gemmini+OPU) bitstream the demo standardizes on int8
    builds for all three networks (matches past 1+4+2 captures in
    benchmarks/results/). The XPU-RT spec's `dispatch_deps_path` may
    point at a fp32 profile for mlp_control because that's where the
    profile data lives — that's a *scheduling*-time concern; the
    *build* is int8.
    """
    with open(spec_path) as f:
        spec = json.load(f)
    nets = spec.get("networks", {})
    if isinstance(nets, list):
        nets_iter = [(n["name"], n) for n in nets]
    else:
        nets_iter = list(nets.items())
    # Order: yolov8_nano first (one-shot), then dronet, then mlp_control —
    # matches periodicity (least-periodic first) for the harness walker.
    def _key(item):
        name, entry = item
        period = entry.get("period", 0) or 0
        return (period, name)
    sorted_nets = sorted(nets_iter, key=_key)
    models = [name for name, _ in sorted_nets]
    quants = ["int8" for _ in sorted_nets]
    return ",".join(models), ",".join(quants)


def _resolve_workload_spec(bundle: dict[str, Any]) -> Path:
    p = bundle.get("networks_json")
    if not p:
        raise SystemExit("bundle is missing networks_json")
    full = (XPURT_ROOT / p) if not Path(p).is_absolute() else Path(p)
    if not full.is_file():
        raise SystemExit(f"networks_json not found: {full}")
    return full


def _extract_trace(uartlog: Path, dst: Path) -> bool:
    """Cut the MODELBLASTER_XPURT_TRACE_BEGIN..END block out of a uartlog.

    Returns True if a trace block was found and written. The block is
    written verbatim (CSV header preserved).
    """
    if not uartlog.is_file():
        return False
    text = uartlog.read_text(errors="replace")
    begin = "=== MODELBLASTER_XPURT_TRACE_BEGIN ==="
    end = "=== MODELBLASTER_XPURT_TRACE_END ==="
    if begin not in text or end not in text:
        return False
    i = text.index(begin) + len(begin)
    j = text.index(end, i)
    body = text[i:j].strip()
    if not body:
        return False
    dst.write_text(body + "\n")
    return True


def _find_uartlog(stdout_log: Path, job_started_at: float) -> Path | None:
    """Locate the uartlog FireSim wrote for our run.

    Strategies in priority order:
      1. parse the explicit `firesim: reading per-run uartlog at <path>`
         line firesim_runner.py prints (most reliable, exact match).
      2. parse `job_id=<N>` (queue submit log) and look for the
         matching `*-q<N>/*/uartlog` results dir.
      3. fall back to the newest uartlog under the results tree with
         mtime > job_started_at (FIRESIM_QUEUE=1 serializes runs, so
         the newest one after our subprocess started IS ours).
    """
    import re
    text = stdout_log.read_text(errors="replace") if stdout_log.is_file() else ""
    results_root = Path("/scratch2/agustin/chipyard/sims/firesim/deploy/results-workload")

    m = re.search(r"firesim: reading per-run uartlog at (\S+)", text)
    if m:
        p = Path(m.group(1))
        if p.is_file():
            return p

    m = re.search(r"job_id=(\d+)", text)
    if m:
        job_id = m.group(1)
        for candidate in results_root.glob(f"*-q{job_id}/*/uartlog"):
            if candidate.is_file():
                return candidate

    newest = None
    newest_mtime = job_started_at
    for u in results_root.glob("**/uartlog"):
        try:
            mt = u.stat().st_mtime
        except OSError:
            continue
        if mt > newest_mtime:
            newest_mtime = mt
            newest = u
    return newest


def _measured_makespan_us(trace_csv: Path) -> float | None:
    """Read the candidate's measured wall-clock makespan (last
    actual_end_cycles in the trace, divided by the assumed clock)."""
    if not trace_csv.is_file():
        return None
    import csv
    last_end = 0
    with open(trace_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            ae = row.get("actual_end_cycles", "").strip()
            if not ae:
                continue
            try:
                ae = int(ae)
            except ValueError:
                continue
            if ae > last_end:
                last_end = ae
    if last_end == 0:
        return None
    # Clock = 1 GHz (FireSim default for the chipyard SoC). Convert to µs.
    return last_end / 1000.0


def _predicted_makespan_us(fixture: Path) -> float | None:
    try:
        j = json.loads(fixture.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    d = j.get("dispatches", {})
    if not isinstance(d, dict):
        return None
    last_end = 0.0
    for entry in d.values():
        st = float(entry.get("start_time", 0.0))
        du = float(entry.get("duration", 0.0))
        end = st + du
        if end > last_end:
            last_end = end
    return last_end * 1e3  # ms -> µs (XPU-RT stores ms)


def run_one(
    candidate: dict[str, Any],
    workload_spec: Path,
    out_dir: Path,
    runner: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Build + run one candidate. Returns a manifest entry."""
    cid = candidate["id"]
    cand_dir = out_dir / cid
    cand_dir.mkdir(parents=True, exist_ok=True)

    if candidate["realizable_by"] != "xpurt":
        return {
            "id": cid,
            "axis": candidate.get("axis"),
            "status": "skipped",
            "reason": (
                f"realizable_by={candidate['realizable_by']!r} not yet "
                "supported by this driver (axis-C fusion codegen pending)"
            ),
        }

    fixture = _fixture_path(candidate)
    if not fixture.is_file():
        return {
            "id": cid,
            "axis": candidate.get("axis"),
            "status": "missing-fixture",
            "fixture": str(fixture),
        }

    cpu_p, cpu_e = _profile_hw_to_kinds(candidate["profile_hw"])
    models, quants = _models_quants_for_networks_json(workload_spec)

    sched_name = f"bundle_{cid}".replace("-", "_")
    env = {
        **os.environ,
        "FIRESIM_QUEUE": "1",
        "RUNNER": runner,
        "XPURT_TRACE": "1",
        "BACKENDS": "gemmini,rvv_opu",
        "REGISTRY": str(REPO_ROOT / "cores/chipyard_gemmini_opu_hetero.json"),
        "CPU_P_KIND": cpu_p,
        "CPU_E_KIND": cpu_e,
        "MODELS": models,
        "QUANTS": quants,
        "QUANT": quants.split(",")[0],
        "SCHEDULE_JSON": str(fixture),
        "SCHED_NAME": sched_name,
        # FORCE_REGEN=0 so the first candidate's (model, backend) staging
        # is reused; the per-candidate dispatch table + main.c are still
        # regenerated every run since SCHEDULE_JSON / SCHED_NAME differ.
        "FORCE_REGEN": "0",
    }

    stdout_log = cand_dir / "run_stdout.log"
    started_at = time.time()
    entry = {
        "id": cid,
        "axis": candidate.get("axis"),
        "solver": candidate.get("solver"),
        "scheduler": candidate.get("scheduler"),
        "profile_hw": candidate.get("profile_hw"),
        "fixture": str(fixture),
        "stdout_log": str(stdout_log),
        "predicted_makespan_us": _predicted_makespan_us(fixture),
    }
    if dry_run:
        entry["status"] = "dry-run"
        return entry

    # `uv run` activates the venv inside the subprocess so the
    # bare `python -m modelblaster.pipeline.*` invocations in
    # run.sh resolve. Without it, run.sh hits conda's python which
    # doesn't have the modelblaster package installed.
    cmd = ["uv", "run", "bash", str(REPO_ROOT / "examples/xpurt_demo/run.sh")]
    with open(stdout_log, "w") as f:
        proc = subprocess.run(
            cmd, env=env, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT)
    wall_s = time.time() - started_at
    entry["wall_s"] = round(wall_s, 1)

    # ELF copy only after a successful build — checking just for file
    # existence picks up stale artifacts from prior runs and lies in
    # the manifest.
    if proc.returncode == 0:
        elf_src = REPO_ROOT / f"examples/xpurt_demo/{env['QUANT']}/build/gemmini_rvv_opu_firesim/zephyr/zephyr.elf"
        if elf_src.is_file():
            elf_dst = cand_dir / "zephyr.elf"
            try:
                shutil.copy2(elf_src, elf_dst)
                entry["elf"] = str(elf_dst)
            except OSError as e:
                entry["elf_copy_error"] = str(e)

    uartlog = _find_uartlog(stdout_log, started_at)
    if uartlog and uartlog.is_file():
        uartlog_dst = cand_dir / "uartlog"
        shutil.copy2(uartlog, uartlog_dst)
        entry["uartlog"] = str(uartlog_dst)
        trace_csv = cand_dir / "xpurt_trace.csv"
        if _extract_trace(uartlog_dst, trace_csv):
            entry["trace_csv"] = str(trace_csv)
            mk = _measured_makespan_us(trace_csv)
            if mk is not None:
                entry["measured_makespan_us"] = mk

    if proc.returncode != 0:
        entry["status"] = "error"
        entry["returncode"] = proc.returncode
    else:
        entry["status"] = "ok"
    return entry


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", required=True, type=Path,
                   help="Path to firesim_batch.json (xpurt.candidate_bundle/v1).")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Output dir for per-candidate ELF/uartlog/trace.")
    p.add_argument("--include", default="",
                   help="Comma-separated allow-list of candidate ids to run "
                        "(plus 'baseline'). Default: all of them.")
    p.add_argument("--runner", default="firesim", choices=("firesim", "spike"),
                   help="run.sh RUNNER value. Default firesim.")
    p.add_argument("--dry-run", action="store_true",
                   help="Resolve fixtures + env but don't build or run.")
    args = p.parse_args(argv)

    bundle = json.loads(args.batch.read_text())
    # `contract` is informational on this file (XPU-RT's iterate_firesim
    # emits the bundle without an explicit contract key today). The
    # shape is fixed via `candidates[]` / `networks_json`. The baseline
    # config comes from the sibling iteration_result.json.
    if "candidates" not in bundle:
        raise SystemExit(
            "bundle is missing required key (candidates) — this driver "
            "expects xpurt.candidate_bundle/v1 shape")
    workload_spec = _resolve_workload_spec(bundle)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_candidates = [_baseline_candidate(bundle, args.batch)] + list(bundle.get("candidates", []))
    if args.include:
        wanted = set(s.strip() for s in args.include.split(",") if s.strip())
        all_candidates = [c for c in all_candidates if c["id"] in wanted]
        if not all_candidates:
            raise SystemExit(f"--include={args.include!r} matched no candidates")

    manifest = {
        "bundle": str(args.batch),
        "workload_spec": str(workload_spec),
        "runner": args.runner,
        "deadline_us": bundle.get("deadline_us"),
        "candidates": [],
    }

    for c in all_candidates:
        print(f"\n=== candidate {c['id']} (axis={c.get('axis')}) ===", flush=True)
        entry = run_one(c, workload_spec, args.out_dir, args.runner,
                        dry_run=args.dry_run)
        manifest["candidates"].append(entry)
        status = entry.get("status")
        pred = entry.get("predicted_makespan_us")
        meas = entry.get("measured_makespan_us")
        wall = entry.get("wall_s")
        print(f"  status={status} predicted={pred} measured={meas} wall_s={wall}",
              flush=True)
        # Write the manifest after every candidate so a crash partway
        # through doesn't lose what's already done.
        (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nwrote manifest: {args.out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
