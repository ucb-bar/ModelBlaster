"""Arm A driver: deterministic pipeline, curated kernels, no LLM.

For a given workload row in `config/workloads.yaml`, invokes the
matching `examples/<model>/run.sh` with BACKEND=reference and the
workload's TARGET/QUANT/RUNNER, captures the harness's stdout, parses
it for VERIFY/PROFILE/WALL_CYCLES markers, and writes the
aggregator's per-run artifacts under
`benchmarks/results/A/<workload-id>/<run-id>/`.

The shell + Python split is deliberate: shell handles the
`_run_lib.sh` env-var contract (target normalization, f16
auto-promotion, spike-fork env-var routing, the dev-box's stale
Vitis-cmake PATH dodge); Python handles parsing and the
provider-agnostic artifact schema.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from modelblaster.benchmarks.runners import spike as spike_runner
from modelblaster.benchmarks.runners import firesim as firesim_runner


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS_ROOT = REPO_ROOT / "benchmarks"
RESULTS_DIR = BENCHMARKS_ROOT / "results"
ARM_ID = "A"


@dataclass(frozen=True)
class Workload:
    id: str
    model: str
    target: str
    quant: str
    runner: str
    slice: Optional[str] = None
    blocked_by: Optional[str] = None


def _load_workload(workload_id: str) -> Workload:
    with open(BENCHMARKS_ROOT / "config" / "workloads.yaml") as f:
        raw = yaml.safe_load(f)["workloads"]
    for r in raw:
        if r["id"] == workload_id:
            return Workload(
                id=r["id"],
                model=r["model"],
                target=r["target"],
                quant=r["quant"],
                runner=r["runner"],
                slice=r.get("slice"),
                blocked_by=r.get("blocked_by"),
            )
    raise SystemExit(f"workload not found: {workload_id}")


def _run_id() -> str:
    """ISO-8601 UTC, safe for filesystem use."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _build_env(workload: Workload) -> dict[str, str]:
    """Env vars the example run.sh expects. The shell's auto-promotion
    handles _f16 IR -> rvv_f16/scalar_f16; we pass the workload's
    target as-is."""
    env = os.environ.copy()
    env["MODEL_NAME"] = workload.model
    env["TARGET"] = workload.target
    env["QUANT"] = workload.quant
    env["BACKEND"] = "reference"
    env["RUNNER"] = workload.runner
    env["OPTIMIZE"] = "0"
    env["GLOBAL_CURATED_DIR"] = str(REPO_ROOT / "kernels")
    return env


def _example_run_sh(model: str) -> Path:
    p = REPO_ROOT / "examples" / model / "run.sh"
    if not p.exists():
        raise SystemExit(f"missing example run script: {p}")
    return p


def _peak_rss_mb_of(rusage_before, rusage_after) -> float:
    """Linux `ru_maxrss` is in KB and is the max over the lifetime of
    all reaped children, so a single call's peak is the delta against
    the snapshot before the call. Returns MB."""
    delta_kb = max(0, rusage_after.ru_maxrss - rusage_before.ru_maxrss)
    return delta_kb / 1024.0


def _select_runner(name: str):
    if name == "spike":
        return spike_runner
    if name == "firesim":
        return firesim_runner
    raise SystemExit(f"unsupported runner: {name} (expected spike|firesim)")


def _require_tools(runner: str) -> None:
    """The harness invokes `west build`, which lives in the Zephyr
    toolchain. The repo's quick-start expects `conda activate zephyr`
    + `scripts/set_envvars_sdk.sh` to be sourced before any run.sh
    invocation. Surface that prerequisite up front rather than letting
    the subprocess fail mid-build with a less actionable message."""
    needed = ["west"]
    if runner == "spike":
        needed.append("spike")
    missing = [t for t in needed if shutil.which(t) is None]
    if missing:
        raise SystemExit(
            "missing tools on PATH: " + ", ".join(missing) + ". "
            "Activate the zephyr build env first, e.g.:\n"
            "  source tools/miniforge3/etc/profile.d/conda.sh && "
            "conda activate zephyr\n"
            "  source scripts/set_envvars_sdk.sh"
        )


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Arm A driver: curated kernels, no LLM.")
    ap.add_argument("--workload", required=True,
                    help="workload id from config/workloads.yaml")
    ap.add_argument("--run-id", default=None,
                    help="override run-id directory name (default: UTC timestamp)")
    args = ap.parse_args(argv)

    workload = _load_workload(args.workload)
    if workload.blocked_by:
        print(f"workload {workload.id} is blocked_by: {workload.blocked_by}",
              file=sys.stderr)
        return 2

    runner = _select_runner(workload.runner)

    run_id = args.run_id or _run_id()
    out_dir = RESULTS_DIR / ARM_ID / workload.id / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    run_sh = _example_run_sh(workload.model)
    env = _build_env(workload)

    started_at = datetime.now(timezone.utc).isoformat()
    rusage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.monotonic()

    _require_tools(workload.runner)

    proc = subprocess.run(
        ["bash", str(run_sh)],
        env=env, cwd=str(REPO_ROOT),
        capture_output=True, text=True,
    )

    wall_clock_s = time.monotonic() - t0
    rusage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    ended_at = datetime.now(timezone.utc).isoformat()

    (out_dir / "stdout.log").write_text(proc.stdout)
    (out_dir / "stderr.log").write_text(proc.stderr)

    parsed = runner.parse_stdout(proc.stdout)
    runner.write_accuracy(out_dir, parsed["verify"])
    runner.write_profile_csv(out_dir, parsed["profile"])
    runner.write_wall_cycles(out_dir, parsed["wall_cycles"])

    # Best-effort env snapshot for reproducibility. Strip the ambient
    # PATH/PYTHONPATH noise that would change run-to-run.
    keep = re.compile(r"^(MODEL_NAME|TARGET|QUANT|BACKEND|RUNNER|OPTIMIZE|"
                      r"GLOBAL_CURATED_DIR|PROFILE_|FIRESIM_|MODELBLASTER_)")
    env_snapshot = {k: v for k, v in env.items() if keep.match(k)}
    (out_dir / "env.txt").write_text(
        "\n".join(f"{k}={v}" for k, v in sorted(env_snapshot.items())) + "\n"
    )

    run_record = {
        "schema_version": 1,
        "arm": ARM_ID,
        "workload_id": workload.id,
        "run_id": run_id,
        "git_sha": _git_sha(),
        "started_at": started_at,
        "ended_at": ended_at,
        "wall_clock_s": round(wall_clock_s, 3),
        "peak_rss_mb": round(_peak_rss_mb_of(rusage_before, rusage_after), 1),
        "exit_status": "ok" if proc.returncode == 0 else f"exit_{proc.returncode}",
        "model": workload.model,
        "target": workload.target,
        "quant": workload.quant,
        "runner": workload.runner,
    }
    with open(out_dir / "run.json", "w") as f:
        json.dump(run_record, f, indent=2)

    # Update the `latest` symlink atomically. Use a tmp + rename so a
    # concurrent aggregator does not catch a missing-target window.
    base = out_dir.parent
    tmp = base / ".latest.new"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    tmp.symlink_to(run_id)
    tmp.replace(base / "latest")

    if proc.returncode != 0:
        print(f"run.sh exited non-zero ({proc.returncode}); "
              f"artifacts saved under {out_dir}", file=sys.stderr)
        return proc.returncode

    print(f"OK: {workload.id} -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
