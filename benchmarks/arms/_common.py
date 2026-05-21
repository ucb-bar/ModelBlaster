"""Shared orchestration between arm drivers.

Every arm:

* looks up its workload row in `config/workloads.yaml`,
* builds an environment for the example's run.sh,
* shells out, captures stdout/stderr, times wall-clock and peak RSS,
* parses the harness markers into the aggregator's artifact schema
  (run.json, accuracy.json, profile_<runner>.csv, wall_cycles.txt,
  env.txt, stdout.log, stderr.log),
* atomically swings the `<workload>/latest` symlink.

The arm-specific bits are: which env vars to set, which tools to
require on PATH, and any per-arm rollups (Arm B synthesizes
`llm_tokens.json` from the per-call JSONL the bedrock client emits).
This module is the orchestrator; the arm files are thin policy.
"""

from __future__ import annotations

import json
import os
import re
import resource
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from modelblaster.benchmarks.runners import spike as spike_runner
from modelblaster.benchmarks.runners import firesim as firesim_runner


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS_ROOT = REPO_ROOT / "benchmarks"
RESULTS_DIR = BENCHMARKS_ROOT / "results"


@dataclass(frozen=True)
class Workload:
    id: str
    model: str
    target: str
    quant: str
    runner: str
    slice: Optional[str] = None
    blocked_by: Optional[str] = None
    firesim_eval: bool = False


def load_workload(workload_id: str) -> Workload:
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
                firesim_eval=bool(r.get("firesim_eval", False)),
            )
    raise SystemExit(f"workload not found: {workload_id}")


def new_run_id() -> str:
    """ISO-8601 UTC, filesystem-safe."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def example_run_sh(model: str) -> Path:
    p = REPO_ROOT / "examples" / model / "run.sh"
    if not p.exists():
        raise SystemExit(f"missing example run script: {p}")
    return p


def require_tools(tools: list[str]) -> None:
    """Fail fast when a required tool is missing on PATH. The Zephyr
    build env (west, spike) lives behind a conda activation; surface
    that prerequisite up front rather than after several minutes of
    codegen."""
    missing = [t for t in tools if shutil.which(t) is None]
    if missing:
        raise SystemExit(
            "missing tools on PATH: " + ", ".join(missing) + ". "
            "Activate the zephyr build env first, e.g.:\n"
            "  source tools/miniforge3/etc/profile.d/conda.sh && "
            "conda activate zephyr\n"
            "  source scripts/set_envvars_sdk.sh"
        )


def required_tools_for(runner: str) -> list[str]:
    needed = ["west"]
    if runner == "spike":
        needed.append("spike")
    # firesim is invoked by the runner script's own logic; the wrapper
    # already exits informatively when its sub-tools are missing, so we
    # do not require firesim on PATH here.
    return needed


def select_runner(name: str):
    if name == "spike":
        return spike_runner
    if name == "firesim":
        return firesim_runner
    raise SystemExit(f"unsupported runner: {name} (expected spike|firesim)")


def peak_rss_mb_delta(before, after) -> float:
    """Linux ru_maxrss is the high-water mark across all reaped children
    of this process; the delta against a snapshot taken just before the
    subprocess call approximates that call's peak. Returns MB."""
    delta_kb = max(0, after.ru_maxrss - before.ru_maxrss)
    return delta_kb / 1024.0


_ENV_KEEP_PATTERN = re.compile(
    r"^(MODEL_NAME|TARGET|QUANT|BACKEND|RUNNER|OPTIMIZE|"
    r"BEAM|EXPANSIONS|ITERATIONS|FIRESIM_EVAL|"
    r"GLOBAL_CURATED_DIR|BEDROCK_CALLS_LOG|"
    r"PROFILE_|FIRESIM_|MODELBLASTER_)"
)


def write_env_snapshot(out_dir: Path, env: dict[str, str]) -> None:
    """Persist the harness-relevant env vars only; the ambient PATH /
    PYTHONPATH / TERM / ... would change between machines and have no
    bearing on reproduction."""
    keep = {k: v for k, v in env.items() if _ENV_KEEP_PATTERN.match(k)}
    (out_dir / "env.txt").write_text(
        "\n".join(f"{k}={v}" for k, v in sorted(keep.items())) + "\n"
    )


def write_run_json(
    out_dir: Path,
    *,
    arm: str,
    workload: Workload,
    run_id: str,
    started_at: str,
    ended_at: str,
    wall_clock_s: float,
    peak_rss_mb: float,
    exit_status: str,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    record: dict[str, Any] = {
        "schema_version": 1,
        "arm": arm,
        "workload_id": workload.id,
        "run_id": run_id,
        "git_sha": git_sha(),
        "started_at": started_at,
        "ended_at": ended_at,
        "wall_clock_s": round(wall_clock_s, 3),
        "peak_rss_mb": round(peak_rss_mb, 1),
        "exit_status": exit_status,
        "model": workload.model,
        "target": workload.target,
        "quant": workload.quant,
        "runner": workload.runner,
    }
    if extra:
        record.update(extra)
    with open(out_dir / "run.json", "w") as f:
        json.dump(record, f, indent=2)


def update_latest_symlink(run_dir: Path) -> None:
    """Atomically swing <workload-dir>/latest -> <run-id> so a
    concurrent aggregator never catches a missing-target window."""
    base = run_dir.parent
    tmp = base / ".latest.new"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    tmp.symlink_to(run_dir.name)
    tmp.replace(base / "latest")


@dataclass
class RunOutcome:
    out_dir: Path
    returncode: int
    started_at: str
    ended_at: str
    wall_clock_s: float
    peak_rss_mb: float


def execute_run_sh(
    *, arm: str, workload: Workload, env: dict[str, str], run_id: str,
) -> RunOutcome:
    """Run the example's run.sh under the given env, capture both
    streams, parse the harness markers, and write all per-cell
    artifacts. Caller owns the post-run rollup (e.g. llm_tokens.json
    for Arm B); this function handles the common artifact set."""
    runner = select_runner(workload.runner)
    require_tools(required_tools_for(workload.runner))

    out_dir = RESULTS_DIR / arm / workload.id / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc).isoformat()
    rusage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.monotonic()

    run_sh = example_run_sh(workload.model)
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

    write_env_snapshot(out_dir, env)

    return RunOutcome(
        out_dir=out_dir,
        returncode=proc.returncode,
        started_at=started_at,
        ended_at=ended_at,
        wall_clock_s=wall_clock_s,
        peak_rss_mb=peak_rss_mb_delta(rusage_before, rusage_after),
    )


def finalize(
    outcome: RunOutcome, *, arm: str, workload: Workload, run_id: str,
    extra_run_json: Optional[dict[str, Any]] = None,
) -> int:
    """Write run.json, swing the latest symlink, return an exit code
    suitable for the driver's main()."""
    write_run_json(
        outcome.out_dir,
        arm=arm,
        workload=workload,
        run_id=run_id,
        started_at=outcome.started_at,
        ended_at=outcome.ended_at,
        wall_clock_s=outcome.wall_clock_s,
        peak_rss_mb=outcome.peak_rss_mb,
        exit_status=("ok" if outcome.returncode == 0
                     else f"exit_{outcome.returncode}"),
        extra=extra_run_json,
    )
    update_latest_symlink(outcome.out_dir)

    if outcome.returncode != 0:
        print(f"run.sh exited non-zero ({outcome.returncode}); "
              f"artifacts saved under {outcome.out_dir}", file=sys.stderr)
        return outcome.returncode

    print(f"OK: {workload.id} -> {outcome.out_dir}")
    return 0
