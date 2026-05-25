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
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from modelblaster.benchmarks.runners import spike as spike_runner
from modelblaster.benchmarks.runners import firesim as firesim_runner
from modelblaster.benchmarks.runners import _hetero_artifacts


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
    # Hetero workloads need a pre-generated XPU-RT schedule and a
    # cores registry; both are pinned on the workload row so the
    # driver knows what to hand to examples/xpurt_demo/run.sh.
    xpurt_schedule_path: Optional[str] = None
    xpurt_cores_registry: Optional[str] = None
    xpurt_backends: Optional[str] = None


# Targets that don't have a per-model run.sh of their own; the harness
# instead dispatches through the multi-backend XPU-RT demo runner.
HETERO_TARGETS = frozenset({"hetero_gemmini_opu"})


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
                xpurt_schedule_path=r.get("xpurt_schedule_path"),
                xpurt_cores_registry=r.get("xpurt_cores_registry"),
                xpurt_backends=r.get("xpurt_backends"),
            )
    raise SystemExit(f"workload not found: {workload_id}")


def apply_runner_override(
    workload: Workload, override: Optional[str],
) -> Workload:
    """Return a Workload with `runner` swapped to `override` if set.

    The override is the iteration-speed knob the user reaches for when
    a workload's default runner is FireSim but they only want functional
    correctness + the inner-loop verify to ride spike. On the
    hetero_gemmini_opu target this auto-selects spike-hetero via
    `hetero_env_overlay`; the resulting cycle counts on accelerator
    ops are NOT authoritative, so this override is for iteration only,
    not baseline capture.
    """
    if override is None or override == workload.runner:
        return workload
    if override not in ("spike", "firesim"):
        raise SystemExit(
            f"--runner-override must be spike|firesim, got {override!r}"
        )
    if (workload.target in HETERO_TARGETS and override == "spike"):
        # Surface the cycle-source caveat on stderr so it lands in the
        # cell's run dir alongside the data.
        print(
            f"NOTE: --runner-override spike on hetero target "
            f"{workload.target}: cycles on accelerator ops are functional-"
            f"only (spike-hetero atomic exec). Use firesim for the "
            f"baseline capture.",
            file=sys.stderr,
        )
    return replace(workload, runner=override)


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


def xpurt_demo_run_sh() -> Path:
    p = REPO_ROOT / "examples" / "xpurt_demo" / "run.sh"
    if not p.exists():
        raise SystemExit(f"missing xpurt_demo runner: {p}")
    return p


# Default cores registry + backend set for the `hetero_gemmini_opu`
# target. Per-workload overrides come from the workload row's
# `xpurt_cores_registry` / `xpurt_backends` fields.
_HETERO_GEMMINI_OPU_DEFAULTS = {
    "registry": "cores/chipyard_gemmini_opu_hetero.json",
    "backends": "gemmini,rvv_opu",
}


def _hetero_defaults(target: str) -> dict[str, str]:
    if target == "hetero_gemmini_opu":
        return _HETERO_GEMMINI_OPU_DEFAULTS
    return {}


def hetero_env_overlay(workload: Workload, env: dict[str, str]) -> None:
    """Add the env vars the multi-backend `xpurt_demo/run.sh` expects
    on top of the per-workload env the caller already populated. Sets:

      MODELS, BACKENDS, REGISTRY, SCHEDULE_JSON, QUANT, RUNNER,
      MODELBLASTER_HETERO_SPIKE (when RUNNER=spike on the
      hetero_gemmini_opu target).
    """
    defaults = _hetero_defaults(workload.target)

    env["MODELS"] = workload.model
    env["BACKENDS"] = workload.xpurt_backends or defaults.get("backends", "")
    env["QUANT"] = workload.quant
    env["RUNNER"] = workload.runner

    registry_rel = workload.xpurt_cores_registry or defaults.get("registry")
    if registry_rel:
        env["REGISTRY"] = str(REPO_ROOT / registry_rel)
    if workload.xpurt_schedule_path:
        env["SCHEDULE_JSON"] = str(
            (REPO_ROOT / workload.xpurt_schedule_path).resolve()
        )

    # spike-hetero is the merlin-side wrapper that loads both Gemmini
    # and Saturn-OPU extensions into one spike process. Point at it so
    # `examples/xpurt_demo/run.sh` picks it up for the hetero target;
    # the user can override by setting MODELBLASTER_HETERO_SPIKE
    # themselves before invoking the driver.
    if (workload.target == "hetero_gemmini_opu"
            and workload.runner == "spike"
            and "MODELBLASTER_HETERO_SPIKE" not in env):
        candidate = Path("/scratch2/agustin/merlin/tools/spike-hetero/spike-hetero")
        if candidate.exists():
            env["MODELBLASTER_HETERO_SPIKE"] = str(candidate)

    # Turn on the harness's per-entry execution trace so the aggregator
    # has makespan / per-tile utilization metrics to read. The trace
    # block is otherwise compiled out (saves a few hundred bytes of
    # data + the cycle-counter reads around each dispatch).
    env.setdefault("XPURT_TRACE", "1")


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
    r"MODELS|BACKENDS|REGISTRY|SCHEDULE_JSON|"
    r"GLOBAL_CURATED_DIR|"
    r"BEDROCK_CALLS_LOG|GEMINI_CALLS_LOG|CLAUDE_CODE_CALLS_LOG|"
    r"LLM_PROVIDER|GEMINI_MODEL|CLAUDE_CODE_MODEL|"
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

    if workload.target in HETERO_TARGETS:
        if not workload.xpurt_schedule_path:
            raise SystemExit(
                f"workload {workload.id!r} targets {workload.target} but "
                f"its row in workloads.yaml has no xpurt_schedule_path. "
                f"Generate a schedule via FreshScheduler "
                f"(/scratch2/dima/misc_sw/FreshScheduler/scripts/"
                f"run_xpurt_schedule.py) and point the workload at it."
            )
        hetero_env_overlay(workload, env)
        run_sh = xpurt_demo_run_sh()
    else:
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
    runner.write_profile_csv(out_dir, parsed["profile"],
                             trace_rows=parsed.get("xpurt_trace_rows"))
    runner.write_wall_cycles(out_dir, parsed["wall_cycles"])
    runner.write_xpurt_trace(out_dir, parsed.get("xpurt_trace"))

    # Hetero-only: emit a static cross-tile bytes estimate by joining
    # graph.json (per-op output tensor sizes) with the workload's
    # schedule.json (per-dispatch slot label). The aggregator's
    # cross_tile_bytes extractor reads the resulting file.
    if workload.target in HETERO_TARGETS:
        _emit_cross_tile_estimate(workload, out_dir)

    # All cells: snapshot the pipeline's passes_applied.json so the
    # aggregator can answer "which fusion/fold passes fired during
    # this IR build?" without re-running extract_graph.
    _copy_passes_applied(workload, out_dir)

    # Arm B-* only: snapshot the optimize-loop trajectory so the
    # aggregator can read per-candidate progress, yield rate, and
    # token cost. Single-target workloads write to
    # generated/<target>/; the hetero path writes per-model files
    # under each model's generated/<target>/ via the xpurt_demo
    # regen step.
    _copy_optimize_artifacts(workload, workload.target, out_dir)

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


def _emit_cross_tile_estimate(workload: Workload, out_dir: Path) -> None:
    """Resolve the canonical graph.json + schedule.json paths for a
    hetero workload and call the runner-side helper to compute the
    cross-tile bytes upper bound. Silent on missing inputs --
    `cross_tile_estimate.json` simply won't be created and the
    aggregator's nullable_if rule leaves the metric blank."""
    if workload.xpurt_schedule_path is None:
        return
    schedule_path = (REPO_ROOT / workload.xpurt_schedule_path).resolve()
    # Convention: per-model IRs land at examples/<model>/<quant>/generated/graph.json.
    # That's what extract_graph and the xpurt_demo regen step both
    # produce, so it's the right place to read from.
    graph_path = (REPO_ROOT / "examples" / workload.model
                  / workload.quant / "generated" / "graph.json")
    _hetero_artifacts.write_cross_tile_estimate(
        out_dir,
        graph_path if graph_path.exists() else None,
        schedule_path if schedule_path.exists() else None,
    )


def _copy_passes_applied(workload: Workload, out_dir: Path) -> None:
    """Snapshot the IR extractor's passes_applied.json into the cell
    run dir so the aggregator can read it via metrics.yaml's
    relative-path source. The pipeline writes it next to graph.json
    in examples/<model>/<quant>/generated/; if it's missing (older
    IR build), we skip silently and the dashboard column stays blank."""
    src = (REPO_ROOT / "examples" / workload.model
           / workload.quant / "generated" / "passes_applied.json")
    if src.exists():
        (out_dir / "passes_applied.json").write_text(src.read_text())


def _copy_optimize_artifacts(workload: Workload, target: str,
                             out_dir: Path) -> None:
    """For Arm B-* runs (BACKEND=llm + OPTIMIZE=1), snapshot
    optimize_summary.json + beam_search_trajectory.jsonl from
    examples/<model>/<quant>/generated/<target>/ into the cell run
    dir. Arm A never produces them (no LLM calls); Arm B-* always
    does when the optimize loop runs."""
    src_dir = (REPO_ROOT / "examples" / workload.model
               / workload.quant / "generated" / target)
    for fname in ("optimize_summary.json", "beam_search_trajectory.jsonl"):
        src = src_dir / fname
        if src.exists():
            (out_dir / fname).write_text(src.read_text())


def synthesize_llm_tokens(
    calls_log: Path, out_path: Path, *, provider: str,
) -> None:
    """Roll up a JSONL of per-call LLM usage records into the
    aggregator's `llm_tokens.json` schema. Provider-agnostic: works
    against any JSONL whose lines have `model_id`, `input_tokens`,
    `output_tokens`, and `cache_read_input_tokens` fields (the shape
    that `pipeline.bedrock_client` and `pipeline.gemini_client` both
    emit). The aggregator's dollars_equivalent extractor then prices
    each model_id against `config/pricing.yaml`.

    `provider` is recorded in the rollup but does not affect pricing
    — pricing.yaml keys on model_id, which is what matters for cost.
    """
    if not calls_log.exists():
        # No LLM calls happened (e.g. all-curated runs); emit a zero
        # rollup so the aggregator does not flag the cell as
        # "missing tokens" when the run was legitimately free.
        empty: dict[str, Any] = {
            "schema_version": 1,
            "provider": provider,
            "tokens_input_cached": 0,
            "tokens_input_uncached": 0,
            "tokens_output": 0,
            "n_calls": 0,
            "by_model": {},
        }
        out_path.write_text(json.dumps(empty, indent=2))
        return

    per_model: dict[str, dict[str, int]] = defaultdict(
        lambda: {"input_cached": 0, "input_uncached": 0, "output": 0,
                 "calls": 0}
    )
    total_in_cached = total_in_uncached = total_out = total_calls = 0
    with open(calls_log) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            mid = rec.get("model_id") or "unknown"
            slot = per_model[mid]
            cached = int(rec.get("cache_read_input_tokens") or 0)
            input_t = int(rec.get("input_tokens") or 0)
            output_t = int(rec.get("output_tokens") or 0)
            # Bedrock/Gemini both report `input_tokens` as the total
            # input (cache-read inclusive); subtract to land in the
            # uncached/cached buckets the cost extractor expects.
            uncached = max(0, input_t - cached)
            slot["input_cached"] += cached
            slot["input_uncached"] += uncached
            slot["output"] += output_t
            slot["calls"] += 1
            total_in_cached += cached
            total_in_uncached += uncached
            total_out += output_t
            total_calls += 1

    rollup: dict[str, Any] = {
        "schema_version": 1,
        "provider": provider,
        "tokens_input_cached": total_in_cached,
        "tokens_input_uncached": total_in_uncached,
        "tokens_output": total_out,
        "n_calls": total_calls,
        "by_model": {k: dict(v) for k, v in per_model.items()},
    }
    out_path.write_text(json.dumps(rollup, indent=2))
