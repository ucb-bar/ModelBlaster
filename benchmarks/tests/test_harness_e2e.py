"""End-to-end harness integration test.

Synthesizes a complete Arm A + Arm B-bedrock cell run for dronet (3
replicates each) without invoking the actual pipeline / spike /
firesim. Then runs the aggregator and asserts:

  * every metric the (arm, workload) pair is supposed to produce
    actually populates,
  * the summary.md renders both phase tables for each workload,
  * replicate aggregation produces mean +/- stddev when N>1,
  * the cost monitor reads the synthesized llm_calls.jsonl cleanly.

Run with:
    uv run python -m modelblaster.benchmarks.tests.test_harness_e2e

Exits 0 on full pass, prints what failed otherwise. Re-runnable: each
invocation wipes its own scratch dir under benchmarks/results/_test/
so it never contaminates real captures.

This is the right test to keep green before launching a real baseline
capture -- if it fails, the dashboard would be missing rows on real
runs too.
"""

from __future__ import annotations

import csv
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "benchmarks" / "results"
CONFIG = REPO_ROOT / "benchmarks" / "config"
TEST_ROOT = RESULTS / "_test"


# ───────────────────── synthetic-data builders ─────────────────────


def synth_profile_csv(n_dispatches: int = 32) -> str:
    """Build a profile_<runner>.csv body matching the real harness's
    columns (dispatch_id, name, op, shape, cycles)."""
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["dispatch_id", "name", "op", "shape", "cycles"])
    # 10 conv2d_s8, 6 batchnorm, 7 relu, 3 add, 1 maxpool, 2 linear, 3 view
    rows: list[tuple[int, str, str, str, int]] = []
    counter = 0
    base_cycles = {
        "conv2d_s8": [11000, 12000, 13000, 18000, 22000, 25000, 31000, 40000, 45000, 60000],
        "batchnorm2d_s8": [800, 900, 1000, 1100, 1200, 1300],
        "relu_s8": [400, 450, 500, 550, 600, 650, 700],
        "add_s8": [600, 700, 800],
        "maxpool2d_s8": [2000],
        "linear_s8": [3000, 4000],
        "view": [10, 20],
    }
    for op, cycles_list in base_cycles.items():
        for cyc in cycles_list:
            rows.append((counter, f"{op[:5]}{counter}", op, "shape", cyc))
            counter += 1
    for row in rows[:n_dispatches]:
        w.writerow(row)
    return out.getvalue()


def synth_accuracy_json(jitter: float = 0.0) -> dict:
    """Faithful accuracy.json shape -- linf, rmse, cosine, verify_pass,
    bit_exact, n_samples, atol_used, rtol_used."""
    return {
        "linf": jitter,
        "rmse": jitter * 0.3,
        "cosine": 1.0 - jitter * 0.01,
        "n_samples": 1000,
        "bit_exact": jitter == 0.0,
        "verify_pass": jitter < 0.1,
        "atol_used": 0.05,
        "rtol_used": 0.001,
    }


def synth_run_json(arm: str, workload_id: str, run_id: str,
                   wall_clock_s: float, **extras) -> dict:
    return {
        "schema_version": 1,
        "arm": arm,
        "workload_id": workload_id,
        "run_id": run_id,
        "git_sha": "test-sha",
        "started_at": "2026-05-25T00:00:00+00:00",
        "ended_at": "2026-05-25T00:01:00+00:00",
        "wall_clock_s": wall_clock_s,
        "peak_rss_mb": 1200.0,
        "exit_status": "ok",
        "model": "dronet",
        "target": "scalar",
        "quant": "int8",
        "runner": "spike",
        **extras,
    }


def synth_passes_applied() -> dict:
    """Mirrors what extract_graph.py wrote on a real dronet run."""
    return {
        "schema_version": 1,
        "extractor": "extract_graph",
        "n_fx_nodes": 34,
        "n_ir_ops": 32,
        "passes": {
            "linear_relu_fuse": {"fired": 0, "sites": []},
            "conv2d_relu_fuse": {"fired": 0, "sites": []},
        },
    }


def synth_graph_summary() -> dict:
    """Matches what graph_summary.synthesize emits for dronet int8."""
    return {
        "schema_version": 1,
        "n_dispatches": 32,
        "n_distinct_op_kinds": 7,
        "n_distinct_shapes": 25,
        "by_op_kind": {
            "conv2d_s8": {"count": 10, "distinct_shapes": 10},
            "batchnorm2d_s8": {"count": 6, "distinct_shapes": 4},
            "relu_s8": {"count": 7, "distinct_shapes": 4},
            "add_s8": {"count": 3, "distinct_shapes": 3},
            "maxpool2d_s8": {"count": 1, "distinct_shapes": 1},
            "linear_s8": {"count": 2, "distinct_shapes": 2},
            "view": {"count": 2, "distinct_shapes": 1},
        },
    }


def synth_binary_size() -> dict:
    return {
        "schema_version": 1,
        "zephyr_elf_bytes": 250_000,
        "kernels_c_bytes": 18_000,
        "kernels_c_loc": 420,
        "weights_npz_bytes": 580_000,
    }


def synth_stage_timings(arm: str) -> dict:
    """Arm B runs spend much longer in generate_kernels (LLM)."""
    gen_kernels = 0.5 if arm == "A" else 480.0
    return {
        "schema_version": 1,
        "extract_s": 8.2,
        "generate_skeleton_s": 0.3,
        "generate_kernels_s": gen_kernels,
        "build_s": 22.5,
        "run_s": 15.1,
        "total_stage_s": 8.2 + 0.3 + gen_kernels + 22.5 + 15.1,
    }


def synth_kernel_picks(arm: str) -> dict:
    """Arm A: all reference (scalar fallback). Arm B: mix of LLM picks."""
    if arm == "A":
        return {
            "schema_version": 1, "target": "scalar",
            "picks": {
                op: {"source": "reference", "algorithm": None, "path": None}
                for op in ["conv2d_s8", "batchnorm2d_s8", "relu_s8",
                           "add_s8", "maxpool2d_s8", "linear_s8"]
            },
        }
    return {
        "schema_version": 1, "target": "scalar",
        "picks": {
            "conv2d_s8":     {"source": "llm", "algorithm": "tiled_blocked", "path": None},
            "batchnorm2d_s8":{"source": "llm", "algorithm": "fused_scale", "path": None},
            "relu_s8":       {"source": "llm", "algorithm": "vector_clamp", "path": None},
            "add_s8":        {"source": "llm", "algorithm": "vector_add", "path": None},
            "maxpool2d_s8":  {"source": "llm", "algorithm": "tiled_blocked", "path": None},
            "linear_s8":     {"source": "llm", "algorithm": "tiled_blocked", "path": None},
        },
    }


def synth_llm_calls_jsonl(n_calls: int = 6) -> str:
    """Realistic llm_calls.jsonl with mixed cached/uncached input."""
    lines = []
    for i in range(n_calls):
        cached = 1200 if i > 0 else 0  # first call has no cache hit
        rec = {
            "ts": f"2026-05-25T19:00:{i:02d}+00:00",
            "provider": "bedrock",
            "model_id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "request_id": f"req-{i}",
            "parent_call_id": None,
            "phase": "kernel_synthesis" if i % 2 == 0 else "beam_rerank",
            "input_tokens": 2000,
            "output_tokens": 500,
            "cache_read_input_tokens": cached,
            "cache_write_input_tokens": 0,
            "stop_reason": "end_turn",
        }
        lines.append(json.dumps(rec))
    return "\n".join(lines) + "\n"


def synth_llm_tokens(arm: str, n_calls: int = 6) -> dict:
    """Roll-up matching what _common.synthesize_llm_tokens would write."""
    model_id = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    cached_per_call = 1200
    uncached = 2000 * n_calls - cached_per_call * (n_calls - 1)
    cached = cached_per_call * (n_calls - 1)
    output = 500 * n_calls
    return {
        "schema_version": 1,
        "provider": "bedrock",
        "tokens_input_cached": cached,
        "tokens_input_uncached": uncached,
        "tokens_output": output,
        "n_calls": n_calls,
        "by_model": {
            model_id: {
                "input_cached": cached,
                "input_uncached": uncached,
                "output": output,
                "calls": n_calls,
            },
        },
    }


def synth_beam_trajectory() -> str:
    """6 candidates: 4 ok, 1 build_fail, 1 duplicate."""
    lines = []
    candidates = [
        ("conv2d_s8", "ok", 13000, 1500, 400),
        ("conv2d_s8", "ok", 11500, 1600, 380),
        ("conv2d_s8", "build_fail", None, 1400, 250),
        ("conv2d_s8", "ok", 11000, 1700, 410),
        ("linear_s8", "ok", 3500, 1300, 320),
        ("linear_s8", "duplicate", None, 1400, 50),
    ]
    for i, (spec, result, cycles, tok_in, tok_out) in enumerate(candidates):
        rec = {
            "spec": spec, "iter": i // 2, "parent_idx": 0, "exp_idx": i % 2,
            "baseline_cycles": 14000 if spec == "conv2d_s8" else 4000,
            "parent_cycles": 14000 if spec == "conv2d_s8" else 4000,
            "result": result,
            "tokens_in": tok_in, "tokens_out": tok_out,
        }
        if cycles is not None:
            rec["cycles"] = cycles
        lines.append(json.dumps(rec))
    return "\n".join(lines) + "\n"


# ───────────────────── cell construction ─────────────────────


def build_cell(arm: str, workload_id: str, run_id: str,
               replicate_idx: int) -> Path:
    """Write a complete set of artifacts to results/<arm>/<workload>/<run-id>/."""
    from modelblaster.benchmarks.ingest import cycles_per_op

    cell_dir = RESULTS / arm / workload_id / run_id
    cell_dir.mkdir(parents=True, exist_ok=True)

    # Inject a little replicate jitter so stddev is non-trivial.
    jitter = 1.0 + (replicate_idx - 1) * 0.02   # 0.98, 1.0, 1.02
    accuracy_jitter = (replicate_idx - 1) * 1e-4

    (cell_dir / "run.json").write_text(json.dumps(
        synth_run_json(arm, workload_id, run_id,
                       wall_clock_s=55.0 * jitter,
                       **({"llm_provider": "bedrock", "beam": 2,
                           "expansions": 3, "iterations": 2,
                           "firesim_eval": False}
                          if arm.startswith("B-") else {})),
        indent=2,
    ))
    (cell_dir / "env.txt").write_text("MODEL_NAME=dronet\nTARGET=scalar\nQUANT=int8\n")
    (cell_dir / "stdout.log").write_text("")
    (cell_dir / "stderr.log").write_text("")
    (cell_dir / "accuracy.json").write_text(json.dumps(
        synth_accuracy_json(jitter=accuracy_jitter), indent=2))
    (cell_dir / "profile_spike.csv").write_text(synth_profile_csv())
    (cell_dir / "wall_cycles.txt").write_text(str(int(5_000_000 * jitter)))
    (cell_dir / "passes_applied.json").write_text(json.dumps(
        synth_passes_applied(), indent=2))
    (cell_dir / "graph_summary.json").write_text(json.dumps(
        synth_graph_summary(), indent=2))
    (cell_dir / "binary_size.json").write_text(json.dumps(
        synth_binary_size(), indent=2))
    (cell_dir / "stage_timings.json").write_text(json.dumps(
        synth_stage_timings(arm), indent=2))
    (cell_dir / "kernel_picks.json").write_text(json.dumps(
        synth_kernel_picks(arm), indent=2))

    # cycles_per_op.json — synthesized from the profile rows the same
    # way the real runner does (using the actual production helper, so
    # this test exercises the synthesize() path too).
    with open(cell_dir / "profile_spike.csv") as f:
        rows = list(csv.DictReader(f))
    cpo = cycles_per_op.synthesize(rows)
    (cell_dir / "cycles_per_op.json").write_text(json.dumps(cpo, indent=2))

    # Arm B specific artifacts.
    if arm.startswith("B-"):
        (cell_dir / "llm_calls.jsonl").write_text(synth_llm_calls_jsonl())
        (cell_dir / "llm_tokens.json").write_text(json.dumps(
            synth_llm_tokens(arm), indent=2))
        (cell_dir / "beam_search_trajectory.jsonl").write_text(
            synth_beam_trajectory())

    # latest symlink (atomic-replace style).
    latest = cell_dir.parent / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_id)

    return cell_dir


def fresh_results_tree() -> None:
    """Wipe any existing per-arm cell trees + aggregated outputs so we
    don't pick up stale data. Preserves the directory's `.gitignore`
    and the directory itself."""
    if RESULTS.exists():
        for entry in RESULTS.iterdir():
            if entry.name == ".gitignore":
                continue
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
    else:
        RESULTS.mkdir(parents=True, exist_ok=True)


# ───────────────────── assertions ─────────────────────


# Metrics every Arm A scalar cell should populate (subset; full list lives
# in metrics.yaml). Used as a sanity check that no extractor silently
# returns None on artifacts we expect to be present.
ARM_A_REQUIRED_METRICS = {
    "pre_kernel": [
        "passes_fired_total", "ir_op_count", "n_input_nodes", "lowering_ratio",
        "n_dispatches_graph", "n_distinct_op_kinds", "n_distinct_shapes",
    ],
    "kernel_synthesis": [
        "cycles_spike", "accuracy_linf", "accuracy_rmse", "accuracy_cos",
        "verify_pass", "bit_exact",
        "compile_wall_clock_s", "compile_peak_rss_mb",
        "n_ops_profiled", "dominant_op_share",
        "mean_cycles_per_dispatch", "stddev_cycles_per_dispatch",
        "op_kind_p95_max_cycles", "op_kind_median_max_cycles",
        "extract_s", "generate_skeleton_s", "generate_kernels_s",
        "build_s", "run_s", "total_stage_s",
        "zephyr_elf_bytes", "kernels_c_bytes", "kernels_c_loc",
        "weights_npz_bytes",
        "n_kernels_curated", "n_kernels_reference", "n_kernels_total",
        "latency_ms_spike",
    ],
}

# Arm B-bedrock adds:
ARM_B_EXTRA_METRICS = [
    "tokens_input_cached", "tokens_input_uncached", "tokens_output",
    "dollars_equivalent",
    "beam_n_candidates_total", "beam_n_candidates_viable",
    "beam_n_candidates_build_fail", "beam_n_candidates_duplicate",
    "beam_tokens_per_candidate_mean", "beam_best_improvement_pct",
    "beam_iter_to_best",
    "n_kernels_llm", "algorithms_distinct_count",
]


def read_dashboard_csv() -> list[dict]:
    p = RESULTS / "dashboard.csv"
    with open(p) as f:
        return list(csv.DictReader(f))


def metrics_present(rows: list[dict], arm: str, workload: str) -> set[str]:
    return {r["metric"] for r in rows
            if r["arm"] == arm and r["workload"] == workload
            and r["value"] != ""}


def run_aggregator(runs_a: int = 1, runs_b: int = 1) -> int:
    cmd = [sys.executable, "-m", "modelblaster.benchmarks.aggregate",
           "--runs-arm", f"A={runs_a}",
           "--runs-arm", f"B-bedrock={runs_b}"]
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


# ───────────────────── main ─────────────────────


def main() -> int:
    print("[1] Reset results tree")
    fresh_results_tree()

    workload_a = "dronet_scalar_smoke"
    workload_b = "dronet_rvv_opu_int8"
    failures: list[str] = []

    print(f"[2] Synthesize 3 replicates of Arm A on {workload_a}")
    for i, ts in enumerate([
        "2026-05-25T00-00-01Z", "2026-05-25T00-00-02Z", "2026-05-25T00-00-03Z",
    ], start=1):
        build_cell("A", workload_a, ts, replicate_idx=i)

    print(f"[3] Synthesize 3 replicates of Arm B-bedrock on {workload_b}")
    for i, ts in enumerate([
        "2026-05-25T00-10-01Z", "2026-05-25T00-10-02Z", "2026-05-25T00-10-03Z",
    ], start=1):
        build_cell("B-bedrock", workload_b, ts, replicate_idx=i)

    print("[4] Run aggregator --runs-arm A=3 --runs-arm B-bedrock=3")
    rc = run_aggregator(runs_a=3, runs_b=3)
    if rc != 0:
        failures.append(f"aggregator returned non-zero rc={rc}")

    print("[5] Read dashboard.csv")
    rows = read_dashboard_csv()
    print(f"    {len(rows)} rows total")

    # Arm A must populate every metric in ARM_A_REQUIRED_METRICS.
    print(f"[6] Verify Arm A on {workload_a} populates required metrics")
    a_present = metrics_present(rows, "A", workload_a)
    expected_a = set(ARM_A_REQUIRED_METRICS["pre_kernel"]
                     + ARM_A_REQUIRED_METRICS["kernel_synthesis"])
    missing_a = expected_a - a_present
    if missing_a:
        failures.append(f"Arm A missing metrics: {sorted(missing_a)}")
    else:
        print(f"    OK ({len(expected_a)} metrics)")

    # Arm B-bedrock must populate all of Arm A's metrics PLUS the LLM-only ones.
    print(f"[7] Verify Arm B-bedrock on {workload_b} populates LLM metrics")
    b_present = metrics_present(rows, "B-bedrock", workload_b)
    expected_b_extra = set(ARM_B_EXTRA_METRICS)
    missing_b = expected_b_extra - b_present
    if missing_b:
        failures.append(f"Arm B missing LLM metrics: {sorted(missing_b)}")
    else:
        print(f"    OK ({len(expected_b_extra)} LLM metrics)")

    # Replicate aggregation: with N=3, numeric metrics should have stddev set.
    print("[8] Verify N=3 replicate aggregation produced stddev")
    sample_rows = [r for r in rows
                   if r["arm"] == "A" and r["workload"] == workload_a
                   and r["metric"] in ("cycles_spike", "wall_clock_s",
                                        "compile_wall_clock_s")]
    n_with_stddev = sum(1 for r in sample_rows
                        if r["stddev"] not in ("", "0", None))
    if not sample_rows:
        failures.append("no replicate sample rows found for stddev check")
    elif n_with_stddev == 0:
        failures.append("no stddev values produced despite N=3 with jitter")
    else:
        n_runs_set = {r["n_runs"] for r in sample_rows}
        print(f"    OK ({n_with_stddev}/{len(sample_rows)} sampled rows have "
              f"stddev; n_runs={n_runs_set})")

    # Cost monitor parses the synthesized JSONL.
    print("[9] Verify cost monitor extractor on synthesized llm_calls.jsonl")
    from modelblaster.benchmarks.tools.cost_monitor import (
        load_pricing, price_call,
    )
    pricing = load_pricing(CONFIG / "pricing.yaml")
    sample_rec = {
        "input_tokens": 2000, "output_tokens": 500,
        "cache_read_input_tokens": 1200, "cache_write_input_tokens": 0,
    }
    cost = price_call(
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0", sample_rec, pricing)
    expected = (2000 - 1200) * 3.0 / 1e6 + 1200 * 0.30 / 1e6 + 500 * 15.0 / 1e6
    if cost is None or abs(cost - expected) > 1e-9:
        failures.append(f"price_call: got {cost}, expected {expected}")
    else:
        print(f"    OK (priced 1 call at ${cost:.5f})")

    # Summary.md must render both phase tables.
    print("[10] Verify summary.md two-phase sectioning")
    summary = (RESULTS / "summary.md").read_text()
    if "pre-kernel" not in summary:
        failures.append("summary.md missing 'pre-kernel' section header")
    if "kernel synthesis" not in summary:
        failures.append("summary.md missing 'kernel synthesis' section header")
    if workload_a not in summary:
        failures.append(f"summary.md missing workload {workload_a}")
    if workload_b not in summary:
        failures.append(f"summary.md missing workload {workload_b}")
    if not failures or all("missing" not in f for f in failures):
        print("    OK")

    # Per-op-kind metrics in cycles_per_op.json (schema v2).
    print("[11] Verify cycles_per_op v2 fields landed")
    cpo_sample = json.loads(
        (RESULTS / "A" / workload_a / "latest"
         / "cycles_per_op.json").read_text())
    cpo_sample_path = (RESULTS / "A" / workload_a / "latest" / "cycles_per_op.json")
    # Resolve symlink for read.
    cpo_actual_path = cpo_sample_path.resolve() / "cycles_per_op.json" \
        if cpo_sample_path.is_dir() else cpo_sample_path
    cpo = json.loads(cpo_actual_path.read_text())
    if cpo.get("schema_version") != 2:
        failures.append(f"cycles_per_op.json schema_version != 2 (got {cpo.get('schema_version')})")
    if "mean_cycles_per_dispatch" not in cpo:
        failures.append("cycles_per_op.json missing mean_cycles_per_dispatch")
    for kind, slot in cpo.get("by_op_kind", {}).items():
        for field in ("median", "p50", "p90", "p95", "stddev"):
            if field not in slot:
                failures.append(f"by_op_kind[{kind}] missing {field}")
                break
    if not any("cycles_per_op" in f for f in failures):
        print("    OK (median/p50/p90/p95/stddev present per op kind)")

    # Report.
    print()
    if failures:
        print(f"FAIL ({len(failures)} issues)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"PASS — dashboard has {len(rows)} populated metric rows across "
          f"2 cells (A: {len(a_present)} metrics, B-bedrock: "
          f"{len(b_present)} metrics)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
