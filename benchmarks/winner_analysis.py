"""Per-(model, target) winner-by-metric + port-back ablation gate.

Reads `results/dashboard.csv` and emits `results/winner.md` showing
which arm wins on which metric for each workload. Ablation support
(`--ablate <piece>`) is the gate used by the agentic-arm port-back
decision: a CompGen file ports into ModelBlaster only when disabling
it in Arm C flips a winner on at least one (arm, workload) cell.

Stub for now: the full implementation lands when Arm C produces
artifacts. Run anyway against the current dashboard to surface
already-computable wins between Arms A and B.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Optional


# Metrics where "lower is better" — used to invert the winner check.
LOWER_IS_BETTER = frozenset({
    "cycles_firesim", "cycles_spike", "accuracy_linf", "accuracy_rmse",
    "compile_wall_clock_s", "compile_peak_rss_mb",
    "tokens_input_cached", "tokens_input_uncached", "tokens_output",
    "dollars_equivalent",
    "makespan_cycles", "cross_tile_bytes",
    "decisions_rejected_by_validator",
})

# Metrics where "higher is better."
HIGHER_IS_BETTER = frozenset({
    "accuracy_cos",
    "warm_cache_hit_rate",
    "accelerator_utilization_gemmini", "accelerator_utilization_opu",
    "deadline_met_rate",
})


BENCHMARKS_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = BENCHMARKS_ROOT / "results"


def _load_dashboard(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _winner(values_by_arm: dict[str, float], metric: str
            ) -> Optional[str]:
    if not values_by_arm:
        return None
    if metric in LOWER_IS_BETTER:
        return min(values_by_arm, key=values_by_arm.get)
    if metric in HIGHER_IS_BETTER:
        return max(values_by_arm, key=values_by_arm.get)
    return None


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="winner-by-metric analysis")
    ap.add_argument("--dashboard", default=str(RESULTS_DIR / "dashboard.csv"))
    ap.add_argument("--out", default=str(RESULTS_DIR / "winner.md"))
    ap.add_argument("--ablate", default=None,
                    help="(unimplemented) compare Arm C with and without "
                         "this piece disabled; lands when Arm C is in tree")
    args = ap.parse_args(argv)

    if args.ablate:
        print("--ablate is not implemented yet (lands with Arm C).",
              file=__import__("sys").stderr)
        return 2

    rows = _load_dashboard(Path(args.dashboard))
    # by_workload[workload][metric] = {arm_id: value}
    by_workload: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    workload_meta: dict[str, dict[str, str]] = {}
    for r in rows:
        wl = r["workload"]
        workload_meta.setdefault(wl, {
            "target": r["target"], "model": r["model"], "runner": r["runner"],
        })
        try:
            by_workload[wl][r["metric"]][r["arm"]] = float(r["value"])
        except ValueError:
            continue  # non-numeric metric values are skipped

    lines: list[str] = ["# Winner-by-metric\n"]
    if not rows:
        lines.append("No dashboard rows yet — nothing to compare.")
        Path(args.out).write_text("\n".join(lines))
        print(f"wrote {args.out} (empty: no dashboard rows)")
        return 0

    for wl in sorted(by_workload):
        meta = workload_meta[wl]
        lines.append(f"## `{wl}` &nbsp;({meta['model']}/{meta['target']}/{meta['runner']})")
        any_decision = False
        for metric, by_arm in sorted(by_workload[wl].items()):
            if len(by_arm) < 2:
                continue  # need >= 2 arms to declare a winner
            w = _winner(by_arm, metric)
            if w is None:
                continue
            any_decision = True
            arms_str = ", ".join(
                f"{aid}={v:g}" for aid, v in sorted(by_arm.items())
            )
            lines.append(f"- **{metric}** winner: `{w}` &nbsp;({arms_str})")
        if not any_decision:
            lines.append("_only one arm reporting; no winner to declare yet._")
        lines.append("")

    Path(args.out).write_text("\n".join(lines))
    print(f"wrote {args.out} ({len(by_workload)} workloads compared)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
