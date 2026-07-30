"""Per-(arm, workload) winner-by-metric + delta-vs-baseline + port-back ablation gate.

Reads `results/dashboard.csv` and emits `results/winner.md` with:

* **Per-workload winners** -- for each metric where two or more arms
  reported a value, which arm wins (LOWER_IS_BETTER or
  HIGHER_IS_BETTER classification).
* **Changes vs baseline** -- when ``--baseline <path>`` is supplied,
  every (arm, workload, metric) in the current dashboard is diffed
  against the matching row in the baseline CSV. Improvements + the
  biggest regressions land in a sorted table at the top of winner.md
  so "did my change help?" is answerable at a glance.

The port-back ablation gate (``--ablate <piece>``) is a Phase 4
addition that lands with Arm C; today it exits informatively.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Optional


# Metrics where "lower is better" -- used to invert the winner check
# and to label baseline deltas as improvements or regressions.
LOWER_IS_BETTER = frozenset({
    "cycles_firesim", "cycles_spike", "accuracy_linf", "accuracy_rmse",
    "compile_wall_clock_s", "compile_peak_rss_mb",
    "tokens_input_cached", "tokens_input_uncached", "tokens_output",
    "dollars_equivalent",
    "makespan_cycles", "cross_tile_bytes",
    "decisions_rejected_by_validator",
    "n_ops_profiled",
})

# Metrics where "higher is better."
HIGHER_IS_BETTER = frozenset({
    "accuracy_cos",
    "warm_cache_hit_rate",
    "accelerator_utilization_gemmini", "accelerator_utilization_opu",
    "accelerator_utilization_rvv", "accelerator_utilization_scalar",
    "deadline_met_rate",
    "decisions_total",
})


BENCHMARKS_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = BENCHMARKS_ROOT / "results"

# Skip metrics from the regression report whose absolute delta is
# within this fraction of the baseline value -- they're floating-point
# noise, not signal. Counted in absolute terms via max(|baseline|, 1.0)
# so metrics with zero baseline still get a sensible threshold.
_NEGLIGIBLE_REL_DELTA = 1e-3


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


def _direction(metric: str, delta: float) -> str:
    """`delta = current - baseline`. Classify against the metric's
    is-better direction. Returns 'improved', 'regressed', or
    'neutral'."""
    if metric in LOWER_IS_BETTER:
        if delta < 0:
            return "improved"
        if delta > 0:
            return "regressed"
    elif metric in HIGHER_IS_BETTER:
        if delta > 0:
            return "improved"
        if delta < 0:
            return "regressed"
    return "neutral"


def _fmt_delta(metric: str, baseline: float, current: float) -> str:
    delta = current - baseline
    if baseline == 0:
        pct = "" if delta == 0 else "  (—)"
    else:
        pct = f"  ({delta / abs(baseline) * 100:+.1f}%)"
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:,.4g}{pct}"


def _format_value(v: float) -> str:
    if v == 0:
        return "0"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 1:
        return f"{v:.4g}"
    if abs(v) >= 1e-3:
        return f"{v:.4f}"
    return f"{v:.3e}"


def _render_baseline_section(current_rows: list[dict[str, str]],
                             baseline_rows: list[dict[str, str]],
                             baseline_path: Path) -> list[str]:
    base_by_key: dict[tuple[str, str, str], float] = {}
    for r in baseline_rows:
        try:
            base_by_key[(r["arm"], r["workload"], r["metric"])] = float(r["value"])
        except (KeyError, ValueError):
            continue

    deltas: list[tuple[str, str, str, float, float, float, str]] = []
    for r in current_rows:
        try:
            current_value = float(r["value"])
        except (KeyError, ValueError):
            continue
        key = (r["arm"], r["workload"], r["metric"])
        if key not in base_by_key:
            continue
        baseline = base_by_key[key]
        delta = current_value - baseline
        rel = abs(delta) / max(abs(baseline), 1.0)
        if rel < _NEGLIGIBLE_REL_DELTA:
            continue  # noise floor
        direction = _direction(r["metric"], delta)
        deltas.append((r["arm"], r["workload"], r["metric"],
                       baseline, current_value, delta, direction))

    if not deltas:
        return [
            "## Changes vs baseline",
            "",
            f"Baseline: `{baseline_path}`",
            "",
            "No metrics differ from the baseline above the "
            f"{_NEGLIGIBLE_REL_DELTA*100:.1f}% relative-noise threshold.",
            "",
        ]

    # Sort: regressions first (largest first), then improvements (largest first),
    # then neutral.
    def _sort_key(t):
        direction = t[6]
        rel_mag = abs(t[5]) / max(abs(t[3]), 1.0)
        priority = {"regressed": 0, "improved": 1, "neutral": 2}[direction]
        return (priority, -rel_mag)

    deltas.sort(key=_sort_key)

    lines: list[str] = []
    lines.append("## Changes vs baseline")
    lines.append("")
    lines.append(f"Baseline: `{baseline_path}`. Showing every (arm, workload, "
                 "metric) cell whose value differs from the baseline by more "
                 f"than {_NEGLIGIBLE_REL_DELTA*100:.1f}% (relative).")
    lines.append("")
    lines.append("| arm | workload | metric | baseline | current | delta | direction |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for arm, wl, metric, baseline, current, delta, direction in deltas:
        marker = {"improved": "✓", "regressed": "✗", "neutral": "·"}[direction]
        lines.append(
            f"| {arm} | {wl} | {metric} | {_format_value(baseline)} | "
            f"{_format_value(current)} | {_fmt_delta(metric, baseline, current)} | "
            f"{marker} {direction} |"
        )
    lines.append("")
    return lines


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="winner-by-metric analysis")
    ap.add_argument("--dashboard", default=str(RESULTS_DIR / "dashboard.csv"))
    ap.add_argument("--baseline", default=None,
                    help="path to a previously-captured dashboard.csv; when "
                         "supplied, the report leads with a diff against it "
                         "(per-cell improvements + regressions + neutrals)")
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

    if args.baseline:
        baseline_rows = _load_dashboard(Path(args.baseline))
        if not baseline_rows:
            lines.append(f"_baseline path `{args.baseline}` not found / empty;"
                         " skipping diff._\n")
        else:
            lines.extend(_render_baseline_section(rows, baseline_rows,
                                                  Path(args.baseline)))

    if not rows:
        lines.append("No dashboard rows yet — nothing to compare.")
        Path(args.out).write_text("\n".join(lines))
        print(f"wrote {args.out} (empty: no dashboard rows)")
        return 0

    lines.append("## Per-workload winners")
    lines.append("")
    for wl in sorted(by_workload):
        meta = workload_meta[wl]
        lines.append(f"### `{wl}` &nbsp;({meta['model']}/{meta['target']}/{meta['runner']})")
        any_decision = False
        for metric, by_arm in sorted(by_workload[wl].items()):
            if len(by_arm) < 2:
                continue
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
    print(f"wrote {args.out} ({len(by_workload)} workloads compared"
          f"{', diff against ' + args.baseline if args.baseline else ''})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
