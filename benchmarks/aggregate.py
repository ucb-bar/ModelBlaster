"""Walk results/ and produce dashboard.csv + summary.md.

The aggregator is arm-agnostic: every extractor reads only files it
declares in `config/metrics.yaml`, and the same `run.json` schema is
used by all arm drivers. Adding an arm means appending a row in
`config/arms.yaml` and a driver file in `arms/`; no aggregator code
change is required.

Layout consumed:
    results/<arm-id>/<workload-id>/<run-id>/<artifact>
    results/<arm-id>/<workload-id>/latest  -> <run-id>

By default the aggregator follows `latest`. `--run-id <id>` pins a
snapshot; multi-replicate mean+stddev across the N latest runs is a
later extension and is not implemented yet.

Outputs:
    results/dashboard.csv   long format, one row per (arm, workload, metric)
    results/summary.md      readable pivot, per workload, all arms, all metrics

The cycle-source-honesty policy is enforced here: on accelerator
targets the aggregator records `cycles_spike` but flags it as
non-authoritative in `summary.md` and refuses to derive a winner row
from it. FireSim is the only source of truth for cycle deltas on
those targets.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import importlib
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


# Targets whose cycle counts on spike are NOT authoritative because
# the corresponding spike extensions execute the accelerator ops
# atomically (no microarchitectural pipeline model).
ACCEL_TARGETS = frozenset({
    "rvv_opu", "gemmini", "gemmini_q31", "hetero_gemmini_opu",
})


BENCHMARKS_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = BENCHMARKS_ROOT / "config"
RESULTS_DIR = BENCHMARKS_ROOT / "results"


# ───────────────────── config loading ─────────────────────


@dataclass(frozen=True)
class Arm:
    id: str
    name: str
    driver: str
    requires_llm: Optional[str]
    gated_by: Optional[str]


@dataclass(frozen=True)
class Workload:
    id: str
    model: str
    target: str
    quant: str
    runner: str
    slice: Optional[str] = None
    blocked_by: Optional[str] = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Metric:
    name: str
    source: Optional[str]
    extractor: Optional[str]
    unit: str
    arms: frozenset[str]
    nullable_if: Optional[str]
    derived_from: tuple[str, ...]
    note: str
    # Two-phase taxonomy from metrics.yaml:
    #   pre_kernel       — graph-side, deterministic across arms.
    #   kernel_synthesis — depends on which kernel C the synthesis loop
    #                      emitted (so it diverges per arm).
    # Unknown / missing values render at the bottom of each workload's
    # table under an "other" header so typos surface.
    phase: str = "kernel_synthesis"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_arms() -> list[Arm]:
    raw = _load_yaml(CONFIG_DIR / "arms.yaml")["arms"]
    return [
        Arm(
            id=r["id"],
            name=r["name"],
            driver=r["driver"],
            requires_llm=r.get("requires_llm"),
            gated_by=r.get("gated_by"),
        )
        for r in raw
    ]


def load_workloads() -> list[Workload]:
    raw = _load_yaml(CONFIG_DIR / "workloads.yaml")["workloads"]
    return [
        Workload(
            id=r["id"],
            model=r["model"],
            target=r["target"],
            quant=r["quant"],
            runner=r["runner"],
            slice=r.get("slice"),
            blocked_by=r.get("blocked_by"),
            tags=tuple(r.get("tags", [])),
        )
        for r in raw
    ]


def load_metrics() -> list[Metric]:
    raw = _load_yaml(CONFIG_DIR / "metrics.yaml")["metrics"]
    out = []
    for name, m in raw.items():
        out.append(Metric(
            name=name,
            source=m.get("source"),
            extractor=m.get("extractor"),
            unit=m.get("unit", ""),
            arms=frozenset(m.get("arms", [])),
            nullable_if=m.get("nullable_if"),
            derived_from=tuple(m.get("derived_from", [])),
            note=m.get("note", ""),
            phase=m.get("phase", "kernel_synthesis"),
        ))
    return out


def load_pricing() -> dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "pricing.yaml")


def load_clocks() -> dict[str, Any]:
    """Per-target clock_hz used to derive latency_ms_* from cycles_*."""
    return _load_yaml(CONFIG_DIR / "clocks.yaml")


def load_matrix_rules() -> dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "matrix.yaml")


def _clock_hz_for(clocks: dict[str, Any], runner: str,
                  target: str) -> Optional[int]:
    """Resolve clock_hz for a (runner, target) pair from clocks.yaml.
    `null` in the config means "do not derive latency for this pair"
    -- the caller treats it as "no latency available."""
    rt = clocks.get(runner) or {}
    per_target = rt.get("per_target") or {}
    if target in per_target:
        v = per_target[target]
        return int(v) if v is not None else None
    default = rt.get("default_clock_hz")
    return int(default) if default is not None else None


def _compute_latency_ms_for(cell: "Cell", run_dir: Path,
                            clocks: dict[str, Any],
                            runner: str) -> Optional[float]:
    """Read the appropriate profile CSV's total cycles for a (runner,
    target) pair, divide by clock_hz, return milliseconds. Returns
    None when the profile CSV is absent for this runner or the clock
    is explicitly null."""
    clock_hz = _clock_hz_for(clocks, runner, cell.workload.target)
    if clock_hz is None or clock_hz <= 0:
        return None
    src = run_dir / f"profile_{runner}.csv"
    if not src.exists():
        return None
    from modelblaster.benchmarks.ingest import profile_csv as _pc
    cycles = _pc.sum_cycles(src)
    if cycles is None:
        return None
    return float(cycles) / float(clock_hz) * 1000.0


# ───────────────────── matrix expansion ─────────────────────


def _matches_rule(rule: dict[str, Any], arm: Arm, workload: Workload) -> bool:
    """A rule matches when every key in the rule matches the
    (arm, workload) pair. Unknown keys cause the rule to never match
    (loudly preferable to silently dropping work)."""
    def _eq(field_value: Any, rule_value: Any) -> bool:
        if isinstance(rule_value, list):
            return field_value in rule_value
        return field_value == rule_value

    for key, val in rule.items():
        if key == "reason":
            continue
        if key == "arm":
            if not _eq(arm.id, val):
                return False
        elif key == "workload_id":
            if not _eq(workload.id, val):
                return False
        elif key == "workload_id_pattern":
            patterns = val if isinstance(val, list) else [val]
            if not any(fnmatch.fnmatch(workload.id, p) for p in patterns):
                return False
        elif key == "workload_tag":
            tags_required = val if isinstance(val, list) else [val]
            if not any(t in workload.tags for t in tags_required):
                return False
        elif key == "target":
            if not _eq(workload.target, val):
                return False
        elif key == "runner":
            if not _eq(workload.runner, val):
                return False
        elif key == "model":
            if not _eq(workload.model, val):
                return False
        else:
            # Unknown predicate — fail closed so typos surface.
            return False
    return True


def expand_matrix(arms: list[Arm], workloads: list[Workload],
                  rules: dict[str, Any]) -> list[tuple[Arm, Workload]]:
    """Cross-product arms x workloads, then apply excludes and
    includes from matrix.yaml."""
    excludes = rules.get("exclude") or []
    includes = rules.get("include") or []
    out = []
    for arm in arms:
        for wl in workloads:
            keep = True
            for rule in excludes:
                if _matches_rule(rule, arm, wl):
                    keep = False
                    break
            if not keep:
                for rule in includes:
                    if _matches_rule(rule, arm, wl):
                        keep = True
                        break
            if keep:
                out.append((arm, wl))
    return out


# ───────────────────── nullable_if evaluation ─────────────────────


def _nullable(metric: Metric, workload: Workload) -> bool:
    """True when this workload's fields say the metric is not produced
    here (e.g. heterogeneous-only metric on a single-tile target).
    Distinct from "missing" — these cells render as `—` without being
    flagged."""
    if not metric.nullable_if:
        return False
    ctx = {
        "runner": workload.runner,
        "target": workload.target,
        "model": workload.model,
        "quant": workload.quant,
        "scenario": "periodic",  # placeholder until periodic schedules exist
    }
    try:
        return bool(eval(metric.nullable_if, {"__builtins__": {}}, ctx))
    except Exception as e:
        print(f"warning: nullable_if eval failed for {metric.name}: {e}",
              file=sys.stderr)
        return False


# ───────────────────── extractor resolution ─────────────────────


_EXTRACTOR_CACHE: dict[str, Any] = {}


def _resolve_extractor(spec: str):
    """Resolve `module:function` to a callable, cached. None when the
    spec cannot be imported."""
    if spec in _EXTRACTOR_CACHE:
        return _EXTRACTOR_CACHE[spec]
    if ":" not in spec:
        _EXTRACTOR_CACHE[spec] = None
        return None
    mod_name, func_name = spec.split(":", 1)
    try:
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, func_name)
    except (ImportError, AttributeError):
        fn = None
    _EXTRACTOR_CACHE[spec] = fn
    return fn


# ───────────────────── per-cell metric collection ─────────────────────


@dataclass
class Cell:
    arm: Arm
    workload: Workload
    # `run_dir` is the canonical run for per-cell artifacts the
    # aggregator reads beyond the numeric metric extraction (e.g. the
    # cycles_per_op.json used in the per-op breakdown table). For
    # N=1 it's the latest; for N>1 it's the most recent of the N.
    run_dir: Optional[Path]
    # All N run dirs the metrics are averaged over (ordered most
    # recent first). N=1 -> single-entry list.
    run_dirs: list[Path] = field(default_factory=list)
    # Mean of each metric across the N runs.
    values: dict[str, Any] = field(default_factory=dict)
    # Sample standard deviation (Bessel-corrected) per metric, when
    # N>1 and the metric is numeric across all N. Absent for N=1.
    stddevs: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _find_run_dirs(arm: Arm, workload: Workload,
                   pin_run_id: Optional[str],
                   n_runs: int) -> list[Path]:
    """Return up to `n_runs` most-recent run directories for this
    (arm, workload). N=1 with no pin returns the `latest` symlink
    target (or the most-recent subdir if no symlink exists). N>1
    walks every run-id subdir and sorts by name -- ISO-8601 UTC
    timestamps as filenames are lexicographically ordered."""
    base = RESULTS_DIR / arm.id / workload.id
    if not base.exists():
        return []
    if pin_run_id is not None:
        d = base / pin_run_id
        return [d] if d.exists() else []
    subs = [d for d in base.iterdir() if d.is_dir() and d.name != "latest"]
    if not subs:
        return []
    if n_runs <= 1:
        latest = base / "latest"
        if latest.is_symlink():
            return [latest.resolve()]
        return [max(subs, key=lambda d: d.stat().st_mtime)]
    # Most-recent first by name; tiebreak on mtime for safety.
    subs.sort(key=lambda d: (d.name, d.stat().st_mtime), reverse=True)
    return subs[:n_runs]


def _arm_matches(arm_id: str, allowed: frozenset[str]) -> bool:
    """An arm's metrics-yaml allow-list match. Supports family
    prefixes: a metric tagged `arms: [B, C]` matches every arm whose
    id starts with `B-` (B-bedrock, B-gemini, B-claude) plus the
    exact arm `B` or `C`. A metric tagged with a specific id
    (`arms: [B-bedrock]`) only matches that exact id. This keeps
    provider-agnostic metrics (tokens, cost) wired across the whole
    Arm-B family without re-listing every provider variant."""
    if arm_id in allowed:
        return True
    if "-" in arm_id:
        family = arm_id.split("-", 1)[0]
        if family in allowed:
            return True
    return False


def _collect_metric(cell: Cell, metric: Metric, pricing: dict[str, Any],
                    clocks: Optional[dict[str, Any]] = None,
                    ) -> tuple[Optional[Any], Optional[float]]:
    """Extract `metric` from every run dir on `cell` and reduce to
    a (mean, stddev) pair. `stddev` is None when N=1 or the metric
    is non-numeric (e.g. a string returned by an extractor). For
    derived metrics the per-run values come from re-evaluating the
    derivation against each run dir's tokens log -- so
    dollars_equivalent also gets a stddev when N>1."""
    if not _arm_matches(cell.arm.id, metric.arms):
        return None, None
    if _nullable(metric, cell.workload):
        return None, None

    samples: list[Any] = []
    for run_dir in cell.run_dirs:
        v = _extract_one(cell, metric, pricing, run_dir, clocks)
        if v is not None:
            samples.append(v)

    if not samples:
        return None, None

    numeric = [float(x) for x in samples
               if isinstance(x, (int, float)) and not isinstance(x, bool)]
    if numeric and len(numeric) == len(samples):
        mean = sum(numeric) / len(numeric)
        stddev = None
        if len(numeric) > 1:
            var = sum((x - mean) ** 2 for x in numeric) / (len(numeric) - 1)
            stddev = math.sqrt(var)
        # Preserve int output when all inputs are int and the mean
        # is exact (avoids "12500.0" in the dashboard).
        if (all(isinstance(x, int) and not isinstance(x, bool)
                for x in samples)
                and mean == int(mean)):
            return int(mean), stddev
        return mean, stddev

    # Non-numeric metric -- emit the most-recent value and skip stddev.
    return samples[0], None


def _extract_one(cell: Cell, metric: Metric, pricing: dict[str, Any],
                 run_dir: Path,
                 clocks: Optional[dict[str, Any]] = None) -> Optional[Any]:
    # Derived metric (e.g. dollars_equivalent, latency_ms_*).
    if metric.derived_from:
        if metric.name == "dollars_equivalent":
            return _compute_cost_usd_for(cell, run_dir, pricing)
        if metric.name == "latency_ms_firesim":
            return _compute_latency_ms_for(cell, run_dir, clocks or {},
                                           "firesim")
        if metric.name == "latency_ms_spike":
            return _compute_latency_ms_for(cell, run_dir, clocks or {},
                                           "spike")
        return None
    # File-backed metric.
    if metric.source is None or metric.extractor is None:
        return None
    src = run_dir / metric.source
    if not src.exists():
        return None
    fn = _resolve_extractor(metric.extractor)
    if fn is None:
        cell.notes.append(f"unresolved extractor {metric.extractor}")
        return None
    try:
        return fn(src)
    except Exception as e:
        cell.notes.append(f"{metric.name} extractor raised {type(e).__name__}: {e}")
        return None


def compute_cost_usd(cell: Cell, pricing: dict[str, Any]) -> Optional[float]:
    """Back-compat wrapper -- prices the latest run only. The
    per-replicate version lives in `_compute_cost_usd_for`."""
    if cell.run_dir is None:
        return None
    return _compute_cost_usd_for(cell, cell.run_dir, pricing)


def _compute_cost_usd_for(cell: Cell, run_dir: Path,
                          pricing: dict[str, Any]) -> Optional[float]:
    """USD equivalent of one run dir's token usage. Reads per-model
    breakdown from llm_tokens.json::by_model so different LLMs in the
    same run are priced separately. Returns None when the run dir
    has no tokens or the breakdown is unavailable."""
    src = run_dir / "llm_tokens.json"
    if not src.exists():
        return None
    try:
        with open(src) as f:
            data = json.load(f)
    except Exception:
        return None
    breakdown = data.get("by_model") or {}
    if not breakdown:
        # Fall back to flat totals priced against the run's recorded model.
        model_id = data.get("model_id") or pricing.get("fallback", {}).get(
            "_default_model_id"
        )
        if model_id is None:
            return None
        breakdown = {model_id: {
            "input_cached": data.get("tokens_input_cached", 0),
            "input_uncached": data.get("tokens_input_uncached", 0),
            "output": data.get("tokens_output", 0),
        }}

    models_table = pricing.get("models", {})
    fallback = pricing.get("fallback", {})
    total = 0.0
    for model_id, counts in breakdown.items():
        rates = models_table.get(model_id) or fallback
        if rates.get("placeholder"):
            # Placeholder rates yield no signal; surface the gap via a note.
            cell.notes.append(f"pricing placeholder for {model_id}")
            continue
        in_uncached = counts.get("input_uncached", 0) or 0
        in_cached = counts.get("input_cached", 0) or 0
        out_t = counts.get("output", 0) or 0
        r_in = rates.get("input_uncached")
        r_cached = rates.get("cache_read")
        r_out = rates.get("output")
        if r_in is None or r_out is None:
            cell.notes.append(f"missing rate for {model_id}")
            continue
        total += in_uncached * r_in / 1_000_000.0
        total += in_cached * (r_cached if r_cached is not None else r_in) \
            / 1_000_000.0
        total += out_t * r_out / 1_000_000.0
    return total


# ───────────────────── dashboard rendering ─────────────────────


def write_dashboard_csv(cells: list[Cell], metrics: list[Metric],
                        out_path: Path) -> None:
    """Long-format CSV: one row per (arm, workload, metric). `stddev`
    is empty when N=1 or the metric is non-numeric. `n_runs` is the
    actual replicate count for this cell (may be less than the
    aggregator's --runs request when only some runs exist)."""
    header = [
        "arm", "workload", "model", "target", "quant", "runner",
        "metric", "value", "stddev", "n_runs", "unit",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for cell in cells:
            n_runs = len(cell.run_dirs)
            for m in metrics:
                v = cell.values.get(m.name)
                if v is None:
                    continue
                stddev = cell.stddevs.get(m.name)
                w.writerow([
                    cell.arm.id,
                    cell.workload.id,
                    cell.workload.model,
                    cell.workload.target,
                    cell.workload.quant,
                    cell.workload.runner,
                    m.name,
                    v,
                    "" if stddev is None else f"{stddev:.6g}",
                    n_runs,
                    m.unit,
                ])


def write_summary_md(cells: list[Cell], metrics: list[Metric],
                     arms: list[Arm], workloads: list[Workload],
                     out_path: Path) -> None:
    by_wl: dict[str, dict[str, Cell]] = {}
    for c in cells:
        by_wl.setdefault(c.workload.id, {})[c.arm.id] = c

    lines: list[str] = []
    lines.append("# Benchmark dashboard\n")
    lines.append(
        "One section per workload, split into two phases:\n"
        "\n"
        "- **pre-kernel**: graph-side metrics (fusion / fold pass fires,\n"
        "  IR op counts, static cross-tile traffic). Deterministic across\n"
        "  arms — A and B should agree here; divergence is a bug.\n"
        "- **kernel synthesis**: properties of the compiled artifact\n"
        "  (cycles, accuracy, makespan, LLM token cost, beam trajectory).\n"
        "  Diverges per arm because the synthesis strategy differs.\n"
        "\n"
        "Each per-workload section is followed by a top-op breakdown\n"
        "and (when hetero) a per-op-x-tile rollup.\n"
    )
    lines.append("")

    # Stable phase order: pre_kernel first, then kernel_synthesis, then
    # anything else (catches typos in metrics.yaml's phase field).
    phase_order = ["pre_kernel", "kernel_synthesis"]
    phase_titles = {
        "pre_kernel": "pre-kernel — graph compilation",
        "kernel_synthesis": "kernel synthesis — compiled-artifact + LLM loop",
    }
    metrics_by_phase: dict[str, list[Metric]] = {}
    for m in metrics:
        metrics_by_phase.setdefault(m.phase, []).append(m)
    phases_present = [p for p in phase_order if p in metrics_by_phase]
    for p in metrics_by_phase:
        if p not in phases_present:
            phases_present.append(p)

    for wl in workloads:
        if wl.id not in by_wl:
            continue
        cells_in_wl = by_wl[wl.id]
        arm_ids = [a.id for a in arms if a.id in cells_in_wl]
        if not arm_ids:
            continue

        lines.append(f"## `{wl.id}`")
        meta = (f"model `{wl.model}`, target `{wl.target}`, "
                f"quant `{wl.quant}`, runner `{wl.runner}`")
        if wl.slice:
            meta += f", slice `{wl.slice}`"
        if wl.blocked_by:
            meta += f"  &nbsp;**[blocked_by: {wl.blocked_by}]**"
        lines.append(meta)
        lines.append("")

        for phase in phases_present:
            phase_metrics = metrics_by_phase[phase]
            phase_rows: list[str] = []
            for m in phase_metrics:
                row_vals = []
                any_value = False
                for aid in arm_ids:
                    cell = cells_in_wl[aid]
                    v = cell.values.get(m.name)
                    if v is None:
                        row_vals.append("—")
                    else:
                        any_value = True
                        stddev = cell.stddevs.get(m.name)
                        n_runs = len(cell.run_dirs)
                        if n_runs <= 1:
                            row_vals.append(_fmt(v))
                        elif stddev is None or stddev <= 1e-9 * max(abs(float(v)), 1.0):
                            row_vals.append(f"{_fmt(v)} (N={n_runs})")
                        else:
                            row_vals.append(
                                f"{_fmt(v)} ± {_fmt(stddev)} (N={n_runs})"
                            )
                if not any_value:
                    continue

                name = m.name
                if name == "cycles_spike" and wl.target in ACCEL_TARGETS:
                    name = f"{name} *(not authoritative on {wl.target})*"

                phase_rows.append("| " + " | ".join(
                    [name] + row_vals + [m.unit]
                ) + " |")

            if not phase_rows:
                continue

            lines.append(f"### {phase_titles.get(phase, phase)}")
            lines.append("")
            header = ["metric"] + arm_ids + ["unit"]
            sep = ["---"] + ["---"] * len(arm_ids) + ["---"]
            lines.append("| " + " | ".join(header) + " |")
            lines.append("| " + " | ".join(sep) + " |")
            lines.extend(phase_rows)
            lines.append("")

        notes_seen: set[str] = set()
        for aid in arm_ids:
            for n in cells_in_wl[aid].notes:
                if n in notes_seen:
                    continue
                notes_seen.add(n)
                lines.append(f"- _note ({aid}):_ {n}")
        if notes_seen:
            lines.append("")

        per_op_lines = _render_per_op_breakdown(cells_in_wl, arm_ids)
        if per_op_lines:
            lines.append("**Top op kinds by cycle share:**")
            lines.append("")
            lines.extend(per_op_lines)
            lines.append("")

        per_tile_lines = _render_per_op_x_tile(cells_in_wl, arm_ids)
        if per_tile_lines:
            lines.append("**Per op kind x tile (cycles attributed by XPURT trace):**")
            lines.append("")
            lines.extend(per_tile_lines)
            lines.append("")

    # Section for workloads with no data at all.
    missing = [wl.id for wl in workloads if wl.id not in by_wl]
    if missing:
        lines.append("## Workloads with no results yet")
        for wid in missing:
            wl = next(w for w in workloads if w.id == wid)
            tail = f"  [blocked_by: {wl.blocked_by}]" if wl.blocked_by else ""
            lines.append(f"- `{wid}` ({wl.target}/{wl.runner}/{wl.quant}){tail}")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def _render_per_op_breakdown(cells_in_wl: dict[str, "Cell"],
                             arm_ids: list[str]) -> list[str]:
    """For each arm that has a `cycles_per_op.json`, list the top-5
    op kinds by cycle share. Returns a list of markdown lines (table
    + rows). When no arm has the artifact, returns an empty list and
    the dashboard renderer skips the section."""
    from modelblaster.benchmarks.ingest import cycles_per_op as cpo

    rows: list[tuple[str, list[tuple[str, float, int]]]] = []
    for aid in arm_ids:
        cell = cells_in_wl[aid]
        if cell.run_dir is None:
            continue
        src = cell.run_dir / "cycles_per_op.json"
        if not src.exists():
            continue
        try:
            top = cpo.top_op_breakdown(src, k=5)
        except Exception as e:
            cell.notes.append(f"cycles_per_op extractor raised: {e}")
            continue
        if top:
            rows.append((aid, top))

    if not rows:
        return []

    lines: list[str] = []
    lines.append("| arm | op kind | share | cycles |")
    lines.append("| --- | --- | --- | --- |")
    for aid, top in rows:
        for op, share, total in top:
            lines.append(f"| {aid} | {op} | {share*100:.1f}% | {_fmt(total)} |")
    return lines


def _render_per_op_x_tile(cells_in_wl: dict[str, "Cell"],
                          arm_ids: list[str]) -> list[str]:
    """Hetero cells only: surface the by_op_kind_x_tile rollup from
    cycles_per_op.json so "conv2d_s8 on gemmini vs conv2d_s8 on
    rvv_opu" is one glance. Returns an empty list when no cell has
    tile-attributed data (single-target workloads, or hetero runs
    without the XPURT trace block in stdout)."""
    rows: list[tuple[str, list[tuple[str, dict]]]] = []
    for aid in arm_ids:
        cell = cells_in_wl[aid]
        if cell.run_dir is None:
            continue
        src = cell.run_dir / "cycles_per_op.json"
        if not src.exists():
            continue
        try:
            with open(src) as f:
                data = json.load(f)
        except Exception:
            continue
        by_tile = data.get("by_op_kind_x_tile") or {}
        if not by_tile:
            continue
        # Sort by share descending so the dominant op-on-tile is first.
        items = sorted(by_tile.items(),
                       key=lambda kv: kv[1].get("share", 0.0),
                       reverse=True)
        rows.append((aid, items))

    if not rows:
        return []

    lines: list[str] = []
    lines.append("| arm | op@tile | count | total cycles | share | mean |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for aid, items in rows:
        for key, slot in items:
            lines.append(
                f"| {aid} | {key} | {slot.get('count', 0)} | "
                f"{_fmt(int(slot.get('total', 0)))} | "
                f"{slot.get('share', 0.0)*100:.1f}% | "
                f"{slot.get('mean', 0.0):.0f} |"
            )
    return lines


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if v == 0:
            return "0"
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        if abs(v) >= 1:
            return f"{v:.4g}"
        # Fractions / probabilities read better in decimal than in
        # scientific notation (0.93 vs 9.3e-01). Only fall back to
        # scientific for genuinely tiny floats where decimals would
        # lose precision.
        if abs(v) >= 1e-3:
            return f"{v:.4f}"
        return f"{v:.3e}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


# ───────────────────── CLI ─────────────────────


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="aggregate benchmark results")
    ap.add_argument("--run-id", default=None,
                    help="pin to a specific run-id under each cell "
                         "(default: follow `latest`)")
    ap.add_argument("--runs", type=int, default=1,
                    help="default replicate count for all arms. When > 1, "
                         "average each metric over the N most-recent run-ids "
                         "per cell and emit mean ± stddev. Mutually exclusive "
                         "with --run-id.")
    ap.add_argument("--runs-arm", action="append", default=None,
                    metavar="ARM=N",
                    help="override --runs for a specific arm (repeatable). "
                         "E.g. --runs-arm A=3 --runs-arm B-bedrock=5. Useful "
                         "when deterministic arms need fewer replicates than "
                         "LLM-driven arms.")
    ap.add_argument("--include-arm", action="append", default=None,
                    help="restrict to these arms (repeatable). Default: all "
                         "arms in arms.yaml")
    args = ap.parse_args(argv)

    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")
    if args.run_id and args.runs != 1:
        raise SystemExit("--run-id pins one run; --runs > 1 averages over many. "
                         "Pick one.")

    # Parse --runs-arm overrides into a {arm_id: n_runs} dict.
    runs_by_arm: dict[str, int] = {}
    for entry in (args.runs_arm or []):
        if "=" not in entry:
            raise SystemExit(f"--runs-arm expects ARM=N, got: {entry!r}")
        aid, n_str = entry.split("=", 1)
        try:
            n = int(n_str)
        except ValueError:
            raise SystemExit(f"--runs-arm N must be an int, got: {n_str!r}")
        if n < 1:
            raise SystemExit(f"--runs-arm N must be >= 1, got: {n}")
        if args.run_id and n != 1:
            raise SystemExit("--run-id pins one run; --runs-arm N>1 conflicts.")
        runs_by_arm[aid] = n

    arms = load_arms()
    workloads = load_workloads()
    metrics = load_metrics()
    pricing = load_pricing()
    clocks = load_clocks()
    rules = load_matrix_rules()

    if args.include_arm:
        keep = set(args.include_arm)
        arms = [a for a in arms if a.id in keep]

    pairs = expand_matrix(arms, workloads, rules)

    cells: list[Cell] = []
    for arm, wl in pairs:
        n_runs = runs_by_arm.get(arm.id, args.runs)
        run_dirs = _find_run_dirs(arm, wl, args.run_id, n_runs)
        cell = Cell(
            arm=arm, workload=wl,
            run_dir=run_dirs[0] if run_dirs else None,
            run_dirs=run_dirs,
        )
        for m in metrics:
            mean, stddev = _collect_metric(cell, m, pricing, clocks)
            if mean is not None:
                cell.values[m.name] = mean
                if stddev is not None:
                    cell.stddevs[m.name] = stddev
        cells.append(cell)

    write_dashboard_csv(cells, metrics, RESULTS_DIR / "dashboard.csv")
    write_summary_md(cells, metrics, arms, workloads,
                     RESULTS_DIR / "summary.md")

    populated = sum(1 for c in cells if c.values)
    if runs_by_arm:
        per_arm = ", ".join(f"{a}={n}" for a, n in sorted(runs_by_arm.items()))
        n_runs_msg = f", runs: default={args.runs} ({per_arm})"
    elif args.runs == 1:
        n_runs_msg = ""
    else:
        n_runs_msg = f", up to {args.runs} runs/cell"
    print(f"wrote {RESULTS_DIR / 'dashboard.csv'} and "
          f"{RESULTS_DIR / 'summary.md'} "
          f"({populated}/{len(cells)} cells populated{n_runs_msg})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
