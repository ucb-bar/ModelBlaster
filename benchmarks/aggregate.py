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
        ))
    return out


def load_pricing() -> dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "pricing.yaml")


def load_matrix_rules() -> dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "matrix.yaml")


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
    run_dir: Optional[Path]
    values: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _find_run_dir(arm: Arm, workload: Workload,
                  pin_run_id: Optional[str]) -> Optional[Path]:
    base = RESULTS_DIR / arm.id / workload.id
    if not base.exists():
        return None
    if pin_run_id is not None:
        d = base / pin_run_id
        return d if d.exists() else None
    latest = base / "latest"
    if latest.is_symlink() or latest.is_dir():
        return latest.resolve() if latest.is_symlink() else latest
    # Fall back to the most recent subdir by mtime if no `latest` symlink.
    subs = [d for d in base.iterdir() if d.is_dir()]
    if not subs:
        return None
    return max(subs, key=lambda d: d.stat().st_mtime)


def _collect_metric(cell: Cell, metric: Metric, pricing: dict[str, Any]
                    ) -> Optional[Any]:
    # Arm filter.
    if cell.arm.id not in metric.arms:
        return None
    # Workload-driven nullable.
    if _nullable(metric, cell.workload):
        return None
    # Derived metric (e.g. dollars_equivalent).
    if metric.derived_from:
        if metric.name == "dollars_equivalent":
            return compute_cost_usd(cell, pricing)
        return None
    # File-backed metric.
    if cell.run_dir is None or metric.source is None or metric.extractor is None:
        return None
    src = cell.run_dir / metric.source
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
    """USD equivalent of the cell's token usage. Reads per-model
    breakdown from llm_tokens.json::by_model so different LLMs in the
    same run are priced separately. Returns None when the cell has no
    tokens or the breakdown is unavailable."""
    if cell.run_dir is None:
        return None
    src = cell.run_dir / "llm_tokens.json"
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
    header = [
        "arm", "workload", "model", "target", "quant", "runner",
        "metric", "value", "unit",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for cell in cells:
            for m in metrics:
                v = cell.values.get(m.name)
                if v is None:
                    continue
                w.writerow([
                    cell.arm.id,
                    cell.workload.id,
                    cell.workload.model,
                    cell.workload.target,
                    cell.workload.quant,
                    cell.workload.runner,
                    m.name,
                    v,
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
        "One section per workload. Each section's table compares the arms\n"
        "side by side across the metrics that apply.\n"
    )
    lines.append("")

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

        header = ["metric"] + arm_ids + ["unit"]
        sep = ["---"] + ["---"] * len(arm_ids) + ["---"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(sep) + " |")

        for m in metrics:
            row_vals = []
            any_value = False
            for aid in arm_ids:
                v = cells_in_wl[aid].values.get(m.name)
                row_vals.append("—" if v is None else _fmt(v))
                if v is not None:
                    any_value = True
            if not any_value:
                continue  # skip metrics that are blank across all arms

            # Cycle-source-honesty: tag spike-on-accel as non-authoritative.
            name = m.name
            if name == "cycles_spike" and wl.target in ACCEL_TARGETS:
                name = f"{name} *(not authoritative on {wl.target})*"

            lines.append("| " + " | ".join([name] + row_vals + [m.unit]) + " |")
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
    ap.add_argument("--include-arm", action="append", default=None,
                    help="restrict to these arms (repeatable). Default: all "
                         "arms in arms.yaml")
    args = ap.parse_args(argv)

    arms = load_arms()
    workloads = load_workloads()
    metrics = load_metrics()
    pricing = load_pricing()
    rules = load_matrix_rules()

    if args.include_arm:
        keep = set(args.include_arm)
        arms = [a for a in arms if a.id in keep]

    pairs = expand_matrix(arms, workloads, rules)

    cells: list[Cell] = []
    for arm, wl in pairs:
        run_dir = _find_run_dir(arm, wl, args.run_id)
        cell = Cell(arm=arm, workload=wl, run_dir=run_dir)
        for m in metrics:
            v = _collect_metric(cell, m, pricing)
            if v is not None:
                cell.values[m.name] = v
        cells.append(cell)

    write_dashboard_csv(cells, metrics, RESULTS_DIR / "dashboard.csv")
    write_summary_md(cells, metrics, arms, workloads,
                     RESULTS_DIR / "summary.md")

    populated = sum(1 for c in cells if c.values)
    print(f"wrote {RESULTS_DIR / 'dashboard.csv'} and "
          f"{RESULTS_DIR / 'summary.md'} "
          f"({populated}/{len(cells)} cells populated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
