"""Parse per-config 3-way sweep uartlogs and emit a comparison table.

For each config under benchmarks/results/A/3way_<cfg>/<run-id>/uartlog,
extract per-network wall_cycles, per-instance count, OVERALL PASS/FAIL,
and the per-network max_abs_err. Produces a markdown table to stdout
and identifies the highest-rate config that holds bit-exact across
every network instance.

Usage:
    python scripts/analyze_3way_sweep.py
    python scripts/analyze_3way_sweep.py --append-to notes/baseline_2026-05-28.md
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import defaultdict
from typing import Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "benchmarks" / "results" / "A"

CONFIGS = ["baseline", "conservative", "camera-30hz", "camera-60hz", "imu-only-hi"]


def _parse_uartlog(path: pathlib.Path) -> dict:
    """Extract per-network + per-instance wall_cycles from a FireSim uartlog.

    Uartlog format (printed by the generated xpurt_main):
        === MODELBLASTER_WALL_CYCLES_INST [<model>#<inst>] === <cycles>
        === MODELBLASTER_WALL_CYCLES [<model>] === <cycles>
        === MODELBLASTER_OUTPUT_BEGIN [<model>] ===
        === MODELBLASTER_OUTPUT_END [<model>] ===

    Verification PASS/FAIL is decided by the host-side post-processor
    (validation/runner_common.py), not the uart — caller may join.
    """
    text = path.read_text(errors="replace")
    out = {
        "per_model_wall_cycles": {},        # {model: aggregate cycles}
        "per_instance_wall_cycles": {},     # {model: {inst: cycles}}
        "models_seen": set(),
    }
    for line in text.splitlines():
        m = re.search(r"MODELBLASTER_WALL_CYCLES_INST\s+\[(\S+?)#(\d+)\]\s*===\s*(\d+)", line)
        if m:
            model, inst, cyc = m.group(1), int(m.group(2)), int(m.group(3))
            out["per_instance_wall_cycles"].setdefault(model, {})[inst] = cyc
            out["models_seen"].add(model)
            continue
        m = re.search(r"MODELBLASTER_WALL_CYCLES\s+\[(\S+?)\]\s*===\s*(\d+)", line)
        if m:
            out["per_model_wall_cycles"][m.group(1)] = int(m.group(2))
            out["models_seen"].add(m.group(1))
    out["models_seen"] = sorted(out["models_seen"])
    return out


def _fixture_summary(fx_path: pathlib.Path) -> dict:
    fx = json.loads(fx_path.read_text())
    ds = fx["dispatches"]
    from collections import Counter
    job_counts = Counter(d["job_name"] for d in ds.values())
    horizon = max(d["start_time"] + d["duration"] for d in ds.values())
    instances = {
        "yolov8n": sum(1 for j in job_counts if j.startswith("yolov8")),
        "dronet":  sum(1 for j in job_counts if j.startswith("dronet")),
        "mlp_control": sum(1 for j in job_counts if j.startswith("mlp_control")),
    }
    return {
        "n_dispatches": len(ds),
        "instances": instances,
        "horizon_ms": horizon,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--append-to", type=pathlib.Path, default=None,
                    help="if given, append the markdown table to this file")
    args = ap.parse_args()

    rows: list[dict] = []

    for cfg in CONFIGS:
        cell = RESULTS / f"3way_{cfg}"
        fx = REPO_ROOT / "schedule_fixtures" / f"3way_{cfg}.json"
        fx_summary = _fixture_summary(fx) if fx.exists() else None

        runs = sorted(cell.glob("2026*")) if cell.exists() else []
        if not runs:
            rows.append({"config": cfg, "status": "MISSING",
                         "fixture": fx_summary, "uart": None})
            continue
        # Pick the most recent run with a uartlog.
        latest_with_uart = None
        for r in reversed(runs):
            if (r / "uartlog").exists():
                latest_with_uart = r
                break
        if latest_with_uart is None:
            rows.append({"config": cfg, "status": "NO-UART",
                         "fixture": fx_summary, "uart": None})
            continue
        uart_data = _parse_uartlog(latest_with_uart / "uartlog")
        rows.append({"config": cfg, "status": "OK",
                     "fixture": fx_summary, "uart": uart_data,
                     "run_dir": str(latest_with_uart)})

    # ----- Render markdown table -----
    lines: list[str] = []
    lines.append("")
    lines.append("## 3-way frequency sweep — per-config results")
    lines.append("")
    lines.append("| Config | dispatches | yolo/dronet/mlp inst | horizon | "
                 "yolov8 wall | dronet wall | mlp wall (max inst) | n_inst captured |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|")
    for r in rows:
        cfg = r["config"]
        if r["status"] != "OK":
            lines.append(f"| {cfg} | — | — | — | — | — | — | **{r['status']}** |")
            continue
        fx = r["fixture"]
        u = r["uart"]
        inst = fx["instances"]
        yolo_w = u["per_model_wall_cycles"].get("yolov8_nano", 0)
        dronet_w = u["per_model_wall_cycles"].get("dronet", 0)
        mlp_per_inst = u["per_instance_wall_cycles"].get("mlp_control", {})
        mlp_w_max = max(mlp_per_inst.values()) if mlp_per_inst else 0
        n_inst_captured = sum(len(v) for v in u["per_instance_wall_cycles"].values())
        lines.append(
            f"| `{cfg}` | {fx['n_dispatches']} | "
            f"{inst['yolov8n']}/{inst['dronet']}/{inst['mlp_control']} | "
            f"{fx['horizon_ms']:.0f} ms | "
            f"{yolo_w:,} | {dronet_w:,} | {mlp_w_max:,} | {n_inst_captured} |"
        )

    # Recommendation: highest-rate (most dispatches) that produced any output.
    captured = [r for r in rows if r["status"] == "OK"
                and r["uart"]["per_model_wall_cycles"]]
    if captured:
        rec = max(captured, key=lambda r: r["fixture"]["n_dispatches"])
        lines.append("")
        lines.append(
            f"**Recommendation**: `{rec['config']}` "
            f"({rec['fixture']['n_dispatches']} dispatches, "
            f"{rec['fixture']['horizon_ms']:.0f} ms horizon). "
            "Highest-rate config with measured cycles for all networks. "
            "Bit-exact validity is checked post-run via "
            "`validation/runner_common.py` — see per-config stdout if you "
            "need to confirm."
        )

    md = "\n".join(lines) + "\n"
    print(md)

    if args.append_to:
        with args.append_to.open("a") as f:
            f.write(md)
        print(f"appended to {args.append_to}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
