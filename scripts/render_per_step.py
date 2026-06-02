"""For each candidate in artifacts/bundle/<run>/manifest.json, emit:

  <run>/<id>/predicted_gantt.png   — schedule from the XPU-RT fixture
  <run>/<id>/measured_gantt.png    — actual cycles from xpurt_trace.csv (if present)
  <run>/<id>/predicted_vs_actual.png — side-by-side overlay (when both exist)

Wired into the demo orchestrator so the user gets visual progress at
every loop step without manual invocation.

Usage:

    python3 scripts/render_per_step.py \\
        --manifest artifacts/bundle/longrun/manifest.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
XPURT = Path("/scratch2/agustin/XPU-RT")


def _render_predicted(fixture: Path, out: Path, deadline_ms: float | None,
                      title: str, x_max_ms: float = 80.0) -> bool:
    cmd = [
        sys.executable, str(REPO / "scripts/render_annotated_gantt.py"),
        "--fixture", str(fixture),
        "--out", str(out),
        "--title", title,
        "--x-max-ms", str(x_max_ms),
    ]
    if deadline_ms is not None:
        cmd += ["--deadline-ms", str(deadline_ms)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  predicted render failed: {r.stderr.strip()}", file=sys.stderr)
        return False
    return out.is_file()


def _render_measured(trace: Path, out: Path, title: str,
                     x_max_ms: float | None = None) -> bool:
    if not trace.is_file():
        return False
    cmd = [
        sys.executable, str(REPO / "scripts/render_annotated_gantt.py"),
        "--trace", str(trace),
        "--out", str(out),
        "--title", title,
        "--clock-mhz", "1000",
    ]
    if x_max_ms is not None:
        cmd += ["--x-max-ms", str(x_max_ms)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  measured render failed: {r.stderr.strip()}", file=sys.stderr)
        return False
    return out.is_file()


def _render_overlay(trace: Path, out: Path) -> bool:
    """XPU-RT plot_gantt --trace renders predicted vs actual side by side
    from the same CSV (predicted_start_ms / predicted_duration_ms columns
    are the schedule fixture's, actual_start_cycles / actual_end_cycles
    are the harness's). Single PNG, two stacked Gantts."""
    if not trace.is_file():
        return False
    cmd = [
        sys.executable, str(XPURT / "xpu-rt/plot_gantt.py"),
        "--trace", str(trace),
        "--out", str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  plot_gantt --trace failed: {r.stderr.strip()}", file=sys.stderr)
        return False
    return out.is_file()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    args = p.parse_args(argv)

    m = json.loads(args.manifest.read_text())
    deadline_us = m.get("deadline_us")
    deadline_ms = deadline_us / 1000.0 if deadline_us else None
    out_root = args.manifest.parent

    print(f"render_per_step: {args.manifest} (deadline={deadline_us} us)")
    rendered = 0
    for cand in m.get("candidates", []):
        cid = cand["id"]
        d = out_root / cid
        d.mkdir(parents=True, exist_ok=True)
        cand_label = (
            f"{cid} · axis={cand.get('axis', '?')} · "
            f"{cand.get('scheduler') or cand.get('solver') or '?'}"
        )

        fixture = cand.get("fixture")
        if fixture and Path(fixture).is_file():
            ok = _render_predicted(
                Path(fixture), d / "predicted_gantt.png",
                deadline_ms,
                title=f"{cand_label}  ·  PREDICTED schedule (from XPU-RT fixture)")
            if ok:
                rendered += 1

        trace = cand.get("trace_csv")
        if trace and Path(trace).is_file():
            ok = _render_measured(
                Path(trace), d / "measured_gantt.png",
                title=f"{cand_label}  ·  MEASURED on FireSim")
            if ok:
                rendered += 1
            # Side-by-side overlay (predicted + actual on one PNG).
            _render_overlay(Path(trace), d / "predicted_vs_actual.png")
            rendered += 1

        print(f"  {cid}: predicted={d/'predicted_gantt.png' if fixture else '-'} "
              f"measured={d/'measured_gantt.png' if trace else '-'}")

    print(f"\nrendered {rendered} PNG(s) under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
