"""Post-process a bundle run: render predicted-vs-actual Gantts + re-advise.

After `scripts/run_xpurt_bundle.py` (or its shell wrapper) has produced
a `manifest.json` with per-candidate `xpurt_trace.csv` files, this
script closes the agentic loop:

  1. For each candidate, emit a measured `SchedulerReport`
     (`scripts/emit_measured_report.py`) by overlaying its trace onto
     the predicted report XPU-RT already produced.
  2. Render predicted-vs-actual Gantts using
     `scripts/plot_xpurt_trace.py` (already in the repo).
  3. Run `$XPURT_ROOT/xpu-rt/advisor.py` on each measured
     report so the verdict reflects real numbers.
  4. Write `artifacts/bundle/round1_report.md` summarizing the
     predicted vs measured comparison + the re-advise verdict per
     candidate. This is the artifact `/close-loop` (XPU-RT skill)
     hands back to the user.

Usage:

    python3 scripts/close_xpurt_loop.py \\
        --manifest artifacts/bundle/manifest.json \\
        --deadline-us 65
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_xpurt_root() -> Path:
    """Locate the XPU-RT checkout. Same contract as the copy in
    `scripts/decision_loop.py` -- keep the two in sync. Identified by the
    two entry points this script calls: xpu-rt/advisor.py and
    xpu-rt/plot_gantt.py.

    Replaces a hardcoded /scratch2/<user>/XPU-RT, which made this script
    runnable by exactly one person.
    """
    env = os.environ.get("XPURT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    for cand in (REPO_ROOT.parent, REPO_ROOT.parent / "XPU-RT"):
        if (cand / "xpu-rt" / "advisor.py").is_file():
            return cand.resolve()
    return REPO_ROOT.parent.resolve()


XPURT_ROOT = _resolve_xpurt_root()


def _predicted_report_path(fixture: Path) -> Path:
    """The schedule fixture has a sibling `*_report.json` produced by
    run_xpurt_schedule.py (SchedulerReport v2)."""
    stem = fixture.stem
    if stem.endswith("_profiled"):
        report_stem = stem[: -len("_profiled")] + "_profiled_report"
    else:
        report_stem = stem + "_report"
    return fixture.parent / f"{report_stem}.json"


def _emit_measured(candidate: dict, deadline_us: float | None,
                   out_dir: Path, clock_mhz: float = 1000.0) -> Path | None:
    fixture = Path(candidate["fixture"])
    trace = candidate.get("trace_csv")
    if not trace:
        return None
    trace = Path(trace)
    if not trace.is_file():
        print(f"  {candidate['id']}: trace missing ({trace})", file=sys.stderr)
        return None
    pred = _predicted_report_path(fixture)
    if not pred.is_file():
        print(f"  {candidate['id']}: predicted report missing ({pred})",
              file=sys.stderr)
        return None
    out = out_dir / "measured_report.json"
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts/emit_measured_report.py"),
        "--predicted-report", str(pred),
        "--trace", str(trace),
        "--out", str(out),
        # Was omitted, so the measured report was always built at the
        # script's 1000 MHz default while the Gantt beside it honoured
        # --clock-mhz. On the K1 (24 MHz rdtime) that silently
        # under-reported every measured time by 41.7x.
        "--clock-mhz", str(clock_mhz),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  {candidate['id']}: emit_measured_report failed: "
              f"{r.stderr.strip()}", file=sys.stderr)
        return None
    return out


def _render_gantt(trace_csv: Path, out_png: Path,
                  clock_mhz: float = 1000.0) -> bool:
    """Render predicted-vs-actual Gantt using XPU-RT's plot_gantt.py.

    XPU-RT's plotter accepts the extracted CSV directly via `--trace`,
    while ModelBlaster's plot_xpurt_trace.py expects the full
    spike/firesim stdout (with the BEGIN/END markers). The CSVs we
    have are already extracted, so the XPU-RT plotter is the right
    consumer.
    """
    cmd = [
        sys.executable, str(XPURT_ROOT / "xpu-rt/plot_gantt.py"),
        "--trace", str(trace_csv),
        "--out", str(out_png),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  plot_gantt --trace failed: {r.stderr.strip()}",
              file=sys.stderr)
        return False
    return out_png.is_file()


def _run_advisor(report: Path, deadline_us: float | None,
                 out_json: Path) -> dict | None:
    cmd = [
        sys.executable, str(XPURT_ROOT / "xpu-rt/advisor.py"),
        "--report", str(report),
        "--json", "--emit", str(out_json),
    ]
    if deadline_us is not None:
        cmd += ["--deadline-us", str(deadline_us)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  advisor failed: {r.stderr.strip()}", file=sys.stderr)
        return None
    try:
        return json.loads(out_json.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"  advisor json read failed: {e}", file=sys.stderr)
        return None


def _summarize_candidate(candidate: dict, advice: dict | None) -> str:
    cid = candidate["id"]
    axis = candidate.get("axis", "?")
    pred = candidate.get("predicted_makespan_us")
    meas = candidate.get("measured_makespan_us")
    label = candidate.get("scheduler") or candidate.get("solver") or "?"
    pred_str = f"{pred:.1f}" if pred is not None else "?"
    meas_str = f"{meas:.1f}" if meas is not None else "?"
    delta_str = "?"
    if pred and meas:
        delta_pct = 100.0 * (meas - pred) / pred
        sign = "+" if delta_pct >= 0 else ""
        delta_str = f"{sign}{delta_pct:.1f}%"
    verdict = "—"
    if advice:
        verdict = advice.get("verdict") or advice.get("granularity") or "—"
        if "recommendations" in advice:
            recs = advice["recommendations"]
            if recs:
                verdict += f" [{', '.join(r.get('kind','?') for r in recs[:3])}]"
    return (f"| `{cid}` | {axis} | {label} | {pred_str} | {meas_str} | "
            f"{delta_str} | {verdict} |")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path,
                   help="artifacts/bundle/manifest.json from run_xpurt_bundle.")
    p.add_argument("--deadline-us", type=float, default=None,
                   help="Optional deadline for the advisor (defaults to "
                        "the bundle's deadline_us).")
    p.add_argument("--out", type=Path,
                   help="Output report markdown. Default: manifest dir / "
                        "round1_report.md.")
    p.add_argument("--clock-mhz", type=float, default=1000.0)
    args = p.parse_args(argv)

    manifest = json.loads(args.manifest.read_text())
    out_dir = args.manifest.parent
    out_md = args.out or out_dir / "round1_report.md"
    deadline_us = args.deadline_us if args.deadline_us is not None \
        else manifest.get("deadline_us")

    rows: list[str] = []
    rows.append("# XPU-RT ⇄ ModelBlaster loop — round 1")
    rows.append("")
    rows.append(f"- bundle: `{manifest['bundle']}`")
    rows.append(f"- workload: `{manifest['workload_spec']}`")
    rows.append(f"- runner: `{manifest['runner']}`")
    rows.append(f"- deadline_us: {deadline_us}")
    rows.append("")
    rows.append("## Per-candidate")
    rows.append("")
    rows.append("| id | axis | label | predicted µs | measured µs | Δ% | advisor verdict |")
    rows.append("|----|------|-------|-------------:|------------:|---:|-----------------|")

    for cand in manifest.get("candidates", []):
        cid = cand["id"]
        cand_dir = out_dir / cid
        cand_dir.mkdir(parents=True, exist_ok=True)
        advice = None
        # Process anything with a trace CSV — covers status=ok (full
        # run) and status=partial-trace (FireSim wall-clock cutoff). A
        # truncated trace is still useful for predicted-vs-measured
        # comparison on the prefix that ran.
        if cand.get("status") in ("ok", "partial-trace") and cand.get("trace_csv"):
            measured = _emit_measured(cand, deadline_us, cand_dir,
                                      clock_mhz=args.clock_mhz)
            if measured:
                gantt = cand_dir / "predicted_vs_actual.png"
                _render_gantt(Path(cand["trace_csv"]), gantt,
                              clock_mhz=args.clock_mhz)
                advice_json = cand_dir / "measured_advice.json"
                advice = _run_advisor(measured, deadline_us, advice_json)
        rows.append(_summarize_candidate(cand, advice))

    rows.append("")
    rows.append("## Notes")
    rows.append("")
    rows.append("- Predicted µs = XPU-RT scheduler's makespan estimate "
                "from the schedule fixture (start_time + duration of latest dispatch).")
    rows.append("- Measured µs = harness `xpurt_trace.csv` last "
                "`actual_end_cycles` / `clock_mhz`.")
    rows.append("- Advisor verdict run on the **measured** report so "
                "remedies (rebalance / coarsen / finer) reflect what "
                "FireSim actually showed.")
    rows.append("- `predicted_vs_actual.png` per candidate dir overlays "
                "the two timelines on the same axes.")
    out_md.write_text("\n".join(rows) + "\n")
    print(f"close_xpurt_loop: wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
