"""Pre-run budget check for LLM spend.

Reads `benchmarks/.budget.json` + the `mb-cost report` cumulative,
applies the recorded-vs-actual offset, and returns the remaining
budget. Exits non-zero if we're already over the warn / stop
thresholds.

Use before launching any Bedrock-backed kernel generation
(Phase 1d / 1e roadmap):

    python3 scripts/budget_check.py --estimate-usd 5.00
    # exit 0  -> $5.00 fits in remaining; safe to proceed
    # exit 2  -> would push us past stop_new_jobs_at_usd; abort
    # exit 3  -> already past hard_cap; abort
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BUDGET = REPO / "benchmarks/.budget.json"


def _read_budget() -> dict:
    if not BUDGET.is_file():
        raise SystemExit(f"missing {BUDGET}; create it with max_usd + actual_offset_usd")
    return json.loads(BUDGET.read_text())


def _recorded_cumul_usd() -> float:
    """Parse `uv run mb-cost report` for the CUMULATIVE line."""
    try:
        r = subprocess.run(
            ["uv", "run", "mb-cost", "report"],
            cwd=REPO, capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"warning: could not run mb-cost report ({e}); "
              f"using as_of recorded_to_date instead", file=sys.stderr)
        return _read_budget()["recorded_to_date_usd"]
    m = re.search(r"CUMULATIVE: \$([0-9]+\.[0-9]+)", r.stdout)
    return float(m.group(1)) if m else _read_budget()["recorded_to_date_usd"]


def actual_spent_usd(b: dict, recorded: float) -> float:
    """Recorded + the user-reported under-count offset."""
    return recorded + b.get("actual_offset_usd", 0.0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--estimate-usd", type=float, default=0.0,
                   help="Estimated additional cost of the run about to launch.")
    p.add_argument("--json", action="store_true",
                   help="Emit a JSON summary on stdout (still gates exit code).")
    args = p.parse_args(argv)

    b = _read_budget()
    recorded = _recorded_cumul_usd()
    actual = actual_spent_usd(b, recorded)
    remaining = b["max_usd"] - actual
    projected = actual + args.estimate_usd

    summary = {
        "max_usd": b["max_usd"],
        "recorded_to_date_usd": round(recorded, 2),
        "actual_offset_usd": b.get("actual_offset_usd", 0.0),
        "actual_to_date_usd": round(actual, 2),
        "remaining_usd": round(remaining, 2),
        "estimate_usd": args.estimate_usd,
        "projected_total_usd": round(projected, 2),
        "warn_at_usd": b["policy"]["warn_at_usd"],
        "stop_new_jobs_at_usd": b["policy"]["stop_new_jobs_at_usd"],
        "hard_cap_usd": b["policy"]["hard_cap"],
    }

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"BUDGET STATUS")
        print(f"  max:        ${b['max_usd']:.2f}")
        print(f"  recorded:   ${recorded:.2f}  (mb-cost report)")
        print(f"  + offset:   ${b.get('actual_offset_usd', 0.0):.2f}  (user-reported under-count)")
        print(f"  actual:     ${actual:.2f}")
        print(f"  remaining:  ${remaining:.2f}")
        if args.estimate_usd > 0:
            print(f"  + run est.: ${args.estimate_usd:.2f}")
            print(f"  projected:  ${projected:.2f}")

    if projected >= b["policy"]["hard_cap"]:
        print(f"\n[BUDGET HARD CAP] projected ${projected:.2f} >= cap "
              f"${b['policy']['hard_cap']:.2f} — REFUSE", file=sys.stderr)
        return 3
    if projected >= b["policy"]["stop_new_jobs_at_usd"]:
        print(f"\n[BUDGET STOP] projected ${projected:.2f} >= stop-new-jobs "
              f"${b['policy']['stop_new_jobs_at_usd']:.2f} — REFUSE", file=sys.stderr)
        return 2
    if projected >= b["policy"]["warn_at_usd"]:
        print(f"\n[BUDGET WARN] projected ${projected:.2f} >= warn "
              f"${b['policy']['warn_at_usd']:.2f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
