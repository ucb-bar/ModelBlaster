"""Parse the per-dispatch cycle table from an example's run.sh spike output.

When `examples/<model>/run.sh` finishes a spike run it prints a table:

    profile -> .../profile.csv
    name   op         shape                cycles      %
    ---------------------------------------------------
    mlp.0  linear_s8  M=1;K=16;N=256        12881    9.4
    mlp.1  elu_s8     n=256                 20498   13.0
    ...
    ---------------------------------------------------
    TOTAL                                  102725
    wall_clock_cycles=7300 (mtime)

This script extracts that table into a JSON shape the decision loop
can re-use:

    {
      "network": "mlp_control",
      "wall_clock_cycles": 7300,
      "total_dispatch_cycles": 102725,
      "dispatches": [
        {"name": "mlp.0", "op": "linear_s8", "shape": "M=1;K=16;N=256",
         "cycles": 12881, "pct_of_total": 12.5},
        ...
      ]
    }

Used by `scripts/measure_candidate.sh` to expose measured cycles to
the schedule re-runner, replacing the stale profile-DB lookup that
caused the prior bookkeeping fiction.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def parse_spike_log(text: str) -> list[dict]:
    """Return list of {name, op, shape, cycles, pct} rows from the
    profile table, in original order."""
    rows: list[dict] = []
    in_table = False
    saw_header = False
    for line in text.splitlines():
        if line.startswith("profile -> "):
            in_table = True
            continue
        if not in_table:
            continue
        stripped = line.strip()
        if stripped.startswith("-----"):
            saw_header = True
            continue
        if stripped.startswith("TOTAL"):
            break
        if not saw_header:
            continue
        if not stripped:
            continue
        # Row: name  op  [shape]  cycles  pct
        # Shape column may be empty (fused ops with multi-shape signature).
        # Parse from the right: pct + cycles are always the last two cols.
        parts = re.split(r"\s{2,}", stripped)
        if len(parts) < 4:
            continue
        try:
            pct = float(parts[-1])
            cycles = int(parts[-2])
        except ValueError:
            continue
        name = parts[0]
        op = parts[1] if len(parts) >= 4 else ""
        shape = parts[2] if len(parts) >= 5 else ""
        rows.append({
            "name": name,
            "op": op,
            "shape": shape,
            "cycles": cycles,
            "pct_of_total": pct,
        })
    return rows


def parse_total(text: str) -> int | None:
    m = re.search(r"^TOTAL\s+(\d+)$", text, re.MULTILINE)
    return int(m.group(1)) if m else None


def parse_wall_clock(text: str) -> int | None:
    m = re.search(r"wall_clock_cycles=(\d+)", text)
    return int(m.group(1)) if m else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spike-log", required=True, type=Path)
    ap.add_argument("--out-cycles", required=True, type=Path)
    ap.add_argument("--network", required=True,
                    help="network name for the output JSON's 'network' field")
    args = ap.parse_args(argv)

    text = args.spike_log.read_text()
    dispatches = parse_spike_log(text)
    total = parse_total(text)
    wall = parse_wall_clock(text)

    if not dispatches:
        print(f"ingest_measured_cycles: no dispatch rows found in "
              f"{args.spike_log}", file=sys.stderr)
        return 2

    out = {
        "network": args.network,
        "wall_clock_cycles": wall,
        "total_dispatch_cycles": total or sum(r["cycles"] for r in dispatches),
        "n_dispatches": len(dispatches),
        "dispatches": dispatches,
    }

    args.out_cycles.parent.mkdir(parents=True, exist_ok=True)
    args.out_cycles.write_text(json.dumps(out, indent=2))
    print(f"ingest_measured_cycles: wrote {args.out_cycles} "
          f"(n={len(dispatches)}, total={out['total_dispatch_cycles']} cyc, "
          f"wall={wall})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
