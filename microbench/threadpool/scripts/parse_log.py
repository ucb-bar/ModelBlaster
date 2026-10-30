#!/usr/bin/env python3
"""Extract the THREADPOOL_BENCH CSV block from a raw harness log.

Each bench harness emits one or more blocks:
    === THREADPOOL_BENCH_BEGIN ===
    harness,variant,n_workers,range,reps,...
    <rows>
    === THREADPOOL_BENCH_END ===

Some harnesses emit multiple blocks (e.g. the pthreadpool one prints a
local-sem ping-pong baseline as a separate single-row block before the
main sweep). We concatenate them into one CSV with a single header.
"""

from __future__ import annotations

import argparse
import re
import sys

BEGIN_RE = re.compile(r"=== THREADPOOL_BENCH_BEGIN ===")
END_RE = re.compile(r"=== THREADPOOL_BENCH_END ===")


def extract_blocks(text: str) -> list[list[str]]:
    """Return the list of CSV-line lists between every BEGIN/END pair."""
    out = []
    start = 0
    while True:
        m = BEGIN_RE.search(text, start)
        if not m:
            break
        e = END_RE.search(text, m.end())
        if not e:
            break
        body = text[m.end():e.start()].strip()
        rows = [ln for ln in body.splitlines() if ln.strip()]
        out.append(rows)
        start = e.end()
    return out


def merge_blocks(blocks: list[list[str]]) -> list[str]:
    if not blocks:
        return []
    # Use the first block's header; assume every other block has the
    # same schema (the harnesses all use bench_emit_csv from
    # bench_common.h, so this is guaranteed at compile time).
    header = blocks[0][0]
    rows = [header]
    for blk in blocks:
        rows.extend(blk[1:])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True,
                    help="raw harness stdout (spike's tee'd output or "
                         "FireSim uartlog)")
    ap.add_argument("--out", required=True,
                    help="path to write the merged CSV")
    args = ap.parse_args()

    with open(args.raw) as f:
        text = f.read()
    blocks = extract_blocks(text)
    if not blocks:
        print(f"FAIL: no THREADPOOL_BENCH block in {args.raw}", file=sys.stderr)
        return 1
    rows = merge_blocks(blocks)
    with open(args.out, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"wrote {len(rows) - 1} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
