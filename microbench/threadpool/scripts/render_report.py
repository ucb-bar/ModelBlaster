#!/usr/bin/env python3
"""Render the markdown report from firesim_overhead.csv.

Walks the merged CSV produced by run_all.sh and emits:
  - headline overhead table (per-call median cycles by harness × N)
  - breakdown table for the harnesses that expose it (k_thread,
    pthreads_raw): wake->first_worker, wake->all_finished,
    finish->observed
  - the k_sem ping-pong baseline so per-sem cost is visible
  - a brief recommendation block

Reads cycle counts in raw cycles; converts to microseconds at 1 GHz
(matches FireSim quad-rocket-saturn target — see project memory entry
firesim_pthreadpool_overhead.md).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict


def cyc_to_us(cyc: int, mhz: float = 1000.0) -> float:
    return cyc / mhz / 1000.0  # mhz is target MHz; cyc / mhz = ns; /1000 = us


def fmt_cyc_us(cyc: int) -> str:
    if cyc <= 0:
        return "—"
    us = cyc / 1e6  # 1 GHz: 1e6 cyc = 1 ms; per-call values mostly 1k–13M
    return f"{int(cyc):,} ({us*1000:.1f} us)"


def fmt_short(cyc: int) -> str:
    if cyc <= 0:
        return "—"
    return f"{int(cyc):,}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
                    help="path to firesim_overhead.csv")
    ap.add_argument("--out", required=True,
                    help="path to write REPORT.md")
    args = ap.parse_args()

    rows = []
    with open(args.csv) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            for k in ("n_workers", "range", "reps",
                      "per_call_min", "per_call_med", "per_call_max",
                      "wake_to_first_worker", "wake_to_all_finished",
                      "finish_to_observed"):
                r[k] = int(r[k])
            rows.append(r)

    if not rows:
        print("no rows in CSV", file=sys.stderr)
        return 1

    # Pull out the local-sem baseline.
    base = next((r for r in rows if r["harness"] == "k_sem_pingpong"), None)
    main_rows = [r for r in rows if r["harness"] != "k_sem_pingpong"]

    # Group by (harness, variant).
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in main_rows:
        groups[(r["harness"], r["variant"])].append(r)

    nw_set = sorted({r["n_workers"] for r in main_rows})
    rg_set = sorted({r["range"] for r in main_rows})

    out = []
    out.append("# Threadpool overhead microbenchmarks — FireSim "
               "quad-rocket-saturn")
    out.append("")
    out.append("Cross-hart synchronization overhead for three dispatch "
               "primitives, measured on FireSim's "
               "alveo_u250_firesim-quad-rocket-saturn-no-nic-l2-llc4mb-ddr3 "
               "hwconfig (4× RVV rocket, 1 GHz target, mtime at 1 MHz). "
               "Per-call cycles are master-thread `rdcycle` deltas around "
               "one dispatch. Each row is the median of "
               f"{main_rows[0]['reps']} timed iterations after "
               "32 warmup iterations.")
    out.append("")
    if base:
        out.append("## k_sem ping-pong baseline (single-hart)")
        out.append("")
        out.append("Round-trip `k_sem_give → k_sem_take` on the master "
                   "hart, no IPI. Establishes the floor cost of the "
                   "primitive itself.")
        out.append("")
        out.append("| metric | cycles |")
        out.append("|---|---|")
        out.append(f"| min | {fmt_short(base['per_call_min'])} |")
        out.append(f"| median | {fmt_short(base['per_call_med'])} |")
        out.append(f"| max | {fmt_short(base['per_call_max'])} |")
        out.append("")

    out.append("## Per-call dispatch cycles (median)")
    out.append("")
    headers = ["harness", "variant", "range"] + [f"N={n}" for n in nw_set]
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    seen_ord = []  # preserve a stable harness ordering
    for (h, v) in groups:
        if (h, v) not in seen_ord:
            seen_ord.append((h, v))
    # Sort: native first, then pthreads_raw, then pthreadpool variants.
    order_map = {"k_thread": 0, "pthreads_raw": 1, "pthreadpool": 2}
    seen_ord.sort(key=lambda hv: (order_map.get(hv[0], 99),
                                  0 if hv[1] == "default" else 1
                                  if hv[1] == "k_sem" else 2))
    for (h, v) in seen_ord:
        for rg in rg_set:
            cells = [h, v, str(rg)]
            for nw in nw_set:
                m = next((r for r in groups[(h, v)]
                          if r["n_workers"] == nw and r["range"] == rg),
                         None)
                cells.append(fmt_short(m["per_call_med"]) if m else "—")
            out.append("| " + " | ".join(cells) + " |")
    out.append("")
    out.append("Cycles are at the 1 GHz target clock; divide by 1e6 for ms. "
               "Column N is the worker count (master included for "
               "pthreadpool, master-only-dispatch for the others).")
    out.append("")

    out.append("## Wake / finish breakdown (k_thread, pthreads_raw)")
    out.append("")
    out.append("Worker stamps `sys_clock_cycle_get_64()` (mtime, the "
               "global tick counter — coherent across harts unlike "
               "rdcycle) right after returning from "
               "`k_sem_take(start_sem)` (= first observable wake) and "
               "again right after computing its slice (= done). All "
               "deltas are master-relative **mtime ticks**; on FireSim "
               "1 tick = 1 µs target = 1000 target cycles at 1 GHz. "
               "Multiply by 1000 to compare with the per-call cycle "
               "column above.")
    out.append("")
    out.append("| harness | N | range | wake→first_worker (µs) | "
               "wake→all_finished (µs) | finish→observed (µs) |")
    out.append("|---|---|---|---|---|---|")
    for (h, v) in seen_ord:
        if h == "pthreadpool":
            continue  # opaque internals — fields are 0
        for nw in nw_set:
            for rg in rg_set:
                m = next((r for r in groups[(h, v)]
                          if r["n_workers"] == nw and r["range"] == rg),
                         None)
                if not m:
                    continue
                out.append(f"| {h} | {nw} | {rg} | "
                           f"{fmt_short(m['wake_to_first_worker'])} | "
                           f"{fmt_short(m['wake_to_all_finished'])} | "
                           f"{fmt_short(m['finish_to_observed'])} |")
    out.append("")

    # Compute pthreadpool overhead vs k_thread baseline at largest N.
    out.append("## Cost attribution")
    out.append("")
    out.append("Comparing per-call medians at N=4 (pulls in the worst-case "
               "wake fanout) for the smallest range (range=32 — work is "
               "negligible, so per-call cycles ≈ pure dispatch cost):")
    out.append("")
    rows_n4 = [r for r in main_rows if r["n_workers"] == 4
               and r["range"] == 32]
    out.append("| harness | variant | per-call median (cyc) | "
               "vs k_thread |")
    out.append("|---|---|---|---|")
    base_kt = next((r for r in rows_n4 if r["harness"] == "k_thread"),
                   None)
    base_med = base_kt["per_call_med"] if base_kt else None
    for r in rows_n4:
        ratio = ""
        if base_med and base_med > 0:
            ratio = f"{r['per_call_med'] / base_med:.2f}×"
        out.append(f"| {r['harness']} | {r['variant']} | "
                   f"{fmt_short(r['per_call_med'])} | {ratio} |")
    out.append("")

    out.append("## Recommendation")
    out.append("")
    out.append("- For ops smaller than ~1 M cycles per worker share, "
               "**stay sequential** — even the cheapest cross-hart "
               "primitive (k_thread + k_sem) costs millions of cycles "
               "per dispatch on this RTL.")
    out.append("- When parallelism *is* warranted, the relative ordering "
               "(cheapest first) at N=4 is `k_thread (k_sem)` ≤ "
               "`pthreads_raw (k_sem)` ≪ `pthreadpool (default)`. "
               "Switching pthreadpool to `spin` removes the condvar "
               "fallback, recovering the closing pthread_cond_wait cost "
               "but leaving pthreadpool's queue/state-machine overhead "
               "intact.")
    out.append("- Concrete pthreadpool fix path, in order of impact: "
               "(1) bump `PTHREADPOOL_SPIN_WAIT_ITERATIONS` so the master "
               "spin-wait covers worst-case cross-hart latency on this "
               "RTL — see the `spin` variant for the upper bound; "
               "(2) replace pthread_cond_wait with futex-equivalent "
               "(POSIX_CONFSTR_FUTEX or Zephyr `k_futex`); (3) replace "
               "the wrapper's pthread_create-spawned workers with "
               "pre-pinned k_threads, side-stepping the POSIX layer "
               "entirely (the `k_thread` row above is the floor).")
    out.append("")
    out.append("Generated from "
               f"`{os.path.relpath(args.csv, os.path.dirname(args.out))}`.")
    out.append("")

    with open(args.out, "w") as f:
        f.write("\n".join(out))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
