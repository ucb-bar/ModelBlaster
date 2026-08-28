#!/usr/bin/env python3
"""Refuse a build that is labelled with a vector backend but runs scalar code.

Why this exists
---------------
`generate_kernels.py` looks curated kernels up by EXACT op name:

    kernels/<backend>/<backend>_<op>_<algorithm>.c

When no file matches, selection falls back to the scalar reference
implementation, records `"source": "reference"` in kernel_picks.json, and the
build succeeds. Nothing reads that field, so the fallback is invisible.

Fusion is what makes this bite. The curated RVV library has `conv2d_s8`,
`batchnorm2d_s8` and `silu_s8` as separate kernels, but the graph fuses them
into `conv2d_batchnorm2d_silu_s8` -- an op name the library has never heard of.
Every constituent had a vector kernel and the fused op still ran scalar.

Measured on the K1 before this gate existed:

    yolov8_nano  rvv_x60   57 of 90 dispatches on reference   99.8% of 4974.8 ms
    dronet       rvv_x60    3 of 21 dispatches on reference   86.7% of   62.6 ms

yolov8_nano measured 0.81x against the pure-scalar build -- SLOWER -- because it
paid the vector build's overhead while executing scalar code. That number sat in
a results table looking like a real finding about RVV being a poor fit for the
op mix. It was a missing file.

A gate is the right shape for this because the failure is silent, cheap to
check, and recurs every time fusion introduces a new op name. Weighting by
dispatch count (or by measured time, with --profile) is what separates "a
trivial op has no vector kernel, fine" from "the model runs scalar".
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys

#: Ops that are legitimately left on the reference implementation: they are
#: memory-bound reshapes or single-element operations where a vector kernel
#: would buy nothing. Anything NOT in here that carries weight is a defect.
_BENIGN = {
    "reshape", "flatten", "transpose", "squeeze", "unsqueeze",
    "identity", "quantize", "dequantize",
}


def _op_weights(graph_path):
    """dispatch count per op, from the graph the kernels were generated from."""
    with open(graph_path) as fh:
        graph = json.load(fh)
    counts = collections.Counter()
    for node in graph.get("ops") or []:
        counts[node.get("op", "?")] += 1
    return counts


def _profile_weights(profile_csv):
    """measured ms per op, and which ops actually ran on the reference.

    Returns (weights, measured_fallback_ops).

    The second value comes from the profile's `implementation` column, which
    profile_writer fills from the build's kernel_picks.json at PROFILE time.
    That is what makes it ground truth rather than a claim: it is recorded
    alongside the timing, by the run that produced it, so it cannot describe a
    different build than the one measured.

    It matters because a kernel_picks.json read later can outlive its sources:
    mlp_control's claimed `linear_s8: reference` while the profile taken
    afterwards showed a curated kernel, and reporting the stale claim would
    have invented a regression. When both are available the profile wins and
    the disagreement is reported as staleness.

    Do NOT infer this from module_name. Its trailing segment is the shape tag,
    and an op with no recorded shape used to render as `..._<op>_scalar`,
    which reads exactly like "ran the scalar reference".
    """
    ms = collections.Counter()
    ran_reference = set()
    with open(profile_csv) as fh:
        for row in csv.DictReader(fh):
            op = row.get("op", "?")
            try:
                ms[op] += float(row.get("mean_time_ns") or 0) / 1e6
            except ValueError:
                pass
            impl = (row.get("implementation") or "").strip()
            if impl:
                if impl.split("/")[0] == "reference":
                    ran_reference.add(op)
            elif (row.get("module_name") or "").endswith("_scalar"):
                # Legacy profiles only. This is a WEAK signal and it is wrong
                # in both directions: the trailing module_name segment is the
                # SHAPE tag, and _shape_concise() used to emit the literal
                # "scalar" for an op with no recorded shape. The fused convs
                # had no shape AND ran the reference, so the two coincided
                # and the label looked reliable -- until real RVV kernels
                # landed (22.9x measured) and the name still said "_scalar".
                # Trust it only when `implementation` is absent entirely.
                ran_reference.add(op)
    return ms, ran_reference


def check(gen_dir, graph_path, profile_csv=None, threshold=5.0, allow=()):
    picks_path = os.path.join(gen_dir, "kernel_picks.json")
    with open(picks_path) as fh:
        picks = json.load(fh)

    target = picks.get("target", "?")
    if target == "scalar":
        print(f"OK   {gen_dir}: target is scalar; nothing to check")
        return True

    measured_fallback = None
    if profile_csv:
        weights, measured_fallback = _profile_weights(profile_csv)
        unit = "ms"
    else:
        weights, unit = _op_weights(graph_path), "dispatches"
    total = sum(weights.values())
    if not total:
        print(f"WARN {gen_dir}: no weights found; cannot judge coverage",
              file=sys.stderr)
        return True

    claimed = {op for op, pick in picks.get("picks", {}).items()
               if pick.get("source") == "reference"}
    on_reference = claimed if measured_fallback is None else measured_fallback

    if measured_fallback is not None and claimed != measured_fallback:
        stale = ", ".join(sorted(claimed ^ measured_fallback)) or "-"
        print(f"     note: {picks_path} disagrees with the measured run on "
              f"[{stale}]; it predates this profile, so the measurement is used")

    fallbacks = []
    for op in sorted(on_reference):
        if op in _BENIGN or op in allow:
            continue
        w = weights.get(op, 0)
        if w:
            fallbacks.append((op, w, 100.0 * w / total))

    fallbacks.sort(key=lambda t: -t[1])
    worst = fallbacks[0][2] if fallbacks else 0.0

    if not fallbacks:
        print(f"OK   {gen_dir} [{target}]: every weighted op has a {target} kernel")
        return True

    lead = "FAIL" if worst >= threshold else "warn"
    share = sum(f[2] for f in fallbacks)
    print(f"{lead} {gen_dir} [{target}]: {len(fallbacks)} op(s) on the SCALAR "
          f"reference, {share:.1f}% of {total:.1f} {unit}")
    for op, w, pct in fallbacks:
        print(f"       {op:<40} {w:>9.1f} {unit}  {pct:>5.1f}%")

    if worst >= threshold:
        print(f"\nA build targeting '{target}' is executing scalar code for the ops "
              f"above.\nThe curated lookup is by EXACT op name, so the fix is to add\n"
              f"    kernels/{target}/{target}_<op>_<algorithm>.c\n"
              f"(or an alias dir -- see backends.curated_aliases). A fused op needs its\n"
              f"OWN kernel; having one for each constituent does not compose.\n"
              f"If a fallback here is genuinely acceptable, pass --allow <op> so the\n"
              f"exemption is recorded in the command rather than assumed.",
              file=sys.stderr)
        return False
    return True


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gen_dirs", nargs="+",
                    help="directories holding kernel_picks.json")
    ap.add_argument("--graph", help="graph.json (default: <gen_dir>/../graph.json)")
    ap.add_argument("--profile", help="results.csv, to weight by measured ms")
    ap.add_argument("--threshold", type=float, default=5.0,
                    help="fail if any single op on reference exceeds this %% "
                         "of the weight (default 5)")
    ap.add_argument("--allow", action="append", default=[],
                    help="op that may stay on reference (repeatable)")
    args = ap.parse_args()

    ok = True
    for d in args.gen_dirs:
        graph = args.graph or os.path.join(os.path.dirname(d.rstrip("/")), "graph.json")
        if not os.path.exists(graph) and not args.profile:
            print(f"WARN {d}: no graph.json at {graph}; skipping", file=sys.stderr)
            continue
        ok &= check(d, graph, args.profile, args.threshold, set(args.allow))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
