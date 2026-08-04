"""Build a QNN-route-to-ONNX-route op-name map by structural alignment.

Yolov8n's schedule was profiled on a TFLite-route DLC whose per-op
`module_name`s look like `pad_0`, `convolution_0`,
`elementwise_product_0`, ... — none of which appear in the
`yolov8n.onnx` file (whose nodes are `/model.0/conv/Conv`,
`/model.0/act/Mul`, ...). slice_to_subonnx.py needs an ONNX node name
to extract sub-graphs from; without a translation step it falls back
to the partition file's coarse 2-segment cut.

The key insight (PARTITIONING_GUIDE §1 of the doc, plus what we
observe): both routes describe the *same* underlying network, in the
*same* topological order, modulo a small set of converter-injected
ops. The TFLite route inserts an explicit `pad_<n>` ahead of each
Conv2d (because TFLite Conv with asymmetric SAME padding emits the
Pad as a separate op); ONNX folds that Pad into the Conv attribute.
Conversely, ONNX Split → TFLite materialises as a `strided_slice` op.

So a greedy walk over both lists, using a small bucket-equivalence
table to match op-type at each step and skipping over the converter-
injected ops on either side, produces a stable `module_name → ONNX
node name` map.

Output format matches `slice_to_subonnx.py --name-map`:

    {
      "ops": [
        {"name": "<schedule_module_name>", "_orig_name": "<onnx_node_name>"},
        ...
      ]
    }

CLI:
    python3 build_qnn_to_onnx_namemap.py \\
        --per-op-csv qnn_models/boards/qrb5165_v66/per_op_stats/yolov8n__DSP.csv \\
        --onnx qnn_models/yolov8n.onnx \\
        --out qnn_models/runtime/gen/qrb5165_dronet_yolov8/yolov8n_qnn_to_onnx.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys


# Canonicalise op-type from each side onto a small shared bucket
# vocabulary so structural matching can compare apples to apples.
# Buckets are deliberately coarse: any op-type that "feels the same"
# to the scheduler maps to one bucket. The matcher prioritises
# bucket-equivalence over exact-type equivalence — most CSV op_type
# columns are already bucketed (Conv / Pool / Activation / ElementWise /
# Reshape/Layout / Split/Slice / ...).

_ONNX_TO_BUCKET = {
    "Conv":             "Conv",
    "ConvTranspose":    "Conv",
    "Gemm":             "FC/MatMul",
    "MatMul":           "FC/MatMul",
    "MaxPool":          "Pool",
    "AveragePool":      "Pool",
    "GlobalAveragePool":"Pool",
    "Sigmoid":          "Activation",
    "Tanh":             "Activation",
    "Relu":             "Activation",
    "HardSwish":        "Activation",
    "HardSigmoid":      "Activation",
    "LeakyRelu":        "Activation",
    "Softmax":          "Softmax",        # distinct bucket — TFLite tags it specially
    "Add":              "ElementWise",
    "Mul":              "ElementWise",
    "Sub":              "ElementWise",
    "Div":              "ElementWise",
    "Pow":              "ElementWise",
    "Min":              "ElementWise",
    "Max":              "ElementWise",
    "BatchNormalization":"Norm",
    "LayerNormalization":"Norm",
    "InstanceNormalization":"Norm",
    "Reshape":          "Reshape/Layout",
    "Transpose":        "Reshape/Layout",
    "Squeeze":          "Reshape/Layout",
    "Unsqueeze":        "Reshape/Layout",
    "Flatten":          "Reshape/Layout",
    "Cast":             "Reshape/Layout",
    "Pad":              "Pad",            # converter-injected, special
    "Resize":           "Resize",
    "Upsample":         "Resize",
    "Concat":           "Concat",
    "Split":            "Split/Slice",
    "Slice":            "Split/Slice",
    "StridedSlice":     "Split/Slice",
    "Gather":           "Gather",
    "ScatterND":        "Scatter",
    "Where":            "Where",
}

_QNN_OPNAME_TO_BUCKET = {
    # Match by the prefix of `module_name` after stripping suffix.
    # TFLite-route module names follow `<op_name>:OpId_<n> (us)`.
    "pad":                          "Pad",
    "convolution":                  "Conv",
    "depthwise_convolution":        "Conv",
    "elementwise_product":          "ElementWise",   # x*y — used by SiLU's mul
    "elementwise_sum":              "ElementWise",
    "elementwise_difference":       "ElementWise",
    "elementwise_div":              "ElementWise",
    "elementwise_neuron":           "Activation",
    "elementwiseneuron":            "Activation",
    "fully_connected":              "FC/MatMul",
    "matmul":                       "FC/MatMul",
    "pool":                         "Pool",
    "max_pool":                     "Pool",
    "avg_pool":                     "Pool",
    "softmax":                      "Softmax",
    "concat":                       "Concat",
    "reshape":                      "Reshape/Layout",
    "transpose":                    "Reshape/Layout",
    "strided_slice":                "Split/Slice",
    "split":                        "Split/Slice",
    "slice":                        "Split/Slice",
    "resize":                       "Resize",
    "batchnorm":                    "Norm",
}


def _csv_op_bucket(op_type: str, op_name: str) -> str:
    """Best-guess bucket from the per-op CSV's `op_type` column +
    `op_name` prefix. The CSV's op_type is usually already bucketed
    (Conv / Pool / Activation / ElementWise / Reshape/Layout /
    Split/Slice / Concat / Norm). Prefer that; fall back to op-name
    prefix for the few buckets the CSV doesn't distinguish."""
    t = op_type.strip()
    # CSV uses some known shorthand variants; normalise them.
    if t in ("Conv", "Pool", "Activation", "ElementWise",
             "Reshape/Layout", "Split/Slice", "Concat", "Norm",
             "FC/MatMul", "Softmax", "Pad", "Resize", "Gather",
             "Scatter", "Where"):
        return t
    # CSV's exact value sometimes is the QNN type ("ElementWiseNeuron").
    name_prefix = re.split(r"[:_0-9]", op_name.strip(), maxsplit=1)[0].lower()
    for k, v in _QNN_OPNAME_TO_BUCKET.items():
        if op_name.strip().lower().startswith(k):
            return v
    if name_prefix:
        for k, v in _QNN_OPNAME_TO_BUCKET.items():
            if name_prefix.startswith(k):
                return v
    return t or "Other"


def _strip_qnn_suffix(s: str) -> str:
    """Strip ":OpId_<n> (us|cycles)" tails or leading underscore from
    a CSV row's op_name so it matches what the schedule uses as
    `module_name`."""
    s = re.sub(r":OpId_\d+\s*\([^)]*\)\s*$", "", s).strip()
    while s.startswith("_"):
        s = s[1:]
        break
    return s


def _onnx_bucket(op_type: str) -> str:
    return _ONNX_TO_BUCKET.get(op_type, op_type)


def align(qnn_seq: list[tuple[str, str]],
           onnx_seq: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Greedy bucket-aligned walk. Each side's element is
    (bucket, name). Pairs each QNN op with the next ONNX op of
    matching bucket; skips Pad-only QNN ops (no ONNX equivalent) and
    Cast/Reshape-only ONNX ops that don't surface in QNN.

    Returns a list of (qnn_name, onnx_name) — onnx_name is empty when
    the QNN op has no ONNX counterpart (typically a TFLite-injected
    Pad).
    """
    qi = oi = 0
    out: list[tuple[str, str]] = []
    skipped_qnn: list[str] = []
    while qi < len(qnn_seq):
        qb, qn = qnn_seq[qi]
        # Try the obvious: bucket matches the next ONNX op.
        if oi < len(onnx_seq) and onnx_seq[oi][0] == qb:
            out.append((qn, onnx_seq[oi][1]))
            qi += 1; oi += 1
            continue
        # QNN-side Pad with no matching ONNX Pad — TFLite injects these
        # but ONNX folds them into the next Conv. Skip the QNN Pad.
        if qb == "Pad" and (oi >= len(onnx_seq) or onnx_seq[oi][0] != "Pad"):
            out.append((qn, ""))     # unmapped — slicer will treat as "skip op"
            skipped_qnn.append(qn)
            qi += 1
            continue
        # ONNX-side filler the QNN route doesn't have — skip the ONNX op
        # without consuming a QNN entry. Common cases: Cast, leftover
        # Reshape/Transpose during conversion.
        if oi < len(onnx_seq) and onnx_seq[oi][0] in ("Reshape/Layout",
                                                       "Cast", "Pad"):
            oi += 1
            continue
        # Try a small look-ahead in ONNX (up to 5 ops) for a bucket match.
        found = False
        for la in range(1, min(6, len(onnx_seq) - oi)):
            if onnx_seq[oi + la][0] == qb:
                # Skip ONNX-side filler in between.
                oi += la
                out.append((qn, onnx_seq[oi][1]))
                qi += 1; oi += 1
                found = True
                break
        if not found:
            print(f"  align: stuck at QNN[{qi}]={qn!r} ({qb}) vs "
                  f"ONNX[{oi}]={onnx_seq[oi][1] if oi < len(onnx_seq) else 'EOF'} "
                  f"({onnx_seq[oi][0] if oi < len(onnx_seq) else 'EOF'}) "
                  f"— leaving unmapped",
                  file=sys.stderr)
            out.append((qn, ""))
            qi += 1
    # If ONNX has trailing ops, that's fine — they're network outputs we
    # didn't have a QNN counterpart for in the schedule (rare).
    if skipped_qnn:
        print(f"  {len(skipped_qnn)} QNN ops left unmapped (TFLite-injected "
              f"Pads / layout converters): {skipped_qnn[:5]}{'...' if len(skipped_qnn)>5 else ''}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-op-csv", required=True,
                    help="schedule's per-op stats CSV (TFLite-route op order)")
    ap.add_argument("--onnx",       required=True,
                    help="source ONNX file (ONNX-route op order)")
    ap.add_argument("--out",        required=True,
                    help="emit name-map JSON in slice_to_subonnx.py's --name-map shape")
    args = ap.parse_args()

    try:
        import onnx
    except ImportError as e:
        sys.exit(f"need onnx: {e}")

    # Schedule sequence.
    qnn_seq: list[tuple[str, str]] = []
    with open(args.per_op_csv) as f:
        for r in csv.DictReader(f):
            nm = _strip_qnn_suffix(r["op_name"])
            if not nm or nm.lower().startswith("input opid") or nm.lower().startswith("misc"):
                continue
            bucket = _csv_op_bucket(r.get("op_type", ""), nm)
            qnn_seq.append((bucket, nm))

    # ONNX sequence — graph order is topological for export-style ONNX.
    m = onnx.load(args.onnx)
    onnx_seq: list[tuple[str, str]] = []
    for n in m.graph.node:
        onnx_seq.append((_onnx_bucket(n.op_type), n.name))

    print(f"qnn (schedule) ops: {len(qnn_seq)}")
    print(f"onnx ops          : {len(onnx_seq)}")

    pairs = align(qnn_seq, onnx_seq)
    n_mapped = sum(1 for _, o in pairs if o)
    print(f"\nmapped: {n_mapped}/{len(pairs)} QNN ops to ONNX names")

    out = {"ops": [
        {"name": q, "_orig_name": o} for q, o in pairs if o
    ]}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}: {len(out['ops'])} pairs")
    print(f"  (the unmapped QNN ops are TFLite-injected Pads — slice_to_subonnx.py")
    print(f"   will skip them in segment construction; ONNX folds them into Conv attrs)")


if __name__ == "__main__":
    main()
