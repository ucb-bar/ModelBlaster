"""Rewrite the decode/prefill RMSNorm decomposition into a form QNN DSP
backend can compose.

The original ONNX has, per layer-norm site:
    Pow(x, 2) -> ReduceMean -> Add(eps) -> Sqrt -> Reciprocal -> Mul(x, ...) -> Mul(gamma, ...)

The `Reciprocal` op is what the DSP op-package list lacks. The simplest fix
is to replace it with a `Div(1.0, x)` op, which DSP supports as the standard
elementwise divide. Mathematically equivalent.

For each `Reciprocal` node:
  - emit a constant initializer with value 1.0 (broadcast-shaped)
  - emit a Div node taking (1.0, x) producing the same output name
  - remove the Reciprocal node

Usage:
    python fuse_rmsnorm.py --in path.onnx --out path_fused.onnx
    python fuse_rmsnorm.py --in-dir path/  --out-dir path_fused/  (batch)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def replace_reciprocal_with_div(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """Replace every Reciprocal(x) with Div(1.0, x). Returns (model, count)."""
    graph = model.graph
    new_nodes = []
    n_replaced = 0
    consts_to_add = []
    init_names = {init.name for init in graph.initializer}

    for node in graph.node:
        if node.op_type != "Reciprocal":
            new_nodes.append(node)
            continue
        x_name = node.input[0]
        y_name = node.output[0]
        # Make a scalar constant initializer with value 1.0 of fp32 dtype.
        const_name = f"__one_for_{node.name or n_replaced}"
        # Use scalar (rank 0) for broadcasting compatibility.
        const_tensor = numpy_helper.from_array(np.array(1.0, dtype=np.float32),
                                                 name=const_name)
        consts_to_add.append(const_tensor)
        new_div = helper.make_node(
            "Div",
            inputs=[const_name, x_name],
            outputs=[y_name],
            name=(node.name or f"div_recip_{n_replaced}") + "_div",
        )
        new_nodes.append(new_div)
        n_replaced += 1

    if n_replaced > 0:
        # Replace graph.node in place
        del graph.node[:]
        graph.node.extend(new_nodes)
        graph.initializer.extend(consts_to_add)

    return model, n_replaced


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", help="Single ONNX file")
    ap.add_argument("--out", dest="out_path", help="Output path (single mode)")
    ap.add_argument("--in-dir", dest="in_dir", help="Directory of dsp_seg_*.onnx files")
    ap.add_argument("--out-dir", dest="out_dir", help="Output directory")
    args = ap.parse_args()

    if args.in_dir:
        in_dir = Path(args.in_dir)
        out_dir = Path(args.out_dir) if args.out_dir else (in_dir / "rmsnorm_fused")
        out_dir.mkdir(exist_ok=True, parents=True)
        n_files = 0
        for onnx_file in sorted(in_dir.glob("dsp_seg_*.onnx")):
            model = onnx.load(str(onnx_file))
            model, count = replace_reciprocal_with_div(model)
            out_path = out_dir / onnx_file.name
            if count > 0:
                onnx.save(model, str(out_path))
                try:
                    onnx.checker.check_model(model, full_check=False)
                except Exception as e:
                    print(f"  warn: checker for {onnx_file.name}: {e}")
                print(f"  {onnx_file.name}: {count} Reciprocal -> Div")
            else:
                import shutil
                shutil.copy2(str(onnx_file), str(out_path))
                print(f"  {onnx_file.name}: no Reciprocal (unchanged)")
            n_files += 1
        print(f"\nProcessed {n_files} files into {out_dir}")
    else:
        if not args.in_path or not args.out_path:
            ap.error("either --in/--out (single) or --in-dir/--out-dir (batch)")
        model = onnx.load(str(args.in_path))
        model, count = replace_reciprocal_with_div(model)
        onnx.save(model, str(args.out_path))
        print(f"{args.in_path}: {count} Reciprocal -> Div -> {args.out_path}")


if __name__ == "__main__":
    main()
