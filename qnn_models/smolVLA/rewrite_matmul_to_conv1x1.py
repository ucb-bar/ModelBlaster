"""Rewrite MatMul ops in vision slices as Conv2d(1×1) for better HTA/DSP tiling.

On Hexagon v66, the QNN Conv2d implementation has heavily-optimized VTCM tiling
that the generic MatMul path lacks. Converting `MatMul(x[B,M,K], W[K,N])` to
`Conv2d(x_4d[B,K,M,1], W_4d[N,K,1,1])` is mathematically equivalent but may
route through a faster kernel.

For SigLIP ViT layers:
  - fc1: MatMul([1,1024,768], [768,3072]) → Conv(in=768, out=3072, k=1) on [1,768,1024,1]
  - fc2: MatMul([1,1024,3072], [3072,768]) → Conv(in=3072, out=768, k=1) on [1,3072,1024,1]
  - QKV: MatMul([1,1024,768], [768,2304]) → Conv(in=768, out=2304, k=1) on [1,768,1024,1]
  - proj: MatMul([1,1024,768], [768,768])  → Conv(in=768, out=768, k=1) on [1,768,1024,1]
  - attn_qk: MatMul([1,12,1024,64], [1,12,64,1024]) — batched, skip (not simple linear)
  - attn_v:  MatMul([1,12,1024,1024], [1,12,1024,64]) — batched, skip

Only rewrites MatMuls where one input is a constant initializer (i.e., weight
matrices from Linear layers). Batched attention MatMuls are left untouched.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _get_initializer(graph, name: str):
    for init in graph.initializer:
        if init.name == name:
            return init
    return None


def _get_value_info(graph, name: str):
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        if vi.name == name:
            return vi
    return None


def _tensor_shape(graph, name: str) -> list[int] | None:
    vi = _get_value_info(graph, name)
    if vi is None:
        return None
    return [d.dim_value for d in vi.type.tensor_type.shape.dim]


def rewrite_matmul_to_conv1x1(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """Rewrite eligible MatMul ops to Conv(1×1). Returns (new_model, count)."""
    graph = model.graph
    init_names = {init.name for init in graph.initializer}

    new_nodes = []
    rewritten = 0

    for node in graph.node:
        if node.op_type != "MatMul":
            new_nodes.append(node)
            continue

        x_name, w_name = node.input[0], node.input[1]

        # Only rewrite if W is a constant (initializer) — this is a Linear layer
        if w_name not in init_names:
            new_nodes.append(node)
            continue

        # Get shapes
        x_shape = _tensor_shape(graph, x_name)
        w_init = _get_initializer(graph, w_name)
        w_array = numpy_helper.to_array(w_init)

        if x_shape is None or len(x_shape) != 3:
            new_nodes.append(node)
            continue

        B, M, K = x_shape
        if w_array.ndim != 2:
            new_nodes.append(node)
            continue

        K2, N = w_array.shape
        if K != K2:
            new_nodes.append(node)
            continue

        out_name = node.output[0]
        prefix = node.name or f"rewrite_{rewritten}"

        # Step 1: Transpose x from (B, M, K) → (B, K, M) via Transpose
        x_transposed = f"{prefix}_x_transposed"
        transpose_node = helper.make_node(
            "Transpose", inputs=[x_name], outputs=[x_transposed],
            perm=[0, 2, 1], name=f"{prefix}_transpose_in"
        )

        # Step 2: Reshape (B, K, M) → (B, K, M, 1) for Conv2d input
        x_4d = f"{prefix}_x_4d"
        reshape_shape_name = f"{prefix}_reshape_to_4d_shape"
        reshape_shape = numpy_helper.from_array(
            np.array([B, K, M, 1], dtype=np.int64), name=reshape_shape_name
        )
        graph.initializer.append(reshape_shape)
        reshape_in_node = helper.make_node(
            "Reshape", inputs=[x_transposed, reshape_shape_name],
            outputs=[x_4d], name=f"{prefix}_reshape_in"
        )

        # Step 3: Create Conv weight: (N, K, 1, 1)
        conv_w_name = f"{prefix}_conv_weight"
        conv_w = numpy_helper.from_array(
            w_array.T.reshape(N, K, 1, 1).copy(), name=conv_w_name
        )
        graph.initializer.append(conv_w)

        # Step 4: Conv node (no bias)
        conv_out = f"{prefix}_conv_out"
        conv_node = helper.make_node(
            "Conv", inputs=[x_4d, conv_w_name], outputs=[conv_out],
            kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0],
            name=f"{prefix}_conv1x1"
        )

        # Step 5: Reshape (B, N, M, 1) → (B, N, M)
        squeezed = f"{prefix}_squeezed"
        reshape_shape_3d_name = f"{prefix}_reshape_to_3d_shape"
        reshape_shape_3d = numpy_helper.from_array(
            np.array([B, N, M], dtype=np.int64), name=reshape_shape_3d_name
        )
        graph.initializer.append(reshape_shape_3d)
        reshape_out_node = helper.make_node(
            "Reshape", inputs=[conv_out, reshape_shape_3d_name],
            outputs=[squeezed], name=f"{prefix}_reshape_out"
        )

        # Step 6: Transpose (B, N, M) → (B, M, N) to match original output shape
        transpose_out_node = helper.make_node(
            "Transpose", inputs=[squeezed], outputs=[out_name],
            perm=[0, 2, 1], name=f"{prefix}_transpose_out"
        )

        new_nodes.extend([
            transpose_node, reshape_in_node, conv_node,
            reshape_out_node, transpose_out_node
        ])

        # Remove old weight initializer (replaced by conv weight)
        # Keep it — other nodes might reference it. Actually for extracted
        # slices each weight is used exactly once, but safer to leave it.

        rewritten += 1

    if rewritten > 0:
        del graph.node[:]
        graph.node.extend(new_nodes)

    return model, rewritten


def validate_rewrite(orig_path: str, new_model: onnx.ModelProto):
    """Check that the rewritten model produces the same output."""
    import onnxruntime as ort

    # Save temp
    tmp_path = "/tmp/_conv1x1_validate.onnx"
    onnx.save(new_model, tmp_path)

    sess_orig = ort.InferenceSession(orig_path, providers=["CPUExecutionProvider"])
    sess_new = ort.InferenceSession(tmp_path, providers=["CPUExecutionProvider"])

    np.random.seed(42)
    inputs = {}
    for inp in sess_orig.get_inputs():
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        inputs[inp.name] = np.random.randn(*shape).astype(np.float32)

    out_orig = sess_orig.run(None, inputs)
    out_new = sess_new.run(None, inputs)

    max_diff = max(np.max(np.abs(a - b)) for a, b in zip(out_orig, out_new))
    os.remove(tmp_path)
    return max_diff


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="Input ONNX file")
    ap.add_argument("-o", "--output", help="Output path (default: <input>_conv1x1.onnx)")
    ap.add_argument("--validate", action="store_true", help="Run numeric validation")
    ap.add_argument("--batch", action="store_true",
                    help="Process all dsp_seg_*.onnx in a directory")
    args = ap.parse_args()

    if args.batch:
        input_dir = Path(args.input)
        out_dir = input_dir / "conv1x1"
        out_dir.mkdir(exist_ok=True)
        for onnx_file in sorted(input_dir.glob("dsp_seg_*.onnx")):
            model = onnx.load(str(onnx_file))
            model, count = rewrite_matmul_to_conv1x1(model)
            out_path = out_dir / onnx_file.name
            if count > 0:
                onnx.save(model, str(out_path))
                msg = f"  {onnx_file.name}: {count} MatMul → Conv1x1"
                if args.validate:
                    diff = validate_rewrite(str(onnx_file), model)
                    msg += f"  max|Δ|={diff:.2e} {'PASS' if diff < 1e-4 else 'FAIL'}"
                print(msg)
            else:
                # Copy unchanged
                import shutil
                shutil.copy2(str(onnx_file), str(out_path))
                print(f"  {onnx_file.name}: no MatMul with const weights (unchanged)")
    else:
        model = onnx.load(args.input)
        model, count = rewrite_matmul_to_conv1x1(model)
        out_path = args.output or args.input.replace(".onnx", "_conv1x1.onnx")
        onnx.save(model, out_path)
        print(f"Rewrote {count} MatMul → Conv1x1: {out_path}")
        if args.validate and count > 0:
            diff = validate_rewrite(args.input, model)
            print(f"  Numeric validation: max|Δ|={diff:.2e} {'PASS' if diff < 1e-4 else 'FAIL'}")


if __name__ == "__main__":
    main()
