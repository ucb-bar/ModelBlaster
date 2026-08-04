"""Extract pure Conv1x1 ops from vision slices as standalone HTA-targeted segments.

Each Linear layer in the ViT (fc1, fc2, QKV, output proj) has been rewritten
to Conv(1×1) by rewrite_matmul_to_conv1x1.py, but with Transpose/Reshape
wrappers that HTA can't execute. This script extracts each Conv op as a
standalone model:
  - Input: (B, C_in, M, 1) — NCHW, HTA-native
  - Output: (B, C_out, M, 1) — NCHW
  - Op: single Conv2d with kernel_shape=[1,1]

The surrounding Transpose/Reshape ops become CPU trampolines.

Usage:
  python extract_hta_convs.py --slices-dir vision_slices_v3/conv1x1 --out-dir vision_slices_v3/hta_convs
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _get_attr(node, name, default=None):
    for attr in node.attribute:
        if attr.name == name:
            return list(attr.ints) if attr.ints else [attr.i] if attr.i else default
    return default


def extract_conv_ops(model_path: str, out_dir: str, include_non_1x1: bool = True) -> list[dict]:
    """Extract each Conv node as a standalone ONNX model.

    Args:
        include_non_1x1: If True, also extract non-1x1 convolutions (e.g. patch embed)
            with their correct kernel_shape/strides. If False, skip them.
    """
    model = onnx.load(model_path)
    graph = model.graph
    seg_name = Path(model_path).stem

    init_map = {init.name: init for init in graph.initializer}
    vi_map = {vi.name: vi for vi in list(graph.value_info) + list(graph.input)}
    extracted = []

    for node in graph.node:
        if node.op_type != "Conv":
            continue

        conv_name = node.name
        x_name = node.input[0]
        w_name = node.input[1]

        w_init = init_map.get(w_name)
        if w_init is None:
            continue
        w_array = numpy_helper.to_array(w_init)
        N, C_in, kH, kW = w_array.shape

        kernel_shape = _get_attr(node, "kernel_shape", [kH, kW])
        strides = _get_attr(node, "strides", [1, 1])
        pads = _get_attr(node, "pads", [0, 0, 0, 0])

        is_1x1 = (kernel_shape == [1, 1])

        if not is_1x1 and not include_non_1x1:
            continue

        if is_1x1:
            # Rewritten MatMul→Conv1x1: input is (B, C_in, M, 1)
            M = 1024
            in_shape = [1, C_in, M, 1]
            out_shape = [1, N, M, 1]
            label = "conv1x1"
        else:
            # Real convolution (e.g. patch embed): derive shapes from graph
            in_vi = vi_map.get(x_name)
            if in_vi is not None:
                in_shape = [d.dim_value for d in in_vi.type.tensor_type.shape.dim]
            else:
                # Fallback for patch embed: SigLIP uses [1,3,512,512]
                in_shape = [1, C_in, 512, 512]

            # Compute output spatial dims
            H_in, W_in = in_shape[2], in_shape[3]
            H_out = (H_in + pads[0] + pads[2] - kH) // strides[0] + 1
            W_out = (W_in + pads[1] + pads[3] - kW) // strides[1] + 1
            out_shape = [1, N, H_out, W_out]
            label = f"conv{kH}x{kW}"

        X = helper.make_tensor_value_info("x", TensorProto.FLOAT, in_shape)
        Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, out_shape)

        new_w = numpy_helper.from_array(w_array.copy(), name="weight")

        inputs = ["x", "weight"]
        initializers = [new_w]
        if len(node.input) > 2 and node.input[2] != "":
            b_name = node.input[2]
            if b_name in init_map:
                b_array = numpy_helper.to_array(init_map[b_name])
                new_b = numpy_helper.from_array(b_array.copy(), name="bias")
                inputs.append("bias")
                initializers.append(new_b)

        conv_node = helper.make_node(
            "Conv", inputs=inputs, outputs=["y"],
            kernel_shape=kernel_shape, strides=strides, pads=pads,
            name=label
        )

        new_graph = helper.make_graph(
            [conv_node], f"{seg_name}_{conv_name}",
            inputs=[X], outputs=[Y],
            initializer=initializers
        )
        new_model = helper.make_model(
            new_graph, opset_imports=[helper.make_opsetid("", 17)]
        )
        new_model.ir_version = 8
        onnx.checker.check_model(new_model, full_check=False)

        out_name = f"{seg_name}_{conv_name}.onnx"
        out_path = os.path.join(out_dir, out_name)
        onnx.save(new_model, out_path)

        extracted.append({
            "segment": seg_name,
            "conv_name": conv_name,
            "in_channels": C_in,
            "out_channels": N,
            "kernel": kernel_shape,
            "strides": strides,
            "in_shape": in_shape,
            "out_shape": out_shape,
            "label": label,
            "file": out_name,
            "weight_size_mb": w_array.nbytes / 1e6,
        })

    return extracted


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slices-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    slices_dir = Path(args.slices_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_extracted = []
    for onnx_file in sorted(slices_dir.glob("dsp_seg_*.onnx")):
        extracted = extract_conv_ops(str(onnx_file), str(out_dir))
        all_extracted.extend(extracted)

    print(f"Extracted {len(all_extracted)} Conv ops as standalone HTA segments:")
    print(f"{'File':<55} {'Shape':>25} {'Kernel':>8} {'Weight MB':>10}")
    print("-" * 100)
    for info in all_extracted:
        shape = f"({info['in_channels']}→{info['out_channels']})"
        ks = f"{info['kernel'][0]}x{info['kernel'][1]}"
        print(f"  {info['file']:<53} {shape:>25} {ks:>8} {info['weight_size_mb']:>8.1f}")

    from collections import Counter
    shapes = Counter(
        f"{i['in_channels']}→{i['out_channels']} k={i['kernel'][0]}x{i['kernel'][1]}"
        for i in all_extracted
    )
    print(f"\nUnique shapes:")
    for shape, count in shapes.most_common():
        print(f"  {shape}: {count}")
    print(f"Total: {len(all_extracted)} Conv models in {out_dir}")


if __name__ == "__main__":
    main()
