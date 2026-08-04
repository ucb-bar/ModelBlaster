#!/usr/bin/env python3
"""Export DroNet to ONNX format.

Usage:
    python export_onnx.py [--large] [--checkpoint PATH] [--output PATH]
"""

import argparse
import os

import torch

from dronet import DronetTorch


def main():
    parser = argparse.ArgumentParser(description="Export DroNet to ONNX")
    parser.add_argument("--large", action="store_true",
                        help="Use 224x224 model instead of 112x112")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a .pth checkpoint to load weights from")
    parser.add_argument("--output", type=str, default=None,
                        help="Output ONNX path (default: dronet.onnx in this dir)")
    args = parser.parse_args()

    small = not args.large
    img_dim = 112 if small else 224

    model = DronetTorch(img_dims=(img_dim, img_dim), img_channels=3,
                        output_dim=1, small=small)

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state)

    model.eval()

    dummy_input = torch.randn(1, 3, img_dim, img_dim)
    output_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "dronet.onnx")

    torch.onnx.export(
        model, dummy_input, output_path,
        input_names=["input"],
        output_names=["steer", "collision"],
        opset_version=18,
    )

    size_kb = os.path.getsize(output_path) / 1024
    print(f"Exported: {output_path} ({size_kb:.1f} KB)")
    print(f"  Input:  [1, 3, {img_dim}, {img_dim}]")
    print(f"  Outputs: steer [1,1], collision [1,1]")


if __name__ == "__main__":
    main()
