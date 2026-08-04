#!/usr/bin/env python3
"""Export MobileNetV2 to ONNX format.

Usage:
    python export_mobilenet.py [--output PATH]
"""

import argparse
import os

import torch
import torchvision.models as models


def main():
    parser = argparse.ArgumentParser(description="Export MobileNetV2 to ONNX")
    parser.add_argument("--output", type=str, default=None,
                        help="Output ONNX path (default: mobilenet_v2.onnx in this dir)")
    args = parser.parse_args()

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.eval()

    # Standard ImageNet input: [1, 3, 224, 224]
    dummy_input = torch.randn(1, 3, 224, 224)
    output_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "mobilenet_v2.onnx")

    torch.onnx.export(
        model, dummy_input, output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
    )

    size_kb = os.path.getsize(output_path) / 1024
    print(f"Exported: {output_path} ({size_kb:.1f} KB)")
    print(f"  Input:  [1, 3, 224, 224]")
    print(f"  Output: [1, 1000] (ImageNet classes)")


if __name__ == "__main__":
    main()
