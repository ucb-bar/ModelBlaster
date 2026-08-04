#!/usr/bin/env python3
"""Export YOLOv8s to ONNX format.

Usage:
    python export_yolo.py [--output PATH] [--model yolov8s]
"""

import argparse
import os

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Export YOLO to ONNX")
    parser.add_argument("--model", type=str, default="yolov8s",
                        help="YOLO model name (default: yolov8s)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output ONNX path")
    args = parser.parse_args()

    model = YOLO(f"{args.model}.pt")

    output_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), f"{args.model}.onnx")

    model.export(format="onnx", opset=18, imgsz=640, simplify=True)

    # ultralytics saves next to the .pt file; move if needed
    default_out = f"{args.model}.onnx"
    if os.path.exists(default_out) and default_out != output_path:
        os.rename(default_out, output_path)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nExported: {output_path} ({size_mb:.1f} MB)")
    print(f"  Input:  [1, 3, 640, 640]")


if __name__ == "__main__":
    main()
