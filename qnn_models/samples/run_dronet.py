#!/usr/bin/env python3
"""Run DroNet inference on QRB5165 via ONNX Runtime.

Usage (on board):
    python run_dronet.py [--model PATH] [--iters N]
"""

import argparse
import time

import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser(description="Run DroNet on QRB5165")
    parser.add_argument("--model", type=str, default="/root/models/dronet/dronet.onnx",
                        help="Path to ONNX model")
    parser.add_argument("--iters", type=int, default=100,
                        help="Number of benchmark iterations")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    for inp in sess.get_inputs():
        print(f"  Input:  {inp.name}  shape={inp.shape}  dtype={inp.type}")
    for out in sess.get_outputs():
        print(f"  Output: {out.name}  shape={out.shape}  dtype={out.type}")

    # Build dummy input matching the model's expected shape
    input_meta = sess.get_inputs()[0]
    input_shape = input_meta.shape
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    print("\nWarmup (5 runs)...")
    for _ in range(5):
        sess.run(None, {input_meta.name: dummy_input})

    # Benchmark
    print(f"Benchmarking ({args.iters} runs)...")
    times = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        steer, collision = sess.run(None, {input_meta.name: dummy_input})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times = np.array(times)
    print(f"\nResults:")
    print(f"  Mean:   {times.mean():.3f} ms")
    print(f"  Median: {np.median(times):.3f} ms")
    print(f"  Std:    {times.std():.3f} ms")
    print(f"  Min:    {times.min():.3f} ms")
    print(f"  Max:    {times.max():.3f} ms")
    print(f"  FPS:    {1000.0 / times.mean():.1f}")

    print(f"\nSample output:")
    print(f"  Steering angle:        {steer[0][0]:.6f}")
    print(f"  Collision probability: {collision[0][0]:.6f}")


if __name__ == "__main__":
    main()
