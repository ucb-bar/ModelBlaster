#!/usr/bin/env python3
import os
import sys
import json
import argparse


def _str2bool(v):
    """Argparse-friendly bool parser that accepts true/false strings."""
    if isinstance(v, bool):
        return v
    if v is None:
        return True
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {v!r}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_dir", type=str, required=True)
    parser.add_argument("--cuda_code_path", type=str, required=True)
    parser.add_argument("--op_atol", type=float, default=1e-3)
    parser.add_argument("--op_rtol", type=float, default=1e-3)
    # Keep defaults small; level-wide runners call this many times.
    parser.add_argument("--rep_time", type=int, default=10)
    parser.add_argument("--warmup_time", type=int, default=1)
    parser.add_argument("--eval_type", type=str, default="kernelbench")
    # NOTE: argparse `type=bool` is a footgun; this supports e.g. "--multi_init_settings false"
    parser.add_argument("--multi_init_settings", type=_str2bool, default=False)
    parser.add_argument("--multi_input_settings", type=_str2bool, default=False)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument(
        "--ext_dir",
        type=str,
        default=os.environ.get("TORCH_EXTENSIONS_DIR", os.path.expanduser("~/.cache/torch_extensions")),
        help="Torch extension cache directory (can also set TORCH_EXTENSIONS_DIR).",
    )
    # Keep prior behavior (profiling enabled) but allow turning it off.
    parser.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable profiling (torch_prof/ncu_prof/clang_tidy) after correctness+timing.",
    )
    parser.add_argument("--torch_prof", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ncu_prof", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clang_tidy", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    return args


def main() -> int:
    args = parse_args()

    try:
        from robust_kbench.primitives.evaluate import (
            eval_torch_runtime,
            correct_cuda_kernel,
            eval_cuda_kernel,
            prof_cuda_kernel,
        )
    except ModuleNotFoundError as e:
        print("ERROR: Could not import `robust_kbench`.", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        return 2

    torch_results, torch_compile_results = eval_torch_runtime(
        task_dir=args.task_dir,
        multi_init_settings=args.multi_init_settings,
        multi_input_settings=args.multi_input_settings,
        warmup_time=args.warmup_time,
        repetition_time=args.rep_time,
        eval_type=args.eval_type,
        timeout=args.timeout,
        gpu_id=0,
        ext_dir=args.ext_dir,
        forward=not args.backward,
        debug=False,
    )

    correct_results = correct_cuda_kernel(
        task_dir=args.task_dir,
        cuda_code_path=args.cuda_code_path,
        op_atol=args.op_atol,
        op_rtol=args.op_rtol,
        multi_init_settings=args.multi_init_settings,
        multi_input_settings=args.multi_input_settings,
        ext_dir=args.ext_dir,
        forward=not args.backward,
        timeout=args.timeout,
    )
    print(f"==> Correctness results: {args.task_dir}")
    print(f"    --- CUDA file: {args.cuda_code_path}")
    print("    --- CUDA correctness test data")
    print(json.dumps(correct_results, indent=4))

    if correct_results.get("summary", {}).get("correct", False):
        cuda_results = eval_cuda_kernel(
            task_dir=args.task_dir,
            cuda_code_path=args.cuda_code_path,
            warmup_time=args.warmup_time,
            repetition_time=args.rep_time,
            eval_type=args.eval_type,
            multi_init_settings=args.multi_init_settings,
            multi_input_settings=args.multi_input_settings,
            timeout=args.timeout,
            forward=not args.backward,
            ext_dir=args.ext_dir,   # the old script omitted this: eval built in ~/.cache/torch_extensions (home quota)
        )
        print(f"==> Evaluation results: {args.task_dir}")
        print(f"    --- CUDA file: {args.cuda_code_path}")
        print("    --- CUDA correctness test data")
        print(json.dumps(cuda_results, indent=4))

        speedup_native = torch_results["summary"]["avg_mean_time"] / cuda_results["summary"]["avg_mean_time"]
        speedup_compile = torch_compile_results["summary"]["avg_mean_time"] / cuda_results["summary"]["avg_mean_time"]
        print(f"==> Torch native avg_mean_time (ms): {torch_results['summary']['avg_mean_time']}")
        print(f"==> Torch compile avg_mean_time (ms): {torch_compile_results['summary']['avg_mean_time']}")
        print(f"==> CUDA avg_mean_time (ms): {cuda_results['summary']['avg_mean_time']}")
        print(f"==> Speedup CUDA over Torch native: {speedup_native}x")
        print(f"==> Speedup CUDA over Torch compile: {speedup_compile}x")

        if args.profile:
            prof_results = prof_cuda_kernel(
                cuda_code_path=args.cuda_code_path,
                task_dir=args.task_dir,
                torch_prof=args.torch_prof,
                ncu_prof=args.ncu_prof,
                clang_tidy=args.clang_tidy,
            )
            print(f"==> Profiling results: {args.task_dir}")
            print(f"    --- CUDA file: {args.cuda_code_path}")
            print("    --- CUDA profiling test data")
            print(json.dumps(prof_results, indent=4))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
