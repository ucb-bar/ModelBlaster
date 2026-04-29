"""Run a built zephyr.elf on spike and compare its output to PyTorch.

Single-model mode (single-model harness): spike stdout has one block
between
    === AGENTS_OUTPUT_BEGIN ===
    <one float per line>
    === AGENTS_OUTPUT_END ===
and the runner compares against the golden in --io.

Multi-model mode (--models name1,name2,...): spike stdout has N blocks
tagged by model name; the runner compares each block against
    agents/examples/<name>/<quant>/generated/io.npz

Parsing/compare/IREE-emit logic lives in agents.validation.runner_common
so the firesim runner reuses it.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import Optional

from agents.validation.runner_common import (
    BEGIN,                # re-exported for back-compat (profile_kernel.py)
    END,
    IREEProfileArgs,
    has_output_marker,
    parse_output,         # re-exported
    parse_profile,        # re-exported
    report_run,
)

__all__ = [
    "BEGIN", "END",
    "find_spike", "run_spike",
    "parse_output", "parse_profile",
    "main",
]


DEFAULT_SPIKE_FALLBACK = "/scratch2/dima/misc_sw/spike"


def find_spike(explicit: Optional[str] = None) -> str:
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(f"--spike {explicit} not found")
        return explicit
    p = shutil.which("spike")
    if p:
        return p
    if os.path.exists(DEFAULT_SPIKE_FALLBACK):
        return DEFAULT_SPIKE_FALLBACK
    raise FileNotFoundError(
        "spike not on PATH and fallback path missing — pass --spike <path>"
    )


def run_spike(spike: str, elf: str, timeout: float = 60.0,
              extra_args: tuple[str, ...] = ()) -> str:
    cmd = [spike, *extra_args, elf]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout
    )
    out = proc.stdout + proc.stderr
    if not has_output_marker(out):
        raise RuntimeError(
            f"spike output missing AGENTS_OUTPUT_BEGIN marker. cmd={cmd!r}\n"
            f"--- output ---\n{out}"
        )
    return out


def _detect_backend(spike_args: list[str], default: str = "scalar") -> str:
    """Infer the HW-backend tag from spike's --isa=... if present.
    rv64gcv* → rvv; otherwise scalar."""
    for a in spike_args:
        if a.startswith("--isa="):
            iso = a.split("=", 1)[1].lower()
            if "v" in iso.replace("rv64", "")[:6]:
                return "rvv"
    return default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--elf", required=True)
    ap.add_argument("--io", default=None,
                    help="io.npz path (single-model mode)")
    ap.add_argument("--models", default=None,
                    help="comma-separated model names for multi-model mode")
    ap.add_argument("--quant", default="fp32")
    ap.add_argument("--repo-root", default=None,
                    help="repo root for resolving per-model io.npz paths")
    ap.add_argument("--spike", default=None)
    ap.add_argument("--atol", type=float, default=None)
    ap.add_argument("--rtol", type=float, default=None)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--spike-arg", action="append", default=[],
                    help="extra arg passed to spike (repeatable)")
    ap.add_argument("--profile-csv", default=None)
    ap.add_argument("--profile-out-root", default=None,
                    help="root of the IREE-shape per-dispatch profile tree")
    ap.add_argument("--profile-source", default="spike")
    ap.add_argument("--profile-cpu", default=None)
    ap.add_argument("--profile-backend", default=None,
                    help="HW backend label (scalar/rvv); auto-detected "
                         "from --spike-arg=--isa=... if omitted")
    ap.add_argument("--profile-cores", default="0,1,2,3")
    ap.add_argument("--profile-clock-mhz", type=float, default=1000.0)
    args = ap.parse_args()

    if not args.models and not args.io:
        ap.error("must pass either --io (single-model) or --models (multi)")

    spike = find_spike(args.spike)
    print(f"spike: {spike}")
    out = run_spike(spike, args.elf, timeout=args.timeout,
                    extra_args=tuple(args.spike_arg))

    backend_tag = args.profile_backend or _detect_backend(args.spike_arg)
    repo_root = args.repo_root or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    iree_args = IREEProfileArgs(
        profile_out_root=args.profile_out_root,
        profile_source=args.profile_source,
        profile_cpu=args.profile_cpu,
        profile_cores=args.profile_cores,
        profile_clock_mhz=args.profile_clock_mhz,
        quant=args.quant,
    )
    models_list = (args.models.split(",") if args.models else None)
    ok = report_run(
        out,
        models=models_list,
        io_path=args.io,
        quant=args.quant,
        atol=args.atol, rtol=args.rtol,
        profile_csv=args.profile_csv,
        iree_args=iree_args,
        backend_tag=backend_tag,
        repo_root=repo_root,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
