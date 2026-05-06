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
    report_pool_sweep_run,
    report_run,
    _VERIFY_RE,           # in-binary verify summary (modern harness)
    _WALL_RE,             # AGENTS_WALL_CYCLES — last line of every block
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
    timed_out = False
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        out = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        # exc.stdout/stderr hold whatever spike printed before it was killed.
        # With text=True they should be str, but CPython's second communicate()
        # call after kill() can return bytes on some paths — decode defensively.
        def _as_str(x: object) -> str:
            if x is None:
                return ''
            if isinstance(x, bytes):
                return x.decode('utf-8', errors='replace')
            return str(x)
        out = _as_str(exc.stdout) + _as_str(exc.stderr)
        print(
            f"WARNING: spike timed out after {timeout:.0f}s — "
            f"using {len(out)} chars of partial output",
            file=sys.stderr,
        )
    # The harness reached its end-of-bench point if it emitted an
    # AGENTS_WALL_CYCLES line (always last), an AGENTS_OUTPUT_END
    # (legacy per-element dump), or an AGENTS_VERIFY line (modern
    # in-binary compare path). Any of those means the run completed
    # cleanly enough to parse.
    has_modern = bool(_VERIFY_RE.search(out))
    has_wall = bool(_WALL_RE.search(out))
    if not timed_out and not (has_output_marker(out) or has_modern or has_wall):
        raise RuntimeError(
            f"spike output missing AGENTS_VERIFY / AGENTS_WALL_CYCLES "
            f"/ AGENTS_OUTPUT_BEGIN markers. cmd={cmd!r}\n"
            f"--- output ---\n{out}"
        )
    if timed_out and not (has_output_marker(out) or has_modern or has_wall):
        raise RuntimeError(
            f"spike timed out after {timeout:.0f}s with no parseable output. "
            f"cmd={cmd!r}"
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
    ap.add_argument("--pool-sizes", default=None,
                    help="comma-list of pool sizes the harness was built "
                         "with (multi-model pool-sweep). Switches the "
                         "runner to walk [<model>@p<N>] tags and emit "
                         "per-(model, pool) profiles under topo_<cores>.")
    ap.add_argument("--save-output", default=None,
                    help="path to write the full captured spike stdout. "
                         "Useful for harvesting AGENTS_XPURT_TRACE blocks "
                         "(built with -DAGENTS_XPURT_TRACE=ON) for the "
                         "trace plotter — spike output is otherwise "
                         "hidden behind subprocess.run(capture_output).")
    args = ap.parse_args()

    if not args.models and not args.io:
        ap.error("must pass either --io (single-model) or --models (multi)")
    if args.pool_sizes and not args.models:
        ap.error("--pool-sizes requires --models")

    spike = find_spike(args.spike)
    print(f"spike: {spike}")
    out = run_spike(spike, args.elf, timeout=args.timeout,
                    extra_args=tuple(args.spike_arg))
    if args.save_output:
        with open(args.save_output, "w") as f:
            f.write(out)
        print(f"spike: saved {len(out)} bytes of stdout to {args.save_output}")

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
    if args.pool_sizes:
        pool_sizes = [int(p) for p in args.pool_sizes.split(",") if p.strip()]
        ok = report_pool_sweep_run(
            out,
            models=models_list,
            pool_sizes=pool_sizes,
            quant=args.quant,
            atol=args.atol, rtol=args.rtol,
            iree_args=iree_args,
            backend_tag=backend_tag,
            repo_root=repo_root,
        )
    else:
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
