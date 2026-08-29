"""Run a built native_sim zephyr.exe on the host and compare to PyTorch.

The `native` RUNNER builds the harness for Zephyr's native_sim board (a plain
x86-64 host executable). This runs at native speed with host-backed memory, so
it can validate the reference kernels at the FULL stock KernelBench dimensions
— which overflow spike's 256 MB Zephyr RAM region.

The harness prints the identical MODELBLASTER_OUTPUT / _VERIFY / _PROFILE /
_WALL_CYCLES markers to stdout regardless of board, so parsing/compare reuses
`runner_common.report_run` exactly as spike_runner does — only the "run the
binary" step differs (exec the .exe instead of spawning spike).
"""

from __future__ import annotations

import argparse
import os
import subprocess

from modelblaster.validation.runner_common import IREEProfileArgs, report_run


def run_native(exe: str, timeout: float = 600.0) -> str:
    """Execute the native_sim binary, returning combined stdout+stderr.

    native_sim apps that fall into an idle loop after printing never exit on
    their own; a timeout is expected and non-fatal — the VERIFY markers are
    emitted before any such loop, so whatever we captured is sufficient.
    """
    if not os.path.exists(exe):
        raise FileNotFoundError(f"native executable {exe} not found")
    try:
        # The harness computes, prints its markers, then exit(0)s (see main.c's
        # CONFIG_ARCH_POSIX path), so no -stop_at is needed; the timeout only
        # guards against an unexpected hang.
        proc = subprocess.run([exe], capture_output=True, text=True,
                              timeout=timeout)
        return proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        err = exc.stderr or ""
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        if isinstance(err, bytes):
            err = err.decode(errors="replace")
        return out + err


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--elf", required=True,
                    help="path to the native_sim zephyr.exe")
    ap.add_argument("--io", default=None, help="single-model io.npz golden")
    ap.add_argument("--quant", default="fp32")
    ap.add_argument("--atol", type=float, default=None)
    ap.add_argument("--rtol", type=float, default=None)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--repo-root", default=None)
    args = ap.parse_args()

    out = run_native(args.elf, timeout=args.timeout)

    print("=== MODELBLASTER_RAW_NATIVE_BEGIN ===")
    print(out, end="" if out.endswith("\n") else "\n")
    print("=== MODELBLASTER_RAW_NATIVE_END ===")

    repo_root = args.repo_root or os.path.abspath(
        os.path.join(os.path.dirname(__file__), ".."))
    ok = report_run(
        out,
        models=None,
        io_path=args.io,
        quant=args.quant,
        atol=args.atol, rtol=args.rtol,
        profile_csv=None,
        iree_args=IREEProfileArgs(
            profile_out_root=None, profile_source="native",
            profile_cpu=None, profile_cores=0, profile_clock_mhz=1000.0,
            quant=args.quant),
        backend_tag="native",
        repo_root=repo_root,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
