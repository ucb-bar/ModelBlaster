"""Run a ModelBlaster harness on a physical SpaceMiT K1 over SSH.

Mirrors spike_runner's CLI so the two are interchangeable from run.sh, and
delegates every bit of parsing, comparison and profile emission to
runner_common. Per that module's own docstring, the only thing a runner adds is
*how* it gets the harness's stdout -- here that is scp + ssh instead of a
simulator subprocess.

Two things are genuinely different on real hardware and are handled here rather
than left to the caller:

**Core pinning is not optional.** Profiles are per-core, and on the K1 cores 0-3
carry the IME extension while 4-7 do not (measured: `smt.vmadot` SIGILLs on
cluster 1). An unpinned run averages over whatever cores happened to be idle,
and can execute a cluster-0 kernel on cluster 1. `--cpu` sets
MODELBLASTER_CPU, which the Linux harness applies with sched_setaffinity.

**Cycle counts are rdtime ticks, not core cycles.** Reading the cycle CSR from
userspace raises SIGILL on this kernel, so the Linux harness uses rdtime -- a
fixed 24 MHz on this board. `--profile-clock-mhz` therefore defaults to 24, not
to the 1.6 GHz core clock. Getting this wrong silently scales every profile by
67x, which is why it is the default rather than something the caller must
remember.

No credentials appear here. The host comes from --host / MODELBLASTER_K1_HOST
and authentication is whatever ssh is already configured to do.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modelblaster.validation.runner_common import (  # noqa: E402
    IREEProfileArgs, report_pool_sweep_run, report_run,
)

DEFAULT_CLOCK_MHZ = 24.0  # rdtime, measured on the K1


def deploy_and_run(host: str, elf: str, remote_root: str, *,
                   cpu: str | None, timeout: float,
                   env: dict[str, str] | None = None) -> str:
    """scp the harness to the board, run it, return its stdout."""
    name = os.path.basename(elf)
    remote_bin = f"{remote_root}/bin/{name}"
    subprocess.run(["ssh", host, f"mkdir -p {shlex.quote(remote_root)}/bin"],
                   check=True, timeout=120)
    subprocess.run(["scp", "-q", elf, f"{host}:{remote_bin}"],
                   check=True, timeout=timeout)
    prefix = ""
    if cpu:
        prefix += f"MODELBLASTER_CPU={shlex.quote(cpu)} "
    for k, v in (env or {}).items():
        prefix += f"{k}={shlex.quote(v)} "
    cmd = f"chmod +x {shlex.quote(remote_bin)} && {prefix}{shlex.quote(remote_bin)}"
    proc = subprocess.run(["ssh", host, cmd], capture_output=True, text=True,
                          timeout=timeout)
    if proc.returncode != 0:
        # Print rather than raise: a harness that ran and then failed its own
        # verify still produced parseable markers worth reporting.
        print(f"k1: harness exited {proc.returncode}", file=sys.stderr)
        if proc.stderr:
            print(proc.stderr[-2000:], file=sys.stderr)
    return proc.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--elf", required=True, help="harness binary to deploy")
    ap.add_argument("--host", default=os.environ.get("MODELBLASTER_K1_HOST", "k1"))
    ap.add_argument("--remote-root",
                    default=os.environ.get("MODELBLASTER_K1_REMOTE_ROOT",
                                           "/root/mb_k1"))
    ap.add_argument("--cpu", default=os.environ.get("MODELBLASTER_K1_CPU"),
                    help="physical core to pin to (0-3 = cluster 0 with IME, "
                         "4-7 = cluster 1 without)")
    ap.add_argument("--io", default=None)
    ap.add_argument("--models", default=None)
    ap.add_argument("--quant", default="fp32")
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--atol", type=float, default=None)
    ap.add_argument("--rtol", type=float, default=None)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--profile-csv", default=None)
    ap.add_argument("--profile-out-root", default=None)
    ap.add_argument("--profile-source", default="k1")
    ap.add_argument("--profile-cpu", default="spacemit_x60")
    ap.add_argument("--profile-backend", default=None)
    ap.add_argument("--profile-cores", default=None,
                    help="topology label for the profile tree; defaults to --cpu")
    ap.add_argument("--profile-clock-mhz", type=float, default=DEFAULT_CLOCK_MHZ,
                    help="rdtime frequency, NOT the core clock (default 24 MHz)")
    ap.add_argument("--pool-sizes", default=None)
    ap.add_argument("--save-output", default=None)
    args = ap.parse_args()

    if not args.models and not args.io:
        ap.error("must pass either --io (single-model) or --models (multi)")
    if args.pool_sizes and not args.models:
        ap.error("--pool-sizes requires --models")

    print(f"k1: host={args.host} elf={args.elf} cpu={args.cpu or 'unpinned'}")
    out = deploy_and_run(args.host, args.elf, args.remote_root,
                         cpu=args.cpu, timeout=args.timeout)
    if args.save_output:
        with open(args.save_output, "w") as f:
            f.write(out)
        print(f"k1: saved {len(out)} bytes of stdout to {args.save_output}")

    # Re-emit with markers intact so benchmarks/runners/k1.py can parse the
    # captured run.sh stdout, exactly as the spike runner does.
    print("=== MODELBLASTER_RAW_K1_BEGIN ===")
    print(out, end="" if out.endswith("\n") else "\n")
    print("=== MODELBLASTER_RAW_K1_END ===")

    repo_root = args.repo_root or os.path.abspath(
        os.path.join(os.path.dirname(__file__), ".."))
    iree_args = IREEProfileArgs(
        profile_out_root=args.profile_out_root,
        profile_source=args.profile_source,
        profile_cpu=args.profile_cpu,
        profile_cores=(args.profile_cores if args.profile_cores is not None
                       else (args.cpu or "0")),
        profile_clock_mhz=args.profile_clock_mhz,
        quant=args.quant,
    )
    models_list = args.models.split(",") if args.models else None
    if args.pool_sizes:
        ok = report_pool_sweep_run(
            out, models=models_list,
            pool_sizes=[int(p) for p in args.pool_sizes.split(",") if p.strip()],
            quant=args.quant, atol=args.atol, rtol=args.rtol,
            iree_args=iree_args, backend_tag=args.profile_backend,
            repo_root=repo_root)
    else:
        ok = report_run(
            out, models=models_list, io_path=args.io, quant=args.quant,
            atol=args.atol, rtol=args.rtol, profile_csv=args.profile_csv,
            iree_args=iree_args, backend_tag=args.profile_backend,
            repo_root=repo_root)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
