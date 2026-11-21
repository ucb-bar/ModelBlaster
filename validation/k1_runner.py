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

The same runner serves the SpaceMiT K3 (MODELBLASTER_BOARD=k3): host, remote
root, profile target and clock come from validation/board_config.py, and the
binary is launched under `taskset -c 0-7` (X100) so that any A100 worker
threads are registered by the runtime itself. Nothing here writes
/proc/set_ai_thread.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modelblaster.validation import board_config  # noqa: E402
from modelblaster.validation.runner_common import (  # noqa: E402
    IREEProfileArgs, report_pool_sweep_run, report_run,
)

DEFAULT_CLOCK_MHZ = 24.0  # rdtime, measured on the K1 (K3 is the same; board_config owns the per-board value)


def _retrying(cmd: list[str], *, flaky: bool, **kw) -> "subprocess.CompletedProcess":
    """subprocess.run, retried on ssh's connection-failure status (255) when the
    board's link is known flaky (K3). K1 keeps its single attempt, unchanged.

    A hung connection is the other failure shape on the K3 link: ssh gets past
    TCP connect and then stalls, so ConnectTimeout never fires and the
    subprocess timeout does. That is the same transport fault as 255 and is
    retried the same way (after a pause, so a board that dropped off for a
    minute has time to come back); the last attempt re-raises."""
    attempts = 4 if flaky else 1
    for i in range(attempts):
        last = i == attempts - 1
        try:
            proc = subprocess.run(cmd, **kw)
        except subprocess.TimeoutExpired:
            if last:
                raise
            print(f"k1: {cmd[0]} timed out, retry {i + 1}/{attempts - 1}", file=sys.stderr)
            time.sleep(30)
            continue
        if proc.returncode != 255 or last:
            return proc
        print(f"k1: {cmd[0]} exit 255, retry {i + 1}/{attempts - 1}", file=sys.stderr)
        time.sleep(30)
    return proc  # unreachable


def deploy_and_run(host: str, elf: str, remote_root: str, *,
                   cpu: str | None, timeout: float,
                   env: dict[str, str] | None = None,
                   launch_cpus: str | None = None) -> str:
    """scp the harness to the board, run it, return its stdout."""
    name = os.path.basename(elf)
    remote_bin = f"{remote_root}/bin/{name}"
    flaky = board_config.resolve().name != "k1"
    # K3's ssh drops connections; give the handshake room (K1: ssh defaults).
    opts = ["-o", "ConnectTimeout=60"] if flaky else []
    # MODELBLASTER_SSH_CONFIG: an ssh config file (passed as -F to ssh AND scp)
    # for boards reached some way the user's ~/.ssh/config does not describe,
    # e.g. a port-forwarded tunnel. Unset = ssh's normal config lookup.
    ssh_cfg = os.environ.get("MODELBLASTER_SSH_CONFIG")
    if ssh_cfg:
        opts = ["-F", ssh_cfg, *opts]
    _retrying(["ssh", *opts, host, f"mkdir -p {shlex.quote(remote_root)}/bin"],
              flaky=flaky, timeout=120).check_returncode()
    _retrying(["scp", *opts, "-q", elf, f"{host}:{remote_bin}"],
              flaky=flaky, timeout=timeout).check_returncode()
    prefix = ""
    if cpu:
        prefix += f"MODELBLASTER_CPU={shlex.quote(cpu)} "
    for k, v in (env or {}).items():
        prefix += f"{k}={shlex.quote(v)} "
    # launch_cpus (K3: "0-7"): the process STARTS on those CPUs. taskset sits
    # after the env assignments so MODELBLASTER_CPU still reaches the binary.
    launcher = f"taskset -c {shlex.quote(launch_cpus)} " if launch_cpus else ""
    cmd = (f"chmod +x {shlex.quote(remote_bin)} && "
           f"{prefix}{launcher}{shlex.quote(remote_bin)}")
    # A retry here would re-RUN the harness, so only the setup steps above
    # retry; the run itself is a single attempt on every board.
    proc = subprocess.run(["ssh", *opts, host, cmd], capture_output=True,
                          text=True, timeout=timeout)
    if proc.returncode != 0:
        # Print rather than raise: a harness that ran and then failed its own
        # verify still produced parseable markers worth reporting.
        print(f"k1: harness exited {proc.returncode}", file=sys.stderr)
        if proc.stderr:
            print(proc.stderr[-2000:], file=sys.stderr)
    return proc.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    cfg = board_config.resolve()
    ap.add_argument("--elf", required=True, help="harness binary to deploy")
    ap.add_argument("--host", default=cfg.host)
    ap.add_argument("--remote-root",
                    default=cfg.remote_root)
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
    ap.add_argument("--profile-source", default=cfg.name)
    ap.add_argument("--profile-cpu", default=cfg.profile_cpu)
    ap.add_argument("--profile-hw-name", default=os.environ.get("PROFILE_HW_NAME"),
                    help="the <hw> level of the profile tree (default: the "
                         "backend name). Lets X100 and A100 runs of one backend "
                         "land in separate trees, e.g. rvv_k3_x100 / "
                         "rvv_k3_a100.")
    ap.add_argument("--launch-cpus", default=cfg.launch_cpus or None,
                    help="taskset mask the binary is started under "
                         "(K3: 0-7 = X100)")
    ap.add_argument("--profile-backend", default=None)
    ap.add_argument("--profile-cores", default=None,
                    help="topology label for the profile tree; defaults to --cpu")
    ap.add_argument("--profile-clock-mhz", type=float, default=cfg.clock_mhz,
                    help="rdtime frequency, NOT the core clock (default 24 MHz)")
    ap.add_argument("--gen-dir", default=None,
                    help="build dir holding kernel_picks.json. Records WHAT "
                         "ran per op in the profile's `implementation` "
                         "column; without it that column is empty and the "
                         "profile cannot say whether a dispatch was actually "
                         "vectorised.")
    ap.add_argument("--pool-sizes", default=None)
    ap.add_argument("--save-output", default=None)
    ap.add_argument("--model-name", default=None,
                    help="model name for the profile tree. Without it the "
                         "name is guessed by walking up from io.npz, which "
                         "mislabels any layout other than "
                         "examples/<model>/<quant>/generated/.")
    ap.add_argument("--iters", type=int, default=None,
                    help="run the model this many times on the board and take "
                         "the MEDIAN per-dispatch cycles, dropping iteration 0 "
                         "as warmup (first-touch faulting of the const weight "
                         "arrays lands entirely in it). Default: one run, one "
                         "sample. See the note at the call site about what this "
                         "does to the verify marker for a stateful model.")
    args = ap.parse_args()

    if not args.models and not args.io:
        ap.error("must pass either --io (single-model) or --models (multi)")
    if args.pool_sizes and not args.models:
        ap.error("--pool-sizes requires --models")

    print(f"k1: host={args.host} elf={args.elf} cpu={args.cpu or 'unpinned'}")
    # --iters > 1 makes the harness run the model N times and print one profile
    # block per iteration; runner_common.parse_profile_reps then takes the
    # median and drops iteration 0 as warmup. deploy_and_run already accepted an
    # env dict, but nothing on the single-model path filled it, so the
    # repetitions could not be asked for from here at all.
    #
    # For a STATEFUL model this changes what the verify marker means: the golden
    # describes iteration 0, and h_state/c_state persist, so the last
    # iteration's output legitimately differs and max_abs_err will be nonzero.
    # Verify with --iters 1 and time with --iters N; do not read the two off one
    # run.
    board_env = {}
    if args.iters and args.iters > 1:
        board_env["MODELBLASTER_ITERS"] = str(args.iters)
    # Opt-in placement audit of multi-core runs (harness_linux
    # MODELBLASTER_POOL_TRACE): forwarded only when set, so every other run is
    # unchanged.
    if os.environ.get("MODELBLASTER_POOL_TRACE"):
        board_env["MODELBLASTER_POOL_TRACE"] = os.environ["MODELBLASTER_POOL_TRACE"]
    out = deploy_and_run(args.host, args.elf, args.remote_root,
                         cpu=args.cpu, timeout=args.timeout,
                         env=board_env or None, launch_cpus=args.launch_cpus)
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
        gen_dir=args.gen_dir,
    )
    # --profile-cores is the (possibly width-relative) topology label; --cpu is
    # where the process really ran. Keep the real list next to the build so a
    # topo_0_1_2_3 profile can still be traced to harts 8-11.
    if args.gen_dir and cfg.name != "k1":  # K1 output stays as it was
        try:
            with open(os.path.join(args.gen_dir, "run_meta.json"), "w") as f:
                json.dump({"board": cfg.name, "host": args.host,
                           "pinned_cpus": args.cpu,
                           "launch_cpus": args.launch_cpus,
                           "topo_label": iree_args.profile_cores,
                           "profile_hw_name": args.profile_hw_name,
                           "profile_cpu": args.profile_cpu}, f, indent=1)
        except OSError as e:
            print(f"k1: could not write run_meta.json: {e}", file=sys.stderr)
    models_list = args.models.split(",") if args.models else None
    # The <hw> tree level IS the backend tag in profile_writer.output_path, so
    # an hw-name override is applied by substituting it here.
    backend_tag = args.profile_hw_name or args.profile_backend
    if args.pool_sizes:
        ok = report_pool_sweep_run(
            out, models=models_list,
            pool_sizes=[int(p) for p in args.pool_sizes.split(",") if p.strip()],
            quant=args.quant, atol=args.atol, rtol=args.rtol,
            iree_args=iree_args, backend_tag=backend_tag,
            repo_root=repo_root)
    else:
        ok = report_run(
            out, models=models_list, io_path=args.io, quant=args.quant,
            atol=args.atol, rtol=args.rtol, profile_csv=args.profile_csv,
            iree_args=iree_args, backend_tag=backend_tag,
            repo_root=repo_root, model_name=args.model_name)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
