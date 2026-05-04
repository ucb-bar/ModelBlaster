"""Run a built zephyr.elf on FireSim and parse the harness's stdout.

Mirrors agents.validation.spike_runner: same CLI shape, same OUTPUT/PROFILE/
WALL_CYCLES parsing (via runner_common), same IREE-shape profile
emission. The only thing that differs is *how* we get the harness's
stdout — instead of running spike in-process, we stage the elf into the
FireSim sim-slot, run `firesim runworkload`, and tail the uartlog until
the OUTPUT_END marker(s) we expect arrive (or a timeout fires).

Pre-conditions (one-time per session, same as the manual flow):
    cd /scratch2/dima/chipyard-fsim/sims/firesim
    source /scratch2/dima/chipyard-fsim/env.sh
    source ./sourceme-manager.sh --skip-ssh-setup
    firesim infrasetup       # only when the FPGA bitstream is stale

Then per run:
    python -m agents.validation.firesim_runner \\
        --elf <path-to-zephyr.elf> \\
        --io  <path-to-io.npz> \\
        --profile-out-root gen/profile --profile-source firesim \\
        --profile-cpu firesim_rocket_saturn --profile-cores 0,1,2,3 \\
        --profile-clock-mhz 1000.0
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Optional

from agents.validation.runner_common import (
    IREEProfileArgs,
    has_output_marker,
    output_block_count,
    report_pool_sweep_run,
    report_run,
    wall_cycles_count,
)


# Defaults match the user's fixed FireSim install layout. All overridable
# via env vars / CLI flags so the runner stays portable.
DEFAULT_FIRESIM_ROOT = os.environ.get(
    "FIRESIM_ROOT", "/scratch2/dima/chipyard-fsim/sims/firesim")
DEFAULT_FIRESIM_ENV = os.environ.get(
    "FIRESIM_ENV", "/scratch2/dima/chipyard-fsim/env.sh")
DEFAULT_FIRESIM_SLOT = os.environ.get(
    "FIRESIM_SLOT", "firesim_rundir/sim_slot_0")
DEFAULT_FIRESIM_BINARY_BASENAME = "zephyr0-zephyr.elf"


def _firesim_paths(root: str, slot: str) -> dict:
    sim_slot = os.path.join(root, slot)
    return {
        "root": root,
        "sim_slot": sim_slot,
        "uartlog": os.path.join(sim_slot, "uartlog"),
        "elf_target": os.path.join(sim_slot, DEFAULT_FIRESIM_BINARY_BASENAME),
    }


def _firesim_cmd(firesim_env: str, firesim_root: str, sub_cmd: str) -> list[str]:
    """Build a `bash -c 'deactivate parent conda env; source ...; firesim'`
    invocation. firesim's env.sh activates chipyard's `.conda-env`; if
    the parent shell already has another conda env active (e.g. the
    agent's `zephyr` env), conda activate stacks rather than replaces,
    so chipyard's PYTHONPATH never wins and firesim itself can't
    `import argcomplete`. We unset the leaking CONDA_* vars and prepend
    the parent's miniforge condabin so env.sh's `type conda` and
    `conda activate <path>` reach chipyard's env on a clean state. Keep
    HOME/PATH otherwise so xdma drivers / FPGA permissions still work."""
    # Inherit parent env, then strip the conda-stack pieces in the
    # subshell prologue so `conda activate` in env.sh starts fresh.
    inner = (
        f"set -e; "
        # Drop any inherited active conda env so chipyard's activate
        # starts from a clean conda state.
        f"unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER "
        f"      CONDA_PYTHON_EXE CONDA_SHLVL CONDA_EXE _CE_M _CE_CONDA; "
        # Defensively make sure conda itself is still callable (the
        # parent env's bin is likely first; condabin contains the
        # `conda` shim that env.sh's `type conda` test relies on).
        f"export PATH=/scratch2/dima/miniforge3/condabin:$PATH; "
        f"source {firesim_env}; "
        f"cd {firesim_root}; "
        f"source ./sourceme-manager.sh --skip-ssh-setup; "
        f"firesim {sub_cmd}"
    )
    return ["bash", "-c", inner]


def _stage_elf(elf: str, paths: dict) -> None:
    """Copy the built elf over the staged binary in the sim slot.
    `firesim infrasetup` does this from deploy/workloads/zephyr/
    zephyr.elf, but a cp lets us skip the slow re-flash when only the
    binary changed."""
    if not os.path.isfile(elf):
        raise FileNotFoundError(f"--elf {elf} not found")
    target = paths["elf_target"]
    os.makedirs(paths["sim_slot"], exist_ok=True)
    shutil.copyfile(elf, target)
    os.chmod(target, 0o755)


def _truncate_uartlog(paths: dict) -> None:
    """Zero the uartlog so the streaming reader only sees output from
    THIS run."""
    p = paths["uartlog"]
    if os.path.exists(p):
        with open(p, "w") as f:
            f.truncate(0)


def _firesim_kill(firesim_env: str, firesim_root: str) -> None:
    """Best-effort tear-down. firesim kill exits 0 even when there's
    nothing to kill, so we don't check the return code."""
    subprocess.run(_firesim_cmd(firesim_env, firesim_root, "kill"),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _firesim_run_async(firesim_env: str, firesim_root: str,
                       log_path: str) -> subprocess.Popen:
    """Spawn `firesim runworkload` and return the Popen handle. We do
    NOT wait on it from this side; the run is treated as done when the
    expected OUTPUT_END markers appear in the uartlog. firesim's own
    stdout/stderr go to `log_path` so we can diagnose silent failures
    (e.g. screen session not starting). The actual UART output lives
    in firesim_rundir/sim_slot_0/uartlog regardless."""
    log_f = open(log_path, "w")
    return subprocess.Popen(
        _firesim_cmd(firesim_env, firesim_root, "runworkload"),
        stdout=log_f, stderr=subprocess.STDOUT,
    )


def _expected_end_count(models: Optional[list[str]],
                        pool_sizes: Optional[list[int]] = None) -> int:
    """How many OUTPUT_END markers do we wait for? Multi-model harness
    emits one per model; single-model harness emits a single bare END.
    Pool-sweep harness emits len(models) * len(pool_sizes) blocks."""
    if pool_sizes:
        assert models, "pool_sizes implies models"
        return len(models) * len(pool_sizes)
    return max(1, len(models)) if models else 1


def run_firesim(elf: str, *, models: Optional[list[str]] = None,
                pool_sizes: Optional[list[int]] = None,
                firesim_root: str = DEFAULT_FIRESIM_ROOT,
                firesim_env: str = DEFAULT_FIRESIM_ENV,
                firesim_slot: str = DEFAULT_FIRESIM_SLOT,
                timeout: float = 600.0,
                poll_interval: float = 1.0,
                stage_elf: bool = True,
                kill_first: bool = True,
                verbose: bool = True) -> str:
    """Run `elf` on FireSim, return the captured uartlog.

    The run is considered "done" as soon as the harness has printed all
    expected `=== AGENTS_OUTPUT_END ===` markers (one per model in
    multi-model mode). At that point we issue `firesim kill` to release
    the FPGA — we don't wait for the simulator to exit on its own
    (some Zephyr binaries spin-loop after the printout)."""
    paths = _firesim_paths(firesim_root, firesim_slot)
    if not os.path.isfile(firesim_env):
        raise FileNotFoundError(
            f"FIRESIM_ENV not found at {firesim_env}; "
            f"set FIRESIM_ENV or pass --firesim-env"
        )
    if not os.path.isdir(firesim_root):
        raise FileNotFoundError(
            f"FIRESIM_ROOT not found at {firesim_root}; "
            f"set FIRESIM_ROOT or pass --firesim-root"
        )
    if kill_first:
        if verbose:
            print(f"firesim: kill any prior sim", flush=True)
        _firesim_kill(firesim_env, firesim_root)
    if stage_elf:
        if verbose:
            print(f"firesim: stage {elf} -> {paths['elf_target']}", flush=True)
        _stage_elf(elf, paths)
    _truncate_uartlog(paths)
    expected_ends = _expected_end_count(models, pool_sizes)
    if verbose:
        print(f"firesim: runworkload (waiting for {expected_ends} "
              f"AGENTS_WALL_CYCLES marker{'s' if expected_ends>1 else ''})",
              flush=True)

    runworkload_log = os.path.join(paths["sim_slot"],
                                   "_agents_runworkload.log")
    proc = _firesim_run_async(firesim_env, firesim_root, runworkload_log)
    deadline = time.monotonic() + timeout
    last_size = 0
    last_progress = time.monotonic()
    # Fast-fail: if Zephyr's fatal-error printer fires (load fault, store
    # fault, illegal instruction, etc.) the workload will never reach the
    # OUTPUT marker. Detect that in the uartlog and short-circuit the
    # poll loop instead of waiting for the full timeout. Saves ~3 minutes
    # per LLM-generated kernel that builds clean for spike but
    # mis-addresses on the FPGA.
    _fault_markers = (
        "Load access fault",
        "Store access fault",
        "Illegal instruction",
        "Instruction access fault",
        ">>> ZEPHYR FATAL ERROR",
        "k_oops",
    )
    fault_seen_at: Optional[float] = None
    try:
        while True:
            if time.monotonic() > deadline:
                # Pull the runworkload stderr/stdout into the message so
                # the user can see *why* firesim never produced output
                # (FPGA flash stale, screen failed, conda env issue, ...)
                try:
                    with open(runworkload_log) as f:
                        rwl_tail = f.read()[-2000:]
                except FileNotFoundError:
                    rwl_tail = "(no log captured)"
                raise TimeoutError(
                    f"firesim run exceeded {timeout}s; uartlog "
                    f"({last_size} bytes) at {paths['uartlog']}.\n"
                    f"--- last 2KB of `firesim runworkload` log ---\n"
                    f"{rwl_tail}"
                )
            text = ""
            try:
                with open(paths["uartlog"]) as f:
                    text = f.read()
            except FileNotFoundError:
                pass
            if len(text) != last_size:
                last_size = len(text)
                last_progress = time.monotonic()
            # Stop on the LAST block's AGENTS_WALL_CYCLES line — that's
            # the trailing per-block sentinel (OUTPUT_END comes earlier
            # in the same block, so racing on it cut the last block's
            # PROFILE+WALL prints off when we killed the sim).
            # harness_multi emits no OUTPUT_BEGIN/END (only VERIFY+PROFILE+WALL),
            # so allow WALL_CYCLES count alone to satisfy in multi-model mode.
            wall_done = wall_cycles_count(text) >= expected_ends
            if wall_done and (has_output_marker(text) or models):
                if verbose:
                    print("firesim: all expected blocks complete "
                          f"({expected_ends} WALL_CYCLES seen)",
                          flush=True)
                break
            # Fast-fail on Zephyr fatal-error printer.
            if fault_seen_at is None:
                for marker in _fault_markers:
                    if marker in text:
                        fault_seen_at = time.monotonic()
                        if verbose:
                            print(f"firesim: detected '{marker}' in "
                                  f"uartlog — workload faulted, "
                                  f"will short-circuit after a brief "
                                  f"settle window",
                                  flush=True)
                        break
            # Once a fault is detected, give the kernel a short window
            # to finish printing the fault frame (regs, stack), then
            # raise. Don't break on first sight — we want the diagnostic
            # in the message we hand back.
            if fault_seen_at is not None and (
                time.monotonic() - fault_seen_at > 5.0
            ):
                tail = text[-2000:]
                raise RuntimeError(
                    f"firesim workload faulted (Zephyr fatal-error "
                    f"printer triggered). uartlog tail:\n{tail}"
                )
            # Surface a heartbeat if uartlog is silent for a long stretch
            # so the user sees we're alive.
            if (verbose and time.monotonic() - last_progress > 30.0
                    and last_size > 0):
                last_progress = time.monotonic()
                print(f"  ... uartlog at {last_size} bytes, still waiting",
                      flush=True)
            time.sleep(poll_interval)
    finally:
        _firesim_kill(firesim_env, firesim_root)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    with open(paths["uartlog"]) as f:
        return f.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--elf", required=True)
    ap.add_argument("--io", default=None,
                    help="io.npz path (single-model mode)")
    ap.add_argument("--models", default=None,
                    help="comma-separated model names for multi-model mode")
    ap.add_argument("--quant", default="fp32")
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--atol", type=float, default=None)
    ap.add_argument("--rtol", type=float, default=None)
    ap.add_argument("--timeout", type=float, default=600.0,
                    help="max seconds to wait for the harness's OUTPUT_END "
                         "marker(s) before tearing down (default 600)")
    ap.add_argument("--poll-interval", type=float, default=1.0,
                    help="how often to re-read the uartlog while waiting "
                         "(default 1s)")
    ap.add_argument("--firesim-root", default=DEFAULT_FIRESIM_ROOT,
                    help="path to <chipyard>/sims/firesim")
    ap.add_argument("--firesim-env", default=DEFAULT_FIRESIM_ENV,
                    help="path to <chipyard>/env.sh")
    ap.add_argument("--firesim-slot", default=DEFAULT_FIRESIM_SLOT,
                    help="rundir-relative path to the slot dir "
                         "(default firesim_rundir/sim_slot_0)")
    ap.add_argument("--no-stage-elf", action="store_true",
                    help="skip the cp into firesim's slot — assume the "
                         "binary was already staged (e.g. by a separate "
                         "infrasetup)")
    ap.add_argument("--no-kill-first", action="store_true",
                    help="don't run `firesim kill` before this run")
    ap.add_argument("--profile-csv", default=None)
    ap.add_argument("--profile-out-root", default=None)
    ap.add_argument("--profile-source", default="firesim")
    ap.add_argument("--profile-cpu", default=None,
                    help="CPU label (default: firesim_rocket_saturn — match "
                         "the alveo_u250 quad-rocket-saturn hwconfig)")
    ap.add_argument("--profile-backend", required=False, default="rvv",
                    help="HW backend label (scalar/rvv). FireSim's quad-"
                         "rocket-saturn build supports rvv natively, so "
                         "default rvv. Override for non-vector hwconfigs.")
    ap.add_argument("--profile-cores", default="0,1,2,3",
                    help="hart layout for the topo_<...> directory "
                         "(default 0,1,2,3 — quad-core hwconfig).")
    ap.add_argument("--profile-clock-mhz", type=float, default=1000.0,
                    help="clock rate used to convert per-op cycles to ns. "
                         "Default 1000.0 = 1 GHz, the typical Rocket clock.")
    ap.add_argument("--pool-sizes", default=None,
                    help="comma-list of pool sizes the harness was built "
                         "with (multi-model pool-sweep). Switches the "
                         "runner to walk [<model>@p<N>] tags and emit "
                         "per-(model, pool) profiles under topo_<cores>.")
    args = ap.parse_args()

    if not args.models and not args.io:
        ap.error("must pass either --io (single-model) or --models (multi)")
    if args.pool_sizes and not args.models:
        ap.error("--pool-sizes requires --models")
    if args.profile_cpu is None:
        args.profile_cpu = "firesim_rocket_saturn"

    pool_sizes = None
    if args.pool_sizes:
        pool_sizes = [int(p) for p in args.pool_sizes.split(",") if p.strip()]

    out = run_firesim(
        args.elf,
        models=(args.models.split(",") if args.models else None),
        pool_sizes=pool_sizes,
        firesim_root=args.firesim_root,
        firesim_env=args.firesim_env,
        firesim_slot=args.firesim_slot,
        timeout=args.timeout,
        poll_interval=args.poll_interval,
        stage_elf=not args.no_stage_elf,
        kill_first=not args.no_kill_first,
    )
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
    if pool_sizes:
        ok = report_pool_sweep_run(
            out,
            models=models_list,
            pool_sizes=pool_sizes,
            quant=args.quant,
            atol=args.atol, rtol=args.rtol,
            iree_args=iree_args,
            backend_tag=args.profile_backend,
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
            backend_tag=args.profile_backend,
            repo_root=repo_root,
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
