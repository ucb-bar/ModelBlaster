"""Run a built zephyr.elf on FireSim and parse the harness's stdout.

Mirrors modelblaster.validation.spike_runner: same CLI shape, same OUTPUT/PROFILE/
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
    python -m modelblaster.validation.firesim_runner \\
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

from modelblaster.validation.runner_common import (
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
# config_runtime.yaml's recipe_arg_overrides::default_simulation_dir
# decides where the simulator actually runs (it copies the staged binary
# into <default_simulation_dir>/sim_slot_<N>/, writes uartlog there, etc.).
# When unset / commented out, it defaults to firesim_rundir/sim_slot_<N>
# under the firesim install. On this host it's overridden to
# /scratch2/agustin/FIRESIM_RUNS_DIR. We read it dynamically rather than
# hard-coding so the integration follows whatever the user has wired up.
DEFAULT_FIRESIM_SIM_DIR_ENV = "FIRESIM_SIM_DIR"


def _resolve_sim_dir(firesim_root: str) -> str:
    """Read `default_simulation_dir` from config_runtime.yaml; fall
    back to the legacy `<firesim_root>/firesim_rundir/` when unset."""
    env_override = os.environ.get(DEFAULT_FIRESIM_SIM_DIR_ENV)
    if env_override:
        return env_override
    config_path = os.path.join(firesim_root, "deploy", "config_runtime.yaml")
    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            sim_dir = (cfg.get("run_farm", {})
                          .get("recipe_arg_overrides", {})
                          .get("default_simulation_dir"))
            if sim_dir:
                return sim_dir
        except Exception:
            pass
    return os.path.join(firesim_root, "firesim_rundir")
# FireSim's runworkload stages the workload's `common_bootbinary` into
# the sim_slot prefixed with `<workload_name><N>-`, where N is the
# node index (0 for single-node sims). The TSI bridge in the
# simulator binary then looks for the prefixed name in its cwd
# (sim_slot dir). Our staged workload is `modelblaster-firesim.json`
# with `common_bootbinary: zephyr0-zephyr.elf`; the runtime expects
# `modelblaster-firesim0-zephyr0-zephyr.elf` to land in sim_slot_0.
DEFAULT_FIRESIM_WORKLOAD_NAME = os.environ.get(
    "FIRESIM_WORKLOAD_NAME", "modelblaster-firesim")
DEFAULT_FIRESIM_BOOTBINARY = "zephyr0-zephyr.elf"
DEFAULT_FIRESIM_BINARY_BASENAME = (
    f"{DEFAULT_FIRESIM_WORKLOAD_NAME}0-{DEFAULT_FIRESIM_BOOTBINARY}")


def _firesim_paths(root: str, slot: str) -> dict:
    """Return all paths the runner cares about.

    `root` is the firesim install root (e.g.
    `/scratch2/agustin/chipyard/sims/firesim`). `slot` is the trailing
    sim_slot path relative to the actual simulation dir
    (which comes from config_runtime.yaml::default_simulation_dir).

    Three distinct file locations the runner touches:
      - `workload_bootbinary`: deploy/workloads/<workload>/<bootbinary>.
        `firesim infrasetup` rsyncs from here. WE WRITE OUR ELF HERE.
      - `sim_slot`: <default_simulation_dir>/sim_slot_<N>/. The simulator
        runs here, writes uartlog here, and infrasetup stages a copy of
        the bootbinary here (prefixed with <workload><N>-).
      - `legacy_sim_slot`: firesim_rundir/sim_slot_<N>/. Old default,
        kept for compatibility with workflows that don't override
        default_simulation_dir.
    """
    # Strip the leading firesim_rundir/ if the slot was set against the
    # legacy default; the real simulation dir provides the prefix.
    sub_slot = slot
    if sub_slot.startswith("firesim_rundir/"):
        sub_slot = sub_slot[len("firesim_rundir/"):]
    real_sim_dir = _resolve_sim_dir(root)
    sim_slot = os.path.join(real_sim_dir, sub_slot)
    # Also expose the legacy path so logs / older overrides still resolve.
    legacy_sim_slot = os.path.join(root, "firesim_rundir", sub_slot)
    return {
        "root": root,
        "sim_slot": sim_slot,
        "legacy_sim_slot": legacy_sim_slot,
        # The live uartlog the simulator writes to during runworkload.
        "uartlog": os.path.join(sim_slot, "uartlog"),
        # Where the simulator's TSI loader looks at runtime.
        "elf_target": os.path.join(sim_slot, DEFAULT_FIRESIM_BINARY_BASENAME),
        # The canonical source path infrasetup reads.
        "workload_bootbinary": os.path.join(
            root, "deploy", "workloads",
            DEFAULT_FIRESIM_WORKLOAD_NAME,
            DEFAULT_FIRESIM_BOOTBINARY),
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
    """Copy the built elf over BOTH staging locations FireSim uses:

    - ``deploy/workloads/<workload>/<bootbinary>`` is what
      ``firesim infrasetup`` reads when populating the run-farm node's
      filesystem (the "canonical" source path).
    - ``firesim_rundir/<slot>/<workload><N>-<bootbinary>`` is what the
      TSI loader inside the simulator binary opens at boot. Without
      this, infrasetup re-creates the prefixed file from the canonical
      source -- a no-op when the source hasn't changed -- so a direct
      cp lets us skip slow re-flashes on iteration.
    """
    if not os.path.isfile(elf):
        raise FileNotFoundError(f"--elf {elf} not found")
    # Sim-slot path (TSI loader's view).
    target = paths["elf_target"]
    os.makedirs(paths["sim_slot"], exist_ok=True)
    shutil.copyfile(elf, target)
    os.chmod(target, 0o755)
    # Workload-bundle path (infrasetup's source-of-truth).
    workload_dest = paths.get("workload_bootbinary")
    if workload_dest:
        os.makedirs(os.path.dirname(workload_dest), exist_ok=True)
        shutil.copyfile(elf, workload_dest)
        os.chmod(workload_dest, 0o755)


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


def _firesim_infrasetup(firesim_env: str, firesim_root: str,
                        verbose: bool = True) -> None:
    """Run `firesim infrasetup` to reset XDMA / FPGA state.

    `runworkload` alone does not reset the FPGA — after a prior aborted
    sim, XDMA buffers and bridge state are stale, and the next
    `runworkload` intermittently crashes mid-run (mtval surfaces as a
    bus error rather than a Zephyr-side bug).  `infrasetup` re-stages
    the bitstream pieces and clears the half-configured state.
    """
    if verbose:
        print(f"firesim: infrasetup (XDMA / FPGA reset)", flush=True)
    res = subprocess.run(
        _firesim_cmd(firesim_env, firesim_root, "infrasetup"),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if res.returncode != 0 and verbose:
        # Don't fatal — some setups treat infrasetup as already-done
        # and exit nonzero with a benign message.  Surface the tail for
        # diagnosis.
        tail = (res.stdout or b"")[-1200:].decode("utf-8", errors="replace")
        print(f"firesim infrasetup rc={res.returncode}; tail:\n{tail}",
              flush=True)


FIRESIM_QUEUE_BIN = os.environ.get(
    "FIRESIM_QUEUE_BIN",
    "/scratch2/agustin/firesim_queue/bin/firesim-queue")


def _use_firesim_queue() -> bool:
    """Whether to route runworkload through the shared FPGA queue.

    Checked at every call (not memoized) so a test can flip
    FIRESIM_QUEUE on/off without restarting the runner. The
    queue is only enabled when BOTH the env var is set AND the
    queue binary exists -- if the queue install moves, ModelBlaster
    falls back to direct invocation rather than hanging.
    """
    return (os.environ.get("FIRESIM_QUEUE", "0") == "1"
            and os.path.exists(FIRESIM_QUEUE_BIN))


def _firesim_run_async(firesim_env: str, firesim_root: str,
                       log_path: str) -> subprocess.Popen:
    """Spawn `firesim runworkload` and return the Popen handle. We do
    NOT wait on it from this side; the run is treated as done when the
    expected OUTPUT_END markers appear in the uartlog. firesim's own
    stdout/stderr go to `log_path` so we can diagnose silent failures
    (e.g. screen session not starting). The actual UART output lives
    in firesim_rundir/sim_slot_0/uartlog regardless.

    When FIRESIM_QUEUE=1 (and the queue binary at FIRESIM_QUEUE_BIN
    exists) the runworkload command is wrapped by `firesim-queue
    submit --background`. The queue daemon serializes FPGA access
    across users (currently us + merlin), so concurrent runs don't
    clobber each other's bitstream / xdma state. The daemon spawns
    the wrapped `firesim runworkload` inside its own process tree;
    our uartlog watcher + firesim kill still work identically because
    firesim kill pgrep's its target by process name, and the uartlog
    path is unchanged.
    """
    log_f = open(log_path, "w")
    runworkload_cmd = _firesim_cmd(firesim_env, firesim_root, "runworkload")
    if _use_firesim_queue():
        priority = os.environ.get("FIRESIM_QUEUE_PRIORITY", "5")
        argv = [
            FIRESIM_QUEUE_BIN, "submit",
            "--priority", str(priority),
            "--cwd", firesim_root,
            "--project", "modelblaster",
            "--background",
            "--",
        ] + runworkload_cmd
        return subprocess.Popen(argv, stdout=log_f, stderr=subprocess.STDOUT)
    return subprocess.Popen(
        runworkload_cmd, stdout=log_f, stderr=subprocess.STDOUT,
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
    expected `=== MODELBLASTER_OUTPUT_END ===` markers (one per model in
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
    # firesim kill causes the xdma kernel module to re-probe, which resets
    # /dev/xdma* permissions back to root:root 0600. Fix up before runworkload.
    subprocess.run(["sudo", "chmod", "666"] + [
        f"/dev/{n}" for n in os.listdir("/dev") if n.startswith("xdma")
    ], check=False)
    # Re-run infrasetup so XDMA / FPGA state is freshly configured.  Without
    # this, runworkload can pick up half-configured XDMA from a previous
    # aborted sim and crash mid-run with a bus error that's not a Zephyr bug.
    if os.environ.get("FIRESIM_SKIP_INFRASETUP", "0") != "1":
        _firesim_infrasetup(firesim_env, firesim_root, verbose=verbose)
    if stage_elf:
        if verbose:
            print(f"firesim: stage {elf} -> {paths['elf_target']}", flush=True)
        _stage_elf(elf, paths)
    _truncate_uartlog(paths)
    expected_ends = _expected_end_count(models, pool_sizes)
    if verbose:
        print(f"firesim: runworkload (waiting for {expected_ends} "
              f"MODELBLASTER_WALL_CYCLES marker{'s' if expected_ends>1 else ''})",
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
    # Zephyr's fatal-error printer (zephyr_ws/zephyr/arch/riscv/core/fatal.c)
    # emits these verbatim including capitalization. Match exactly — adding
    # `.lower()` would cost CPU per poll iteration and risk picking up
    # markers in trace data. Variants intentionally enumerated:
    #   "Instruction Access fault"  ← capital A (from RISC-V exception
    #                                  table "Instruction Access fault")
    #   "Instruction access fault"  ← legacy lowercase (older Zephyr fork)
    # Both kept so the fast-fail short-circuits on every dialect we've
    # seen on FireSim Saturn.
    _fault_markers = (
        "Load access fault",
        "Store access fault",
        "Store/AMO access fault",
        "Illegal instruction",
        "Instruction access fault",
        "Instruction Access fault",
        "Load Access fault",
        "Store Access fault",
        ">>> ZEPHYR FATAL ERROR",
        "k_oops",
        "HARNESS_FATAL_BEGIN",
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
                # Use errors='replace' because micro-DDS topic data
                # (e.g., XCDR-encoded "rt/<net>/done" payloads) contains
                # non-UTF8 bytes that would otherwise abort the loop
                # right when we want to wait for the fault dump.
                with open(paths["uartlog"], encoding="utf-8",
                          errors="replace") as f:
                    text = f.read()
            except FileNotFoundError:
                pass
            if len(text) != last_size:
                last_size = len(text)
                last_progress = time.monotonic()
            # Stop on the LAST block's MODELBLASTER_WALL_CYCLES line — that's
            # the trailing per-block sentinel. Neither single-model nor
            # multi-model harness emits OUTPUT_BEGIN/END (they use
            # VERIFY+PROFILE+WALL_CYCLES), so WALL_CYCLES count alone
            # is the correct termination condition in all modes.
            wall_done = wall_cycles_count(text) >= expected_ends
            if wall_done:
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
            _settle = float(os.environ.get("FIRESIM_FAULT_SETTLE", "5.0"))
            if fault_seen_at is not None and (
                time.monotonic() - fault_seen_at > _settle
            ):
                tail = text[-4000:]
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

    with open(paths["uartlog"], encoding="utf-8", errors="replace") as f:
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
    # Re-emit the uartlog (with MODELBLASTER_VERIFY / PROFILE_BEGIN /
    # PROFILE_END / WALL_CYCLES markers intact) so downstream parsers
    # in benchmarks/runners/firesim.py:parse_stdout can find them in the
    # captured run.sh stdout. Bracket with our own markers so the
    # extractor can grep for just this block. Mirrors the spike runner's
    # MODELBLASTER_RAW_SPIKE_BEGIN/END dance.
    print("=== MODELBLASTER_RAW_FIRESIM_BEGIN ===")
    print(out, end="" if out.endswith("\n") else "\n")
    print("=== MODELBLASTER_RAW_FIRESIM_END ===")
    repo_root = args.repo_root or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
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
