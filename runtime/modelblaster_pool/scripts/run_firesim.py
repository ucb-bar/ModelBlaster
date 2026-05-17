#!/usr/bin/env python3
"""Run the modelblaster_pool unit-test elf on FireSim, capture uartlog.

Slim adaptation of modelblaster/microbench/threadpool/scripts/run_firesim_bench.py.
Polls the uartlog for the MODELBLASTER_POOL_TEST_END marker (or PASS/FAIL line)
and tears down. The caller (run_test.sh) is responsible for grepping the
captured log for PASS/FAIL.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

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
    inner = (
        f"set -e; "
        f"unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER "
        f"      CONDA_PYTHON_EXE CONDA_SHLVL CONDA_EXE _CE_M _CE_CONDA; "
        f"export PATH=/scratch2/dima/miniforge3/condabin:$PATH; "
        f"source {firesim_env}; "
        f"cd {firesim_root}; "
        f"source ./sourceme-manager.sh --skip-ssh-setup; "
        f"firesim {sub_cmd}"
    )
    return ["bash", "-c", inner]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--elf", required=True)
    ap.add_argument("--firesim-root", default=DEFAULT_FIRESIM_ROOT)
    ap.add_argument("--firesim-env", default=DEFAULT_FIRESIM_ENV)
    ap.add_argument("--firesim-slot", default=DEFAULT_FIRESIM_SLOT)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--poll-interval", type=float, default=2.0)
    ap.add_argument("--raw-out", required=True)
    args = ap.parse_args()

    paths = _firesim_paths(args.firesim_root, args.firesim_slot)
    if not os.path.isfile(args.firesim_env):
        print(f"FATAL: firesim env not found at {args.firesim_env}",
              file=sys.stderr)
        return 2

    print("firesim: kill any prior sim", flush=True)
    subprocess.run(_firesim_cmd(args.firesim_env, args.firesim_root, "kill"),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.makedirs(paths["sim_slot"], exist_ok=True)
    if not os.path.isfile(args.elf):
        print(f"FATAL: --elf {args.elf} not found", file=sys.stderr)
        return 2
    shutil.copyfile(args.elf, paths["elf_target"])
    os.chmod(paths["elf_target"], 0o755)
    if os.path.exists(paths["uartlog"]):
        with open(paths["uartlog"], "w") as f:
            f.truncate(0)

    print("firesim: runworkload", flush=True)
    rwl = os.path.join(paths["sim_slot"], "_agents_pool_runworkload.log")
    log_f = open(rwl, "w")
    proc = subprocess.Popen(
        _firesim_cmd(args.firesim_env, args.firesim_root, "runworkload"),
        stdout=log_f, stderr=subprocess.STDOUT,
    )

    deadline = time.monotonic() + args.timeout
    last_size = 0
    last_progress = time.monotonic()
    try:
        while True:
            if time.monotonic() > deadline:
                try:
                    with open(rwl) as f:
                        rwl_tail = f.read()[-2000:]
                except FileNotFoundError:
                    rwl_tail = "(no log captured)"
                print(f"TIMEOUT after {args.timeout}s; uartlog "
                      f"({last_size} bytes) at {paths['uartlog']}.\n"
                      f"--- last 2KB runworkload log ---\n{rwl_tail}",
                      file=sys.stderr)
                return 3
            text = ""
            try:
                with open(paths["uartlog"]) as f:
                    text = f.read()
            except FileNotFoundError:
                pass
            if len(text) != last_size:
                last_size = len(text)
                last_progress = time.monotonic()
            if "=== MODELBLASTER_POOL_TEST_END ===" in text:
                print("firesim: modelblaster_pool test end marker seen", flush=True)
                break
            if (time.monotonic() - last_progress > 60.0
                    and last_size > 0):
                last_progress = time.monotonic()
                print(f"  ... uartlog at {last_size} bytes, still waiting",
                      flush=True)
            time.sleep(args.poll_interval)
    finally:
        subprocess.run(_firesim_cmd(args.firesim_env, args.firesim_root,
                                    "kill"),
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    with open(paths["uartlog"]) as f:
        raw = f.read()
    with open(args.raw_out, "w") as f:
        f.write(raw)
    print(f"firesim: wrote raw log ({len(raw)} bytes) to {args.raw_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
