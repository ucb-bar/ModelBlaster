#!/usr/bin/env python3
"""Run a microbench .elf on FireSim, capture uartlog, extract CSV.

This is a slim adaptation of agents/validation/firesim_runner.py. The
harness emits one or more THREADPOOL_BENCH_BEGIN/END blocks instead of
the AGENTS_OUTPUT_BEGIN/END blocks the production runner expects, so we
roll our own poll loop that stops on the first "bench_*: done" line —
that's the marker each microbench prints right after the final CSV
block. Polite-wait coordination (against another concurrent FireSim
process holding the bitstream) is the responsibility of the caller
shell script.
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


def _firesim_cmd(firesim_env: str, firesim_root: str,
                 sub_cmd: str) -> list[str]:
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
    ap.add_argument("--csv-out", required=True)
    args = ap.parse_args()

    paths = _firesim_paths(args.firesim_root, args.firesim_slot)
    if not os.path.isfile(args.firesim_env):
        print(f"FATAL: firesim env not found at {args.firesim_env}",
              file=sys.stderr)
        return 2

    # Tear down any prior sim, stage elf, truncate uartlog.
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
    rwl = os.path.join(paths["sim_slot"], "_microbench_runworkload.log")
    log_f = open(rwl, "w")
    proc = subprocess.Popen(
        _firesim_cmd(args.firesim_env, args.firesim_root, "runworkload"),
        stdout=log_f, stderr=subprocess.STDOUT,
    )

    # The harness prints "bench_*: done\n" right after its final
    # THREADPOOL_BENCH_END marker; we poll for that and tear down.
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
            if "=== THREADPOOL_BENCH_END ===" in text and \
               ("bench_pthreadpool: done" in text or
                "bench_zephyr_threads: done" in text or
                "bench_pthreads_raw: done" in text):
                print("firesim: bench done line seen", flush=True)
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

    # Copy raw log out, then parse CSV via the shared parser.
    with open(paths["uartlog"]) as f:
        raw = f.read()
    with open(args.raw_out, "w") as f:
        f.write(raw)

    # Reuse parse_log's extract/merge.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from parse_log import extract_blocks, merge_blocks  # type: ignore

    blocks = extract_blocks(raw)
    if not blocks:
        print(f"FAIL: no THREADPOOL_BENCH block in uartlog "
              f"(raw at {args.raw_out})", file=sys.stderr)
        return 1
    rows = merge_blocks(blocks)
    with open(args.csv_out, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"firesim: wrote {len(rows) - 1} rows to {args.csv_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
