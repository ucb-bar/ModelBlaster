"""Run a built zephyr.elf on a Chipyard Verilator simulator and profile it.

WHY THIS EXISTS: the per-op profile database is populated from cycle counts the
harness prints around each dispatch. Two runners could produce them until now:

  spike_runner.py    functional ISS. No timing model, and measured against
                     FireSim on mlp_control it undercounts by a factor that
                     varies with the operation -- 0.75x on linear, 0.50x on elu.
                     It also cannot execute Gemmini's RoCC instructions without
                     libgemmini.so, which is not built in this tree.
  firesim_runner.py  cycle-exact, but needs an FPGA host and a built bitstream.

Verilator sits between them: cycle-exact on the same RTL the bitstream is built
from -- so Gemmini's mesh, LoopConv, the scratchpad and the DMA are all modelled
-- while needing nothing but a host CPU. It is slower than FireSim by orders of
magnitude, which bounds what it is useful for:

    mlp_control     ~107 K cycles    seconds
    dronet         ~9.0 M cycles     minutes
    yolov8_nano     ~418 M cycles    hours -- use FireSim

That range covers the question Verilator is here to answer, which is how a
configuration's *accelerator shape* changes per-op cycles. A 16x16 Gemmini has no
profile anywhere in this tree, and mesh width is not something Spike can model.

The console arrives over HTIF (`zephyr,console = &htif` in the target DTS), so
the harness's MODELBLASTER_* markers land on the simulator's stdout and
runner_common parses them exactly as it does for spike and FireSim.

Building the simulator is deliberately NOT this module's job -- it is a Chipyard
make flow with its own toolchain environment. Pass the built binary with --sim:

    make -C backends/chipyard/sims/verilator CONFIG=<cfg> \
         CONFIG_PACKAGE=chipyard MODEL=TestHarness \
         MODEL_PACKAGE=chipyard.harness GENERATOR_PACKAGE=chipyard TOP=ChipTop

Note that FireSim *target* configs cannot be Verilated: they instantiate
firesim.lib.bridges.RationalClockBridge, which has no Verilog outside a FireSim
compile. Use a plain `chipyard`-package config with the same accelerator chain.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
from typing import Optional

from modelblaster.validation.runner_common import (
    BEGIN,
    END,
    IREEProfileArgs,
    has_output_marker,
    parse_output,
    parse_profile,
    report_run,
    _VERIFY_RE,
    _WALL_RE,
)

__all__ = [
    "find_sim", "run_verilator", "parse_output", "parse_profile", "main",
]

# Chipyard names the simulator after the harness and config it was built from.
_SIM_GLOB = "simulator-*"


def find_sim(explicit: Optional[str] = None,
             search_dir: Optional[str] = None,
             config: Optional[str] = None) -> str:
    """Locate a built Chipyard Verilator simulator.

    --sim wins. Otherwise glob `search_dir` (default: the chipyard verilator sim
    directory) and, if `config` is given, require it in the filename so a tree
    holding several built simulators cannot silently pick the wrong one.
    """
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(f"--sim {explicit} not found")
        return explicit
    if search_dir is None:
        here = os.path.dirname(os.path.abspath(__file__))
        repo = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
        search_dir = os.path.join(repo, "backends", "chipyard", "sims", "verilator")
    found = sorted(
        p for p in glob.glob(os.path.join(search_dir, _SIM_GLOB))
        if os.path.isfile(p) and os.access(p, os.X_OK) and not p.endswith(".log")
    )
    if config:
        found = [p for p in found if config in os.path.basename(p)]
    if not found:
        raise FileNotFoundError(
            f"no built Verilator simulator matching {_SIM_GLOB}"
            + (f" and config {config!r}" if config else "")
            + f" in {search_dir}. Build one with the make line in this module's "
              f"docstring, or pass --sim."
        )
    if len(found) > 1:
        raise RuntimeError(
            "several built simulators match; disambiguate with --config or "
            "--sim:\n  " + "\n  ".join(found)
        )
    return found[0]


def run_verilator(sim: str, elf: str, timeout: float = 7200.0,
                  max_cycles: Optional[int] = None,
                  extra_args: tuple[str, ...] = (),
                  progress_every: float = 0.0) -> tuple[str, float]:
    """Run `elf` on `sim`, returning (combined output, wall seconds).

    Flags mirror Chipyard's own `run-binary-fast` rule: everything the simulator
    itself consumes sits between +permissive and +permissive-off, and the binary
    follows. No disassembly is requested -- it costs wall time and this runner
    only wants the harness's own printed cycle counts.
    """
    args = ["+permissive"]
    if max_cycles is not None:
        args.append(f"+max-cycles={max_cycles}")
    args += [*extra_args, "+permissive-off", elf]
    cmd = [sim, *args]

    started = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
        out = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True

        def _as_str(x: object) -> str:
            if x is None:
                return ""
            if isinstance(x, bytes):
                return x.decode("utf-8", errors="replace")
            return str(x)

        out = _as_str(exc.stdout) + _as_str(exc.stderr)
        print(
            f"WARNING: verilator timed out after {timeout:.0f}s — using "
            f"{len(out)} chars of partial output. Verilator runs ~5-50 kHz, so a "
            f"9 M-cycle workload can need 30+ minutes; raise --timeout.",
            file=sys.stderr,
        )
    elapsed = time.monotonic() - started

    # Same completion test spike_runner uses: any of the wall-cycles line, the
    # in-binary verify summary, or the legacy output block means the harness got
    # far enough to parse.
    complete = (
        has_output_marker(out) or bool(_VERIFY_RE.search(out))
        or bool(_WALL_RE.search(out))
    )
    if not complete:
        detail = f"timed out after {timeout:.0f}s" if timed_out else "exited"
        raise RuntimeError(
            f"verilator {detail} with no MODELBLASTER_VERIFY / "
            f"MODELBLASTER_WALL_CYCLES / MODELBLASTER_OUTPUT_BEGIN marker. "
            f"cmd={cmd!r}\n--- last 2000 chars ---\n{out[-2000:]}"
        )
    return out, elapsed


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run a zephyr.elf on a Chipyard Verilator simulator and "
                    "emit a per-op cycle profile."
    )
    ap.add_argument("--elf", required=True)
    ap.add_argument("--sim", default=None,
                    help="built simulator binary; auto-discovered if omitted")
    ap.add_argument("--sim-dir", default=None,
                    help="directory to search for the simulator")
    ap.add_argument("--config", default=None,
                    help="Chipyard config name, used to disambiguate --sim-dir")
    ap.add_argument("--io", default=None, help="io.npz path (single-model mode)")
    ap.add_argument("--models", default=None,
                    help="comma-separated model names for multi-model mode")
    ap.add_argument("--quant", default="int8")
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--atol", type=float, default=None)
    ap.add_argument("--rtol", type=float, default=None)
    ap.add_argument("--timeout", type=float, default=7200.0,
                    help="seconds; default 2 h, sized for a ~9 M-cycle workload")
    ap.add_argument("--max-cycles", type=int, default=None,
                    help="pass +max-cycles to the simulator as a backstop")
    ap.add_argument("--sim-arg", action="append", default=[],
                    help="extra simulator arg, inside the permissive region")
    ap.add_argument("--profile-csv", default=None,
                    help="write the per-op cycle profile here")
    ap.add_argument("--save-log", default=None,
                    help="write raw simulator output here")
    args = ap.parse_args()

    sim = find_sim(args.sim, args.sim_dir, args.config)
    elf = os.path.abspath(args.elf)
    if not os.path.exists(elf):
        raise FileNotFoundError(f"--elf {elf} not found")

    print(f"sim  {sim}", file=sys.stderr)
    print(f"elf  {elf}", file=sys.stderr)
    out, elapsed = run_verilator(
        sim, elf, timeout=args.timeout, max_cycles=args.max_cycles,
        extra_args=tuple(args.sim_arg),
    )
    if args.save_log:
        with open(args.save_log, "w") as fh:
            fh.write(out)

    records = parse_profile(out) or []
    cycles = sum(int(r.get("cycles") or 0) for r in records)
    rate = (cycles / elapsed) if elapsed > 0 else 0.0
    print(
        f"verilator: {elapsed:.1f}s wall for {cycles:,} simulated cycles "
        f"({rate / 1000:.1f} kHz) over {len(records)} dispatch(es)",
        file=sys.stderr,
    )

    return report_run(
        out,
        models=args.models.split(",") if args.models else None,
        io=args.io,
        quant=args.quant,
        repo_root=args.repo_root,
        atol=args.atol,
        rtol=args.rtol,
        profile_csv=args.profile_csv,
        source="verilator",
        iree=IREEProfileArgs(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
