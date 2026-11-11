"""Per-board defaults for the Linux board runners (K1 and K3).

Why this is Python and not shell: the K1 runners grew their board knowledge as
scattered `${VAR:-default}` literals, and the K3 port needs the SAME scripts to
serve a second board. Forking them would leave two copies that drift (the K1
copies already carry hard-won fixes). One table here, plus the three derivations
that are easy to get silently wrong -- the topo tag, the profile hw name and the
launch mask -- all unit-testable without a board.

Stdlib only on purpose: the shell scripts run this file by path before any
PYTHONPATH is set up.

Selection is `MODELBLASTER_BOARD` (default `k1`). With it unset every value
here equals the literal the K1 scripts used before this module existed, so K1
behaviour is unchanged.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class BoardConfig:
    name: str
    host: str
    remote_root: str
    registry: str            # repo-relative
    cpu_p_kind: str
    cpu_e_kind: str
    backend: str             # default codegen backend / core-kind backend
    profile_cpu: str         # the <target> level of the profile tree
    out_root_single: str     # run_model_*.sh staging dir (repo-relative)
    out_root_xpurt: str      # run_xpurt_*.sh staging dir (repo-relative)
    clock_mhz: float         # rdtime frequency, NOT the core clock
    topo_relative_default: bool
    launch_cpus: str         # taskset mask the process is STARTED under ("" = none)
    # core-type name -> harts, for deriving a per-core-type profile tree.
    # Empty = one core type, no derivation.
    core_types: tuple[tuple[str, tuple[int, ...]], ...] = ()


_DEFAULTS: dict[str, BoardConfig] = {
    "k1": BoardConfig(
        name="k1", host="k1", remote_root="/root/mb_k1",
        registry="cores/spacemit_k1.json",
        cpu_p_kind="rvv", cpu_e_kind="rvv_c1", backend="rvv_x60",
        profile_cpu="spacemit_x60",
        out_root_single="build/k1", out_root_xpurt="build/k1_xpurt",
        clock_mhz=24.0, topo_relative_default=False, launch_cpus="",
    ),
    "k3": BoardConfig(
        name="k3", host="spacemitk3", remote_root="/home/sudospace/mb_k3",
        registry="cores/spacemit_k3.json",
        cpu_p_kind="x100", cpu_e_kind="a100", backend="rvv_k3",
        profile_cpu="spacemit_k3",
        out_root_single="build/k3", out_root_xpurt="build/k3_xpurt",
        clock_mhz=24.0, topo_relative_default=True,
        # The process starts on the X100; A100 worker threads register
        # themselves with the kernel from inside the runtime. The shell never
        # touches /proc/set_ai_thread.
        launch_cpus="0-7",
        core_types=(("x100", tuple(range(0, 8))), ("a100", tuple(range(8, 16)))),
    ),
}


def resolve(board: str | None = None,
            env: Mapping[str, str] | None = None) -> BoardConfig:
    """Board defaults with explicit env overrides applied.

    Host / remote root overrides are `MODELBLASTER_<BOARD>_HOST` and
    `MODELBLASTER_<BOARD>_REMOTE_ROOT` -- for k1 those are exactly the
    pre-existing `MODELBLASTER_K1_*` names.
    """
    env = os.environ if env is None else env
    board = (board or env.get("MODELBLASTER_BOARD") or "k1").lower()
    if board not in _DEFAULTS:
        raise ValueError(
            f"unknown MODELBLASTER_BOARD {board!r}; known: {sorted(_DEFAULTS)}")
    base = _DEFAULTS[board]
    pfx = f"MODELBLASTER_{board.upper()}_"
    from dataclasses import replace
    return replace(
        base,
        host=env.get(pfx + "HOST") or base.host,
        remote_root=env.get(pfx + "REMOTE_ROOT") or base.remote_root,
    )


def parse_cpulist(spec: str) -> list[int]:
    """"0-3", "0,1,2,3", "0_1_2_3" -> sorted unique ints. Fails loudly."""
    out: list[int] = []
    for part in str(spec).replace(" ", "").replace("_", ",").split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            if not (lo.isdigit() and hi.isdigit()) or int(lo) > int(hi):
                raise ValueError(f"bad cpu range {part!r} in {spec!r}")
            out.extend(range(int(lo), int(hi) + 1))
        elif part.isdigit():
            out.append(int(part))
        else:
            raise ValueError(f"bad cpu id {part!r} in {spec!r}")
    return sorted(set(out))


def topo_tag(cores: Sequence[int], relative: bool) -> str:
    """The `topo_*` directory suffix ("0_1_2_3").

    XPU-RT's profile loader looks a topology up by WIDTH (topo_0_1 means "two
    cores"), not by which physical CPUs measured it. Filing an 8-11 run under
    topo_8_9_10_11 therefore makes it unfindable. `relative` renumbers to
    0..n-1; the real CPUs are still what the process is pinned to.
    """
    cs = sorted(set(cores))
    if not cs:
        raise ValueError("empty core list")
    if relative:
        cs = list(range(len(cs)))
    return "_".join(str(c) for c in cs)


def core_type_of(cfg: BoardConfig, cores: Sequence[int]) -> str | None:
    """Single core type covering all of `cores`; None on boards without types.
    A mixed set is a hard error: its timing belongs to neither type, and a
    profile filed under one of them would silently mislead the scheduler."""
    if not cfg.core_types:
        return None
    found: set[str] = set()
    for c in cores:
        for tname, harts in cfg.core_types:
            if c in harts:
                found.add(tname)
                break
        else:
            raise ValueError(f"cpu {c} belongs to no {cfg.name} core type")
    if len(found) != 1:
        raise ValueError(
            f"cores {sorted(cores)} span core types {sorted(found)}; profile "
            f"each core type separately (or set PROFILE_HW_NAME explicitly)")
    return found.pop()


def derive_hw_name(cfg: BoardConfig, cores: Sequence[int], target: str,
                   explicit: str | None = None) -> str | None:
    """The `<hw>` level of the profile tree, or None = leave the writer's
    default (backend name). Explicit wins; otherwise on a multi-core-type board
    it is `<target>_<core type>` (rvv_k3_x100 / rvv_k3_a100)."""
    if explicit:
        return explicit
    t = core_type_of(cfg, cores)
    return f"{target}_{t}" if t else None


def shell_exports(cfg: BoardConfig) -> str:
    items = {
        "BOARD_NAME": cfg.name, "BOARD_HOST": cfg.host,
        "BOARD_REMOTE_ROOT": cfg.remote_root, "BOARD_REGISTRY": cfg.registry,
        "BOARD_CPU_P_KIND": cfg.cpu_p_kind, "BOARD_CPU_E_KIND": cfg.cpu_e_kind,
        "BOARD_BACKEND": cfg.backend, "BOARD_PROFILE_CPU": cfg.profile_cpu,
        "BOARD_OUT_ROOT_SINGLE": cfg.out_root_single,
        "BOARD_OUT_ROOT_XPURT": cfg.out_root_xpurt,
        "BOARD_CLOCK_MHZ": f"{cfg.clock_mhz:g}",
        "BOARD_TOPO_RELATIVE": "1" if cfg.topo_relative_default else "0",
        "BOARD_LAUNCH_CPUS": cfg.launch_cpus,
    }
    return "\n".join(f"{k}={shlex.quote(v)}" for k, v in items.items())


def _main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--board", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("shell", help="print BOARD_* assignments for `eval`")
    t = sub.add_parser("topo", help="print the topo tag for a core list")
    t.add_argument("cores")
    t.add_argument("--relative", choices=["0", "1"], default=None,
                   help="default: the board's default")
    h = sub.add_parser("hw-name", help="print the profile <hw> name (may be empty)")
    h.add_argument("cores")
    h.add_argument("--target", required=True)
    h.add_argument("--explicit", default=None)
    args = ap.parse_args()
    try:
        cfg = resolve(args.board)
        if args.cmd == "shell":
            print(shell_exports(cfg))
        elif args.cmd == "topo":
            rel = cfg.topo_relative_default if args.relative is None \
                else args.relative == "1"
            print(topo_tag(parse_cpulist(args.cores), rel))
        else:
            print(derive_hw_name(cfg, parse_cpulist(args.cores), args.target,
                                 args.explicit) or "")
    except ValueError as e:
        print(f"board_config: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(_main())
