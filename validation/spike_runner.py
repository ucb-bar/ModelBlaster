"""Run a built zephyr.elf on spike and compare its output to the PyTorch golden.

Single-model mode (single-model harness): spike stdout has one block
between
    === AGENTS_OUTPUT_BEGIN ===
    <one float per line>
    === AGENTS_OUTPUT_END ===
and the runner compares against the golden in --io.

Multi-model mode (--models name1,name2,...): spike stdout has N blocks
tagged by model name, e.g.
    === AGENTS_OUTPUT_BEGIN [mlp_generic] ===
    ...
    === AGENTS_OUTPUT_END [mlp_generic] ===
and the runner compares each block against
    agents/examples/<name>/<quant>/generated/io.npz
(or against an explicit map provided via --model-io <name>:<path>).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from typing import Optional

import numpy as np

from agents.pipeline import profile_writer


BEGIN = "=== AGENTS_OUTPUT_BEGIN ==="
END = "=== AGENTS_OUTPUT_END ==="
PROF_BEGIN = "=== AGENTS_PROFILE_BEGIN ==="
PROF_END = "=== AGENTS_PROFILE_END ==="

# Multi-model markers — same prefix, with " [<name>]" suffix.
_BEGIN_BARE = re.compile(r"=== AGENTS_OUTPUT_BEGIN(?: \[([^\]]+)\])? ===")
_END_BARE = re.compile(r"=== AGENTS_OUTPUT_END(?: \[([^\]]+)\])? ===")
_PROF_BEGIN_BARE = re.compile(r"=== AGENTS_PROFILE_BEGIN(?: \[([^\]]+)\])? ===")
_PROF_END_BARE = re.compile(r"=== AGENTS_PROFILE_END(?: \[([^\]]+)\])? ===")
# Wall-clock total — single integer on the same line as the marker.
_WALL_RE = re.compile(
    r"=== AGENTS_WALL_CYCLES(?: \[([^\]]+)\])? === (\d+)"
)

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
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout
    )
    out = proc.stdout + proc.stderr
    if not _BEGIN_BARE.search(out):
        raise RuntimeError(
            f"spike output missing AGENTS_OUTPUT_BEGIN marker. cmd={cmd!r}\n"
            f"--- output ---\n{out}"
        )
    return out


def _output_block(text: str, tag: Optional[str] = None) -> str:
    """Return the body between BEGIN/END markers. If `tag` is None, match
    the bare (single-model) form; otherwise match `[<tag>]`."""
    if tag is None:
        b = re.escape(BEGIN)
        e = re.escape(END)
    else:
        b = re.escape(f"=== AGENTS_OUTPUT_BEGIN [{tag}] ===")
        e = re.escape(f"=== AGENTS_OUTPUT_END [{tag}] ===")
    m = re.search(rf"{b}\n(.*?)\n{e}", text, re.S)
    if not m:
        raise RuntimeError(
            f"could not find AGENTS_OUTPUT_{{BEGIN,END}} "
            f"{'(bare)' if tag is None else f'[{tag}]'} block"
        )
    return m.group(1)


def parse_output(text: str, tag: Optional[str] = None) -> np.ndarray:
    body = _output_block(text, tag)
    nums = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        nums.append(float(line))
    return np.array(nums, dtype=np.float32)


def parse_wall_cycles(text: str, tag: Optional[str] = None) -> Optional[int]:
    """Return the wall-clock cycles printed by the harness, or None if
    the marker isn't present. `tag` selects the per-model line in
    multi-model mode."""
    for m in _WALL_RE.finditer(text):
        if (m.group(1) or None) == tag:
            return int(m.group(2))
    return None


def parse_profile(text: str, tag: Optional[str] = None) -> Optional[list[dict]]:
    """Parse the AGENTS_PROFILE CSV block. Returns None if absent.
    `tag` selects the per-model block in multi-model mode."""
    if tag is None:
        b = re.escape(PROF_BEGIN)
        e = re.escape(PROF_END)
    else:
        b = re.escape(f"=== AGENTS_PROFILE_BEGIN [{tag}] ===")
        e = re.escape(f"=== AGENTS_PROFILE_END [{tag}] ===")
    m = re.search(rf"{b}\n(.*?)\n{e}", text, re.S)
    if not m:
        return None
    lines = [ln for ln in m.group(1).splitlines() if ln.strip()]
    if not lines:
        return []
    header = [c.strip() for c in lines[0].split(",")]
    out = []
    for ln in lines[1:]:
        cells = [c.strip() for c in ln.split(",")]
        rec = dict(zip(header, cells))
        if "cycles" in rec:
            rec["cycles"] = int(rec["cycles"])
        if "dispatch_id" in rec:
            rec["dispatch_id"] = int(rec["dispatch_id"])
        out.append(rec)
    return out


def write_profile_csv(records: list[dict], path: str) -> None:
    if not records:
        return
    header = list(records[0].keys())
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for rec in records:
            f.write(",".join(str(rec[k]) for k in header) + "\n")


def print_profile_summary(records: list[dict]) -> None:
    if not records:
        print("(no profile records emitted)")
        return
    name_w = max(4, max(len(r["name"]) for r in records))
    op_w = max(2, max(len(r["op"]) for r in records))
    shape_w = max(5, max(len(r["shape"]) for r in records))
    total = sum(int(r["cycles"]) for r in records)
    print(f"{'name':<{name_w}}  {'op':<{op_w}}  {'shape':<{shape_w}}  "
          f"{'cycles':>10}  {'%':>5}")
    print("-" * (name_w + op_w + shape_w + 22))
    for r in records:
        cyc = int(r["cycles"])
        pct = (100.0 * cyc / total) if total else 0.0
        print(f"{r['name']:<{name_w}}  {r['op']:<{op_w}}  {r['shape']:<{shape_w}}  "
              f"{cyc:>10}  {pct:>5.1f}")
    print("-" * (name_w + op_w + shape_w + 22))
    print(f"{'TOTAL':<{name_w + op_w + shape_w + 4}}{total:>10}")


def compare(actual: np.ndarray, golden: np.ndarray,
            atol: float = 1e-5, rtol: float = 1e-4) -> tuple[bool, dict]:
    if actual.shape != golden.shape:
        return False, {
            "error": f"shape mismatch: actual={actual.shape} golden={golden.shape}",
        }
    abs_err = np.abs(actual - golden)
    rel_err = abs_err / np.maximum(np.abs(golden), 1e-12)
    ok = bool(np.all(abs_err <= atol + rtol * np.abs(golden)))
    return ok, {
        "max_abs_err": float(abs_err.max()),
        "max_rel_err": float(rel_err.max()),
        "atol": atol,
        "rtol": rtol,
    }


def _check_one(actual: np.ndarray, golden_npz_path: str,
               atol: Optional[float], rtol: Optional[float],
               label: str = "") -> tuple[bool, dict]:
    """Compare one actual output array against the golden in `golden_npz_path`.
    Returns (ok, stats). Auto-selects bit-exact tolerance for integer goldens."""
    raw_golden = np.load(golden_npz_path)["output"]
    is_int = raw_golden.dtype.kind in ("i", "u")
    golden = raw_golden.astype(np.float32).reshape(-1)
    a = atol if atol is not None else (0.0 if is_int else 1e-5)
    r = rtol if rtol is not None else (0.0 if is_int else 1e-4)
    ok, stats = compare(actual, golden, atol=a, rtol=r)
    if "error" in stats:
        print(f"{label}FAIL: {stats['error']}")
        return False, stats
    print(f"{label}actual: {actual.tolist()}")
    print(f"{label}golden: {golden.tolist()}")
    print(
        f"{label}max_abs_err={stats['max_abs_err']:.3g} "
        f"max_rel_err={stats['max_rel_err']:.3g} "
        f"(atol={stats['atol']:g} rtol={stats['rtol']:g})"
    )
    print(f"{label}{'PASS' if ok else 'FAIL'}")
    return ok, stats


def _model_io_path(repo_root: str, name: str, quant: str) -> str:
    return os.path.join(
        repo_root, "agents", "examples", name, quant, "generated", "io.npz"
    )


def _detect_backend(spike_args: list[str], default: str = "scalar") -> str:
    """Infer the HW-backend tag from spike's --isa=... if present.
    rv64gcv* → rvv; otherwise scalar."""
    for a in spike_args:
        if a.startswith("--isa="):
            iso = a.split("=", 1)[1].lower()
            if "v" in iso.replace("rv64", "")[:6]:  # detect 'v' extension
                return "rvv"
    return default


def _emit_iree_profile(records: list[dict], model: str, args,
                       backend: str) -> str | None:
    """Build a ProfileMeta from CLI args and write an IREE-shape CSV.
    Returns the written path, or None if --profile-out-root is unset."""
    if not args.profile_out_root or not records:
        return None
    cores = [int(c) for c in args.profile_cores.split(",") if c.strip()]
    cpu = args.profile_cpu or args.profile_source
    meta = profile_writer.ProfileMeta(
        model=model,
        quant=args.quant,
        backend=backend,
        cores=cores,
        source=args.profile_source,
        cpu=cpu,
        clock_mhz=args.profile_clock_mhz,
    )
    return profile_writer.write_profile(records, meta, args.profile_out_root)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--elf", required=True)
    ap.add_argument("--io", default=None,
                    help="io.npz path (single-model mode)")
    ap.add_argument("--models", default=None,
                    help="comma-separated model names for multi-model mode "
                         "(e.g. mlp_generic,mlp_control)")
    ap.add_argument("--quant", default="fp32",
                    help="quant axis used to locate per-model io.npz "
                         "(only relevant in --models mode)")
    ap.add_argument("--repo-root", default=None,
                    help="repo root for resolving per-model io.npz paths "
                         "(default: 2 dirs above this script)")
    ap.add_argument("--spike", default=None)
    ap.add_argument("--atol", type=float, default=None,
                    help="defaults to 1e-5 for fp32, 0 for int8 (bit-exact)")
    ap.add_argument("--rtol", type=float, default=None,
                    help="defaults to 1e-4 for fp32, 0 for int8 (bit-exact)")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--spike-arg", action="append", default=[],
                    help="extra arg passed to spike (repeatable)")
    ap.add_argument("--profile-csv", default=None,
                    help="path to write per-kernel profile CSV "
                         "(single-model only; multi-model writes one CSV "
                         "per model under <io>/../<name>_profile.csv)")

    # IREE-shape per-dispatch profile emission. When --profile-out-root
    # is set the runner additionally writes <root>/<backend>/<cpu>/
    # <model>/<model>.<quant>/<spec>/topo_<cores>/results.csv, one per
    # model. XPU-RT ingests this format directly.
    ap.add_argument("--profile-out-root", default=None,
                    help="root of the IREE-shape per-dispatch profile tree")
    ap.add_argument("--profile-source", default="spike",
                    help="provenance tag for the data: spike (default), "
                         "firesim, rtlsim, or a chip name")
    ap.add_argument("--profile-cpu", default=None,
                    help="CPU label used in the output path (defaults to "
                         "--profile-source)")
    ap.add_argument("--profile-backend", default=None,
                    help="HW backend label (scalar/rvv); auto-detected "
                         "from --spike-arg=--isa=... if omitted")
    ap.add_argument("--profile-cores", default="0,1,2,3",
                    help="comma-separated hart layout, used in the topo_<...> "
                         "directory name (default 0,1,2,3 for spike -p4)")
    ap.add_argument("--profile-clock-mhz", type=float, default=1000.0,
                    help="clock rate used to convert per-op cycles to ns "
                         "(default 1000.0 = 1 GHz; representative spike rocket)")
    args = ap.parse_args()

    if not args.models and not args.io:
        ap.error("must pass either --io (single-model) or --models (multi)")

    spike = find_spike(args.spike)
    print(f"spike: {spike}")
    out = run_spike(spike, args.elf, timeout=args.timeout,
                    extra_args=tuple(args.spike_arg))

    backend_tag = args.profile_backend or _detect_backend(args.spike_arg)

    if args.models:
        # Multi-model mode: parse N tagged blocks, compare each.
        repo_root = args.repo_root or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        names = [n.strip() for n in args.models.split(",") if n.strip()]
        all_ok = True
        for name in names:
            print(f"\n--- model: {name} ---")
            actual = parse_output(out, tag=name)
            io_path = _model_io_path(repo_root, name, args.quant)
            if not os.path.exists(io_path):
                print(f"FAIL: golden not found at {io_path}")
                all_ok = False
                continue
            ok, _ = _check_one(actual, io_path, args.atol, args.rtol,
                               label=f"  [{name}] ")
            all_ok = all_ok and ok
            profile = parse_profile(out, tag=name)
            if profile is not None:
                csv_path = os.path.join(
                    os.path.dirname(io_path), f"{name}_profile.csv"
                )
                write_profile_csv(profile, csv_path)
                print(f"  [{name}] profile -> {csv_path}")
                print_profile_summary(profile)
            wall = parse_wall_cycles(out, tag=name)
            if wall is not None:
                print(f"  [{name}] wall_clock_cycles={wall} (mtime)")
            iree_path = _emit_iree_profile(profile or [], name, args,
                                           backend_tag)
            if iree_path:
                print(f"  [{name}] iree_profile -> {iree_path}")
        print()
        print(f"OVERALL: {'PASS' if all_ok else 'FAIL'} ({len(names)} models)")
        return 0 if all_ok else 1

    # Single-model mode (existing behavior).
    actual = parse_output(out)
    ok, _ = _check_one(actual, args.io, args.atol, args.rtol)
    profile = parse_profile(out)
    if profile is not None:
        csv_path = args.profile_csv or os.path.join(
            os.path.dirname(os.path.abspath(args.io)), "profile.csv"
        )
        write_profile_csv(profile, csv_path)
        print()
        print(f"profile -> {csv_path}")
        print_profile_summary(profile)
    wall = parse_wall_cycles(out)
    if wall is not None:
        print(f"wall_clock_cycles={wall} (mtime)")
    if profile is not None and args.profile_out_root:
        # Single-model: derive the model name from the io.npz parent
        # directory layout (.../examples/<model>/<quant>/generated/io.npz).
        io_abs = os.path.abspath(args.io)
        model_name = os.path.basename(
            os.path.dirname(os.path.dirname(os.path.dirname(io_abs)))
        )
        iree_path = _emit_iree_profile(profile, model_name, args, backend_tag)
        if iree_path:
            print(f"iree_profile -> {iree_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
