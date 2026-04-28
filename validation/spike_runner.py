"""Run a built zephyr.elf on spike and compare its output to the PyTorch golden.

Spike's stdout is parsed for the block between the markers
    === AGENTS_OUTPUT_BEGIN ===
    <one float per line>
    === AGENTS_OUTPUT_END ===
which the generated harness/src/main.c emits.
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


BEGIN = "=== AGENTS_OUTPUT_BEGIN ==="
END = "=== AGENTS_OUTPUT_END ==="
PROF_BEGIN = "=== AGENTS_PROFILE_BEGIN ==="
PROF_END = "=== AGENTS_PROFILE_END ==="

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
    if BEGIN not in out:
        raise RuntimeError(
            f"spike output missing BEGIN marker. cmd={cmd!r}\n"
            f"--- output ---\n{out}"
        )
    return out


def parse_output(text: str) -> np.ndarray:
    m = re.search(rf"{re.escape(BEGIN)}\n(.*?)\n{re.escape(END)}", text, re.S)
    if not m:
        raise RuntimeError("could not find AGENTS_OUTPUT_{BEGIN,END} block")
    nums = []
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        nums.append(float(line))
    return np.array(nums, dtype=np.float32)


def parse_profile(text: str) -> Optional[list[dict]]:
    """Parse the AGENTS_PROFILE CSV block. Returns None if absent."""
    m = re.search(
        rf"{re.escape(PROF_BEGIN)}\n(.*?)\n{re.escape(PROF_END)}", text, re.S
    )
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--elf", required=True)
    ap.add_argument("--io", required=True, help="io.npz with 'output' key")
    ap.add_argument("--spike", default=None)
    ap.add_argument("--atol", type=float, default=None,
                    help="defaults to 1e-5 for fp32, 0 for int8 (bit-exact)")
    ap.add_argument("--rtol", type=float, default=None,
                    help="defaults to 1e-4 for fp32, 0 for int8 (bit-exact)")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--spike-arg", action="append", default=[],
                    help="extra arg passed to spike (repeatable)")
    ap.add_argument("--profile-csv", default=None,
                    help="path to write per-kernel profile CSV (default: "
                         "<dir of --io>/profile.csv)")
    args = ap.parse_args()

    spike = find_spike(args.spike)
    print(f"spike: {spike}")
    out = run_spike(spike, args.elf, timeout=args.timeout,
                    extra_args=tuple(args.spike_arg))
    actual = parse_output(out)
    raw_golden = np.load(args.io)["output"]
    is_int = raw_golden.dtype.kind in ("i", "u")
    golden = raw_golden.astype(np.float32).reshape(-1)
    # Auto-select tolerance from golden's dtype: bit-exact for integer ops,
    # standard fp32 round-off for float ops.
    atol = args.atol if args.atol is not None else (0.0 if is_int else 1e-5)
    rtol = args.rtol if args.rtol is not None else (0.0 if is_int else 1e-4)
    ok, stats = compare(actual, golden, atol=atol, rtol=rtol)

    print(f"actual: {actual.tolist()}")
    print(f"golden: {golden.tolist()}")
    if "error" in stats:
        print(f"FAIL: {stats['error']}")
        return 1
    print(
        f"max_abs_err={stats['max_abs_err']:.3g} "
        f"max_rel_err={stats['max_rel_err']:.3g} "
        f"(atol={stats['atol']:g} rtol={stats['rtol']:g})"
    )
    print("PASS" if ok else "FAIL")

    profile = parse_profile(out)
    if profile is not None:
        csv_path = args.profile_csv or os.path.join(
            os.path.dirname(os.path.abspath(args.io)), "profile.csv"
        )
        write_profile_csv(profile, csv_path)
        print()
        print(f"profile -> {csv_path}")
        print_profile_summary(profile)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
