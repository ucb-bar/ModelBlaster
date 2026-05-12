"""Roofline / SOL analysis from per-op profile CSVs.

Walks `gen/profile/<backend>/<target>/<network>/...` results.csv files
and computes:

  - ops:    arithmetic op count per dispatch (closed-form per op kind)
  - bytes:  worst-case streaming traffic per dispatch (in + weights +
            out + small consts; no cache-reuse modeling — pessimistic
            on AI to keep v1 simple)
  - AI:     ops / bytes
  - GOPS:   ops / wall_time
  - SOL:    min(peak_compute, peak_bandwidth × AI) per (backend, target)
  - util%:  GOPS / SOL
  - roof:   "compute" or "memory" (which side of the roofline knee)

Output: per-op table sorted by util ascending (biggest gap first), plus
a per-(backend, target, network) cycle-weighted aggregate. Useful for
spotting where future kernel optimization has the most headroom.

Caveats (see also the v1 caveats discussed in the design):

  * Non-streaming ops like silu_s8 (LUT-bounded) appear "memory bound"
    when the actual bottleneck is LUT scatter latency. Take with salt.
  * Convolution bytes are naive (assumes no weight reuse), so AI is
    pessimistic; real cache reuse pushes effective AI higher. v1
    favors over-reporting "memory bound" — fine for triage, not for
    absolute claims.
  * Requantize stages on gemmini run on the CPU after tiled_conv_auto,
    not on the systolic array. Peak-ops-per-cycle for gemmini reflects
    the array; the requantize tail looks like wasted utilization here.
  * mtime granularity (1 µs at 1 MHz on chipyard) clamps short ops to
    0 cycles, infinite measured GOPS. Rows with cycles==0 are skipped.
  * spike's "cycles" field is functional-sim instruction count, not a
    real perf measurement — the SOL columns are only meaningful when
    target startswith "firesim".

Backend SOL configs are theoretical (DIM, VLEN, clock) and should be
calibrated against a packed peak-microbench when we have one.

Usage:
    python -m agents.scripts.roofline_analysis <profile-root-or-csv>
    python -m agents.scripts.roofline_analysis gen/profile/sweep_v8 \\
        --backend RVV --network yolov8_nano --target firesim_rocket_saturn

The future home for the (ops, bytes) formulas is in extract_graph.py /
reference_kernels.py — once the PyTorch graph parser is the
authoritative source of per-op ops/bytes counts, this script imports
them from there instead of duplicating the closed forms here.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import Callable, Optional


# ---------------------------------------------------------------------- #
# Backend SOL configs                                                    #
# ---------------------------------------------------------------------- #

@dataclass(frozen=True)
class BackendSOL:
    """Theoretical roof for one (backend, target). Tune these as
    microbench data lands.
    """
    name: str
    target: str
    clock_mhz: float
    peak_ops_per_cycle: float       # per-hart sustained int8 OPS at peak
    peak_bytes_per_cycle: float     # effective DRAM bandwidth at peak

    @property
    def peak_compute_gops(self) -> float:
        return self.peak_ops_per_cycle * self.clock_mhz / 1000.0  # 1e6 cyc/s × ops/cyc / 1e9

    @property
    def peak_bandwidth_gbps(self) -> float:
        return self.peak_bytes_per_cycle * self.clock_mhz / 1000.0


# Defaults: per-hart, theoretical. Calibrate against measured peak when
# we have a microbench; for now these are good enough for relative
# triage (which ops are FAR from SOL vs which are close).
SOL_CONFIGS: dict[tuple[str, str], BackendSOL] = {
    # gemmini Saturn DIM=16 systolic. 16×16 array × 2 ops/MAC = 512
    # ops/cyc when fully fed. Memory: shared L2 bus on the dual-rocket
    # bitstream is the bottleneck; 32 B/cyc is a starting estimate
    # (L2 line-wide loads at the systolic feeder rate).
    ("gemmini", "firesim_rocket_saturn"): BackendSOL(
        name="gemmini", target="firesim_rocket_saturn",
        clock_mhz=1000.0, peak_ops_per_cycle=512.0, peak_bytes_per_cycle=32.0,
    ),
    # V256D128_rvv = saturn quad-rocket-saturn-llc4mb bitstream: VLEN=256,
    # DLEN=128. With LMUL=4 the LMUL-effective vlen is 1024 bits → 128
    # int8 lanes; sustained MAC throughput peaks around 64 ops/cyc gated
    # by DLEN=128 (one 128-bit FMA per cycle on this microarch).
    # Memory: ~16 B/cyc effective on the L2 bus.
    ("V256D128_rvv", "firesim_rocket_saturn"): BackendSOL(
        name="V256D128_rvv", target="firesim_rocket_saturn",
        clock_mhz=1000.0, peak_ops_per_cycle=64.0, peak_bytes_per_cycle=16.0,
    ),
    # Legacy alias — kept so older CSVs / scripts that still emit
    # backend="RVV" keep finding a SOL entry.
    ("RVV", "firesim_rocket_saturn"): BackendSOL(
        name="V256D128_rvv", target="firesim_rocket_saturn",
        clock_mhz=1000.0, peak_ops_per_cycle=64.0, peak_bytes_per_cycle=16.0,
    ),
    # scalar Rocket: 1 MAC/cyc, modest L1 throughput.
    ("scalar", "firesim_rocket_saturn"): BackendSOL(
        name="scalar", target="firesim_rocket_saturn",
        clock_mhz=1000.0, peak_ops_per_cycle=1.0, peak_bytes_per_cycle=4.0,
    ),
}


# ---------------------------------------------------------------------- #
# Shape parsing                                                          #
# ---------------------------------------------------------------------- #

_SHAPE_KV_RE = re.compile(r"([A-Za-z_]+)\s*=\s*(-?\d+)")


def parse_shape(shape_str: str) -> dict[str, int]:
    """'N=1;IC=3;IH=160;IW=160;OC=16;OH=80;OW=80;KH=3;KW=3;SH=2;SW=2;PH=1;PW=1'
    → {N:1, IC:3, IH:160, ...}. Tolerant of whitespace and missing
    fields.
    """
    out: dict[str, int] = {}
    for m in _SHAPE_KV_RE.finditer(shape_str or ""):
        try:
            out[m.group(1)] = int(m.group(2))
        except ValueError:
            continue
    return out


# ---------------------------------------------------------------------- #
# Per-op (ops, bytes) formulas                                           #
# ---------------------------------------------------------------------- #

# Element size for the dominant tensor type per op family. Most of our
# ops are int8 (1 byte); a few have fp32 params (γ/β/scale).

def _conv2d_s8(s: dict[str, int]) -> tuple[int, int]:
    N, IC, IH, IW = s.get("N", 1), s.get("IC", 0), s.get("IH", 0), s.get("IW", 0)
    OC, OH, OW = s.get("OC", 0), s.get("OH", 0), s.get("OW", 0)
    KH, KW = s.get("KH", 1), s.get("KW", 1)
    ops = 2 * N * OC * OH * OW * IC * KH * KW    # MACs counted as 2 ops
    in_bytes = N * IC * IH * IW                  # int8
    w_bytes = OC * IC * KH * KW                  # int8
    out_bytes = N * OC * OH * OW                 # int8
    bias_bytes = 4 * OC                          # int32
    return ops, in_bytes + w_bytes + out_bytes + bias_bytes


def _linear_s8(s: dict[str, int]) -> tuple[int, int]:
    M, K, N = s.get("M", 1), s.get("K", 0), s.get("N", 0)
    ops = 2 * M * N * K
    in_bytes = M * K
    w_bytes = K * N
    out_bytes = M * N
    bias_bytes = 4 * N
    return ops, in_bytes + w_bytes + out_bytes + bias_bytes


def _batchnorm2d_s8(s: dict[str, int]) -> tuple[int, int]:
    N, C, H, W = s.get("N", 1), s.get("C", 0), s.get("H", 0), s.get("W", 0)
    n = N * C * H * W
    ops = 2 * n                                  # mul + add per element
    return ops, 2 * n + 8 * C                    # in+out int8 + γ,β fp32


def _silu_or_sigmoid_s8(s: dict[str, int]) -> tuple[int, int]:
    n = s.get("n", 0)
    # silu(x) = x * sigmoid(x) ≈ 5 fp ops in the mathematical formulation
    # (exp, add, reciprocal, mul x2). The actual implementation uses a
    # 256-entry LUT, so the *real* bottleneck is scatter latency rather
    # than arithmetic throughput — but for SOL accounting we count the
    # mathematical work to get a meaningful GOPS number that reflects
    # what would have to happen without the LUT shortcut. Same logic
    # for sigmoid (exp, add, reciprocal = ~3 ops; round up to 5 to keep
    # silu and sigmoid on the same scale). LUT-bounded ops will still
    # report low util — that's correct, the LUT is hiding the work
    # rather than the kernel being slow.
    return 5 * n, 2 * n + 256


def _elementwise_s8(s: dict[str, int]) -> tuple[int, int]:
    n = s.get("n", 0)
    # Single-input elementwise (relu).
    return n, 2 * n


def _add_s8(s: dict[str, int]) -> tuple[int, int]:
    n = s.get("n", 0)
    # Two inputs + one output, one op per element.
    return n, 3 * n


def _maxpool2d_s8(s: dict[str, int]) -> tuple[int, int]:
    N = s.get("N", 1)
    C = s.get("C", s.get("OC", 0))
    OH, OW = s.get("OH", s.get("H", 0)), s.get("OW", s.get("W", 0))
    KH, KW = s.get("KH", 2), s.get("KW", 2)
    in_H = (OH - 1) * s.get("SH", 2) + KH
    in_W = (OW - 1) * s.get("SW", 2) + KW
    ops = N * C * OH * OW * KH * KW              # comparisons
    in_bytes = N * C * in_H * in_W
    out_bytes = N * C * OH * OW
    return ops, in_bytes + out_bytes


def _cat_s8(s: dict[str, int]) -> tuple[int, int]:
    N = s.get("N", 1)
    H, W = s.get("H", 0), s.get("W", 0)
    C_total = s.get("C_total", 0)
    n = N * C_total * H * W
    return 0, 2 * n                              # pure memcpy


def _zero_cost(_: dict[str, int]) -> tuple[int, int]:
    return 0, 0


OP_FORMULAS: dict[str, Callable[[dict[str, int]], tuple[int, int]]] = {
    "conv2d_s8": _conv2d_s8,
    "linear_s8": _linear_s8,
    "batchnorm2d_s8": _batchnorm2d_s8,
    "silu_s8": _silu_or_sigmoid_s8,
    "sigmoid_s8": _silu_or_sigmoid_s8,
    "relu_s8": _elementwise_s8,
    "add_s8": _add_s8,
    "maxpool2d_s8": _maxpool2d_s8,
    "cat2_c1_s8": _cat_s8,
    "cat3_c1_s8": _cat_s8,
    "cat4_c1_s8": _cat_s8,
    "upsample_nearest_s8": _cat_s8,              # also pure memcpy at int8
    "view": _zero_cost,
    "chunk2_c1": _zero_cost,
    "chunk2_c1_s8": _zero_cost,
}


# ---------------------------------------------------------------------- #
# CSV reader                                                             #
# ---------------------------------------------------------------------- #

@dataclass
class OpRow:
    """One dispatch from a profile CSV with derived roofline metrics."""
    backend: str
    target: str
    network: str
    dispatch_id: int
    op: str
    name: str
    shape: str
    cycles: int
    ops: int
    bytes_: int
    arithmetic_intensity: float
    measured_gops: float
    sol_gops_compute: float
    sol_gops_bandwidth: float
    sol_gops: float
    sol_util: float
    roof: str                                    # "compute" / "memory" / "n/a"


def _walk_profile_csvs(root: str):
    """Yield (backend, target, network, csv_path) tuples from the
    standard profile tree:
      gen/profile/<backend>/<target>/<network>/<basename>/<input_tag>/<topo>/results.csv
    """
    if root.endswith(".csv") and os.path.isfile(root):
        # User passed a single CSV; infer (backend, target, network) from the
        # path structure.
        parts = os.path.normpath(os.path.abspath(root)).split(os.sep)
        # Walk back 6 levels: <topo>/<input_tag>/<basename>/<network>/<target>/<backend>
        try:
            idx = parts.index("profile")
            backend = parts[idx + 1]
            target = parts[idx + 2]
            network = parts[idx + 3]
        except (ValueError, IndexError):
            backend, target, network = "?", "?", "?"
        yield backend, target, network, root
        return
    for backend in sorted(os.listdir(root)):
        bd = os.path.join(root, backend)
        if not os.path.isdir(bd):
            continue
        for target in sorted(os.listdir(bd)):
            td = os.path.join(bd, target)
            if not os.path.isdir(td):
                continue
            for network in sorted(os.listdir(td)):
                nd = os.path.join(td, network)
                if not os.path.isdir(nd):
                    continue
                for dirpath, _, files in os.walk(nd):
                    if "results.csv" in files:
                        yield backend, target, network, os.path.join(dirpath, "results.csv")


def _row_metrics(backend: str, target: str, network: str, raw: dict) -> Optional[OpRow]:
    op = (raw.get("op") or "").strip()
    if not op:
        return None
    cycles = int(raw.get("cycles") or 0)
    if cycles <= 0:
        # mtime granularity OR zero-cost op — skip from roofline
        # (no measured throughput to compare against).
        return None
    shape = parse_shape(raw.get("shape", ""))
    formula = OP_FORMULAS.get(op)
    if formula is None:
        ops, bytes_ = 0, 0
    else:
        ops, bytes_ = formula(shape)
    sol = SOL_CONFIGS.get((backend, target))
    if sol is None or ops == 0 or bytes_ == 0:
        return OpRow(
            backend=backend, target=target, network=network,
            dispatch_id=int(raw.get("dispatch_id") or -1),
            op=op, name=raw.get("name", "") or raw.get("module_name", ""),
            shape=raw.get("shape", ""), cycles=cycles,
            ops=ops, bytes_=bytes_,
            arithmetic_intensity=0.0, measured_gops=0.0,
            sol_gops_compute=0.0, sol_gops_bandwidth=0.0,
            sol_gops=0.0, sol_util=0.0, roof="n/a",
        )
    seconds = cycles / (sol.clock_mhz * 1e6)
    ai = ops / bytes_
    measured_gops = ops / seconds / 1e9
    sol_compute = sol.peak_compute_gops
    sol_bandwidth = sol.peak_bandwidth_gbps * ai
    sol_gops = min(sol_compute, sol_bandwidth)
    util = (measured_gops / sol_gops) if sol_gops > 0 else 0.0
    roof = "compute" if sol_compute < sol_bandwidth else "memory"
    return OpRow(
        backend=backend, target=target, network=network,
        dispatch_id=int(raw.get("dispatch_id") or -1),
        op=op, name=raw.get("name", "") or raw.get("module_name", ""),
        shape=raw.get("shape", ""), cycles=cycles,
        ops=ops, bytes_=bytes_,
        arithmetic_intensity=ai, measured_gops=measured_gops,
        sol_gops_compute=sol_compute, sol_gops_bandwidth=sol_bandwidth,
        sol_gops=sol_gops, sol_util=util, roof=roof,
    )


def load_rows(profile_root: str,
              backend_filter: Optional[str] = None,
              target_filter: Optional[str] = None,
              network_filter: Optional[str] = None) -> list[OpRow]:
    rows: list[OpRow] = []
    for backend, target, network, csv_path in _walk_profile_csvs(profile_root):
        if backend_filter and backend != backend_filter:
            continue
        if target_filter and target != target_filter:
            continue
        if network_filter and network != network_filter:
            continue
        with open(csv_path) as f:
            for raw in csv.DictReader(f):
                rec = _row_metrics(backend, target, network, raw)
                if rec is not None:
                    rows.append(rec)
    return rows


# ---------------------------------------------------------------------- #
# Reporters                                                              #
# ---------------------------------------------------------------------- #

def _format_table(rows: list[OpRow], top_n: Optional[int]) -> str:
    if not rows:
        return "(no rows)"
    rows_sorted = sorted(
        [r for r in rows if r.roof != "n/a"],
        key=lambda r: r.sol_util,
    )
    if top_n is not None:
        rows_sorted = rows_sorted[:top_n]
    cols = [
        ("backend",    8),  ("network", 14), ("op",     16),  ("name", 22),
        ("shape",     38),  ("cyc",     10), ("GOPS",    8),  ("SOL",   8),
        ("util%",      7),  ("roof",     7),
    ]
    out = []
    out.append("  ".join(f"{c[0]:<{c[1]}}" for c in cols))
    out.append("  ".join("-" * c[1] for c in cols))
    def _fmt_gops(g: float) -> str:
        # Switch to mGOPS when small enough to lose precision in 7.2f.
        if g >= 0.01:
            return f"{g:>7.2f}"
        return f"{g * 1000:>5.2f}m"

    def _fmt_util(u: float) -> str:
        pct = u * 100
        if pct >= 1.0:
            return f"{pct:>6.1f}"
        if pct >= 0.01:
            return f"{pct:>6.2f}"
        return f"{pct:>6.3f}"

    for r in rows_sorted:
        shape_short = r.shape if len(r.shape) <= 38 else r.shape[:35] + "..."
        name_short = (r.name or "").split("$")[0][:22]
        out.append("  ".join([
            f"{r.backend:<8}",
            f"{r.network:<14}",
            f"{r.op:<16}",
            f"{name_short:<22}",
            f"{shape_short:<38}",
            f"{r.cycles:<10d}",
            _fmt_gops(r.measured_gops),
            f"{r.sol_gops:>7.2f}",
            _fmt_util(r.sol_util),
            f"{r.roof:<7}",
        ]))
    return "\n".join(out)


def _aggregate(rows: list[OpRow]) -> str:
    if not rows:
        return "(no rows)"
    # Cycle-weighted util per (backend, target, network).
    keys: dict[tuple[str, str, str], list[OpRow]] = {}
    for r in rows:
        if r.roof == "n/a":
            continue
        keys.setdefault((r.backend, r.target, r.network), []).append(r)
    out = []
    out.append(f"{'backend':<8}  {'target':<24}  {'network':<14}  "
               f"{'#ops':>6}  {'cycles':>12}  {'avg GOPS':>10}  "
               f"{'avg SOL':>10}  {'wt-util%':>9}")
    out.append("-" * 110)
    for (backend, target, network), recs in sorted(keys.items()):
        total_cyc = sum(r.cycles for r in recs)
        total_ops = sum(r.ops for r in recs)
        clock_hz = (SOL_CONFIGS[(backend, target)].clock_mhz * 1e6
                    if (backend, target) in SOL_CONFIGS else 1e9)
        total_seconds = total_cyc / clock_hz
        agg_gops = total_ops / total_seconds / 1e9 if total_seconds > 0 else 0
        # Cycle-weighted SOL: each op contributes its sol_gops × cycles.
        wt_sol = sum(r.sol_gops * r.cycles for r in recs) / max(total_cyc, 1)
        wt_util = sum(r.sol_util * r.cycles for r in recs) / max(total_cyc, 1)
        out.append(
            f"{backend:<8}  {target:<24}  {network:<14}  "
            f"{len(recs):>6d}  {total_cyc:>12d}  {agg_gops:>10.2f}  "
            f"{wt_sol:>10.2f}  {wt_util * 100:>8.1f}"
        )
    return "\n".join(out)


# ---------------------------------------------------------------------- #
# CLI                                                                    #
# ---------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m agents.scripts.roofline_analysis gen/profile/sweep_v8\n"
            "  python -m agents.scripts.roofline_analysis gen/profile/sweep_v8 \\\n"
            "      --backend RVV --network yolov8_nano --top 20\n"
        ),
    )
    ap.add_argument("profile_root",
                    help="path to a profile root (gen/profile/sweep_v8) or "
                         "a single results.csv")
    ap.add_argument("--backend", help="filter to one backend (gemmini/RVV/scalar)")
    ap.add_argument("--target",  help="filter to one target (firesim_rocket_saturn/spike/...)")
    ap.add_argument("--network", help="filter to one network (yolov8_nano/dronet/...)")
    ap.add_argument("--top", type=int, default=20,
                    help="show this many lowest-util ops (default 20). "
                         "0 = show all.")
    ap.add_argument("--write-csv",
                    help="also write the full per-op roofline as CSV here")
    args = ap.parse_args()

    rows = load_rows(args.profile_root, args.backend, args.target, args.network)

    print("\n=== Roofline aggregate (cycle-weighted utilization) ===\n")
    print(_aggregate(rows))

    print("\n=== Per-op roofline (sorted by util ascending — biggest gap first) ===\n")
    print(_format_table(rows, top_n=(None if args.top == 0 else args.top)))

    if args.write_csv:
        with open(args.write_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["backend", "target", "network", "dispatch_id",
                        "op", "name", "shape", "cycles",
                        "ops", "bytes", "arithmetic_intensity",
                        "measured_gops", "sol_gops_compute",
                        "sol_gops_bandwidth", "sol_gops",
                        "sol_util", "roof"])
            for r in rows:
                w.writerow([r.backend, r.target, r.network, r.dispatch_id,
                            r.op, r.name, r.shape, r.cycles,
                            r.ops, r.bytes_, f"{r.arithmetic_intensity:.4f}",
                            f"{r.measured_gops:.4f}",
                            f"{r.sol_gops_compute:.4f}",
                            f"{r.sol_gops_bandwidth:.4f}",
                            f"{r.sol_gops:.4f}",
                            f"{r.sol_util:.4f}", r.roof])
        print(f"\nWrote {args.write_csv}")


if __name__ == "__main__":
    main()
