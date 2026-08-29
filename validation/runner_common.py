"""Shared parsing / golden-compare / IREE-profile helpers.
Used by both the spike and firesim runners; the only thing each runner
adds on top is *how* it gets the harness's stdout text.

Markers (the agent harness prints these unchanged across simulators):
    === MODELBLASTER_VERIFY [<model>] === max_abs_err=<g> max_rel_err=<g> n=<int>
    === MODELBLASTER_PROFILE_BEGIN[<model>] ===
    dispatch_id,name,op,shape,cycles
    ...
    === MODELBLASTER_PROFILE_END  [<model>] ===
    === MODELBLASTER_WALL_CYCLES  [<model>] === <int>

The single-model harness omits the [<model>] tag and emits one block of
each kind. Multi-model harnesses tag every block.

The legacy `=== MODELBLASTER_OUTPUT_BEGIN [<model>] === / END ===` per-element
output dump is still recognized for back-compat with older harness
binaries, but the modern harness prints `MODELBLASTER_VERIFY` instead — the
in-binary compare against the baked-in test_golden saves shipping the
full output tensor over HTIF UART (which dominated FireSim runtime).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np


BEGIN = "=== MODELBLASTER_OUTPUT_BEGIN ==="
END = "=== MODELBLASTER_OUTPUT_END ==="
PROF_BEGIN = "=== MODELBLASTER_PROFILE_BEGIN ==="
PROF_END = "=== MODELBLASTER_PROFILE_END ==="

# Tagged variants — used by the multi-model harness.
_BEGIN_BARE = re.compile(r"=== MODELBLASTER_OUTPUT_BEGIN(?: \[([^\]]+)\])? ===")
_END_BARE = re.compile(r"=== MODELBLASTER_OUTPUT_END(?: \[([^\]]+)\])? ===")
_PROF_BEGIN_BARE = re.compile(r"=== MODELBLASTER_PROFILE_BEGIN(?: \[([^\]]+)\])? ===")
_PROF_END_BARE = re.compile(r"=== MODELBLASTER_PROFILE_END(?: \[([^\]]+)\])? ===")
_WALL_RE = re.compile(
    r"=== MODELBLASTER_WALL_CYCLES(?: \[([^\]]+)\])? === (\d+)"
)
_VERIFY_RE = re.compile(
    r"=== MODELBLASTER_VERIFY(?: \[([^\]]+)\])? === "
    r"max_abs_err=(\S+) max_rel_err=(\S+) n=(\d+)"
)


def parse_verify(text: str, tag: Optional[str] = None
                 ) -> Optional[dict]:
    """Pull the in-binary verify summary for the given tag (or the
    untagged single-model variant if `tag is None`). Returns a dict
    with keys max_abs_err, max_rel_err, n — or None if the harness
    didn't emit one (legacy binaries; we then have to fall back to
    parse_output)."""
    for m in _VERIFY_RE.finditer(text):
        m_tag = m.group(1)
        if (tag is None and m_tag is None) or (m_tag == tag):
            return {
                "max_abs_err": float(m.group(2)),
                "max_rel_err": float(m.group(3)),
                "n": int(m.group(4)),
            }
    return None


def verify_count(text: str) -> int:
    """How many MODELBLASTER_VERIFY summary lines have appeared. Mirrors
    wall_cycles_count for streamed runners that need an end-of-bench
    sentinel without parsing the per-element output dump."""
    return len(list(_VERIFY_RE.finditer(text)))


def has_output_marker(text: str) -> bool:
    """Quick check that the harness reached the OUTPUT phase. Useful as
    a stop condition for streamed runners."""
    return bool(_BEGIN_BARE.search(text))


def output_block_count(text: str) -> int:
    """Number of distinct OUTPUT_END markers seen — counts how many of
    the expected tagged blocks have completed."""
    return len(list(_END_BARE.finditer(text)))


def wall_cycles_count(text: str) -> int:
    """Number of MODELBLASTER_WALL_CYCLES markers seen. The harness prints
    this AFTER OUTPUT_END and PROFILE_END for each block, so it's the
    correct end-of-block sentinel for a streamed runner: waiting for
    OUTPUT_END alone races the trailing PROFILE/WALL prints and the
    last block's profile gets cut off when the sim is killed."""
    return len(list(_WALL_RE.finditer(text)))


def _output_block(text: str, tag: Optional[str] = None) -> str:
    if tag is None:
        b = re.escape(BEGIN); e = re.escape(END)
    else:
        b = re.escape(f"=== MODELBLASTER_OUTPUT_BEGIN [{tag}] ===")
        e = re.escape(f"=== MODELBLASTER_OUTPUT_END [{tag}] ===")
    m = re.search(rf"{b}\n(.*?)\n{e}", text, re.S)
    if not m:
        raise RuntimeError(
            f"could not find MODELBLASTER_OUTPUT_{{BEGIN,END}} "
            f"{'(bare)' if tag is None else f'[{tag}]'} block"
        )
    return m.group(1)


def parse_output(text: str, tag: Optional[str] = None) -> np.ndarray:
    """Parse the OUTPUT block. Supports two formats:

    Format 1 (small tensors — < ~64 elements):
        =1
        =2
        ...
      One value per line; returned as a 1D float32 array.

    Format 2 (large tensors — sampled to head+tail to stay UART-friendly):
        OUTPUT_SAMPLED size=<N> head=<H> tail=<T>
          [0] <v>
          [1] <v>
          ...
          [N-T] <v>
          ...
          [N-1] <v>
        OUTPUT_SUMMARY sum=... abs_sum=... min=... max=...
      Returns a SPARSE representation: ndarray of shape (n_samples, 2)
      with columns (index, value). check_one() handles this by comparing
      against golden[index] at each sampled position.
    """
    body = _output_block(text, tag)
    # Detect format.
    if "OUTPUT_SAMPLED" in body:
        # Sampled large-tensor format.
        nums: list[tuple[int, float]] = []
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("OUTPUT_SAMPLED") or line.startswith("OUTPUT_SUMMARY"):
                continue
            # Expected: "[<idx>] <val>"
            m = re.match(r"\[(\d+)\]\s+(.+)$", line)
            if m:
                nums.append((int(m.group(1)), float(m.group(2))))
        # Sparse array of (index, value) pairs.
        return np.array(nums, dtype=np.float32)
    # Dense small-tensor format.
    nums_dense = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        nums_dense.append(float(line))
    return np.array(nums_dense, dtype=np.float32)


def parse_wall_cycles(text: str, tag: Optional[str] = None) -> Optional[int]:
    for m in _WALL_RE.finditer(text):
        if (m.group(1) or None) == tag:
            return int(m.group(2))
    return None


def parse_profile(text: str, tag: Optional[str] = None) -> Optional[list[dict]]:
    if tag is None:
        b = re.escape(PROF_BEGIN); e = re.escape(PROF_END)
    else:
        b = re.escape(f"=== MODELBLASTER_PROFILE_BEGIN [{tag}] ===")
        e = re.escape(f"=== MODELBLASTER_PROFILE_END [{tag}] ===")
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


#: Below this many rdtime ticks a dispatch's run-to-run spread is mostly timer
#: quantisation. 240 ticks = 10 us at 24 MHz, so the tick is 0.4% of the sample
#: and a reported CV is about the kernel rather than about the clock.
_CV_MIN_TICKS = 240

#: Per-iteration profile blocks emitted when MODELBLASTER_ITERS > 1.
_ITER_PROF = re.compile(
    r"=== MODELBLASTER_ITER_PROFILE_BEGIN \[(?P<it>\d+)\] ===\n"
    r"(?P<body>.*?)\n=== MODELBLASTER_ITER_PROFILE_END \[\1\] ===",
    re.S)


def parse_profile_reps(text: str) -> Optional[list[dict]]:
    """Median per-dispatch cycles across MODELBLASTER_ITERS repetitions.

    Every cost in the authoritative profile DB was a SINGLE sample: the harness
    prints one PROFILE block, `parse_profile` reads it, and that number becomes
    the service time the scheduler plans against and the advisor compares
    against a free slot. Two closed-loop recommendations have already been
    rejected on n=1 gaps -- they were large (41%, 36%) so the conclusions hold,
    but the project's own criterion asks for a median of N and did not have one.

    The binary could already do the work: MODELBLASTER_ITERS>1 runs the model N
    times and prints a per-iteration profile block. Nothing read them, so the
    repetitions were paid for and discarded.

    Returns None when no per-iteration blocks are present, so the single-block
    path stays exactly as it was. Adds `cycles_n`, `cycles_min`, `cycles_max`
    and `cycles_cv_pct` -- the spread is the point, not decoration: it is what
    tells you whether a 10% "win" is distinguishable from run-to-run noise, and
    an advisor that reports confidence needs it.

    Iteration 0 is DISCARDED when at least three remain. First-touch faulting of
    a const weight array lands entirely in it -- measured on vitfly_lstm, whose
    1.7 MB of weights cost 21.9 ms cold against 2.88 ms warm, a 7.6x difference
    attributable to nothing but page faults.
    """
    blocks = []
    for m in _ITER_PROF.finditer(text):
        lines = [ln for ln in m.group("body").splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        header = [c.strip() for c in lines[0].split(",")]
        recs = []
        for ln in lines[1:]:
            rec = dict(zip(header, [c.strip() for c in ln.split(",")]))
            if "cycles" in rec:
                rec["cycles"] = int(rec["cycles"])
            if "dispatch_id" in rec:
                rec["dispatch_id"] = int(rec["dispatch_id"])
            recs.append(rec)
        blocks.append((int(m.group("it")), recs))
    if not blocks:
        return None

    blocks.sort()
    warm = [recs for it, recs in blocks if it > 0] if len(blocks) >= 4 else \
           [recs for _, recs in blocks]
    dropped = len(blocks) - len(warm)

    # Key on (dispatch_id, name) rather than position: a model whose op count
    # varied between iterations would otherwise average unrelated ops together.
    by_key: dict = {}
    order: list = []
    for recs in warm:
        for r in recs:
            k = (r.get("dispatch_id"), r.get("name"))
            if k not in by_key:
                by_key[k] = (dict(r), [])
                order.append(k)
            by_key[k][1].append(int(r["cycles"]))

    import statistics
    out = []
    for k in order:
        base, samples = by_key[k]
        n = len(samples)
        med = int(statistics.median(samples))
        base["cycles"] = med
        base["cycles_n"] = n
        base["cycles_min"] = min(samples)
        base["cycles_max"] = max(samples)
        base["cycles_cv_pct"] = (
            round(100.0 * statistics.stdev(samples) / med, 2)
            if n > 1 and med else 0.0)
        out.append(base)
    if out:
        # Report CV only for dispatches long enough for it to MEAN anything.
        # rdtime ticks at 24 MHz (41.7 ns), so a 1.5 us dispatch is ~36 ticks
        # and a 0.1 us one is 2-3. At that scale the spread is dominated by
        # quantisation of the timer, not by the kernel: the hybrid fused_full
        # profile showed 244.8% CV on a cast_i8_to_f16 costing 36 ticks, which
        # reads as wild instability and is nothing of the kind. Quoting it as
        # the run's "worst CV" would make every profile look untrustworthy and
        # would bury the figures that do matter -- the 0.5-0.7 ms convolutions
        # in the same run sit at 15-22%, which is a real and interesting number.
        floor = _CV_MIN_TICKS
        big = [r for r in out if r["cycles"] >= floor]
        if big:
            worst = max(big, key=lambda r: r["cycles_cv_pct"])
            print(f"  profile: median of {len(warm)} rep(s)"
                  f"{f' (dropped {dropped} warmup)' if dropped else ''}, "
                  f"worst CV {worst['cycles_cv_pct']:.1f}% "
                  f"({worst.get('op') or worst.get('name')}, "
                  f"{worst['cycles']} ticks)")
        else:
            print(f"  profile: median of {len(warm)} rep(s)"
                  f"{f' (dropped {dropped} warmup)' if dropped else ''}; "
                  f"every dispatch is under {floor} rdtime ticks, so per-dispatch "
                  f"CV is timer quantisation and is not reported")
        small = len(out) - len(big)
        if small:
            print(f"           {small} of {len(out)} dispatch(es) under {floor} "
                  f"ticks; their cycles_cv_pct is in the CSV but is dominated "
                  f"by the 41.7 ns timer tick")
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


def _select_tolerance(raw_golden: np.ndarray,
                      atol: Optional[float], rtol: Optional[float],
                      golden_npz_path: Optional[str] = None,
                      ) -> tuple[float, float]:
    """Common tolerance autoselect — fp16 / int / fp32 defaults match
    historical check_one behavior.

    For fp32, the default atol scales with sqrt(max_K) found in a
    sibling graph.json (if present and atol is not explicitly provided).
    This allows large-N matmul/reduction kernels whose f32 accumulation
    error is O(eps * sqrt(K)) to pass without false negatives.
    """
    import math as _math
    import json as _json
    is_int = raw_golden.dtype.kind in ("i", "u")
    is_fp16 = raw_golden.dtype == np.float16
    if is_int:
        a_default = r_default = 0.0
    elif is_fp16:
        a_default, r_default = 1e-2, 1e-2
        # Scale fp16 atol with sqrt(max_K) for large-N reductions.
        if atol is None and golden_npz_path is not None:
            graph_path = os.path.join(
                os.path.dirname(golden_npz_path), "graph.json"
            )
            if os.path.exists(graph_path):
                try:
                    with open(graph_path) as _gf:
                        _ir = _json.load(_gf)
                    _max_k = max(
                        (op.get("shape", {}).get("K", 1)
                         for op in _ir.get("ops", [])),
                        default=1,
                    )
                    a_default = max(1e-2, 1e-3 * _math.sqrt(_max_k))
                except Exception:
                    pass
    else:
        a_default, r_default = 1e-5, 1e-4
        # Scale fp32 atol with sqrt(max_K) when graph.json is available.
        if atol is None and golden_npz_path is not None:
            graph_path = os.path.join(
                os.path.dirname(golden_npz_path), "graph.json"
            )
            if os.path.exists(graph_path):
                try:
                    with open(graph_path) as _gf:
                        _ir = _json.load(_gf)
                    _max_k = max(
                        (op.get("shape", {}).get("K", 1)
                         for op in _ir.get("ops", [])),
                        default=1,
                    )
                    a_default = max(1e-5, 1e-4 * _math.sqrt(_max_k))
                except Exception:
                    pass
    return (
        atol if atol is not None else a_default,
        rtol if rtol is not None else r_default,
    )


def check_one(actual: np.ndarray, golden_npz_path: str,
              atol: Optional[float], rtol: Optional[float],
              label: str = "",
              verify: Optional[dict] = None) -> tuple[bool, dict]:
    """Validate `actual` against the io.npz golden.

    `verify` (preferred): the in-binary `MODELBLASTER_VERIFY` summary dict
    from `parse_verify`. When supplied, we use the device-computed
    `max_abs_err` / `max_rel_err` directly and skip the on-host
    per-element compare. Pass condition is the OR of the two bounds —
    sufficient for numpy.allclose-style tolerance (every element
    satisfies at least one of `abs_err <= atol` or `rel_err <= rtol`,
    so it satisfies the elementwise `abs_err <= atol + rtol*|g|`).

    `actual`: legacy per-element output array. Used only when
    `verify` is None — older harness binaries still ship the full
    output dump."""
    raw_golden = np.load(golden_npz_path)["output"]
    a, r = _select_tolerance(raw_golden, atol, rtol, golden_npz_path)

    if verify is not None:
        n_expected = int(np.asarray(raw_golden).reshape(-1).size)
        if verify["n"] != n_expected:
            stats = {
                "error": (f"MODELBLASTER_VERIFY n={verify['n']} mismatches "
                          f"golden size {n_expected}"),
                "max_abs_err": float("inf"),
                "max_rel_err": float("inf"),
                "atol": a, "rtol": r,
            }
            print(f"{label}FAIL: {stats['error']}")
            return False, stats
        max_ae = verify["max_abs_err"]
        max_re = verify["max_rel_err"]
        ok = (max_ae <= a) or (max_re <= r)
        stats = {
            "max_abs_err": max_ae,
            "max_rel_err": max_re,
            "atol": a, "rtol": r,
            "source": "in_binary_verify",
        }
        print(
            f"{label}max_abs_err={max_ae:.3g} max_rel_err={max_re:.3g} "
            f"(atol={a:g} rtol={r:g}, in-binary)"
        )
        print(f"{label}{'PASS' if ok else 'FAIL'}")
        return ok, stats

    golden = raw_golden.astype(np.float32).reshape(-1)

    # Sparse "sampled" actual: shape (n, 2) with columns (index, value).
    # Emitted by parse_output() when the harness used OUTPUT_SAMPLED.
    # Compare each sampled value against golden at its index.
    if actual.ndim == 2 and actual.shape[1] == 2:
        idxs = actual[:, 0].astype(np.int64)
        vals = actual[:, 1].astype(np.float32)
        max_idx = int(idxs.max()) if len(idxs) else -1
        if max_idx >= len(golden):
            err = (f"sampled index {max_idx} out of golden size "
                   f"{len(golden)}")
            print(f"{label}FAIL: {err}")
            return False, {"error": err, "atol": a, "rtol": r}
        gold_at = golden[idxs]
        abs_err = np.abs(vals - gold_at)
        rel_err = np.where(
            np.abs(gold_at) > 0,
            abs_err / np.abs(gold_at),
            np.zeros_like(abs_err))
        ok = bool(np.all((abs_err <= a) | (rel_err <= r)))
        max_ae = float(abs_err.max()) if len(abs_err) else 0.0
        max_re = float(rel_err.max()) if len(rel_err) else 0.0
        stats = {
            "max_abs_err": max_ae, "max_rel_err": max_re,
            "atol": a, "rtol": r, "source": "sampled",
            "n_sampled": int(len(idxs)),
            "n_full": int(len(golden)),
        }
        print(f"{label}sampled {len(idxs)} of {len(golden)} elements "
              f"at idx {idxs.tolist()}")
        print(f"{label}actual: {vals.tolist()}")
        print(f"{label}golden: {gold_at.tolist()}")
        print(
            f"{label}max_abs_err={max_ae:.3g} max_rel_err={max_re:.3g} "
            f"(atol={a:g} rtol={r:g}, sampled)"
        )
        print(f"{label}{'PASS' if ok else 'FAIL'}")
        return ok, stats

    # Dense actual: legacy small-tensor compare path.
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


def model_io_path(repo_root: str, name: str, quant: str) -> str:
    return os.path.join(
        repo_root, "examples", name, quant, "generated", "io.npz"
    )


@dataclass
class IREEProfileArgs:
    """Subset of CLI args needed by emit_iree_profile.
    Both runners surface these as flags."""
    profile_out_root: Optional[str]
    profile_source: str
    profile_cpu: Optional[str]
    profile_cores: str
    profile_clock_mhz: float
    quant: str
    #: Build dir holding kernel_picks.json, so the profile can record WHAT
    #: ran per op rather than only how long it took. Optional: without it the
    #: `implementation` column is empty, which reads as unknown.
    gen_dir: Optional[str] = None


def _parse_cores(spec: str) -> list[int]:
    """A core list, accepting the comma and the underscore spelling.

    WHY THIS IS NOT `[int(c) for c in spec.split(",")]`, which is what it was.
    The profile tree writes the topology as `topo_0_1_2_3`, so the underscore
    form is the one a caller reads off a path and hands back. Split only on
    commas, `"0_1_2_3"` reaches `int()` as a single token -- and Python accepts
    underscores as digit separators, so it returns **123**. No exception, and
    the run is filed under `topo_123`: a directory that does not correspond to
    any topology, sorts nowhere near the real ones, and reads as plausible.

    Found by a 4-core ffn_block run landing in `topo_123`. Rejecting the
    underscore would have been enough to make it loud; accepting it is better,
    because the underscore is the spelling the tree itself uses.
    """
    out: list[int] = []
    for tok in str(spec).replace("_", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if not tok.isdigit():
            raise ValueError(
                f"--profile-cores {spec!r}: {tok!r} is not a hart id. Use "
                f"'0,1,2,3' or '0_1_2_3'.")
        out.append(int(tok))
    return out


def emit_iree_profile(records: list[dict], model: str,
                      args: IREEProfileArgs, backend: str) -> Optional[str]:
    """Build a ProfileMeta and write an IREE-shape CSV.
    Returns the written path, or None if --profile-out-root is unset."""
    if not args.profile_out_root or not records:
        return None
    from modelblaster.pipeline import profile_writer
    cores = _parse_cores(args.profile_cores)
    cpu = args.profile_cpu or args.profile_source
    meta = profile_writer.ProfileMeta(
        model=model,
        quant=args.quant,
        backend=backend,
        cores=cores,
        source=args.profile_source,
        cpu=cpu,
        clock_mhz=args.profile_clock_mhz,
        picks=(profile_writer.picks_from_build(args.gen_dir)
               if args.gen_dir else {}),
    )
    return profile_writer.write_profile(records, meta, args.profile_out_root)


def report_pool_sweep_run(text: str, *,
                          models: list[str],
                          pool_sizes: list[int],
                          quant: str,
                          atol: Optional[float], rtol: Optional[float],
                          iree_args: IREEProfileArgs,
                          backend_tag: str,
                          repo_root: str) -> bool:
    """Walk OUTPUT/PROFILE/WALL blocks tagged `[<model>@p<N>]` — one
    block per (model, pool_size) pair from the multi_main_sweep harness.
    For each tag we:
      - parse the model name and pool size
      - compare against the model's golden (same tolerance regardless
        of pool size — pool only changes timing, not numerics)
      - emit the profile under topo_<cores> reflecting THIS pool's hart
        layout (cores = list(range(N))), so XPU-RT's existing per-topo
        loader picks each results.csv up at the right path
    Returns True iff every (model, pool_size) verifies."""
    all_ok = True
    print(f"\n=== pool sweep: {len(models)} models × {len(pool_sizes)} pool sizes ===")
    for ps in pool_sizes:
        for name in models:
            tag = f"{name}@p{ps}"
            print(f"\n--- {tag} ---")
            verify = parse_verify(text, tag=tag)
            actual = None if verify is not None else parse_output(text, tag=tag)
            if verify is None and actual is None:
                print(f"  [{tag}] SKIP: tag not found in spike output "
                      f"(partial run / timeout?)")
                all_ok = False
                continue
            golden_path = model_io_path(repo_root, name, quant)
            if not os.path.exists(golden_path):
                print(f"FAIL: golden not found at {golden_path}")
                all_ok = False
                continue
            ok, _ = check_one(actual, golden_path, atol, rtol,
                              label=f"  [{tag}] ", verify=verify)
            all_ok = all_ok and ok
            profile = parse_profile(text, tag=tag)
            wall = parse_wall_cycles(text, tag=tag)
            if wall is not None:
                print(f"  [{tag}] wall_clock_cycles={wall} (mtime)")
            # Emit per-(model, pool) profile under topo_<cores>. We
            # rebuild IREEProfileArgs so cores reflects this pool size
            # (cores=[0] for p1, [0,1] for p2, [0,1,2,3] for p4, ...).
            if profile is not None and iree_args.profile_out_root:
                cores = ",".join(str(i) for i in range(ps))
                per_pool_args = IREEProfileArgs(
                    profile_out_root=iree_args.profile_out_root,
                    profile_source=iree_args.profile_source,
                    profile_cpu=iree_args.profile_cpu,
                    profile_cores=cores,
                    profile_clock_mhz=iree_args.profile_clock_mhz,
                    quant=iree_args.quant,
                )
                iree_path = emit_iree_profile(
                    profile, name, per_pool_args, backend_tag)
                if iree_path:
                    print(f"  [{tag}] iree_profile -> {iree_path}")
    print()
    print(f"OVERALL: {'PASS' if all_ok else 'FAIL'} "
          f"({len(models) * len(pool_sizes)} runs)")
    return all_ok


def report_run(text: str, *, models: Optional[list[str]],
               io_path: Optional[str], quant: str,
               atol: Optional[float], rtol: Optional[float],
               profile_csv: Optional[str],
               iree_args: IREEProfileArgs, backend_tag: str,
               repo_root: str,
               quants: Optional[list[str]] = None,
               model_name: Optional[str] = None) -> bool:
    """Single entry point used by both runners after they've captured
    the harness stdout. Walks the OUTPUT/PROFILE/WALL blocks, compares
    each against its golden, prints summaries, writes per-model CSV,
    and emits IREE-shape profile data when configured. Returns overall
    PASS/FAIL boolean.

    Multi-quant: when `quants` is given, it's a parallel list to
    `models` providing each model's quant for golden-path resolution.
    Falls back to the single `quant` arg when not given. Useful for
    mixed-quant binaries (int8 dronet + fp32 mlp_control in one
    xpurt_demo build)."""
    if models:
        names = [n.strip() for n in models if n.strip()]
        all_ok = True
        for i, name in enumerate(names):
            print(f"\n--- model: {name} ---")
            # Prefer the in-binary verify summary when present (modern
            # harness — saves shipping the full output tensor over UART);
            # fall back to per-element parse_output for legacy binaries.
            verify = parse_verify(text, tag=name)
            actual = None if verify is not None else parse_output(text, tag=name)
            per_model_quant = (quants[i] if quants and i < len(quants)
                               else quant)
            golden_path = model_io_path(repo_root, name, per_model_quant)
            if not os.path.exists(golden_path):
                print(f"FAIL: golden not found at {golden_path}")
                all_ok = False
                continue
            ok, _ = check_one(actual, golden_path, atol, rtol,
                              label=f"  [{name}] ", verify=verify)
            all_ok = all_ok and ok
            profile = parse_profile(text, tag=name)
            if profile is not None:
                csv_path = os.path.join(
                    os.path.dirname(golden_path), f"{name}_profile.csv"
                )
                write_profile_csv(profile, csv_path)
                print(f"  [{name}] profile -> {csv_path}")
                print_profile_summary(profile)
            wall = parse_wall_cycles(text, tag=name)
            if wall is not None:
                print(f"  [{name}] wall_clock_cycles={wall} (mtime)")
            iree_path = emit_iree_profile(profile or [], name,
                                          iree_args, backend_tag)
            if iree_path:
                print(f"  [{name}] iree_profile -> {iree_path}")
        print()
        print(f"OVERALL: {'PASS' if all_ok else 'FAIL'} ({len(names)} models)")
        return all_ok

    # Single-model.
    if not io_path:
        raise ValueError("io_path required for single-model mode")
    verify = parse_verify(text)
    actual = None if verify is not None else parse_output(text)
    ok, _ = check_one(actual, io_path, atol, rtol, verify=verify)
    # Prefer the median across MODELBLASTER_ITERS repetitions when the run
    # produced them; fall back to the single block otherwise.
    profile = parse_profile_reps(text) or parse_profile(text)
    if profile is not None:
        out_csv = profile_csv or os.path.join(
            os.path.dirname(os.path.abspath(io_path)), "profile.csv"
        )
        write_profile_csv(profile, out_csv)
        print()
        print(f"profile -> {out_csv}")
        print_profile_summary(profile)
    wall = parse_wall_cycles(text)
    if wall is not None:
        print(f"wall_clock_cycles={wall} (mtime)")
    if profile is not None and iree_args.profile_out_root:
        # The model name decides where the profile lands in the tree. It used to
        # be derived purely by walking three directories up from io.npz, which
        # silently mislabels any layout other than
        # examples/<model>/<quant>/generated/io.npz -- a K1 build rooted at
        # build/k1/<model>/<quant>/ filed every profile under "k1". An explicit
        # name wins; the walk remains the default for existing callers.
        if model_name is None:
            io_abs = os.path.abspath(io_path)
            model_name = os.path.basename(
                os.path.dirname(os.path.dirname(os.path.dirname(io_abs)))
            )
        iree_path = emit_iree_profile(profile, model_name, iree_args, backend_tag)
        if iree_path:
            print(f"iree_profile -> {iree_path}")
    return ok
