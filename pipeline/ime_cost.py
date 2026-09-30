"""Shared IME-vs-RVV cost model — "does the K1 IME beat RVV for this shape?".

Single source of truth for the only-if-better rule, used by (1) the kernel
picker's shape-aware guard (generate_kernels._probe_swap) so it does not FORCE
the IME kernel on an op-kind whose shapes lose, and (2) apply_ime_hint.py.

conv: MEASURED lookup keyed by (IC,IH,IW,OC,KH,KW) — the conv win tracks
K=IC*KH*KW and N=OC (4x8 tile fill + pack amortization), NOT M=OH*OW, so it is
never modelled. Unmeasured conv shapes return None -> treated as "not known
better" -> stays RVV.

ONE TABLE PER CONV OP-KIND, because the baseline is part of the measurement.
A speedup is IME-vs-*the RVV kernel that would otherwise run*, and the deployed
graph runs the FUSED RVV kernel for `conv2d_batchnorm2d_silu_s8`, not the
standalone conv. Reading the standalone table for the fused op compares against
a kernel the build never executes: on the deployed yolov8_nano_64x96 that is the
difference between 8 winning shapes and 22 of 28 (l2.m0.cv1 measures 0.41x
standalone and 1.53x fused). So:

    conv2d_s8                     -> artifacts/ime_conv/ime_vs_rvv_conv.csv
    conv2d_batchnorm2d_silu_s8    -> artifacts/ime_fused_conv/ime_vs_rvv_fused_conv.csv

and any OTHER conv2d* op has no table at all -> None -> stays RVV. That is the
only-if-better rule doing its job: there is no IME kernel for those ops, and
borrowing another op's numbers is how the wrong-baseline bug got in.

matmul/linear: the M-anchor curve, itself MEASURED on the K1 board
(kernels/ime/ime_matmul header): M=7->0.25x, 64->1.43x, 128->2.30x, ceil 3.33x.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Tuple

_ANCHORS: List[Tuple[float, float]] = [(1.0, 0.12), (7.0, 0.25), (64.0, 1.43), (128.0, 2.30)]
_CEILING = 3.33
_ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "artifacts")
# Kept as a module attribute: callers (and the docs) refer to "the measured conv
# table" by this name, and it is still the table for the standalone conv.
_MEASURED_CSV = os.path.join(_ARTIFACTS, "ime_conv", "ime_vs_rvv_conv.csv")
_FUSED_CSV = os.path.join(_ARTIFACTS, "ime_fused_conv", "ime_vs_rvv_fused_conv.csv")

#: conv op-kind -> the CSV measured against THAT op's RVV kernel. An op absent
#: here has no measured table and is never placed on the IME.
CONV_TABLES = {
    "conv2d_s8": _MEASURED_CSV,
    "conv2d_batchnorm2d_silu_s8": _FUSED_CSV,
}

INT8_MATMUL = ("linear", "linear_s8", "matmul_s8", "matmul")
# The K1 IME (smt.vmadot) is an INT8 matrix engine — there is NO fp16 matrix
# hardware (cores/spacemit_k1.json). So an fp16 matmul/linear can only reach the
# matrix engine by requantizing to int8, which trades accuracy and MUST be gated
# by the model's accuracy contract (notes/vint_mixed_precision_experiments.md) —
# never a free/silent win.
F16_MATMUL = ("linear_f16", "matmul_f16")
MATMUL_CLASS = INT8_MATMUL + F16_MATMUL


def _matmul_speedup(m: float) -> float:
    if m <= _ANCHORS[0][0]:
        return _ANCHORS[0][1]
    for (m0, s0), (m1, s1) in zip(_ANCHORS, _ANCHORS[1:]):
        if m <= m1:
            return s0 + (s1 - s0) * (m - m0) / (m1 - m0)
    m_last, s_last = _ANCHORS[-1]
    return min(_CEILING, s_last + (_CEILING - s_last) * (1.0 - m_last / m))


def _load_measured_conv(path: str = _MEASURED_CSV) -> Dict[Tuple[int, ...], float]:
    """(IC,IH,IW,OC,KH,KW) -> measured speedup, from one of the CONV_TABLES.

    A shape measured more than once (the fused table has one row per DISPATCH,
    so a repeated shape appears repeatedly) keeps its WORST speedup: the guard
    must not place a shape on the IME on the strength of its luckiest row.
    """
    out: Dict[Tuple[int, ...], float] = {}
    try:
        for r in csv.DictReader(open(path)):
            key = tuple(int(r[k]) for k in ("IC", "IH", "IW", "OC", "KH", "KW"))
            sp = float(r["speedup"])
            out[key] = sp if key not in out else min(out[key], sp)
    except Exception:
        pass
    return out


_MEASURED_CONV = _load_measured_conv(_MEASURED_CSV)
_CONV_TABLE_CACHE: Dict[str, Dict[Tuple[int, ...], float]] = {}


def measured_conv_table(op: str) -> Optional[Dict[Tuple[int, ...], float]]:
    """The measured table for a conv op-kind, or None when it has none."""
    path = CONV_TABLES.get(op)
    if path is None:
        return None
    if op not in _CONV_TABLE_CACHE:
        _CONV_TABLE_CACHE[op] = _load_measured_conv(path)
    return _CONV_TABLE_CACHE[op]


def ime_speedup_for(op: str, shape: Dict[str, int],
                    allow_int8_requant: bool = False) -> Tuple[Optional[float], str]:
    """(speedup, provenance) — None speedup = unknown/not-eligible (=> RVV).

    fp16 matmul/linear reach the int8 IME only via requant: eligible ONLY when
    `allow_int8_requant=True` (the caller has checked the accuracy contract);
    otherwise they stay on RVV fp16 no matter how favorable M is."""
    if op in F16_MATMUL:
        if not allow_int8_requant:
            return None, "fp16: IME needs int8 requant (accuracy-gated) — kept on RVV"
        m = shape.get("M")
        if m is None:
            return None, "no-M"
        return _matmul_speedup(float(m)), "int8-requant (accuracy-gated)"
    if op in INT8_MATMUL:
        m = shape.get("M")
        if m is None:
            return None, "no-M"
        return _matmul_speedup(float(m)), "measured-anchor"
    if op.startswith("conv2d"):
        table = measured_conv_table(op)
        if table is None:
            return None, f"no measured IME-vs-RVV table for {op} -- stays RVV"
        try:
            key = (shape["IC"], shape["IH"], shape["IW"], shape["OC"], shape["KH"], shape["KW"])
        except KeyError:
            return None, "conv-shape-incomplete"
        if key in table:
            return table[key], ("measured-fused" if op != "conv2d_s8" else "measured")
        return None, "unmeasured"
    return None, "not-matmul-class"


def _macs(op: str, s: Dict[str, int]) -> float:
    """Relative work per instance (proxy for RVV cost), to cycle-weight a mixed
    op-kind's aggregate. matmul: M*K*N. (conv is not aggregated here — see below.)"""
    if op in MATMUL_CLASS:
        return float(s.get("M", 1) * s.get("K", 1) * s.get("N", 1))
    return 1.0


def ime_wins_aggregate(op: str, shapes: List[Dict[str, int]],
                       allow_int8_requant: bool = False) -> Tuple[bool, str]:
    """Per-op-KIND verdict for the picker, which must commit ONE kernel to every
    instance of the op. Two regimes:

      * matmul/linear — decide on the WORK-WEIGHTED net benefit across the op's
        shapes: sum_i work_i*(1 - 1/speedup_i) > 0. A few tiny M=1 linears do not
        veto IME when the M=128 GEMMs dominate the cycles (ffn), but a uniformly
        small-M op (attention, M=8) nets negative and stays RVV.
      * conv — INHERENTLY per-layer mixed (yolov8: 20 layers win, 20 lose), so a
        single op-kind choice is wrong for half of them. The picker leaves conv
        on RVV and defers to the per-dispatch scheduler (apply_ime_hint), which
        routes each conv layer to IME only where it measured faster.
    """
    if not shapes:
        return False, "no-shapes"
    if op.startswith("conv2d"):
        return False, "conv is per-layer mixed -> deferred to per-dispatch scheduler"
    net = 0.0
    known = 0
    for s in shapes:
        sp, _ = ime_speedup_for(op, s, allow_int8_requant=allow_int8_requant)
        if sp is None:
            continue
        known += 1
        net += _macs(op, s) * (1.0 - 1.0 / sp)
    if known == 0:
        return False, "no shape has a known IME speedup"
    if net > 0:
        return True, f"work-weighted net IME benefit > 0 across {known} shape(s)"
    return False, f"work-weighted net IME benefit <= 0 across {known} shape(s) -> RVV"


# back-compat alias
ime_wins_all = ime_wins_aggregate


def ime_useful(op: str, shapes: List[Dict[str, int]],
               allow_int8_requant: bool = False) -> Tuple[bool, str]:
    """Table-inclusion verdict for the MULTI-impl build: keep the IME kernel in
    the ime_x60 table iff it is KNOWN faster than RVV on AT LEAST ONE shape of
    this op. The per-dispatch scheduler (comparing measured rvv vs ime profiles)
    then decides WHICH instances actually run on IME — so a losing instance is
    never forced onto IME, and a winning one is not stripped away.

    This is the right guard for the end-to-end runtime (both rvv+ime tables
    linked, walker dispatches by `impl`): attention (uniformly slower) is
    excluded, but conv (20/40 yolov8 layers win) and ffn stay AVAILABLE."""
    if not shapes:
        return False, "no-shapes"
    best = None
    for s in shapes:
        sp, _ = ime_speedup_for(op, s, allow_int8_requant=allow_int8_requant)
        if sp is not None and (best is None or sp > best):
            best = sp
    if best is not None and best > 1.0:
        return True, f"IME faster on >=1 shape (best {best:.2f}x) -> keep in ime table"
    return False, "IME not faster on any known shape -> excluded from ime table"


# ---------------------------------------------------------------------------
# The explicit override: MB_IME_FORCE
# ---------------------------------------------------------------------------
#
# Everything above is the only-if-better rule: an (op, shape) reaches the matrix
# engine only where a MEASURED table says the engine is faster than the RVV
# kernel that would otherwise run, and anything unmeasured stays on RVV. That is
# the right default and it is what every build in this repo ships with.
#
# It is not, however, what an engineer who has just been handed an accelerator
# and no per-shape profiling would do. That engineer turns the accelerator on
# for everything that has a kernel for it. `MB_IME_FORCE=1` builds exactly that
# deployment, so it can be measured rather than argued about:
#
#     MB_IME_FORCE=1 python -m modelblaster.pipeline.generate_kernels ... --target ime_x60
#
# It makes the picker take the IME kernel for EVERY op that HAS one, whatever
# the tables say -- including ops the tables have never measured, and ops the
# tables measure as LOSSES. The build is therefore no longer table-guided, and
# says so in `kernel_picks.json` (`ime_force`, `table_guided: false`,
# `ime_forced_ops`) and in the generator log.
#
# Two things it does NOT change:
#   * the numerics contract -- a forced kernel is still verified like any other,
#     and an IME kernel that is not bit-exact with its RVV counterpart is still
#     a defect, not a tradeoff;
#   * the PLACEMENT contract -- `smt.vmadot` executes on cluster 0 and raises
#     SIGILL on cluster 1 (artifacts/ime_isa_probe/FINDINGS.md), so a forced
#     build can only run on harts 0-3. Forcing the kernel does not make the
#     instruction legal anywhere it was not legal before.
#
# It is a BUILD switch and nothing else reads it: `apply_ime_hint.py` and
# `scripts/make_ime_profile.py` stay table-guided whether or not it is set, so
# a per-dispatch schedule is never silently forced along with a build. What a
# forced build measures, and the deployment it stands for, is in
# docs/Baselines/ros_with_ime.md.
FORCE_ENV = "MB_IME_FORCE"

FORCE_NOTE = (
    "BUILD IS NOT TABLE-GUIDED: MB_IME_FORCE=1 selected the K1 IME kernel for "
    "every op that has one, bypassing the measured IME-vs-RVV tables, so ops "
    "the tables measure as LOSSES and ops they have never measured are on the "
    "matrix engine too. This build executes smt.vmadot and therefore runs on "
    "cluster 0 (harts 0-3) ONLY -- it raises SIGILL on harts 4-7."
)

_FORCE_FALSEY = {"", "0", "false", "False", "FALSE", "no", "off"}


def force_requested() -> bool:
    """True when MB_IME_FORCE asks for a blanket-IME build (default: False)."""
    return os.environ.get(FORCE_ENV, "0") not in _FORCE_FALSEY
