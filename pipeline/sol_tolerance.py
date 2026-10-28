"""SOL-ExecBench's correctness tolerance as an acceptance gate of the flow (docs/sol_gate.md).

SOL's rule (core/bench/correctness.py; docs/sol_execbench.md), per output tensor, the output and the reference in
the problem's native dtype and compared in float64:

  * shape and dtype equal;
  * any inf / NaN in the output fails (unless ``allow_negative_inf`` and both sides are -inf at that element);
  * an all-zero output against a non-zero reference fails;
  * ``|out - ref| <= max_atol + max_rtol * |ref|`` must hold on at least ``required_matched_ratio`` (default 0.99)
    of the elements;
  * ``max_error_cap`` (if set) bounds the maximum error.

The flow's own gate (kb_export's driver) is one scale per tensor (``output_rel * max|golden| + output_abs``); for fp32
problems that is 1.5x-3.5x looser than SOL's ``max_atol`` (1000x+ where it is 1e-5) and a single-pass TF32 winner
slips through it (sol_audit: 92008, 92011). This module is the shared implementation of SOL's rule:

  * :func:`sol_compare` -- numpy / torch, used by the tests, graph_verify, mbir and the CLI;
  * the C++ twin (:data:`DRIVER_SNIPPET`, compiled into the exported driver, unit-tested with a host compiler);
  * where the tolerance and SOL's reference outputs come from (:func:`tolerances_for`, :func:`read_bench_tolerance`,
    :func:`ensure_reference`), and the export / policy state the flow shares (:func:`export_spec`, ``STATE``);
  * a CLI (``python -m modelblaster.pipeline.sol_tolerance --help``).

Everything is OFF by default: with ``--sol-gate off`` (``MB_SOL_GATE`` unset) :func:`export_spec` returns None and an
export is byte-identical to the one without this module.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

DEFAULT_RATIO = 0.99
DEFAULT_ATOL = DEFAULT_RTOL = 1e-2          # graph_verify's / the evaluator's defaults when a key is missing
KEYS = ("max_atol", "max_rtol", "required_matched_ratio", "max_error_cap", "allow_negative_inf")
MODES = ("off", "final", "accept")
REF_FILE = "sol_ref.npz"
FORMAT = "mb-sol/1"
# the SOL dtype names of an output -> the driver's buffer dtypes the contract allows (SOL: output dtype == reference's)
_DTYPE_BUF = {"float32": "f32", "bfloat16": "bf16", "float16": "f16", "int64": "i64", "int32": "i32", "bool": "b8",
              "uint8": "u8"}


# ---------------------------------------------------------------- the rule

def normalize(tol: dict | None) -> dict:
    """SOL_TOLERANCE -> the five keys SOL's model reads. ``required_matched_ratio`` defaults to 0.99; the spelling
    ``required_match_ratio`` (L2 workloads) is ignored by SOL's pydantic model, so it is too (recorded in
    ``ignored_keys``); ``max_atol`` / ``max_rtol`` default to 1e-2 when absent."""
    tol = dict(tol or {})
    out = {"max_atol": float(tol.get("max_atol", DEFAULT_ATOL)),
           "max_rtol": float(tol.get("max_rtol", DEFAULT_RTOL)),
           "required_matched_ratio": DEFAULT_RATIO if tol.get("required_matched_ratio") is None
           else float(tol["required_matched_ratio"]),
           "max_error_cap": None if tol.get("max_error_cap") is None else float(tol["max_error_cap"]),
           "allow_negative_inf": bool(tol.get("allow_negative_inf"))}
    ignored = sorted(k for k in tol if k not in KEYS)
    if ignored:
        out["ignored_keys"] = ignored
    return out


def _np64(x) -> np.ndarray:
    if hasattr(x, "detach"):                                  # a torch tensor (bf16 included)
        x = x.detach().to("cpu").double().numpy()
    return np.asarray(x, np.float64).reshape(-1)


def sol_compare(out, ref, tol: dict | None, *, chunk: int = 1 << 24) -> dict:
    """SOL's correctness check of one output tensor (``out`` against ``ref``; numpy arrays or torch tensors, any
    float dtype, compared in float64). -> {ok, matched_ratio, matched, n, required_ratio, max_abs_err, max_rel_err,
    max_excess, nonfinite_output, all_zero_vs_nonzero_ref, max_error_cap_exceeded}; a shape mismatch is
    {"ok": False, "why": ...}. ``max_abs_err`` is over the finite differences (inf when there is none, as SOL's);
    ``max_rel_err`` is ``|out - ref| / |ref|`` over finite differences at non-zero references; ``max_excess`` the
    worst ``|out - ref| - allowance`` (non-finite counted as 0)."""
    t = normalize(tol)
    o, r = _np64(out), _np64(ref)
    if o.shape != r.shape:
        return {"ok": False, "why": f"shape {o.shape} != {r.shape}"}
    n = int(o.size)
    matched, nonfin, out_nz, ref_nz = 0, False, False, False
    max_abs, max_rel, max_exc, any_fin = 0.0, 0.0, -math.inf, False
    for lo in range(0, n, chunk):
        a, b = o[lo:lo + chunk], r[lo:lo + chunk]
        with np.errstate(invalid="ignore", over="ignore"):
            nf = ~np.isfinite(a)
            neg = (np.isneginf(a) & np.isneginf(b)) if t["allow_negative_inf"] else np.zeros(a.shape, bool)
            d = np.abs(a - b)
            allow = t["max_atol"] + t["max_rtol"] * np.abs(b)
            good = ((d <= allow) | neg) & ~(nf & ~neg)
            slack = d - allow
        matched += int(good.sum())
        nonfin |= bool((nf & ~neg).any())
        out_nz |= bool((a != 0).any())
        ref_nz |= bool((b != 0).any())
        fin = np.isfinite(d)
        if fin.any():
            any_fin = True
            max_abs = max(max_abs, float(d[fin].max()))
            nzr = fin & (b != 0)
            if nzr.any():
                max_rel = max(max_rel, float((d[nzr] / np.abs(b[nzr])).max()))
        s = np.where(np.isfinite(slack), slack, 0.0)
        if s.size:
            max_exc = max(max_exc, float(s.max()))
    ratio = matched / n if n else 1.0
    max_abs = max_abs if (any_fin or n == 0) else math.inf
    allzero = bool(n) and not out_nz and ref_nz
    cap = t["max_error_cap"]
    capbad = cap is not None and max_abs > cap
    ok = ratio >= t["required_matched_ratio"] and not nonfin and not allzero and not capbad
    return {"ok": bool(ok), "matched_ratio": ratio, "matched": matched, "n": n,
            "required_ratio": t["required_matched_ratio"], "max_abs_err": max_abs, "max_rel_err": max_rel,
            "max_excess": 0.0 if max_exc == -math.inf else max_exc, "nonfinite_output": nonfin,
            "all_zero_vs_nonzero_ref": allzero, "max_error_cap_exceeded": bool(capbad)}


def sol_compare_all(outs, refs, tols) -> list[dict]:
    """:func:`sol_compare` per output; ``tols`` one tolerance for all or one per output; a count mismatch fails."""
    outs, refs = list(outs), list(refs)
    if len(outs) != len(refs):
        return [{"ok": False, "why": f"{len(outs)} outputs, the reference has {len(refs)}"}]
    tl = tols if isinstance(tols, (list, tuple)) else [tols] * len(outs)
    return [sol_compare(o, r, t) for o, r, t in zip(outs, refs, tl)]


# ---------------------------------------------------------------- where the tolerance comes from

_ASSIGN = re.compile(r"^SOL_TOLERANCE\s*=", re.M)


def read_bench_tolerance(path) -> dict | None:
    """The bench file's ``SOL_TOLERANCE`` (a literal dict assigned at module level; the file is parsed, not
    imported). None when the file has none (a KernelBench problem)."""
    p = Path(path)
    if not p.is_file():
        return None
    src = p.read_text()
    if not _ASSIGN.search(src):
        return None
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SOL_TOLERANCE"
                                                for t in node.targets):
            try:
                v = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                return None
            return dict(v) if isinstance(v, dict) else None
    return None


def record_in_graph(graph: dict, tol: dict, *, bench: str | None = None, source: str = "bench") -> dict:
    """Put ``graph["sol_tolerance"]``: {format, source, bench, outputs: [tolerance per graph output, in order]}."""
    outs = (graph.get("output") or {})
    n = len(outs.get("tensors") or ([outs["tensor"]] if outs.get("tensor") else [])) or 1
    t = normalize(tol)
    rec = {"format": FORMAT, "source": source, **({"bench": str(bench)} if bench else {}),
           "outputs": [{k: t[k] for k in KEYS} for _ in range(n)]}
    if t.get("ignored_keys"):
        rec["ignored_keys"] = t["ignored_keys"]
    graph["sol_tolerance"] = rec
    return rec


def record_in_ir(ir_dir, bench) -> dict | None:
    """Add ``sol_tolerance`` to ``ir_dir``/graph.json from the bench's SOL_TOLERANCE (extract_graph does this after a
    ``--bench-file`` extraction; the CLI ``record`` does it for an IR extracted earlier). None: the bench has none."""
    tol = read_bench_tolerance(bench)
    if tol is None:
        return None
    gp = Path(ir_dir) / "graph.json"
    g = json.loads(gp.read_text())
    rec = record_in_graph(g, tol, bench=os.path.basename(str(bench)))
    gp.write_text(json.dumps(g, indent=2))
    return rec


def _env_override() -> dict | None:
    """``$MB_SOL_TOLERANCE``: a JSON dict, ``@file.json`` or a bench file -- overrides what the IR recorded."""
    v = os.environ.get("MB_SOL_TOLERANCE")
    if not v:
        return None
    v = v.strip()
    try:
        if v.startswith("@"):
            return json.loads(Path(v[1:]).read_text())
        return json.loads(v) if v.startswith("{") else read_bench_tolerance(v)
    except (OSError, ValueError):
        return None


def _env_bench() -> dict | None:
    """``$MB_SOL_BENCH``: a bench file's SOL_TOLERANCE, for an IR extracted before the tolerance was recorded."""
    b = os.environ.get("MB_SOL_BENCH")
    return read_bench_tolerance(b) if b else None


def tolerances_for(graph: dict, n_outputs: int | None = None) -> list[dict] | None:
    """One normalized tolerance per graph output, or None when the problem has no SOL tolerance (nothing to gate).
    Order: ``$MB_SOL_TOLERANCE`` (an explicit override: a JSON dict, ``@file.json`` or a bench file), then what the
    IR recorded (``graph["sol_tolerance"]``), then ``$MB_SOL_BENCH`` (a bench file, for an IR extracted before the
    tolerance was recorded)."""
    n = n_outputs or len(((graph.get("output") or {}).get("tensors")) or [1])
    env = _env_override()
    if env is not None:
        return [normalize(env)] * n
    rec = graph.get("sol_tolerance")
    outs = (rec or {}).get("outputs") or []
    if outs:
        return [normalize(outs[i] if i < len(outs) else outs[-1]) for i in range(n)]
    env = _env_bench()
    return None if env is None else [normalize(env)] * n


# ---------------------------------------------------------------- SOL's reference outputs

def bench_path_for(ir_dir) -> Path | None:
    """The bench file of an IR: ``$MB_SOL_BENCH``; the farm layout ``<problem dir>/problem.py`` beside ``irs/<name>``;
    ``<ir>/sol_bench.py``."""
    b = os.environ.get("MB_SOL_BENCH")
    if b and os.path.isfile(b):
        return Path(b)
    d = Path(ir_dir).resolve()
    for c in (d / "sol_bench.py", d.parent.parent / "problem.py"):
        if c.is_file() and read_bench_tolerance(c) is not None:
            return c
    return None


def ensure_reference(ir_dir, bench=None, *, device: str = "auto", force: bool = False) -> Path | None:
    """Write ``ir_dir``/sol_ref.npz: the bench's ``sol_reference_outputs(None, device)`` (SOL's reference: native
    dtypes and torch defaults, on the GPU where there is one: matmul TF32 off, cuDNN TF32 on), one float32 array per
    output (exact for bf16 / fp16 / fp32 values) and ``meta`` (JSON: dtypes, device, torch / cuDNN versions). Returns the
    path; None when there is no bench or the call fails (the gate then compares with the MB golden)."""
    ir_dir = Path(ir_dir)
    dst = ir_dir / REF_FILE
    if dst.exists() and not force:
        return dst
    bench = Path(bench) if bench else bench_path_for(ir_dir)
    if bench is None or not Path(bench).is_file():
        return None
    try:
        import importlib.util

        import torch
        spec = importlib.util.spec_from_file_location("_sol_bench_ref", str(bench))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "sol_reference_outputs"):
            return None
        dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        outs = mod.sol_reference_outputs(None, dev)
        arrs, dts = {}, []
        for i, t in enumerate(outs):
            dts.append(str(t.dtype).replace("torch.", ""))
            arrs[f"out{i}"] = t.detach().to("cpu").float().numpy()
        meta = {"device": dev, "dtypes": dts, "torch": torch.__version__,
                "cudnn": getattr(torch.backends.cudnn, "version", lambda: None)() if dev == "cuda" else None,
                "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None, "bench": os.path.basename(str(bench))}
        np.savez(dst, meta=np.array(json.dumps(meta)), **arrs)
        return dst
    except Exception as e:  # noqa: BLE001 -- best effort: the gate falls back to the golden
        print(f"[sol_tolerance] no SOL reference for {ir_dir}: {type(e).__name__}: {str(e)[:300]}", file=sys.stderr,
              flush=True)
        return None


def load_reference(ir_dir=None, ir_name: str | None = None) -> dict | None:
    """{"arrays": [float32 arrays in output order], "dtypes": [...], "device": "cuda"|"cpu", "path": str}: the file
    in ``ir_dir`` (a variant dir symlinks it), else the registered one of ``ir_name`` (``STATE.refs``)."""
    cands = []
    if ir_dir is not None:
        cands.append(Path(ir_dir) / REF_FILE)
    if ir_name and ir_name in STATE.refs:
        cands.append(Path(STATE.refs[ir_name]))
    for c in cands:
        if c.exists():
            z = np.load(c, allow_pickle=False)
            meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
            keys = sorted((k for k in z.files if k.startswith("out")), key=lambda k: int(k[3:]))
            return {"arrays": [z[k] for k in keys], "dtypes": meta.get("dtypes") or [None] * len(keys),
                    "device": meta.get("device"), "path": str(c.resolve()), "meta": meta}
    return None


# ---------------------------------------------------------------- shared state of the flow (one process)

class _State:
    """Settings (from the flow's flags / ``MB_SOL_*`` env) and the per-IR policy the gate decides at run time."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.mode_cfg: str | None = None         # None: read $MB_SOL_GATE
        self.slot_frac: float | None = None
        self.golden_margin: float | None = None
        self.bound: str | None = None
        self.policy: dict[str, dict] = {}        # ir name -> {"outputs": {index: {ref, margin, required}}, "status": ...}
        self.refs: dict[str, str] = {}           # ir name -> sol_ref.npz path (set by the gate for variant IR dirs)
        self.stack: list[str | None] = []        # scoped overrides: "enforce" | "report" | None

    @property
    def mode(self) -> str:
        m = (self.mode_cfg if self.mode_cfg is not None else os.environ.get("MB_SOL_GATE", "off") or "off").strip().lower()
        m = {"0": "off", "no": "off", "false": "off", "1": "accept", "on": "accept", "yes": "accept"}.get(m, m)
        if m not in MODES:
            raise ValueError(f"--sol-gate {m!r}: expected one of {', '.join(MODES)}")
        return m

    def _f(self, v, env, dflt):
        return float(v if v is not None else os.environ.get(env, dflt))

    @property
    def frac(self) -> float:
        """Share of the output's tolerance a single fp32 slot may use (0: no derived slot bound)."""
        return self._f(self.slot_frac, "MB_SOL_SLOT_FRAC", 0.5)

    @property
    def margin(self) -> float:
        """Scale of SOL's atol / rtol when the gate compares with the MB golden instead of SOL's own reference."""
        return self._f(self.golden_margin, "MB_SOL_GOLDEN_MARGIN", 0.5)

    @property
    def bound_mode(self) -> str:
        b = (self.bound or os.environ.get("MB_SOL_BOUND", "and")).strip().lower()
        if b not in ("and", "replace"):
            raise ValueError(f"MB_SOL_BOUND {b!r}: and | replace")
        return b


STATE = _State()


def mode() -> str:
    return STATE.mode


@contextlib.contextmanager
def scope(how: str | None):
    """Force how exports inside the block gate: "enforce" (a failed SOL check fails the run), "report" (the check is
    printed, never fails the run) or None (no SOL tables at all: the plain flow)."""
    STATE.stack.append(how)
    try:
        yield
    finally:
        STATE.stack.pop()


def effective(ir_name: str) -> str | None:
    """None | "report" | "enforce" for an export of ``ir_name`` now."""
    m = STATE.mode
    if m == "off":                                  # off is off, whatever scope a caller opened
        return None
    if STATE.stack:
        return STATE.stack[-1]
    if m == "accept":
        return "enforce" if ir_name in STATE.policy else "report"    # until the seed was measured: report only
    return None                                                      # final: only the gate's own scopes carry tables


# ---------------------------------------------------------------- export: manifest block + driver tables

def output_stats(arr, chunk: int = 1 << 24) -> dict:
    """max / mean |value| of a golden output (finite values), for the slot bound derived from the tolerance (chunked: no
    float64 copy of a big output)."""
    a = arr.detach().to("cpu").numpy() if hasattr(arr, "detach") else np.asarray(arr)
    a = a.reshape(-1)
    mx, tot, n = 0.0, 0.0, 0
    for lo in range(0, a.size, chunk):
        c = np.abs(a[lo:lo + chunk].astype(np.float32 if a.dtype.itemsize <= 4 else np.float64))
        c = c[np.isfinite(c)]
        if c.size:
            mx = max(mx, float(c.max()))
            tot += float(c.sum(dtype=np.float64))
            n += int(c.size)
    return {"max": mx, "mean": tot / n if n else 0.0}


def slot_cap_for(ir, graph: dict, env: dict) -> float | None:
    """The derived fp32 slot bound (:func:`slot_cap`) of an export of ``ir`` (``env``: the walk's tensors, which has every graph
    output whatever partition is exported): accept mode only, once the seed was measured (the policy exists), every export of an IR
    with a SOL tolerance -- a KernelBlaster problem of one op included, whose own output is not a graph output."""
    if STATE.mode != "accept" or effective(ir.name) != "enforce":       # not before the seed was measured (the probe, round 0)
        return None
    names = list(ir.output_names())
    tols = tolerances_for(graph, len(names))
    if tols is None:
        return None
    return slot_cap(tols, [output_stats(env[t]) if t in env else {"max": 0.0, "mean": 0.0} for t in names], STATE.frac)


def slot_cap(tols: list[dict], stats: list[dict], frac: float, floor: float = 2e-5) -> float | None:
    """A conservative per-slot relative bound for fp32 slots from the output tolerances: a slot may use ``frac`` of the
    tightest output's typical per-element allowance relative to that output's scale,
    ``(max_atol + max_rtol * mean|ref|) / max|ref|``; never below ``floor`` (fp32 kernels' own noise: a legitimate
    fp32 kernel reaches ~1e-6 of the scale). A heuristic pre-filter (sensitivity of the output to the slot is
    unknown), not the guarantee: the whole-model SOL gate is. None when ``frac`` <= 0 or nothing is known."""
    if frac <= 0:
        return None
    eps = []
    for t, s in zip(tols, stats):
        if s["max"] > 0:
            eps.append((t["max_atol"] + t["max_rtol"] * s["mean"]) / s["max"])
    return max(floor, frac * min(eps)) if eps else None


def export_spec(ir, graph: dict, out_names: list[str], buffers: list[dict]) -> dict | None:
    """The ``sol`` block of a manifest and the data the driver tables are made from, or None when the export is not
    gated (mode off, no SOL tolerance, no graph output in the partition).

    ``ir`` is a mbir.IR (its ``graph`` carries ``sol_tolerance``, its directory the reference), ``out_names`` the graph
    outputs in order, ``buffers`` the manifest buffers (outputs get their index).
    Keys: ``enforce`` (bool), ``bound`` ("and" | "replace"), ``outputs`` {buffer name: {index, tolerance, ref (the
    SOL reference's element count or None), use ("sol" | "golden"), margin, required}}, ``status``, ``warnings``; ``_ref`` {buffer name: float32 array} is the reference data for data.bin (the writer pops it)."""
    how = effective(ir.name)
    if how is None:
        return None
    tols = tolerances_for(graph, len(out_names))
    if tols is None:
        return None
    pol = STATE.policy.get(ir.name) or {}
    ref = load_reference(getattr(ir, "dir", None), ir.name)
    spec = {"format": FORMAT, "enforce": how == "enforce", "bound": STATE.bound_mode, "outputs": {}, "_ref": {},
            "warnings": [], "mode": STATE.mode}
    if ref is not None:
        spec["reference"] = {"path": ref["path"], "device": ref["device"], "dtypes": ref["dtypes"]}
    for b in buffers:
        if b["kind"] != "output" or b["name"] not in out_names:
            continue
        i = out_names.index(b["name"])
        t = tols[i]
        o = {"index": i, "tolerance": {k: t[k] for k in KEYS}, "ref": None}
        use = (pol.get("outputs", {}).get(str(i)) or pol.get("outputs", {}).get(i) or {})
        rarr = None
        if ref is not None and i < len(ref["arrays"]):
            rarr = ref["arrays"][i]
            dt = (ref["dtypes"][i] if i < len(ref["dtypes"]) else None)
            if rarr.size != int(b["numel"]):
                spec["warnings"].append(f"{b['name']}: the SOL reference has {rarr.size} elements, the buffer {b['numel']}: "
                                        "not used")
                rarr = None
            elif dt in _DTYPE_BUF and _DTYPE_BUF[dt] != b["dtype"]:
                spec["warnings"].append(f"{b['name']}: SOL's reference is {dt}, the buffer {b['dtype']} (SOL's dtype "
                                        "contract fails such an output): reference not used")
                rarr = None
            elif ref.get("device") != "cuda":
                spec["warnings"].append(f"{b['name']}: the SOL reference ran on the CPU (no TF32 as SOL's GPU "
                                        "reference): it is treated as the golden's twin")
        if rarr is not None:
            o["ref"] = int(rarr.size)
            spec["_ref"][b["name"]] = np.ascontiguousarray(rarr, np.float32).reshape(-1)
        want = use.get("ref") or ("sol" if rarr is not None and ref.get("device") == "cuda" else "golden")
        if want == "sol" and rarr is None:
            want = "golden"
        o["use"] = want
        o["margin"] = float(use.get("margin", 1.0 if want == "sol" else STATE.margin))
        o["required"] = float(use.get("required", t["required_matched_ratio"]))
        spec["outputs"][b["name"]] = o
    if not spec["outputs"]:
        return None
    spec["status"] = pol.get("status", "unmeasured")
    return spec


def public(spec: dict | None) -> dict | None:
    """``spec`` without the reference data (what the manifest records)."""
    return None if spec is None else {k: v for k, v in spec.items() if not k.startswith("_")}


# C++ twin of sol_compare: standalone (no CUDA), so a unit test compiles it with the host compiler. The exported
# driver includes it verbatim.
DRIVER_SNIPPET = r'''
// ---- SOL-ExecBench's correctness rule (modelblaster pipeline/sol_tolerance.py: sol_compare's twin) -----------------
struct MbSolTol { double atol, rtol, ratio, cap; int neginf; };                   // cap < 0: no max_error_cap
struct MbSolPart { long matched = 0; int nonfinite = 0, out_nz = 0, ref_nz = 0, any_fin = 0;
                   double max_abs = 0, max_rel = 0, max_exc = -INFINITY; };
struct MbSolVerdict { long n = 0, matched = 0; double ratio = 1, max_abs = 0, max_rel = 0, max_excess = 0;
                      int nonfinite = 0, allzero = 0, cap_bad = 0, ok = 1; };
static void mb_sol_chunk(const float *a, const float *r, long lo, long hi, const MbSolTol &t, MbSolPart &p) {
    for (long i = lo; i < hi; ++i) {
        double o = a[i], f = r[i];
        bool nf = !std::isfinite(o);
        bool nn = t.neginf && std::isinf(o) && o < 0 && std::isinf(f) && f < 0;     // both -inf, allowed
        double d = std::fabs(o - f), allow = t.atol + t.rtol * std::fabs(f);
        bool good = (d <= allow) || nn;                                          // NaN compares false: unmatched
        if (nf && !nn) { good = false; p.nonfinite = 1; }
        if (good) ++p.matched;
        if (o != 0.0) p.out_nz = 1;
        if (f != 0.0) p.ref_nz = 1;
        if (std::isfinite(d)) {
            p.any_fin = 1;
            p.max_abs = std::max(p.max_abs, d);
            if (f != 0.0) p.max_rel = std::max(p.max_rel, d / std::fabs(f));
        }
        double s = d - allow;
        if (!std::isfinite(s)) s = 0.0;
        p.max_exc = std::max(p.max_exc, s);
    }
}
static MbSolVerdict mb_sol_compare(const float *a, const float *r, long n, const MbSolTol &t) {
    unsigned nt = std::min(8u, std::max(1u, std::thread::hardware_concurrency()));
    if (n < (1L << 20)) nt = 1;
    std::vector<MbSolPart> part(nt);
    if (nt == 1) mb_sol_chunk(a, r, 0, n, t, part[0]);
    else {
        std::vector<std::thread> th;
        long step = (n + nt - 1) / nt;
        for (unsigned k = 0; k < nt; ++k)
            th.emplace_back([&, k] { long lo = k * step, hi = std::min(n, lo + step); if (lo < hi) mb_sol_chunk(a, r, lo, hi, t, part[k]); });
        for (auto &x : th) x.join();
    }
    MbSolPart s;
    for (auto &p : part) {
        s.matched += p.matched; s.nonfinite |= p.nonfinite; s.out_nz |= p.out_nz; s.ref_nz |= p.ref_nz; s.any_fin |= p.any_fin;
        s.max_abs = std::max(s.max_abs, p.max_abs); s.max_rel = std::max(s.max_rel, p.max_rel); s.max_exc = std::max(s.max_exc, p.max_exc);
    }
    MbSolVerdict v;
    v.n = n; v.matched = s.matched; v.ratio = n ? (double)s.matched / (double)n : 1.0;
    v.max_abs = (s.any_fin || n == 0) ? s.max_abs : (double)INFINITY;
    v.max_rel = s.max_rel; v.max_excess = s.max_exc == -INFINITY ? 0.0 : s.max_exc;
    v.nonfinite = s.nonfinite; v.allzero = (n > 0 && !s.out_nz && s.ref_nz);
    v.cap_bad = t.cap >= 0 && v.max_abs > t.cap;
    v.ok = v.ratio >= t.ratio && !v.nonfinite && !v.allzero && !v.cap_bad;
    return v;
}
'''

# the driver part that needs the blob / tables (only emitted for a gated export); `tag` as check_outputs'
DRIVER_CHECK = r'''
// SOL's rule on a graph output: against SOL's own reference outputs (when the problem carries them) and against the MB
// golden at SOL's tolerance scaled by B_SOL_MARGIN; B_SOL_USE says which one gates. Prints one MB_SOL line each.
static bool mb_sol_buffer(const char *tag, int b, const float *a, const float *golden, bool flow_ok) {
    if (env_num("MB_SOL", 1) == 0) return flow_ok;
    bool gate_ok = true, gated = false;
    static const char *REF[] = {"sol", "golden", "golden1"};
    for (int k = 0; k < 3; ++k) {                      // 0: SOL's reference, 1: the golden (x margin), 2: the golden at SOL's own atol / rtol
        MbSolTol t = {B_SOL_ATOL[b], B_SOL_RTOL[b], B_SOL_RATIO[b], B_SOL_CAP[b], B_SOL_NEGINF[b]};
        const float *r;
        if (k == 2 && B_SOL_MARGIN[b] == 1.0) continue;                 // the same check as k == 1
        if (k == 0) { if (B_SOL_REF[b] < 0) continue; r = (const float *)(mb_blob + B_SOL_REF[b]); }
        else { r = golden; if (k == 1) { t.atol *= B_SOL_MARGIN[b]; t.rtol *= B_SOL_MARGIN[b]; } }
        int gate = k < 2 && (k == 0) == (B_SOL_USE[b] == 1);
        if (gate) t.ratio = B_SOL_REQ[b];
        MbSolVerdict v = mb_sol_compare(a, r, B_NUMEL[b], t);
        printf("MB_SOL %s ref=%s matched=%.6f required=%.6g max_abs_err=%.6g max_rel_err=%.6g max_excess=%.6g nonfinite=%d "
               "allzero=%d cap_exceeded=%d n=%ld atol=%.6g rtol=%.6g ok=%d gate=%d\n", B_NAME[b], REF[k],
               v.ratio, t.ratio, v.max_abs, v.max_rel, v.max_excess, v.nonfinite, v.allzero, v.cap_bad, v.n, t.atol, t.rtol,
               (int)v.ok, gate);
        if (gate) { gate_ok = v.ok; gated = true; }
    }
    bool enforce = env_num("MB_SOL_ENFORCE", MB_SOL_ENFORCE) != 0;
    if (!enforce || !gated) return flow_ok;
    (void)tag;
    return B_SOL_REPLACE ? gate_ok : (flow_ok && gate_ok);
}
'''


def _num(v) -> str:
    v = float(v)
    return "INFINITY" if math.isinf(v) else repr(v)


def driver_parts(spec: dict, buffers: list[dict], blob_offsets: dict[str, int]) -> tuple[str, str]:
    """(tables, code) the driver template splices in for a gated export. ``blob_offsets`` {buffer: byte offset of its
    SOL reference in data.bin}."""
    nb = len(buffers)
    on = [spec["outputs"].get(b["name"]) if b["kind"] == "output" else None for b in buffers]

    def arr(name, ctype, vals):
        return f"static const {ctype} {name}[] = {{{', '.join(str(v) for v in (list(vals) or [0]))}}};"

    tables = "\n".join([
        f"static const int MB_SOL_ENFORCE = {int(spec['enforce'])}, B_SOL_REPLACE = {int(spec['bound'] == 'replace')};",
        arr("B_SOL", "int", (int(o is not None) for o in on)),
        arr("B_SOL_ATOL", "double", (_num(o["tolerance"]["max_atol"]) if o else 0 for o in on)),
        arr("B_SOL_RTOL", "double", (_num(o["tolerance"]["max_rtol"]) if o else 0 for o in on)),
        arr("B_SOL_RATIO", "double", (_num(o["tolerance"]["required_matched_ratio"]) if o else 1 for o in on)),
        arr("B_SOL_REQ", "double", (_num(o["required"]) if o else 1 for o in on)),
        arr("B_SOL_CAP", "double", (_num(-1 if o["tolerance"]["max_error_cap"] is None else o["tolerance"]["max_error_cap"])
                                    if o else -1 for o in on)),
        arr("B_SOL_NEGINF", "int", (int(o["tolerance"]["allow_negative_inf"]) if o else 0 for o in on)),
        arr("B_SOL_USE", "int", (int(o["use"] == "sol") if o else 0 for o in on)),
        arr("B_SOL_MARGIN", "double", (_num(o["margin"]) if o else 1 for o in on)),
        arr("B_SOL_REF", "long", (blob_offsets.get(b["name"], -1) if o else -1 for b, o in zip(buffers, on))),
    ])
    assert nb == len(on)
    return tables, DRIVER_SNIPPET + DRIVER_CHECK


# ---------------------------------------------------------------- reading a driver's output

_SOL_LINE = re.compile(r"^MB_SOL (\S+) ref=(\w+) matched=(\S+) required=(\S+) max_abs_err=(\S+) max_rel_err=(\S+) "
                       r"max_excess=(\S+) nonfinite=(\d) allzero=(\d) cap_exceeded=(\d) n=(\d+) atol=(\S+) rtol=(\S+) "
                       r"ok=(\d) gate=(\d)", re.M)


def parse_sol_lines(text: str) -> dict:
    """{buffer: {"sol": {...}, "golden": {...}, "gate": "sol" | "golden"}} from a driver run's MB_SOL lines ({} when
    there are none: the run was not gated). A checked output appears once per check of the run (the eager check and the
    graph replay's): the last one stands, a failure in any is kept (``ok`` is the AND)."""
    out: dict = {}
    for m in _SOL_LINE.finditer(text or ""):
        name, ref = m.group(1), m.group(2)
        rec = {"matched_ratio": float(m.group(3)), "required_ratio": float(m.group(4)), "max_abs_err": float(m.group(5)),
               "max_rel_err": float(m.group(6)), "max_excess": float(m.group(7)), "nonfinite_output": m.group(8) == "1",
               "all_zero_vs_nonzero_ref": m.group(9) == "1", "max_error_cap_exceeded": m.group(10) == "1",
               "n": int(m.group(11)), "atol": float(m.group(12)), "rtol": float(m.group(13)), "ok": m.group(14) == "1"}
        d = out.setdefault(name, {})
        if ref in d and not d[ref]["ok"]:
            rec["ok"] = False
        d[ref] = rec
        if m.group(15) == "1":
            d["gate"] = ref
    return out


def verdict(sol: dict | None) -> dict | None:
    """Summary of :func:`parse_sol_lines`: {"ok": every gating check passed, "outputs": {buffer: gating record + "ref"}}."""
    if not sol:
        return None
    outs = {}
    for name, d in sol.items():
        g = d.get("gate")
        if g and g in d:
            outs[name] = {**d[g], "ref": g, **({"golden_ok": d["golden"]["ok"]} if "golden" in d and g != "golden" else {}),
                          **({"sol_ref_ok": d["sol"]["ok"]} if "sol" in d and g != "sol" else {})}
    return {"ok": all(o["ok"] for o in outs.values()) if outs else None, "outputs": outs}


# ---------------------------------------------------------------- CLI

def _load_arrays(p, prefix="out"):
    z = np.load(p, allow_pickle=False)
    keys = sorted((k for k in z.files if re.fullmatch(prefix + r"\d+", k)), key=lambda k: int(k[len(prefix):]))
    return [z[k] for k in keys]


def _cmd_tolerance(a) -> int:
    tol = None
    if a.bench:
        tol = read_bench_tolerance(a.bench)
    elif a.ir:
        g = json.loads((Path(a.ir) / "graph.json").read_text())
        tl = tolerances_for(g)
        tol = tl[0] if tl else None
    print(json.dumps({"tolerance": normalize(tol) if tol is not None else None}, indent=1))
    return 0 if tol is not None else 1


def _cmd_compare(a) -> int:
    tol = json.loads(a.tolerance) if a.tolerance else read_bench_tolerance(a.bench) if a.bench else {}
    outs, refs = _load_arrays(a.out), _load_arrays(a.ref)
    res = sol_compare_all(outs, refs, normalize(tol))
    print(json.dumps({"ok": all(r["ok"] for r in res), "outputs": res}, indent=1))
    return 0 if all(r["ok"] for r in res) else 1


def _cmd_ref(a) -> int:
    p = ensure_reference(a.ir, a.bench, device=a.device, force=a.force)
    print(json.dumps({"reference": str(p) if p else None}))
    return 0 if p else 1


def _cmd_record(a) -> int:
    rec = record_in_ir(a.ir, a.bench)
    print(json.dumps({"sol_tolerance": rec}, indent=1))
    return 0 if rec else 1


def _cmd_parse(a) -> int:
    text = Path(a.log).read_text() if a.log != "-" else sys.stdin.read()
    sol = parse_sol_lines(text)
    v = verdict(sol)
    print(json.dumps({"verdict": v, "raw": sol}, indent=1))
    return 0 if v and v["ok"] else 1


def _cmd_check(a) -> int:
    """Re-check a final pack (``work/final/<name>/{ir, problem}``) against SOL's tolerance: export its graph with its slot
    sources (the winners) and the SOL tables, build, run the driver on the GPU of this host. The audit's check, cheaply."""
    from . import kb_export
    from . import sol_tolerance as S         # not this module: run as ``-m`` it is __main__, and kb_export imports the package's copy
    from .kb_proposals import write_variant_ir
    pack, orig = Path(a.pack), Path(a.ir)
    tmp = Path(a.work or (pack / "sol_check"))
    g = json.loads(((orig if a.seed else pack / "ir") / "graph.json").read_text())      # --seed: the original graph, hand kernels
    vir = write_variant_ir(orig, g, tmp / "ir")
    if a.bench:
        os.environ["MB_SOL_BENCH"] = str(a.bench)
        S.ensure_reference(orig, a.bench, device=a.device)
        S.STATE.refs[g["name"]] = str(orig / S.REF_FILE)
    srcs = {p.stem: p.read_text() for p in sorted((pack / "problem" / "slots").glob("*.cu"))} \
        if (pack / "problem" / "slots").is_dir() else {}
    if a.seed:
        srcs = {}
    S.STATE.mode_cfg = "accept"
    with S.scope("report"):
        d = kb_export.export_problem(vir, tmp / "problem", slot_sources=srcs or None, strategy="whole")
    man = json.loads((d / "manifest.json").read_text())
    if "sol" not in man:
        print(json.dumps({"error": "the export carries no SOL block (no SOL_TOLERANCE: give --bench, or the IR was not "
                                   "extracted from a SOL bench)"}))
        return 2
    if a.export_only:
        print(json.dumps({"problem": str(d), "sol": man["sol"]}, indent=1))
        return 0
    res = kb_export.build_and_run(d, arch=a.arch, runs=1, graph=False)
    v = S.verdict(res.get("sol"))
    print(json.dumps({"passed_flow_gate": res.get("passed"), "verdict": v, "sol": res.get("sol")}, indent=1))
    return 0 if v and v["ok"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("tolerance", help="print the normalized SOL tolerance of a bench file or an IR")
    p.add_argument("--bench"); p.add_argument("--ir"); p.set_defaults(f=_cmd_tolerance)
    p = sub.add_parser("compare", help="SOL's check of an outputs npz (out0.. arrays) against a reference npz")
    p.add_argument("out"); p.add_argument("ref"); p.add_argument("--bench"); p.add_argument("--tolerance", help="JSON")
    p.set_defaults(f=_cmd_compare)
    p = sub.add_parser("ref", help="write IR/sol_ref.npz: the bench's SOL reference outputs (GPU if present)")
    p.add_argument("--ir", required=True); p.add_argument("--bench", required=True)
    p.add_argument("--device", default="auto"); p.add_argument("--force", action="store_true"); p.set_defaults(f=_cmd_ref)
    p = sub.add_parser("record", help="add sol_tolerance to an extracted IR's graph.json from the bench")
    p.add_argument("--ir", required=True); p.add_argument("--bench", required=True); p.set_defaults(f=_cmd_record)
    p = sub.add_parser("parse", help="the SOL verdict of a driver log (MB_SOL lines)")
    p.add_argument("log", help="a file or -"); p.set_defaults(f=_cmd_parse)
    p = sub.add_parser("check", help="re-check a final pack against SOL's tolerance (builds + runs on this host's GPU)")
    p.add_argument("pack", help="work/final/<name> (ir/, problem/slots/*.cu)")
    p.add_argument("--ir", required=True, help="the original extracted IR dir (weights.npz, io.npz)")
    p.add_argument("--bench"); p.add_argument("--arch", default="100"); p.add_argument("--device", default="auto")
    p.add_argument("--work"); p.add_argument("--seed", action="store_true", help="check the original graph on the hand kernels (the seed) instead")
    p.add_argument("--export-only", action="store_true"); p.set_defaults(f=_cmd_check)
    a = ap.parse_args(argv)
    return a.f(a)


if __name__ == "__main__":
    sys.exit(main())
