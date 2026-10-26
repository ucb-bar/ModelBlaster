"""Layout folding: delete `strided_copy` ops by giving GEMM slots strided operands.

The IR keeps every activation in PyTorch's row-major layout, so each
permute/transpose/split that is not a pure view becomes a `strided_copy` op
(a full read + write of the tensor), and a GEMM that wants a transposed or
per-head view reads that materialized copy. KernelBench L3 measurements
(l3_farm/analysis/layout_packing.md) put 4.4% of the summed sweep time in these
copies, concentrated in attention / einsum models (49_Mamba2 57%, 43_MinGPT
30%, 50_ReLUSelfAttention 20%, 44_MiniGPTBlock 15%, Swin/ViT 13-24%).

This pass is a *layout propagation* restricted to what the fixed-signature
slot contract can express without new memory formats: a copy that only feeds
one bmm operand (or that only consumes one bmm output) is folded into the
bmm's addressing, which becomes the new slot kind ``bmm_strided`` (``_f16``):

    C[b0,b1] = A[b0,b1] @ B[b0,b1]   (b0 < B0, b1 < B1, A: MxK, B: KxN, C: MxN)
    A[b0,b1][m,k] = A_ptr[a_off + b0*a_s0 + b1*a_s1 + m*a_sm + k*a_sk]
    B[b0,b1][k,n] = B_ptr[b_off + b0*b_s0 + b1*b_s1 + k*b_sk + n*b_sn]
    C[b0,b1][m,n] -> C_ptr[c_off + b0*c_s0 + b1*c_s1 + m*c_sm + n*c_sn]

One fixed C signature (see reference_kernels_ext.BMM_STRIDED); the strides
are ordinary int shape params, so the existing shape-guarded cache / split
machinery specializes a winner per stride set (the agent sees e.g. a_sk == 1
vs a_sm == 1 as constants and picks ldmatrix(.trans) accordingly). Two batch
levels are enough for multi-head views ([B, nh] with unrelated strides);
an operand is folded only when one of its two inner strides is 1 (a row- or
column-major matrix), so every folded operand is still vector-loadable.

Correctness never depends on the algebra here: every composed map is checked
point-wise against the original copy chain (corners + pseudo-random samples),
the rewritten IR is walked with walk_torch against the original golden by
graph_hints.verify_rewrite, and the whole-model check against PyTorch is
unchanged because graph inputs/outputs keep their layout (only intermediate
tensors disappear).

    python -m modelblaster.pipeline.layout_fold IR_DIR_or_graph.json [--profile profile.json] [--out graph.json]
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

GEMM_OPS = {"bmm": "bmm_strided", "bmm_f16": "bmm_strided_f16"}
COPY_OPS = {"strided_copy", "strided_copy_f16"}
VIEW_OPS = {"view"}
MAXR = 6


# ------------------------------------------------------------------ affine maps
# A map is (off, [(d, s), ...]): element i (row-major over the d's) of the
# result lives at src[off + sum i_k * s_k].

def _prod(xs):
    p = 1
    for x in xs:
        p *= int(x)
    return p


def _contig(ds):
    st, acc = [], 1
    for d in reversed(ds):
        st.append(acc)
        acc *= int(d)
    return st[::-1]


def copy_map(shape: dict) -> tuple[int, list]:
    dims = [(int(shape[f"d{i}"]), int(shape[f"s{i}"])) for i in range(MAXR)]
    return int(shape.get("off", 0)), [(d, s) for d, s in dims if d != 1]


def _eval(m, idx):
    off, dims = m
    return off + sum(i * s for i, (_, s) in zip(idx, dims))


def _unravel(p, ds):
    out = []
    for d in reversed(ds):
        out.append(p % d)
        p //= d
    return out[::-1]


def merge(dims):
    """Drop size-1 dims, merge adjacent dims that are contiguous w.r.t. each other."""
    out = [(d, s) for d, s in dims if d != 1]
    i = len(out) - 1
    while i > 0:
        (d0, s0), (d1, s1) = out[i - 1], out[i]
        if s0 == s1 * d1:
            out[i - 1] = (d0 * d1, s1)
            del out[i]
        i -= 1
    return out


def fit(dims, sizes):
    """Re-express a row-major dim list as exactly ``len(sizes)`` dims of the
    given extents (splitting a dim when an extent divides it, merging adjacent
    contiguous dims). -> strides per size (0 for size 1) or None."""
    src = [list(x) for x in dims if x[0] != 1]
    res = []
    for want in sizes:
        if want == 1:
            res.append(0)
            continue
        got, stride = 1, None
        while got < want:
            if not src:
                return None
            d, s = src[0]
            need = want // got
            if want % got:
                return None
            if d <= need:
                if stride is not None and stride != s * d:
                    return None                       # not mergeable with the previous piece
                stride = s
                got *= d
                src.pop(0)
            else:
                if d % need:
                    return None
                # split (d, s) -> (need, s * d/need), (d/need, s); take the outer piece
                if stride is not None and stride != s * (d // need) * need:
                    return None
                stride = s * (d // need)
                got *= need
                src[0] = [d // need, s]
        if got != want:
            return None
        res.append(stride)
    return res if not src else None


def compose(outer, inner_shape, inner):
    """outer reads the contiguous result of ``inner`` (logical dims
    ``inner_shape``); -> a map straight from inner's source, or None."""
    off2, od = outer
    D = [d for d, _ in inner[1]]
    if _prod(D) != _prod(inner_shape) and inner[1]:
        return None
    cD = _contig(D)
    base = _unravel(off2, D) if D else []
    pieces = []
    work = list(od)
    while work:
        d, t = work.pop(0)
        if d == 1:
            continue
        r = next((r for r in range(len(D)) if cD[r] <= t), None) if t else None
        if t == 0:
            pieces.append((d, 0))
            continue
        if r is None or t % cD[r]:
            return None
        step = t // cD[r]
        if base[r] + (d - 1) * step < D[r]:
            pieces.append((d, step * inner[1][r][1]))
            continue
        # spans several inner dims: split at inner dim r's radix
        if step != 1 or base[r] != 0 or d % D[r]:
            return None
        work[:0] = [(d // D[r], t * D[r]), (D[r], t)]
    off = inner[0] + sum(b * s for b, (_, s) in zip(base, inner[1]))
    cand = (off, pieces)
    return cand if _same(outer, inner, cand) else None


def _samples(ds, n=256, seed=0):
    if not ds:
        yield []
        return
    rng = random.Random(seed)
    yield [0] * len(ds)
    yield [d - 1 for d in ds]
    for _ in range(n):
        yield [rng.randrange(d) for d in ds]


def _same(outer, inner, cand) -> bool:
    """cand == inner o outer on corners + random points (the safety net)."""
    ds = [d for d, _ in outer[1]]
    if [d for d, _ in cand[1]] != ds and _prod(d for d, _ in cand[1]) != _prod(ds):
        return False
    D = [d for d, _ in inner[1]]
    cds = [d for d, _ in cand[1]]
    for idx in _samples(ds):
        p = _eval(outer, idx)
        want = _eval(inner, _unravel(p, D)) if D else inner[0] + p
        lin = sum(i * c for i, c in zip(idx, _contig(ds)))
        if _eval(cand, _unravel(lin, cds)) != want:
            return False
    return True


# ------------------------------------------------------------------ graph walk

def _index(graph):
    prod, readers = {}, {}
    for o in graph["ops"]:
        for t in o["outputs"]:
            prod[t] = o
        for t in o["inputs"]:
            readers.setdefault(t, []).append(o)
    outs = set(graph["output"].get("tensors") or [graph["output"]["tensor"]])
    return prod, readers, outs


def _tshape(graph, t):
    return graph["tensors"][t]["shape"]


def source_map(graph, t, prod, readers, outs):
    """Follow views/copies upward from tensor ``t``: -> (root tensor, map from
    root to t's contiguous elements, [ops that become dead if t's reader stops
    reading through them])."""
    m = (0, [(d, s) for d, s in zip(_tshape(graph, t), _contig(_tshape(graph, t)))])
    m = (0, merge(m[1]))
    chain = []
    cur = t
    while True:
        o = prod.get(cur)
        if o is None or o["op"] not in COPY_OPS | VIEW_OPS:
            break
        src = o["inputs"][0]
        if o["op"] in VIEW_OPS:
            inner = (int(o.get("offset", 0)), [(_prod(_tshape(graph, cur)), 1)])
        else:
            inner = copy_map(o["shape"])
        nm = compose(m, _tshape(graph, cur), inner)
        if nm is None:
            break
        sole = len(readers.get(cur, [])) == 1 and cur not in outs
        if not sole:
            break
        m, cur = nm, src
        chain.append(o)
    return cur, m, chain


def sink_map(graph, t, prod, readers, outs):
    """Follow t's single reader chain of views/copies downward while the
    composite is a bijection (a pure permutation of t). -> (final tensor,
    write map: element i of t (row-major) -> final[off + ...], [ops])."""
    cur, chain = t, []
    best = None
    perm = (0, [(d, s) for d, s in zip(_tshape(graph, t), _contig(_tshape(graph, t)))])
    n = _prod(_tshape(graph, t))
    while True:
        r = readers.get(cur, [])
        if len(r) != 1 or cur in outs or r[0]["op"] not in COPY_OPS | VIEW_OPS:
            break
        o = r[0]
        if o["op"] in VIEW_OPS:
            if o.get("offset"):
                break
            chain.append(o)
            cur = o["outputs"][0]
            if any(c["op"] in COPY_OPS for c in chain):
                best = (cur, list(chain))
            continue
        off, dims = copy_map(o["shape"])
        if off or _prod(d for d, _ in dims) != n:
            break
        chain.append(o)
        cur = o["outputs"][0]
        best = (cur, list(chain))
    if best is None:
        return None
    final, ops = best
    # read map final[j] = t[G(j)], composed back through the chain
    G = (0, merge([(d, s) for d, s in zip(_tshape(graph, final), _contig(_tshape(graph, final)))]))
    for o in reversed(ops):
        inner = copy_map(o["shape"]) if o["op"] in COPY_OPS else (0, [(n, 1)])
        G = compose(G, _tshape(graph, o["outputs"][0]), inner)
        if G is None:
            return None
    g = G
    off, dims = g
    dims = merge(dims)
    order = sorted(range(len(dims)), key=lambda k: -dims[k][1])
    sd = [dims[k] for k in order]
    if off or not sd or sd[-1][1] != 1 or any(sd[i][1] != sd[i + 1][1] * sd[i + 1][0] for i in range(len(sd) - 1)):
        return None                                   # not a bijection onto t
    ost = _contig([d for d, _ in dims])
    w = (0, [(dims[k][0], ost[k]) for k in order])   # t's dims (row-major) -> final strides
    return final, w, ops


# ------------------------------------------------------------------ the fold

def _natural(batch, r, c):
    return (0, [(batch, r * c), (r, c), (c, 1)])


def _gemm_of(op):
    """The bmm an op runs: itself, or the head stage of a fused group."""
    if op["op"] in GEMM_OPS:
        return op
    subs = op.get("sub_ops") or []
    return subs[0] if subs and subs[0]["op"] in GEMM_OPS else None


def _try_fold(graph, op, prod, readers, outs):
    gm = _gemm_of(op)
    fused = gm is not op
    sh = gm["shape"]
    bt, M, K, N = int(sh["batch"]), int(sh["M"]), int(sh["K"]), int(sh["N"])
    a_t, b_t = gm["inputs"][:2]
    c_t = gm["outputs"][0]
    a_root, a_map, a_ops = source_map(graph, a_t, prod, readers, outs)
    b_root, b_map, b_ops = source_map(graph, b_t, prod, readers, outs)
    if not any(o["op"] in COPY_OPS for o in a_ops):
        a_root, a_map, a_ops = a_t, _natural(bt, M, K), []
    if not any(o["op"] in COPY_OPS for o in b_ops):
        b_root, b_map, b_ops = b_t, _natural(bt, K, N), []
    sk = None if fused else sink_map(graph, c_t, prod, readers, outs)
    if sk:
        c_final, c_map, c_ops = sk
    else:
        c_final, c_map, c_ops = c_t, _natural(bt, M, N), []
    if not (a_ops or b_ops or c_ops):
        return None
    cands = {1}
    for m, r, c in ((a_map, M, K), (b_map, K, N), (c_map, M, N)):
        dims = [x for x in m[1] if x[0] != 1]
        acc = 1
        for d, _ in reversed(dims):
            acc *= d
            if acc % (r * c) == 0 and bt % (acc // (r * c)) == 0:
                cands.add(acc // (r * c))
        for d, _ in dims:
            if bt % d == 0:
                cands.add(d)
    for B1 in sorted(cands, reverse=True):
        B0 = bt // B1
        fa = fit(a_map[1], [B0, B1, M, K])
        fb = fit(b_map[1], [B0, B1, K, N])
        fc = fit(c_map[1], [B0, B1, M, N])
        if None in (fa, fb, fc):
            continue
        # every operand must stay a dense row- or column-major matrix
        if 1 not in (fa[2], fa[3]) or 1 not in (fb[2], fb[3]) or 1 not in (fc[2], fc[3]):
            continue
        shape = {"B0": B0, "B1": B1, "M": M, "K": K, "N": N,
                 "a_off": a_map[0], "a_s0": fa[0], "a_s1": fa[1], "a_sm": fa[2], "a_sk": fa[3],
                 "b_off": b_map[0], "b_s0": fb[0], "b_s1": fb[1], "b_sk": fb[2], "b_sn": fb[3],
                 "c_off": c_map[0], "c_s0": fc[0], "c_s1": fc[1], "c_sm": fc[2], "c_sn": fc[3]}
        folded = [o["name"] for o in a_ops + b_ops + c_ops if o["op"] in COPY_OPS]
        g2 = {**gm, "op": GEMM_OPS[gm["op"]], "inputs": [a_root, b_root], "outputs": [c_final], "shape": shape}
        if not fused:
            new = {**g2, "folded": folded}
            new.pop("slot_variant", None)
        else:                                          # head stage of a fused group: A/B only
            new = copy.deepcopy(op)
            new["sub_ops"][0] = g2
            if isinstance(new.get("shape"), dict) and "a" in new["shape"]:
                new["shape"] = {**new["shape"], "a": g2["shape"]}      # stage a's shape (strides): the shape key / guard
            ext = {a_t: a_root, b_t: b_root}
            new["inputs"] = list(dict.fromkeys(ext.get(t, t) for t in op["inputs"]))
            new["op"] = op["op"].replace(f"fused__{gm['op']}__", f"fused__{GEMM_OPS[gm['op']]}__", 1)
            new["folded"] = folded
            new.pop("slot_variant", None)
        return new, a_ops + b_ops + c_ops
    return None


def fold(graph: dict, *, op_us: dict | None = None, only: set | None = None) -> tuple[dict, dict]:
    """Fold strided copies into bmm operands/results. -> (new graph, report)."""
    g = copy.deepcopy(graph)
    rep = {"folded": [], "skipped": [], "removed_copy_us": 0.0}
    changed = True
    while changed:
        changed = False
        prod, readers, outs = _index(g)
        for op in list(g["ops"]):
            if _gemm_of(op) is None or (only and op["name"] not in only):
                continue
            r = _try_fold(g, op, prod, readers, outs)
            if r is None:
                if any((prod.get(t) or {}).get("op") in COPY_OPS for t in op["inputs"]):
                    rep["skipped"].append(op["name"])
                continue
            new, dead = r
            # ops still needed by someone else stay (source_map only walks sole readers)
            dead_names = {o["name"] for o in dead}
            g["ops"] = [new if o is op else o for o in g["ops"] if o["name"] not in dead_names or o is op]
            for o in dead:
                for t in o["outputs"]:
                    if t not in new["inputs"] and t not in new["outputs"]:
                        g["tensors"].pop(t, None)
            if "sub_ops" not in op and op["outputs"][0] != new["outputs"][0]:
                g["tensors"].pop(op["outputs"][0], None)
            us = sum((op_us or {}).get(n, 0.0) for n in new["folded"])
            rep["folded"].append({"op": op["name"], "copies": new["folded"], "copy_us": round(us, 1),
                                  "shape": new["shape"]})
            rep["removed_copy_us"] += us
            changed = True
            break
    rep["removed_copy_us"] = round(rep["removed_copy_us"], 1)
    return g, rep


def fold_hint(source: dict, ops: list[str] | None = None, status: str = "proposed") -> dict:
    """The graph-hint record (graph_hints kind ``fold_layout``)."""
    h = {"id": "layout:fold_strided_gemm", "kind": "fold_layout", "target": "cuda", "status": status,
         "source": [source], "evidence": []}
    if ops:
        h["ops"] = ops
    return h


def propose(graph: dict, op_us: dict, *, total_us: float | None = None, min_gain_pct: float = 2.0,
            ir: str | None = None) -> list[dict]:
    """Cost-driven proposal: worth trying when the copies it removes are
    >= ``min_gain_pct`` of the model (same rule as propose_from_profile)."""
    _, rep = fold(graph, op_us=op_us)
    total = total_us or sum(op_us.values()) or 1.0
    pct = 100.0 * rep["removed_copy_us"] / total
    if not rep["folded"] or pct < min_gain_pct:
        return []
    return [fold_hint({"by": "mb-cost", "ir": ir, "est_gain_us": rep["removed_copy_us"],
                       "est_gain_pct": round(pct, 2), "ops": [f["op"] for f in rep["folded"]]},
                      [f["op"] for f in rep["folded"]])]


def _load(p: Path) -> dict:
    p = Path(p)
    return json.loads((p / "graph.json" if p.is_dir() else p).read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ir")
    ap.add_argument("--profile", help="profile.json (ops: name, us) to estimate the removed time")
    ap.add_argument("--out")
    a = ap.parse_args()
    g = _load(Path(a.ir))
    op_us = {}
    if a.profile:
        op_us = {o["name"]: o["us"] for o in json.loads(Path(a.profile).read_text())["ops"]}
    g2, rep = fold(g, op_us=op_us)
    print(json.dumps(rep, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(g2, indent=1))


if __name__ == "__main__":
    main()
