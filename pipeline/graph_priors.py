"""Dataset priors for graph decisions (fusion / layout / attention outcomes), scoped by partition.

Used by ``graph_search`` (``--graph-priors rank|preapply``; l3_farm/analysis/graph_search_redesign.md). The question a
prior answers: "a hint of this structural kind, in this partition (arch | dtype), was accepted how often?".

Keys (:func:`fusion_keys`): a hint is described at three levels, most specific first --

* ``L0`` the exact stage list without dtype / argument markers (``conv2d+batchnorm2d+relu``),
* ``L1`` stage families (``conv+bn+relu``; ``cat2_c1`` / ``catn`` -> ``cat``, ``ewise2`` / ``add`` / ``mul`` ->
  ``ewise``, ``ewise_scalar`` / ``mul_scalar`` -> ``sewise``, ``conv2d_dw`` -> ``dwconv``),
* ``L2`` the head family plus the tail length (``conv+*2``),

and a structural hint (``fuse_attention``, ``fold_layout``, ``fuse_siblings``, ``split_slot``) by its kind. A prior
uses the most specific level with at least ``min_n`` observations (else the next; else the uninformative seed
Beta(1, 1)). The posterior is Beta(accepted + 1, rejected + 1); ``lb`` is its ``q`` quantile (default 10%).

Sources (:class:`Priors`), summed per key:

1. A5's structural priors (``kb_global.struct_prior``; adapter :func:`_a5_counts`) when the function exists and
   answers; otherwise (until then) the bootstrap table below stands in -- logged in AGENT_COORDINATION.md;
2. the bootstrap table ``graph_priors_l3sweep.json`` (the Level 3 sweep's 137 fusion decisions,
   ``l3_farm/analysis/e7_data/decisions.csv``, partition ``cuda:sm_89|fp16``) -- used only for that partition;
3. this work dir's own decisions (``graph_decisions.jsonl``, written by ``graph_search``) -- any partition.

A table can be built from any decisions.csv with :func:`table_from_decisions` (the replay does it leave-one-problem-out).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOOTSTRAP = HERE / "graph_priors_l3sweep.json"
STRUCTURAL = ("fuse_attention", "fold_layout", "fuse_siblings", "split_slot", "unfuse")

_DTYPE = ("_f16", "_f32", "_s8_pc", "_s8", "_i8", "_bf16")
_FAMILY = {
    "conv2d": "conv", "conv2d_dw": "dwconv", "conv2d_grouped": "gconv", "conv1d": "conv", "conv3d": "conv",
    "batchnorm2d": "bn", "batchnorm1d": "bn", "layernorm": "ln", "groupnorm": "gn",
    "linear": "linear", "matmul": "gemm", "bmm": "gemm", "addmm": "gemm",
    "ewise2": "ewise", "add": "ewise", "mul": "ewise", "sub": "ewise", "div": "ewise",
    "ewise_scalar": "sewise", "mul_scalar": "sewise", "add_scalar": "sewise",
    "catn": "cat", "cat2_c1": "cat", "cat3_c1": "cat", "cat4_c1": "cat",
    "sum_dim": "reduce", "reduce": "reduce", "mean_dim": "reduce",
    "maxpool2d": "pool", "maxpool2d_ex": "pool", "avgpool2d": "pool", "adaptive_avg_pool2d": "pool",
}


def _base(tok: str) -> str:
    for s in _DTYPE:
        if tok.endswith(s):
            return tok[: -len(s)]
    return tok


def stages(fused_kind: str) -> list[str]:
    """``fused__conv2d_f16__batchnorm2d_f16__add_f16__a`` -> ['conv2d', 'batchnorm2d', 'add']; argument markers
    (``a`` / ``b``) and the sibling marker (``par``) are dropped."""
    k = fused_kind[len("fused__"):] if fused_kind.startswith("fused__") else fused_kind
    return [_base(t) for t in k.split("__") if t and len(t) > 1 and t != "par"]


def fusion_keys(hint_or_kind) -> list[str]:
    """[L0, L1, L2] keys of a hint (or a fused kind string); structural kinds: [kind]."""
    if isinstance(hint_or_kind, dict):
        h = hint_or_kind
        if h.get("kind") != "fuse":
            return [str(h.get("kind"))]
        kind = h.get("fused_kind") or str(h.get("id", "")).removeprefix("fuse:")
    else:
        kind = str(hint_or_kind)
        if kind in STRUCTURAL:
            return [kind]
    st = stages(kind)
    if not st:
        return [kind]
    fam = [_FAMILY.get(s, s) for s in st]
    return ["+".join(st), "+".join(fam), f"{fam[0]}+*{len(fam) - 1}"]


# ---------------------------------------------------------------- Beta posterior

def _betacf(a: float, b: float, x: float) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1.0 + aa / c
        c = c if abs(c) > 1e-30 else 1e-30
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1.0 + aa / c
        c = c if abs(c) > 1e-30 else 1e-30
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-12:
            break
    return h


def beta_cdf(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x)
    bt = math.exp(lbt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def beta_quantile(q: float, a: float, b: float) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if beta_cdf(mid, a, b) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def posterior(acc: float, rej: float, q: float = 0.1) -> dict:
    a, b = acc + 1.0, rej + 1.0
    return {"accepted": acc, "rejected": rej, "n": acc + rej, "p_mean": round(a / (a + b), 4),
            "lb": round(beta_quantile(q, a, b), 4), "q": q}


# ---------------------------------------------------------------- tables

def _status_counts(rows) -> dict:
    """[(hint id, status)] -> {key: [accepted, rejected]} at every level."""
    out: dict[str, list] = {}
    for hid, status in rows:
        if status not in ("accepted", "rejected"):
            continue
        hint = {"id": hid, "kind": "fuse", "fused_kind": hid.split(":", 1)[1]} if hid.startswith("fuse:") else \
            {"id": hid, "kind": "split_slot" if hid.startswith("split:") else hid.split(":", 1)[0]}
        for k in dict.fromkeys(fusion_keys(hint)):          # L0 == L1 for most kinds: count once
            c = out.setdefault(k, [0, 0])
            c[0 if status == "accepted" else 1] += 1
    return out


def table_from_decisions(csv_path: Path, *, part: str = "cuda:sm_89|fp16", exclude_problem: str | None = None,
                         samples=("main", "degraded")) -> dict:
    """A priors table from an e7-style decisions.csv (problem, sample, round, hint, status, ...)."""
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r.get("sample") not in samples or r.get("problem") == exclude_problem:
                continue
            rows.append((r["hint"], r["status"]))
    return {"format": "mb-graph-priors/1", "part": part, "source": str(csv_path),
            "exclude_problem": exclude_problem, "counts": _status_counts(rows)}


# ---------------------------------------------------------------- the store

def _a5_counts(hint: dict | None, key: str, part: str, work: Path | None = None):
    """A5's structural-outcome priors (``kb_global.struct_prior(kind, context, *, work, store, part)`` -> a Prior with
    accepted / rejected (or n / rate), its own hierarchical back-off), when the function exists. Only the L0 key of
    a hint is asked (A5 backs off by itself). Returns (accepted, rejected) or None (absent, no data, or an error:
    then the bootstrap table stands in)."""
    try:
        from . import kb_global
    except Exception:  # noqa: BLE001
        return None
    fn = getattr(kb_global, "struct_prior", None)
    if fn is None or hint is None:
        return None
    kind = hint.get("kind", "fuse")
    ctx = tuple(stages(hint.get("fused_kind") or str(hint.get("id", "")).removeprefix("fuse:"))) \
        if kind == "fuse" else (kind,)
    try:
        r = fn(kind, ctx, work=work, part=part)
    except Exception:  # noqa: BLE001 -- knowledge never fails a run
        return None
    get = (lambda k: r.get(k)) if isinstance(r, dict) else (lambda k: getattr(r, k, None))
    acc, rej, n, rate = get("accepted"), get("rejected"), get("n"), get("rate")
    if acc is not None and rej is not None and (float(acc) + float(rej)) > 0:
        return float(acc), float(rej)
    if n and rate is not None:
        return float(rate) * float(n), (1.0 - float(rate)) * float(n)
    return None


class Priors:
    """Partition-scoped priors: ``prior(hint)`` -> {p_mean, lb, n, accepted, rejected, key, level, source}."""

    def __init__(self, part: str, *, tables: list[dict] | None = None, local: Path | None = None,
                 use_a5: bool = True, min_n: int = 3, q: float = 0.1):
        self.part, self.min_n, self.q, self.use_a5 = part, int(min_n), float(q), use_a5
        self.tables = [t for t in (tables or []) if t.get("part") in (None, part)]
        self.local = Path(local) if local else None
        self.sources: set[str] = set()

    @classmethod
    def default(cls, part: str, work: Path | None = None, **kw) -> "Priors":
        tabs = []
        if BOOTSTRAP.exists():
            try:
                tabs.append(json.loads(BOOTSTRAP.read_text()))
            except ValueError:
                pass
        return cls(part, tables=tabs, local=(Path(work) / "graph_decisions.jsonl") if work else None, **kw)

    def _local_counts(self) -> dict:
        if self.local is None or not self.local.exists():
            return {}
        rows = []
        for line in self.local.read_text().splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            # only measured decisions vote (a pre-applied hint is the prior's own output, not evidence)
            if e.get("event") == "decision" and e.get("measured") and e.get("hint"):
                rows.append((e["hint"], e.get("status")))
        return _status_counts(rows)

    def counts(self, key: str, hint: dict | None = None, level: int = 0) -> tuple[float, float, list[str]]:
        """Counts for one key: A5's store when it answers (asked at L0 only; it holds the sweep's outcomes too, so the
        bootstrap table is then not added), else the bootstrap table(s); plus this work dir's measured decisions."""
        acc = rej = 0.0
        src = []
        c = _a5_counts(hint, key, self.part, self.local.parent if self.local else None) \
            if self.use_a5 and level == 0 else None
        if c is not None:
            acc, rej = acc + c[0], rej + c[1]
            src.append("a5")
        else:
            for t in self.tables:
                c = (t.get("counts") or {}).get(key)
                if c:
                    acc, rej = acc + c[0], rej + c[1]
                    src.append(Path(str(t.get("source", "table"))).name)
        c = self._local_counts().get(key)
        if c:
            acc, rej = acc + c[0], rej + c[1]
            src.append("work")
        return acc, rej, src

    def prior(self, hint) -> dict:
        keys = fusion_keys(hint)
        h = hint if isinstance(hint, dict) else {"kind": "fuse", "fused_kind": str(hint)}
        for lvl, k in enumerate(keys):
            acc, rej, src = self.counts(k, h, lvl)
            if acc + rej >= self.min_n:
                self.sources.update(src)
                return {**posterior(acc, rej, self.q), "key": k, "level": lvl, "source": "+".join(src),
                        "part": self.part}
        return {**posterior(0, 0, self.q), "key": keys[0], "level": None, "source": "seed", "part": self.part}


def explore(seed: int, hid: str, rate: float) -> bool:
    """Deterministic exploration draw (a pre-apply-eligible hint is decided by measurement instead)."""
    if rate <= 0:
        return False
    import hashlib
    h = int(hashlib.sha1(f"{seed}:{hid}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < rate


def main():
    import argparse
    ap = argparse.ArgumentParser(description="build a priors table from a decisions.csv")
    ap.add_argument("decisions")
    ap.add_argument("--part", default="cuda:sm_89|fp16")
    ap.add_argument("--out", default=str(BOOTSTRAP))
    a = ap.parse_args()
    t = table_from_decisions(Path(a.decisions), part=a.part)
    Path(a.out).write_text(json.dumps(t, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in t["counts"].items() if "+" in k and "*" not in k}, sort_keys=True))


if __name__ == "__main__":
    main()
