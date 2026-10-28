"""Region WINDOWS: larger / chunked ``fuse_region`` candidates for graphs the flow's finders cannot regionise
(docs/region_windows.md).

``--region-windows off|auto`` (default off; ``$MB_REGION_WINDOWS`` when the flag is absent). Off: nothing here runs, no
hint is proposed, the member cap stays 256 (``region_slots.max_members``), every output is byte-identical to a tree
without this module. ``auto``: once per IR per round (``capabilities.Caps.propose``) the proposer below turns the IR's DAG
into a few *window sets* and hands each over as ONE ordinary ``fuse_region`` hint (``windows``: several explicit member
lists, one region op per group), decided by the existing paired variant-vs-original decision like any other hint.

Why (graph-agent pilot, l3_farm/graph_agent_pilot/SUMMARY.txt): the hints-only arm's gain came almost entirely from
``fuse_region`` with agent-chosen members (34_VanillaRNNHidden: four 64-step recurrence windows; 15_DenseNet121: two whole
dense blocks) while the flow could not even PROPOSE a region for 34 (805 ops) or 15 (246 ops): ``MAX_MEMBERS = 256`` and
only three whole-graph patterns (rnn_stack | attention_block | all).

What is proposed (labels are the hint id suffix, ``region:win:<label>``):

``whole``      one window over every eligible op of the (largest) run, when it has <= ``hard_cap`` members
``chunk<M>``   windows of at most M members for each size M of ``sizes``; cut points are aligned to detected motif
               boundaries (a periodic op-kind sequence: the 64-step recurrence of 34, per-layer repeats of 15) and
               elsewhere to the cut with the fewest tensors crossing it (a single-frontier cut when there is one)
``blocks<M>``  windows of at most M members whose boundaries are single-frontier cuts (exactly one live tensor crosses:
               a whole dense block / residual block / inception module is never cut), packed greedily
``motif<M>``   only the periodic parts, in motif-aligned windows of at most M members; everything else stays plain ops

A window is a CONTIGUOUS run of the IR's op order, so it is convex by construction (no external path leaves and
re-enters); every group is nevertheless validated by the one authority, ``region_slots.fuse_region`` (applied window by
window on a working copy) and a group it refuses is dropped, not proposed. Ops that are already region ops, and dead ops
(``graph_prune``), are never members. Members are stored as ATOM names (``graph_hints.atoms``) so a window survives
fusions accepted later (``_locate`` maps atoms to the current ops).

Ranking: ``est_gain_pct`` = 8% (the region estimate of ``Caps._propose_regions``) of the covered ops' time share (profile)
or op-count share; ties favour fewer, larger windows; at most ``max_candidates`` (default 3) are proposed, always keeping the
best of each label family first. Nothing here touches GPUs or LLMs.
"""
from __future__ import annotations

import copy
import json
import os
import re
import time
from dataclasses import dataclass, field

ENV = "MB_REGION_WINDOWS"
ENV_SIZES = "MB_REGION_WINDOW_SIZES"
ENV_CANDS = "MB_REGION_WINDOW_CANDIDATES"
ENV_HARD = "MB_REGION_WINDOW_HARD_CAP"
ENV_MIN = "MB_REGION_WINDOW_MIN"
MODES = ("off", "auto")
DEFAULT_SIZES = (256,)
DEFAULT_CANDS = 3
DEFAULT_HARD_CAP = 1100           # `whole` is only proposed up to this many members (see docs: 34's 1062 ops export fine)
DEFAULT_MIN = 4                   # a window needs this many non-alias ops, else the ops stay plain
SOURCE = "mb-region-windows"
GAIN_PER_SHARE = 0.08             # the flow's own region estimate (capabilities.Caps._propose_regions)
MAX_PERIOD = 64
MIN_REPS = 3
MIN_MOTIF_OPS = 32                # a periodic segment shorter than this does not align anything (3 x 1 repeated concats)
ALIAS_OPS = ("view", "chunk2_c1")


# ------------------------------------------------------------------------------------------------ configuration
def resolve_mode(value=None) -> str:
    """``--region-windows`` value (None = ``$MB_REGION_WINDOWS``, else off), validated."""
    v = value if value is not None else os.environ.get(ENV, "off")
    v = str(v).strip().lower() or "off"
    if v in ("0", "false", "no"):
        v = "off"
    if v in ("1", "true", "yes", "on"):
        v = "auto"
    if v not in MODES:
        raise ValueError(f"--region-windows must be one of {MODES}, not {v!r}")
    return v


def _int_list(text: str, what: str) -> tuple:
    try:
        out = tuple(int(x) for x in str(text).replace(";", ",").split(",") if x.strip())
    except ValueError:
        raise ValueError(f"{what}={text!r}: expected comma separated integers") from None
    if not out or any(x < 2 for x in out):
        raise ValueError(f"{what}={text!r}: expected sizes >= 2")
    return tuple(dict.fromkeys(out))


@dataclass(frozen=True)
class WindowConfig:
    mode: str = "off"
    sizes: tuple = DEFAULT_SIZES
    max_candidates: int = DEFAULT_CANDS
    hard_cap: int = DEFAULT_HARD_CAP
    min_ops: int = DEFAULT_MIN

    @property
    def on(self) -> bool:
        return self.mode == "auto"

    @classmethod
    def from_cfg(cls, cfg=None, env=None) -> "WindowConfig":
        """Flags (Config fields; 0 / "" = unset) win over the environment, which wins over the defaults."""
        env = os.environ if env is None else env
        mode = resolve_mode(getattr(cfg, "region_windows", None))
        sizes = str(getattr(cfg, "region_window_sizes", "") or env.get(ENV_SIZES, "") or "").strip()
        cands = int(getattr(cfg, "region_window_candidates", 0) or env.get(ENV_CANDS, 0) or 0)
        hard = int(getattr(cfg, "region_window_hard_cap", 0) or env.get(ENV_HARD, 0) or 0)
        mn = int(getattr(cfg, "region_window_min_ops", 0) or env.get(ENV_MIN, 0) or 0)
        return cls(mode=mode, sizes=_int_list(sizes, "region window sizes") if sizes else DEFAULT_SIZES,
                   max_candidates=cands if cands > 0 else DEFAULT_CANDS,
                   hard_cap=hard if hard > 0 else DEFAULT_HARD_CAP, min_ops=mn if mn > 0 else DEFAULT_MIN)

    def key(self) -> str:
        return f"sizes={','.join(map(str, self.sizes))};cands={self.max_candidates};hard={self.hard_cap};min={self.min_ops}"

    def member_cap(self) -> int:
        """The ``fuse_region`` member cap the windows need (``$MB_REGION_MAX_MEMBERS`` is set to this by ``Caps.apply_env``)."""
        return max(256, self.hard_cap, *self.sizes)


# ------------------------------------------------------------------------------------------------ the sequence
def _atoms(name: str) -> list:
    return name.split("+")


def _is_region(o: dict) -> bool:
    return str(o.get("op", "")).startswith("region__")


def eligible_runs(graph: dict) -> list:
    """Maximal runs of consecutive op indices (IR order) that may be window members: live, not a region op. Dead ops
    inside a run are skipped (fuse_region moves them), a region op ends the run."""
    from . import graph_prune
    ops = graph["ops"]
    dead = set(graph_prune.dead_ops(graph))
    runs, cur = [], []
    for i, o in enumerate(ops):
        if _is_region(o):
            if cur:
                runs.append(cur)
            cur = []
        elif i in dead:
            continue
        else:
            cur.append(i)
    if cur:
        runs.append(cur)
    return runs


_KIND_SUFFIX = re.compile(r"_(f16|bf16|f32|i8)$")


def kind_token(o: dict) -> str:
    """A coarse op kind for motif detection: dtype suffixes stripped, concat arity (cat2/cat3/catn) and slot variants
    ignored (the flow specialises per-instance kinds of one repeated layer), fused kinds as their part list."""
    k = str(o.get("op", ""))
    parts = [_KIND_SUFFIX.sub("", p) for p in k.split("__")]
    parts = [re.sub(r"^cat\d+_c1$|^catn$|^cat2_c1$", "cat", p) for p in parts]
    return "__".join(parts)


def cut_costs(graph: dict, idx: list) -> list:
    """``cost[j]`` = number of live tensors crossing the cut BEFORE ``idx[j]`` (``j`` in 0..len(idx)): produced by an op of
    ``idx[:j]`` and read by an op of ``idx[j:]`` or returned by the graph. ``cost[0] = cost[len] = 0`` by convention."""
    ops = graph["ops"]
    n = len(idx)
    first: dict = {}
    last: dict = {}
    for p, i in enumerate(idx):
        for t in list(ops[i].get("outputs", [])):
            first.setdefault(t, p)
        for t in ops[i].get("inputs", []):
            last[t] = p
    gouts = set((graph.get("output") or {}).get("tensors") or [(graph.get("output") or {}).get("tensor")])
    diff = [0] * (n + 2)
    for t, d in first.items():
        u = last.get(t)
        if t in gouts:
            u = n                                      # returned: crosses every later cut
        if u is None or u <= d:
            continue
        diff[d + 1] += 1
        diff[u + 1] -= 1
    cost, run = [], 0
    for j in range(n + 1):
        run += diff[j]
        cost.append(run if 0 < j < n else 0)
    return cost


def alias_reach(graph: dict, idx: list) -> list:
    """``reach[p]`` = the run position of the last reader of the ALIAS output (view / chunk2_c1: pointer arithmetic that must
    stay inside its region) of ``idx[p]`` (``len(idx)`` when the graph returns it), -1 for any other op. A window
    ``[a, b)`` is legal iff ``max(reach[a:b]) < b`` (``fuse_region``: "an alias inside the region is read outside it")."""
    ops = graph["ops"]
    n = len(idx)
    last: dict = {}
    for p, i in enumerate(idx):
        for t in ops[i].get("inputs", []):
            last[t] = p
    gouts = set((graph.get("output") or {}).get("tensors") or [(graph.get("output") or {}).get("tensor")])
    out = []
    for p, i in enumerate(idx):
        r = -1
        if ops[i]["op"] in ALIAS_OPS:
            for t in ops[i].get("outputs", []):
                r = max(r, n if t in gouts else last.get(t, -1))
        out.append(r)
    return out


def legal_end(reach: list, a: int, b: int) -> bool:
    return b > a and max(reach[a:b]) < b


@dataclass
class Motif:
    start: int            # position in the run (0-based)
    period: int
    reps: int

    @property
    def end(self) -> int:
        return self.start + self.period * self.reps


def detect_motifs(tokens: list, max_period: int = MAX_PERIOD, min_reps: int = MIN_REPS) -> list:
    """Periodic segments of a token sequence: non-overlapping ``Motif`` (start, period, reps), found by the longest runs of
    ``t[i] == t[i+p]`` for every period p; the PRIMITIVE (smallest) period wins among equal coverage. A segment needs
    ``min_reps`` full periods. Pure; O(n * max_period)."""
    n = len(tokens)
    segs = []
    for p in range(1, min(max_period, n // min_reps) + 1):
        a = None
        for i in range(n - p + 1):
            same = i < n - p and tokens[i] == tokens[i + p]
            if same and a is None:
                a = i
            elif not same and a is not None:
                reps = (i + p - a) // p
                if reps >= min_reps:
                    segs.append((a, p, reps))
                a = None
    segs.sort(key=lambda s: (-(s[1] * s[2]), s[1], s[0]))
    out, taken = [], []
    for a, p, reps in segs:
        b = a + p * reps
        if any(a < y and x < b for x, y in taken):
            continue
        out.append(Motif(a, p, reps))                   # (multiples of an accepted period overlap it: skipped above)
        taken.append((a, b))
    return sorted(out, key=lambda m: m.start)


# ------------------------------------------------------------------------------------------------ partitioning
def _count_nonalias(ops, idx) -> int:
    return sum(ops[i]["op"] not in ALIAS_OPS for i in idx)


def _legal_cuts(reach: list, s: int, lo: int, hi: int) -> list:
    """The cuts ``c`` in ``[lo, hi]`` for which the window ``[s, c)`` is legal (``legal_end``)."""
    run, out = max(reach[s:lo], default=-1), []
    for c in range(lo, hi + 1):
        if c - 1 >= lo:
            run = max(run, reach[c - 1])
        if run < c:
            out.append(c)
    return out


def _best_cut(cost: list, reach: list, s: int, lo: int, hi: int, prefer_single: bool):
    """A legal cut for a window starting at ``s`` in ``[lo, hi]``: the largest single-frontier cut (cost 1) when
    ``prefer_single`` and one exists, else the cheapest (largest among ties); None when none is legal."""
    cands = _legal_cuts(reach, s, lo, hi)
    if not cands:
        return None
    if prefer_single:
        singles = [c for c in cands if cost[c] == 1]
        if singles:
            return max(singles)
    m = min(cost[c] for c in cands)
    return max(c for c in cands if cost[c] == m)


def fix_boundaries(wins: list, reach: list, size: int) -> list:
    """Make every window legal (``legal_end``): an illegal end moves to the nearest legal cut (both neighbours within ``size``);
    when none exists the window merges with an adjacent next one if the union is legal and fits, else is dropped (its ops
    stay plain)."""
    wins = [list(w) for w in sorted(wins)]
    out: list = []
    k = 0
    while k < len(wins):
        a, b = wins[k]
        if legal_end(reach, a, b):
            out.append((a, b))
            k += 1
            continue
        nxt = wins[k + 1] if k + 1 < len(wins) else None
        limit = (nxt[1] - 1) if nxt and nxt[0] == b else (nxt[0] if nxt else len(reach))
        fixed = None
        for d in range(1, len(reach) + 1):
            for c in (b - d, b + d):
                if a < c <= limit and c - a <= size and legal_end(reach, a, c) and (
                        not (nxt and nxt[0] == b) or nxt[1] - c <= size):
                    fixed = c
                    break
            if fixed:
                break
        if fixed:
            out.append((a, fixed))
            if nxt and nxt[0] == b:
                wins[k + 1][0] = fixed
            k += 1
        elif nxt and nxt[0] == b and nxt[1] - a <= size and legal_end(reach, a, nxt[1]):
            wins[k + 1][0] = a
            k += 1
        else:
            k += 1                                      # dropped
    return out


def chunk_segment(cost: list, reach: list, lo: int, hi: int, m: int, *, mode: str = "chunk") -> list:
    """Windows ``[(a, b), ...]`` partitioning the run positions ``[lo, hi)`` into legal pieces of at most ``m`` ops.
    ``chunk``: cut in the last half of the piece at a single-frontier cut when there is one, else the cheapest; ``blocks``:
    the LARGEST single-frontier cut that fits (a whole block is never cut), else as ``chunk``. Both fall back to the largest
    legal cut of the whole piece; when none is legal within ``m`` (one unsplittable block bigger than ``m``) its ops stay
    plain and the walk resumes at the next position where a window can start."""
    out, s = [], lo
    while s < hi:
        top, half = min(s + m, hi), s + max(m // 2, 1)
        if hi - s <= m and legal_end(reach, s, hi):
            out.append((s, hi))
            break
        top = min(top, hi - 1) if hi - s > m else top
        c = None
        if mode == "blocks":
            singles = [x for x in range(s + 1, top + 1) if cost[x] == 1 and legal_end(reach, s, x)]
            c = max(singles) if singles else None
        if c is None:
            c = _best_cut(cost, reach, s, min(half + 1, top), top, True)
        if c is None:
            c = _best_cut(cost, reach, s, s + 1, top, False)
        if c is None:                                  # an unsplittable section: the ops stay plain, resume after it
            nxt = next((x for x in range(top + 1, hi + 1) if legal_end(reach, s, x)), None)
            if nxt is None:
                break
            s = nxt
            continue
        out.append((s, c))
        s = c
    return out


def coalesce(wins: list, reach: list, size: int) -> list:
    """Merge ADJACENT windows left to right while the union fits ``size`` and is legal (a motif / block split that left
    fragments does not leave 6-op windows next to each other)."""
    out: list = []
    for a, b in sorted(wins):
        if out and out[-1][1] == a and b - out[-1][0] <= size and legal_end(reach, out[-1][0], b):
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def global_legal(reach: list) -> list:
    """``ok[c]`` for the cuts 0..n: no alias produced BEFORE c is read at or after c (a cut between the layers of a dense block
    is not: every later concat reads the earlier layers' views). The units between consecutive legal cuts are atomic."""
    n, run, ok = len(reach), -1, [True]
    for c in range(1, n + 1):
        run = max(run, reach[c - 1])
        ok.append(c == n or run < c)
    return ok


def snap_motifs(motifs: list, reach: list, min_gap: int = 4) -> list:
    """Motifs whose ends respect the atomic units: a motif START moves to the first cut where the gap window before it (when it
    has at least ``min_gap`` ops) is legal, a motif END moves down to the last cut where the motif's own window is legal (a motif
    boundary inside a dense block would cut views that later concats read). Dropped when fewer than ``MIN_REPS`` periods or
    ``MIN_MOTIF_OPS`` ops are left."""
    out, prev_end = [], 0
    for m in sorted(motifs, key=lambda x: x.start):
        a = m.start
        if a - prev_end >= min_gap and not legal_end(reach, prev_end, a):
            a = next((c for c in range(a, m.end + 1) if legal_end(reach, prev_end, c)), m.end)
        b = m.end
        if not legal_end(reach, a, b):
            b = next((c for c in range(b, a, -1) if legal_end(reach, a, c)), a)
        reps = (b - a) // m.period
        if b > a and reps >= MIN_REPS and reps * m.period >= MIN_MOTIF_OPS:
            out.append(Motif(a, m.period, reps))
            prev_end = a + reps * m.period
    return out


def motif_windows(m: Motif, size: int) -> list:
    """Balanced windows of whole periods, each at most ``size`` ops: ``[(a, b), ...]`` in run positions; [] when one period
    does not fit."""
    if m.period > size:
        return []
    per = max(1, size // m.period)
    w = -(-m.reps // per)                              # number of windows
    base, extra = divmod(m.reps, w)
    out, s = [], m.start
    for k in range(w):
        r = base + (1 if k < extra else 0)
        out.append((s, s + r * m.period))
        s += r * m.period
    return out


def _merge_small(wins: list, ops, idx, size: int, min_ops: int) -> list:
    """Windows with fewer than ``min_ops`` non-alias ops merge into an adjacent window when the union fits ``size``, else
    are dropped (their ops stay plain)."""
    wins = list(wins)
    changed = True
    while changed:
        changed = False
        for k, (a, b) in enumerate(wins):
            if _count_nonalias(ops, idx[a:b]) >= min_ops:
                continue
            for nb in (k - 1, k + 1):
                if 0 <= nb < len(wins):
                    x, y = wins[nb]
                    if (x == b or y == a) and max(b, y) - min(a, x) <= size:
                        wins[nb] = (min(a, x), max(b, y))
                        del wins[k]
                        changed = True
                        break
            else:
                del wins[k]
                changed = True
            if changed:
                break
    return wins


def partition_run(graph: dict, idx: list, size: int, *, mode: str, min_ops: int, motifs: list | None = None,
                  cost: list | None = None, reach: list | None = None) -> list:
    """Windows (lists of op indices) over one eligible run ``idx``: ``mode`` chunk | blocks | motif (see module doc)."""
    ops = graph["ops"]
    n = len(idx)
    cost = cost if cost is not None else cut_costs(graph, idx)
    reach = reach if reach is not None else alias_reach(graph, idx)
    motifs = snap_motifs([m for m in (motifs or []) if m.period <= size], reach)
    wins: list = []
    if mode == "motif":
        for m in motifs:
            wins += motif_windows(m, size)
    else:
        pos = 0
        segs = []
        for m in sorted(motifs, key=lambda x: x.start):
            if m.start > pos:
                segs.append(("gap", pos, m.start))
            segs.append(("motif", m, None))
            pos = m.end
        if pos < n:
            segs.append(("gap", pos, n))
        if not segs:
            segs = [("gap", 0, n)]
        for kind, a, b in segs:
            if kind == "motif":
                wins += motif_windows(a, size) if mode == "chunk" else chunk_segment(cost, reach, a.start, a.end, size, mode="blocks")
            else:
                wins += chunk_segment(cost, reach, a, b, size, mode=mode)
    wins = fix_boundaries(sorted(wins), reach, size)
    if mode != "motif":
        wins = coalesce(wins, reach, size)
    wins = _merge_small(wins, ops, idx, size, min_ops)
    wins = fix_boundaries(wins, reach, size)
    return [[idx[p] for p in range(a, b)] for a, b in wins]


# ------------------------------------------------------------------------------------------------ candidates
@dataclass
class Candidate:
    label: str
    groups: list                                     # [[atom names]] one list per window
    stats: dict = field(default_factory=dict)
    est_gain_pct: float = 0.0

    @property
    def id(self) -> str:
        return f"region:win:{self.label}"


def _group_names(ops, window) -> list:
    return list(dict.fromkeys(a for i in window for a in _atoms(ops[i]["name"])))


class _cap_env:
    """``$MB_REGION_MAX_MEMBERS`` raised to ``cap`` inside the block (the windows are proposed and validated for the cap they
    need whatever the caller's environment says), restored after."""

    def __init__(self, cap: int):
        self.cap, self.old = int(cap), None

    def __enter__(self):
        self.old = os.environ.get("MB_REGION_MAX_MEMBERS")
        from .region_slots import max_members
        if max_members() < self.cap:
            os.environ["MB_REGION_MAX_MEMBERS"] = str(self.cap)
        return self

    def __exit__(self, *a):
        if self.old is None:
            os.environ.pop("MB_REGION_MAX_MEMBERS", None)
        else:
            os.environ["MB_REGION_MAX_MEMBERS"] = self.old


def validate_groups(graph: dict, specs: dict, groups: list) -> tuple:
    """(valid groups, [{group head, why}]): every group applied window by window to a working copy with the one authority,
    ``region_slots.fuse_region`` (the graph the NEXT window sees has the previous ones fused). The caller's graph is not
    modified."""
    from . import graph_hints as gh
    from .region_slots import fuse_region
    g = copy.deepcopy(graph)
    ok, bad = [], []
    for grp in groups:
        try:
            cur = gh._locate(g, grp)
            g, _ = fuse_region(g, cur, specs, "ops")
        except (ValueError, StopIteration, KeyError) as e:
            bad.append({"first": grp[0], "last": grp[-1], "n": len(grp), "why": str(e)[:200]})
            continue
        ok.append(grp)
    return ok, bad


def _share(ops, windows, op_us, total_us) -> float:
    if op_us and total_us:
        return min(1.0, sum(op_us.get(ops[i]["name"], 0.0) for w in windows for i in w) / total_us)
    n = len(ops) or 1
    return sum(len(w) for w in windows) / n


def propose_windows(graph: dict, specs: dict, cfg: WindowConfig, **kw) -> tuple:
    """:func:`_propose_windows` with the member cap lifted to what the configuration needs while validating."""
    with _cap_env(cfg.member_cap()):
        return _propose_windows(graph, specs, cfg, **kw)


def _propose_windows(graph: dict, specs: dict, cfg: WindowConfig, *, op_us: dict | None = None,
                     total_us: float | None = None, validate: bool = True) -> tuple:
    """-> (candidates ranked best first (at most ``cfg.max_candidates``), report). ``graph`` must have unique op names
    (``graph_hints.unique_names``). ``op_us`` {op name: us} / ``total_us`` rank by time share (op-count share without).
    ``report``: runs, motifs, every label tried with its window sizes, dropped groups and seconds."""
    t0 = time.time()
    ops = graph["ops"]
    runs = [r for r in eligible_runs(graph) if _count_nonalias(ops, r) >= cfg.min_ops]
    report: dict = {"runs": [len(r) for r in runs], "motifs": [], "tried": [], "dropped": [], "seconds": 0.0}
    if not runs:
        report["seconds"] = round(time.time() - t0, 3)
        return [], report
    sets: dict = {}                                     # label -> [windows of op indices]

    def add(label, wins):
        wins = [w for w in wins if _count_nonalias(ops, w) >= cfg.min_ops]
        if wins:
            sets[label] = wins

    run_info = []
    for r in runs:
        cost = cut_costs(graph, r)
        toks = [kind_token(ops[i]) for i in r]
        motifs = detect_motifs(toks)
        for m in motifs:
            report["motifs"].append({"first": ops[r[m.start]]["name"], "period": m.period, "reps": m.reps, "ops": m.period * m.reps})
        run_info.append((r, cost, motifs))
    big = max(runs, key=len)
    if len(big) <= cfg.hard_cap:
        add("whole", [big])
    for m in cfg.sizes:
        label = f"chunk{m}"
        acc = []
        for r, cost, motifs in run_info:
            acc += partition_run(graph, r, m, mode="chunk", min_ops=cfg.min_ops, motifs=motifs, cost=cost)
        add(label, acc)
    msize = max(cfg.sizes)
    acc = []
    for r, cost, motifs in run_info:
        gl = global_legal(alias_reach(graph, r))
        if sum(cost[c] == 1 and gl[c] for c in range(1, len(r))) >= max(2, len(r) // 50):   # single-frontier cuts: blocks are meaningful
            acc += partition_run(graph, r, msize, mode="blocks", min_ops=cfg.min_ops, motifs=None, cost=cost)
    add(f"blocks{msize}", acc)
    if any(mo for _, _, mo in run_info):
        for m in cfg.sizes:
            acc = []
            for r, cost, motifs in run_info:
                acc += partition_run(graph, r, m, mode="motif", min_ops=cfg.min_ops, motifs=motifs, cost=cost)
            add(f"motif{m}", acc)
    # identical window sets (e.g. chunk256 == whole on a small graph) keep the first label
    seen: dict = {}
    cands: list = []
    for label, wins in sets.items():
        key = json.dumps(sorted([w[0], w[-1], len(w)] for w in wins))
        if key in seen:
            report["tried"].append({"label": label, "duplicate_of": seen[key]})
            continue
        seen[key] = label
        pairs = [(w, _group_names(ops, w)) for w in wins]
        groups = [grp for _, grp in pairs]
        dropped: list = []
        if validate:
            groups, dropped = validate_groups(graph, specs, groups)
        report["tried"].append({"label": label, "windows": len(wins), "sizes": [len(w) for w in wins],
                                "valid": len(groups), "dropped": len(dropped)})
        report["dropped"] += [{"label": label, **d} for d in dropped]
        if not groups:
            continue
        keep = [w for w, grp in pairs if grp in groups]
        share = _share(ops, keep, op_us, total_us)
        c = Candidate(label, groups, {"windows": len(groups), "members": sum(len(g) for g in groups),
                                      "largest": max(len(w) for w in keep), "share": round(share, 4)},
                      est_gain_pct=round(100 * GAIN_PER_SHARE * share, 2))
        cands.append(c)
    order = {"whole": 0, "chunk": 1, "motif": 2, "blocks": 3}
    cands.sort(key=lambda c: (-c.est_gain_pct, c.stats["windows"], order.get(re.sub(r"\d+$", "", c.label), 9), c.label))
    report["seconds"] = round(time.time() - t0, 3)
    return cands[:cfg.max_candidates], report


def hint_for(c: Candidate, ir: str, cfg: WindowConfig | None = None) -> dict:
    """The ordinary ``fuse_region`` hint record of a candidate (``windows``: one explicit member list per window)."""
    return {"id": c.id, "kind": "fuse_region", "pattern": "ops", "windows": c.groups, "target": "cuda", "status": "proposed",
            "source": [{"by": SOURCE, "ir": ir, "label": c.label, **c.stats, "est_gain_pct": c.est_gain_pct}]}
