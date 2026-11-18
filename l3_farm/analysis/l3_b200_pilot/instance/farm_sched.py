"""Scheduling policies for the L3 farm (analysis/scheduling.md). Pure functions over plain dicts (no ssh, no
clock reads), so the decisions are unit-testable with stubs (tests/test_farm_sched.py); farm.py calls them.

config.json "scheduling" (absent = {"mode": "independent"}: file order, a problem starts as soon as a node is
free -- exactly the dispatcher's behaviour before this module existed):

  mode            independent | staged | pooled
  seed            "auto" | [names or numbers]     staged: the seed wave (auto: cheap + diverse, see seed_wave)
  seed_n          3                                size of an automatic seed wave
  wave_size       0                                later waves (0 = seed_n)
  gate_w          2                                open the next wave when >= W problems of the current one merged
  gate_on         any | final                      what counts as merged (any = a round or final merge)
  max_wait_min    45                               ... or when the current wave is this old
  idle_max_min    10                               never leave a ready node idle longer while problems wait
  expensive_frac  0.2                              the most expensive fraction always runs last
  sim_buckets     4                                similarity is compared in buckets (ties -> cheaper first)
  cost_from       auto | history | ir | static     estimator inputs (auto: history, else IR, else source)
  history_dirs    ["results"]                      where earlier runs' work/state.json + IRs are looked up
  cost_base, cost_per_pair   20, 10                estimated LLM calls = base + per_pair x min(pairs, top_k)
  pool            {...}                            pooled mode (farm.py pooled_tick)
  barrier         {...}                            barrier scheduling (X-151, analysis/barrier_scheduling.md): phase 1 (every
                                                   problem runs to the end of its characterization pass and parks), a global
                                                   soft barrier, phase 2 (resume, longest first); see BARRIER_DEFAULTS
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

DEFAULTS = {"mode": "independent", "seed": "auto", "seed_n": 3, "wave_size": 0, "gate_w": 2, "gate_on": "any",
            "max_wait_min": 45, "idle_max_min": 10, "expensive_frac": 0.2, "sim_buckets": 4, "cost_from": "auto",
            "history_dirs": ["results"], "cost_base": 20.0, "cost_per_pair": 10.0}


def conf(cfg: dict) -> dict:
    return {**DEFAULTS, **((cfg or {}).get("scheduling") or {})}


# ---------------------------------------------------------------- estimator

_DT = re.compile(r"_(f16|f32|bf16|i8|fp16|fp32)$")


def op_kind(op: str) -> str:
    return _DT.sub("", op or "")


def ir_pairs(graph: dict) -> list:
    """(op kind, shape json) of every original op of an extracted IR (fused ops expanded into their sub-ops)."""
    out = []
    for o in graph.get("ops", []):
        subs = o.get("sub_ops") or [o]
        for x in subs:
            out.append((op_kind(x.get("op", "")), json.dumps(x.get("shape") or {}, sort_keys=True)))
    return out


_LAYER = re.compile(r"nn\.([A-Z][A-Za-z0-9]+)\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)")
_FN = re.compile(r"(?:torch|F)\.(matmul|bmm|softmax|relu|gelu|tanh|sigmoid|cat|einsum|layer_norm|"
                 r"scaled_dot_product_attention|conv2d|linear|max_pool2d|avg_pool2d|adaptive_avg_pool2d|dropout|"
                 r"exp|cumsum|transpose|permute)\b")
_ALIAS = {"Conv2d": "conv2d", "Linear": "linear", "BatchNorm2d": "batchnorm2d", "ReLU": "relu", "ReLU6": "relu6",
          "MaxPool2d": "maxpool2d", "AvgPool2d": "avgpool2d", "AdaptiveAvgPool2d": "adaptive_avgpool2d",
          "LayerNorm": "layernorm", "GELU": "gelu", "Softmax": "softmax", "Dropout": "dropout", "LSTM": "lstm",
          "GRU": "gru", "RNN": "rnn", "Embedding": "embedding", "Sigmoid": "sigmoid", "Tanh": "tanh"}


def static_pairs(src: str) -> list:
    """Fallback without an IR: layer constructors / functional calls in the problem source (shapes = the
    constructor arguments; loops are not unrolled, so this undercounts deep models)."""
    out = [(_ALIAS.get(m.group(1), m.group(1).lower()), m.group(2).strip()) for m in _LAYER.finditer(src)]
    out += [(m.group(1), "") for m in _FN.finditer(src)]
    return out


def find_ir(name: str, dirs: list) -> Path | None:
    for d in dirs:
        base = Path(d) / name
        for pat in ("irs/*/graph.json", "work/r00/profile/*/ir/graph.json", "work/final/*/ir/graph.json"):
            hits = sorted(base.glob(pat))
            if hits:
                return hits[0]
    return None


def history_calls(name: str, dirs: list) -> float | None:
    for d in dirs:
        st = Path(d) / name / "work" / "state.json"
        if st.exists():
            try:
                rounds = json.loads(st.read_text()).get("rounds") or []
            except (OSError, ValueError):
                continue
            calls = [r.get("llm_calls_total") for r in rounds if r.get("llm_calls_total") is not None]
            if calls:
                return float(max(calls))
    return None


def estimate(name: str, problem_file: Path | None, sc: dict, top_k: int = 0) -> dict:
    """{kinds, pairs, cost, cost_src, ops_src}: cost in estimated LLM calls (relative; for ordering only)."""
    dirs = sc.get("history_dirs") or []
    pairs, src = [], None
    g = find_ir(name, dirs)
    if g is not None:
        try:
            pairs, src = ir_pairs(json.loads(g.read_text())), f"ir:{g}"
        except (OSError, ValueError):
            pairs = []
    if not pairs and problem_file is not None and Path(problem_file).exists():
        pairs, src = static_pairs(Path(problem_file).read_text()), "static"
    distinct = len(set(pairs))
    kinds = sorted({k for k, _ in pairs if k})
    n = min(distinct, top_k) if top_k else distinct
    cost_ir = sc["cost_base"] + sc["cost_per_pair"] * n
    hc = history_calls(name, dirs) if sc.get("cost_from") in ("auto", "history") else None
    if hc is not None:
        cost, csrc = hc, "history"
    else:
        cost, csrc = cost_ir, "ir" if (src or "").startswith("ir:") else "static"
    return {"name": name, "kinds": kinds, "kind_counts": dict(Counter(k for k, _ in pairs)), "pairs": distinct,
            "cost": round(cost, 1), "cost_src": csrc, "ops_src": src}


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.0


def similarity(est: dict, merged: list, ests: dict) -> tuple:
    """(max Jaccard with one merged problem's op kinds, coverage of this problem's kinds by all merged ones)."""
    k = set(est.get("kinds") or [])
    best = max((jaccard(k, ests[m]["kinds"]) for m in merged if m in ests), default=0.0)
    allm = set().union(*[set(ests[m]["kinds"]) for m in merged if m in ests]) if merged else set()
    cov = len(k & allm) / len(k) if k else 0.0
    return round(best, 3), round(cov, 3)


# ---------------------------------------------------------------- staged ordering

def expensive_set(ests: dict, names: list, frac: float) -> set:
    if not names or frac <= 0:
        return set()
    n = max(1, int(math.ceil(len(names) * frac))) if len(names) > 2 else 0
    ranked = sorted(names, key=lambda x: (-ests[x]["cost"], x))
    return set(ranked[:n])


def seed_wave(ests: dict, names: list, sc: dict, order: dict) -> list:
    """Explicit list (names or problem numbers), or automatic: among the non-expensive problems, the
    cheapest first, then greedily the one adding the most op kinds not yet covered (ties: cheaper)."""
    seed = sc.get("seed", "auto")
    if isinstance(seed, list) and seed:
        out = []
        for x in seed:
            m = [n for n in names if n == str(x) or n.split("_")[0] == str(x)]
            out += [n for n in m if n not in out]
        return out
    k = int(sc.get("seed_n", 3))
    exp = expensive_set(ests, names, float(sc.get("expensive_frac", 0.2)))
    pool = sorted((n for n in names if n not in exp), key=lambda n: (ests[n]["cost"], order.get(n, 0)))
    pool = pool[: max(2 * k, int(math.ceil(len(pool) / 2)))]       # the cheaper part (>= 2 candidates per seat)
    out, covered = [], set()
    while pool and len(out) < k:
        # most op kinds not covered yet per unit of estimated cost (ties: cheaper, then file order)
        best = max(pool, key=lambda n: (len(set(ests[n]["kinds"]) - covered) / max(ests[n]["cost"], 1.0),
                                        -ests[n]["cost"], -order.get(n, 0)))
        if out and not (set(ests[best]["kinds"]) - covered):     # nothing new: take the cheapest instead
            best = pool[0]
        out.append(best)
        covered |= set(ests[best]["kinds"])
        pool.remove(best)
    return out


def rank(ests: dict, pending: list, merged: list, sc: dict, order: dict, all_names: list) -> list:
    """Pending problems, best first: non-expensive before expensive; higher similarity to what has merged
    (bucketed); then cheaper; then file order. -> [(name, info)]."""
    exp = expensive_set(ests, all_names, float(sc.get("expensive_frac", 0.2)))
    b = max(1, int(sc.get("sim_buckets", 4)))
    rows = []
    for n in pending:
        sim, cov = similarity(ests[n], merged, ests)
        info = {"sim": sim, "cov": cov, "cost": ests[n]["cost"], "expensive": n in exp,
                "sim_bucket": math.floor(sim * b + 1e-9) / b}
        # the expensive tier only by cost (the most expensive last); the rest by similarity bucket, then cost
        rows.append(((n in exp, 0 if n in exp else -info["sim_bucket"], ests[n]["cost"], order.get(n, 0)), n, info))
    rows.sort(key=lambda r: r[0])
    return [(n, info) for _, n, info in rows]


def merged_names(problems: dict, gate_on: str = "any") -> list:
    out = []
    for n, p in problems.items():
        g = p.get("gk") or {}
        if not g.get("merged_version"):
            continue
        if gate_on == "final" and p.get("status") not in ("done", "failed"):
            continue
        out.append(n)
    return sorted(out, key=lambda n: (problems[n].get("gk") or {}).get("first_merged_at", 0))


def staged_pick(problems: dict, ests: dict, sched: dict, sc: dict, now: float, free: list, idle_since: dict,
                order: dict) -> list:
    """The staged policy's starts for this tick. ``problems``: farm.json problems; ``sched``: persisted state
    (mutated: waves, decisions); ``free``: ready, idle node ids; ``idle_since``: node -> when it became idle.
    -> [(node, name, why)]."""
    names = list(problems)
    pending = [n for n in names if problems[n]["status"] == "pending" and n in ests]
    waves = sched.setdefault("waves", [])
    dec = sched.setdefault("decisions", [])
    out = []
    if not pending:
        return out
    if not waves:
        sw = [n for n in seed_wave(ests, [n for n in names if n in ests], sc, order) if n in pending]
        waves.append({"k": 0, "members": [], "plan": sw, "size": len(sw) or int(sc["seed_n"]), "opened": now,
                      "why": "seed wave (" + ("explicit" if isinstance(sc.get("seed"), list) else "auto: cheap + diverse")
                             + f"): {sw}"})
        dec.append({"t": now, "event": "wave", "k": 0, "why": waves[-1]["why"]})
    size_next = int(sc.get("wave_size") or sc.get("seed_n") or 3)
    W = int(sc.get("gate_w", 2))
    for node in free:
        pending = [n for n in pending if problems[n]["status"] == "pending" and n not in [x[1] for x in out]]
        if not pending:
            break
        cur = waves[-1]
        why = None
        if len(cur["members"]) < cur["size"]:
            plan = [n for n in cur.get("plan", []) if n in pending]
            if plan:
                name, why = plan[0], f"wave {cur['k']} (seed)"
            else:
                merged = merged_names(problems, sc.get("gate_on", "any"))
                r = rank(ests, pending, merged, sc, order, [n for n in names if n in ests])
                name, info = r[0]
                why = (f"wave {cur['k']} rank 1/{len(r)} sim {info['sim']} cov {info['cov']} cost {info['cost']}"
                       f"{' expensive' if info['expensive'] else ''}")
        else:
            merged = merged_names(problems, sc.get("gate_on", "any"))
            # merged, or finished without anything to merge (unsupported / failed / skipped): nothing to wait for
            n_merged = sum(1 for m in cur["members"] if m in merged or
                           problems.get(m, {}).get("status") in ("unsupported", "failed", "skipped", "done"))
            need = min(W, len(cur["members"]))
            age = (now - cur["opened"]) / 60
            idle = (now - idle_since.get(node, now)) / 60
            if n_merged >= need:
                gate = f"gate: {n_merged}/{len(cur['members'])} of wave {cur['k']} merged (W={W})"
            elif age >= float(sc.get("max_wait_min", 45)):
                gate = f"gate: wave {cur['k']} open {age:.0f} min >= max_wait {sc.get('max_wait_min')}"
            elif idle >= float(sc.get("idle_max_min", 10)):
                gate = f"no-idle: node {node} idle {idle:.0f} min >= {sc.get('idle_max_min')} with {len(pending)} waiting"
            else:
                last = sched.get("waiting")
                msg = (f"waiting: wave {cur['k']} {n_merged}/{need} merged, open {age:.0f}/{sc.get('max_wait_min')} min, "
                       f"node {node} idle {idle:.0f}/{sc.get('idle_max_min')} min")
                if last != msg.split(", node")[0]:
                    dec.append({"t": now, "event": "wait", "why": msg})
                    sched["waiting"] = msg.split(", node")[0]
                continue
            r = rank(ests, pending, merged, sc, order, [n for n in names if n in ests])
            nw = {"k": cur["k"] + 1, "members": [], "size": size_next, "opened": now, "why": gate}
            waves.append(nw)
            dec.append({"t": now, "event": "wave", "k": nw["k"], "why": gate,
                        "ranking": [[n, i["sim"], i["cost"], i["expensive"]] for n, i in r[:8]]})
            name, info = r[0]
            why = (f"wave {nw['k']} ({gate}) rank 1/{len(r)} sim {info['sim']} cov {info['cov']} cost {info['cost']}"
                   f"{' expensive' if info['expensive'] else ''}")
            cur = nw
        if not cur["members"]:
            cur["opened"] = now                  # a wave's age counts from its first start
        cur["members"].append(name)
        sched.pop("waiting", None)
        dec.append({"t": now, "event": "start", "name": name, "node": node, "why": why})
        out.append((node, name, why))
    del dec[:-200]
    return out


# ---------------------------------------------------------------- pooled: static lease queues

def lease_queues(running: list, hosts: list, ests: dict, policy: str = "share", current: dict | None = None) -> dict:
    """Which pool hosts each running problem may lease GPUs from (static per start; the per-GPU flock on the
    host serialises timed runs across problems). policy all: every problem -> every host; share: hosts in
    proportion to estimated cost (each problem >= 1 host, the costliest get more). ``current``: the queues of
    problems already running (kept); a newcomer gets the least-used hosts."""
    hosts = sorted(hosts)
    if not running or not hosts:
        return {}
    if policy == "all" or len(running) == 1:
        return {n: list(hosts) for n in running}
    cur = {n: [h for h in q if h in hosts] for n, q in (current or {}).items() if n in running and q}
    tot = sum(max(ests.get(n, {}).get("cost", 1.0), 1.0) for n in running)
    out = dict(cur)
    use = {h: sum(h in q for q in cur.values()) for h in hosts}
    for n in sorted((n for n in running if n not in cur), key=lambda n: -max(ests.get(n, {}).get("cost", 1.0), 1.0)):
        share = max(ests.get(n, {}).get("cost", 1.0), 1.0) / tot
        k = max(1, min(len(hosts), int(round(share * len(hosts)))))
        pick = sorted(hosts, key=lambda h: (use[h], h))[:k]
        out[n] = pick
        for h in pick:
            use[h] += 1
    return out


# ---------------------------------------------------------------- barrier scheduling (X-151; analysis/barrier_scheduling.md)
# Per-problem fields (farm.json): phase p1 | parked | p2 (absent = a problem that never ran under the barrier), parked_at (set when
# the problem parked: it has ARRIVED at the barrier), park_node. status "parked" = parked, pack kept, no slot held.
# Persisted decision state: sched["barrier"] = {open, opened, why, last_arrival, log[]}; once open it never closes.

BARRIER_DEFAULTS = {"enabled": False, "after": "char", "min_frac": 0.9, "max_wait_min": 90.0,
                    "phase1_order": "cheap_first", "phase2_order": "longest_first", "phase1_head": 0}
TERMINAL = ("failed", "skipped", "unsupported")


def bconf(sc: dict) -> dict:
    return {**BARRIER_DEFAULTS, **((sc or {}).get("barrier") or {})}


def barrier_enabled(sc: dict) -> bool:
    return bool(bconf(sc)["enabled"])


def is_parked(p: dict) -> bool:
    """Parked and waiting: status parked, or a pending problem whose phase is still parked (a manual requeue)."""
    return p.get("status") == "parked" or (p.get("status") == "pending" and p.get("phase") == "parked")


def census(problems: dict) -> dict:
    """The barrier's view of the problem set. considered = every problem that is not failed / skipped / unsupported
    (excluded ones are not waited for); arrived = considered problems that parked at some point, or finished
    without parking (nothing left to wait for); waiting = the rest (running in phase 1, not started, retrying)."""
    cons = [n for n, p in problems.items() if p.get("status") not in TERMINAL]
    arr = [n for n in cons if problems[n].get("parked_at") or problems[n].get("status") == "done"]
    return {"considered": cons, "arrived": arr, "waiting": [n for n in cons if n not in arr],
            "parked": [n for n in cons if is_parked(problems[n])]}


def last_arrival(problems: dict, arrived: list) -> float:
    t = [problems[n].get("parked_at") or problems[n].get("finished") or 0 for n in arrived]
    return max(t) if t else 0.0


def barrier_update(problems: dict, bstate: dict, sc: dict, now: float) -> str | None:
    """Decide (idempotently: ``bstate`` is persisted, an open barrier stays open) whether the barrier opens now. Rules, in order:
      all   every considered problem has arrived;
      frac  arrived >= ceil(min_frac x considered) (default 90 %: the stragglers are NOT waited for: they park later and go
            straight into phase 2, no second barrier);
      stall at least one problem arrived and nothing new arrived for ``max_wait_min`` (clock = the LAST arrival, it restarts at
            every new park / finish): a hung or very slow phase 1 never holds the rest back.
    -> the reason when it opened in this call, else None."""
    bc = bconf(sc)
    c = census(problems)
    la = last_arrival(problems, c["arrived"])
    if la:
        bstate["last_arrival"] = la
    if bstate.get("open"):
        return None
    n, a = len(c["considered"]), len(c["arrived"])
    need = max(1, math.ceil(float(bc["min_frac"]) * n - 1e-9)) if n else 0
    why = None
    if n and a >= n:
        why = f"all: {a}/{n} problems arrived"
    elif n and a >= need:
        why = f"frac: {a}/{n} parked >= {float(bc['min_frac']):.0%} (need {need})"
    elif a >= 1 and now - la >= float(bc["max_wait_min"]) * 60:
        why = (f"stall: {a}/{n} parked, no new park for {(now - la) / 60:.0f} min >= max_wait {bc['max_wait_min']} min "
               f"(still in phase 1: {sorted(c['waiting'])[:8]})")
    if why:
        bstate.update(open=True, opened=now, why=why, arrived=a, considered=n)
        bstate.setdefault("log", []).append({"t": now, "event": "open", "why": why})
        del bstate["log"][:-50]
    return why


def phase1_order(names: list, ests: dict, order: dict, bc: dict) -> list:
    """Phase 1 admission: cheap first (estimated LLM calls ascending) so most problems park early and the barrier opens soon;
    optionally (``phase1_head`` k) the k most expensive first, so that the longest problems are not still in phase 1 when the
    barrier opens (a tail shortener: their phase 2 would otherwise start late)."""
    if bc.get("phase1_order") == "file":
        return sorted(names, key=lambda n: order.get(n, 0))
    cheap = sorted(names, key=lambda n: (ests.get(n, {}).get("cost", 0.0), order.get(n, 0)))
    k = int(bc.get("phase1_head") or 0)
    if k > 0:               # the k most expensive problems are admitted first: their phase 1 is as short as any, and they then
        top = sorted(names, key=lambda n: (-ests.get(n, {}).get("cost", 0.0), order.get(n, 0)))[:k]   # start phase 2 at the barrier
        return top + [n for n in cheap if n not in top]
    return cheap


def phase2_order(names: list, ests: dict, order: dict, bc: dict, sc: dict | None = None, merged: list | None = None) -> list:
    """Phase 2 admission, after the barrier: the seeding reason for cheap-first is gone, so the LONGEST (most expensive by the
    staged estimator) first, which shortens the tail; ``rank`` = the staged ranking (similarity, expensive last), ``file``."""
    mode = bc.get("phase2_order", "longest_first")
    if mode == "rank":
        r = rank({n: ests[n] for n in names if n in ests}, [n for n in names if n in ests], merged or [], sc or DEFAULTS, order,
                 [n for n in names if n in ests])
        return [n for n, _ in r] + [n for n in names if n not in ests]
    if mode == "file":
        return sorted(names, key=lambda n: order.get(n, 0))
    return sorted(names, key=lambda n: (-ests.get(n, {}).get("cost", 0.0), order.get(n, 0)))


def barrier_next(problems: dict, ests: dict, bstate: dict, sc: dict, order: dict) -> tuple | None:
    """The next start: (name, why, park). Barrier closed: the cheapest not-yet-started problem, started WITH the park flag
    (phase 1). Open: among the parked problems (resume) and the ones not started yet / retrying (a fresh start without the park
    flag: there is nothing left to wait for), the longest first (phase 2). A problem running in phase 1 is never touched."""
    bc = bconf(sc)
    pend = [n for n, p in problems.items() if p.get("status") == "pending" and not is_parked(p)]
    if not bstate.get("open"):
        o = phase1_order(pend, ests, order, bc)
        return (o[0], f"phase 1 ({bc['phase1_order']}) {1}/{len(o)} cost {ests.get(o[0], {}).get('cost', '?')}", True) if o else None
    cands = pend + [n for n, p in problems.items() if is_parked(p)]
    merged = [n for n, p in problems.items() if p.get("parked_at") or (p.get("gk") or {}).get("merged_version")]
    o = phase2_order(cands, ests, order, bc, sc, merged)
    if not o:
        return None
    n = o[0]
    kind = "resume" if is_parked(problems[n]) else "fresh start"
    return (n, f"phase 2 ({bc['phase2_order']}) {kind} 1/{len(o)} cost {ests.get(n, {}).get('cost', '?')}", False)


def barrier_plan(problems: dict, ests: dict, bstate: dict, sc: dict, order: dict) -> dict:
    """What ``farm.py sched plan`` prints: the phase-1 admission order, the phase-2 order and the barrier decision inputs."""
    bc = bconf(sc)
    c = census(problems)
    p1 = [n for n, p in problems.items() if p.get("status") == "pending" and not is_parked(p)]
    left = [n for n in problems if n in c["considered"] and problems[n].get("status") not in ("done",)]
    n, a = len(c["considered"]), len(c["arrived"])
    return {"phase1": phase1_order(p1, ests, order, bc), "phase2": phase2_order(left, ests, order, bc, sc),
            "considered": n, "arrived": a, "need": max(1, math.ceil(float(bc["min_frac"]) * n - 1e-9)) if n else 0,
            "open": bool(bstate.get("open")), "why": bstate.get("why")}
