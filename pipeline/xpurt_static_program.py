"""Compile an ingested XPU-RT schedule into per-worker static programs.

Input: the entry list `ingest_xpurt_schedule.load()` returns (table order =
start-time order, data deps backward). Output: a plain dict (JSON-able) that
`generate_xpurt_main --walker static` turns into const C arrays, plus edge
statistics.

What is decided here, once, instead of at run time:

  * S1 per-worker lists. One worker per (core_kind, hart) that appears as the
    master OR as a helper hart of any entry. A worker's list holds its own
    entries in table order, and (gang mode) one SLICE item for every composite
    entry it is a helper of, at that entry's table position.

  * S2 edge classification. Edges: data deps, the schedule's time_dependency
    edges, and instance-order edges (every sink of instance k-1 of a network
    -> every root of instance k; and, for a cyclic run, the sinks of the last
    instance of epoch E-1 -> the roots of instance 0 of epoch E), which keep
    two instances of one network off its shared buffers.
      - same-worker edges whose producer precedes the consumer in that
        worker's list are ELIDED: static order already guarantees them;
      - the remaining cross-worker edges are transitively REDUCED over the
        graph (all edges + every worker's order edges): an edge u -> v is
        dropped when v is reachable from u through another path;
      - what is left is the cross-worker wait list of each item.
    The reduction leans on worker order edges, which is sound because a
    cross-worker wait is always against the producer worker's PREFIX counter
    ("everything up to position p is done"), never against a single entry. An
    item that is run out of order (S6) checks its `full` list instead: every
    direct dependency, unreduced.

  * S3 spin or sleep per remaining edge, from the schedule's expected wait:
    the consumer reaches the item when its previous item is planned to end
    (or at its own gate); the producer is planned to end at end(u). If the
    expected wait is at most `spin_ms` the edge is marked SPIN (bounded spin,
    then futex), otherwise SLEEP (one check, then futex).

  * S4 gates. Only roots (items with no data dependency: the periodic
    releases) are gated on absolute time, at their scheduled start.

  * S5 gangs. A composite entry becomes a gang: the master item runs the
    kernel with a gang pool whose fork publishes the job to the helper
    workers' slice items; no pool threads, no hart locks.

  * S6 alternatives. For every item, up to `alt_window` later items of the
    same worker that are independent of it (no path in the dependency graph),
    are not gang members and not gated: the candidates the worker may run
    when the item is blocked.
"""

from __future__ import annotations

from typing import Any

K_NORMAL, K_GANG_MASTER, K_GANG_SLICE = 0, 1, 2


def _reach_bitsets(n: int, succ: list[set[int]], topo: list[int]) -> list[int]:
    """desc[u] = bitset of nodes reachable from u (excluding u)."""
    desc = [0] * n
    for u in reversed(topo):
        b = 0
        for v in succ[u]:
            b |= desc[v] | (1 << v)
        desc[u] = b
    return desc


def build(entries: list[Any], *, gang: bool = True, cyclic: bool = False,
          alt_window: int = 8, spin_ms: float = 0.05) -> dict:
    n = len(entries)
    by_id = {e.entry_id: e for e in entries}
    assert sorted(by_id) == list(range(n)), "entry ids must be 0..N-1 in table order"
    ents = [by_id[i] for i in range(n)]

    # ---- workers -----------------------------------------------------------
    wkeys: list[tuple[str, int]] = []

    def wid(kind: str, hart: int) -> int:
        k = (kind, hart)
        if k not in wkeys:
            wkeys.append(k)
        return wkeys.index(k)

    for e in ents:
        if e.hart < 0:
            raise ValueError(f"entry {e.entry_id}: unbound hart; the static walker needs every entry placed")
        wid(e.core_kind, e.hart)
        if gang:
            for h in e.harts[1:]:
                wid(e.core_kind, h)
    n_w = len(wkeys)

    # ---- per-worker item lists (table order) --------------------------------
    # node ids: entry i -> i ; slice of entry i on helper index k -> n + running index
    items: list[list[dict]] = [[] for _ in range(n_w)]
    node_of_slice: dict[tuple[int, int], int] = {}
    nodes: list[dict] = []          # node -> dict(worker, pos, entry, kind)
    for e in ents:
        nodes.append({})
    gangs: list[dict] = []
    gang_of_harts: dict[tuple[int, ...], int] = {}
    for e in ents:
        composite = len(e.harts) > 1
        w = wid(e.core_kind, e.hart)
        kind = K_GANG_MASTER if (composite and gang) else K_NORMAL
        g = -1
        if composite and gang:
            key = (e.core_kind,) + tuple(e.harts)
            if key not in gang_of_harts:
                gang_of_harts[key] = len(gangs)
                gangs.append(dict(kind=e.core_kind, harts=list(e.harts)))
            g = gang_of_harts[key]
        it = dict(entry=e.entry_id, kind=kind, gang=g, slice=0)
        nodes[e.entry_id] = dict(worker=w, pos=len(items[w]), entry=e.entry_id, kind=kind)
        items[w].append(it)
        if composite and gang:
            for k, h in enumerate(e.harts[1:], start=1):
                hw = wid(e.core_kind, h)
                node = len(nodes)
                node_of_slice[(e.entry_id, k)] = node
                nodes.append(dict(worker=hw, pos=len(items[hw]), entry=e.entry_id, kind=K_GANG_SLICE))
                items[hw].append(dict(entry=e.entry_id, kind=K_GANG_SLICE, gang=g, slice=k, node=node))
    for e in ents:
        items[nodes[e.entry_id]["worker"]][nodes[e.entry_id]["pos"]]["node"] = e.entry_id
    n_nodes = len(nodes)

    # ---- dependency edges (entry -> entry) ----------------------------------
    inst: dict[tuple[str, int], list[int]] = {}
    for e in ents:
        inst.setdefault((e.network, e.instance), []).append(e.entry_id)
    has_succ = [False] * n
    for e in ents:
        for d in e.deps_entry_ids:
            has_succ[d] = True
    roots = {k: [i for i in v if not ents[i].deps_entry_ids] for k, v in inst.items()}
    sinks = {k: [i for i in v if not has_succ[i]] for k, v in inst.items()}

    data_edges: set[tuple[int, int]] = set()
    time_edges: set[tuple[int, int]] = set()
    inst_edges: set[tuple[int, int]] = set()
    epoch_edges: set[tuple[int, int]] = set()     # producer in the previous epoch
    for e in ents:
        for d in e.deps_entry_ids:
            if d >= e.entry_id:
                raise ValueError(f"entry {e.entry_id}: data dep {d} is not backward in table order")
            data_edges.add((d, e.entry_id))
        if e.time_dep_entry_id >= 0:
            if e.time_dep_entry_id >= e.entry_id:
                raise ValueError(f"entry {e.entry_id}: time dep is not backward in table order")
            time_edges.add((e.time_dep_entry_id, e.entry_id))
    n_inst: dict[str, int] = {}
    for (net, k) in inst:
        n_inst[net] = max(n_inst.get(net, 0), k + 1)
    for (net, k), rts in roots.items():
        if k > 0:
            for s in sinks[(net, k - 1)]:
                for r in rts:
                    if s >= r:
                        raise ValueError(
                            f"{net}: instance {k - 1} entry {s} is not before instance {k} entry {r} in "
                            f"table order; the static walker needs non-overlapping instances")
                    inst_edges.add((s, r))
        elif cyclic:
            for s in sinks[(net, n_inst[net] - 1)]:
                for r in rts:
                    epoch_edges.add((s, r))
    # time_dependency is "the previous dispatch on the same hardware target": the
    # same master worker, so static order always implies it. It is counted and
    # elided, and is NOT part of the dependency relation used for run-ahead
    # (it would chain every item to its predecessor and forbid all of it).
    time_cross = {(u, v) for (u, v) in time_edges if nodes[u]["worker"] != nodes[v]["worker"]}
    time_edges -= time_cross        # not an order edge of one target: ignored, and reported
    dep_edges = data_edges | inst_edges                  # same-epoch correctness edges, entry -> entry

    # ---- graph over nodes: dep edges + worker order edges -------------------
    succ: list[set[int]] = [set() for _ in range(n_nodes)]
    for u, v in dep_edges:
        succ[u].add(v)
    for w in range(n_w):
        for a, b in zip(items[w], items[w][1:]):
            succ[a["node"]].add(b["node"])
    # table order (slices at their entry's position) is a topological order
    topo = sorted(range(n_nodes), key=lambda x: (nodes[x]["entry"], 0 if x < n else 1, x))
    rank = {x: i for i, x in enumerate(topo)}
    for u in range(n_nodes):
        for v in succ[u]:
            assert rank[u] < rank[v], "internal: order edge against table order"
    desc = _reach_bitsets(n_nodes, succ, topo)

    stats = dict(entries=n, workers=n_w, gang_entries=sum(1 for e in ents if len(e.harts) > 1 and gang),
                 slice_items=len(nodes) - n, edges_total=0, edges_data=len(data_edges),
                 edges_time=len(time_edges - data_edges), edges_instance=len(inst_edges - data_edges - time_edges),
                 edges_epoch=len(epoch_edges), elided_same_worker=0, removed_by_reduction=0,
                 remaining_cross_worker=0, remaining_cross_worker_data=0, epoch_cross_worker=0,
                 spin_edges=0, sleep_edges=0)
    stats["time_edges_cross_worker_ignored"] = len(time_cross)
    only_time = time_edges - dep_edges
    stats["edges_total"] = len(dep_edges) + len(only_time) + len(epoch_edges)
    stats["elided_same_worker"] += len(only_time)

    wait: dict[int, list[dict]] = {i: [] for i in range(n)}
    full: dict[int, list[dict]] = {i: [] for i in range(n)}
    end_ms = [e.start_time_ms + e.duration_ms for e in ents]

    def arrive_ms(v: int) -> float:
        nv = nodes[v]
        if nv["pos"] == 0:
            return ents[v].start_time_ms
        prev = items[nv["worker"]][nv["pos"] - 1]
        return end_ms[prev["entry"]]

    for (u, v) in sorted(dep_edges):
        wu, wv = nodes[u]["worker"], nodes[v]["worker"]
        full[v].append(dict(worker=wu, pos=nodes[u]["pos"], eoff=0, own=int(wu == wv)))
        if wu == wv:
            assert nodes[u]["pos"] < nodes[v]["pos"]
            stats["elided_same_worker"] += 1
            continue
        # redundant iff v is reachable from another successor of u
        redundant = any((desc[x] >> v) & 1 for x in succ[u] if x != v)
        if redundant:
            stats["removed_by_reduction"] += 1
            continue
        exp_wait = max(0.0, end_ms[u] - arrive_ms(v))
        spin = int(exp_wait <= spin_ms)
        wait[v].append(dict(worker=wu, pos=nodes[u]["pos"], eoff=0, spin=spin,
                            expected_wait_ms=round(exp_wait, 6)))
        stats["remaining_cross_worker"] += 1
        stats["remaining_cross_worker_data"] += int((u, v) in data_edges)
        stats["spin_edges" if spin else "sleep_edges"] += 1
    for (u, v) in sorted(epoch_edges):
        wu, wv = nodes[u]["worker"], nodes[v]["worker"]
        full[v].append(dict(worker=wu, pos=nodes[u]["pos"], eoff=1, own=int(wu == wv)))
        if wu == wv:
            stats["elided_same_worker"] += 1      # the worker finished epoch E-1 before starting E
            continue
        wait[v].append(dict(worker=wu, pos=nodes[u]["pos"], eoff=1, spin=0, expected_wait_ms=0.0))
        stats["epoch_cross_worker"] += 1

    # keep, per (consumer, producer worker, eoff), only the largest position: the prefix counter covers the rest
    def squash(lst: list[dict]) -> list[dict]:
        best: dict[tuple[int, int], dict] = {}
        for x in lst:
            k = (x["worker"], x["eoff"])
            if k not in best or x["pos"] > best[k]["pos"]:
                best[k] = x
            elif x["pos"] == best[k]["pos"]:
                pass
        return sorted(best.values(), key=lambda x: (x["eoff"], x["worker"]))

    # ---- hand-off metric: remaining cross-worker DATA edges per network instance
    handoffs: dict[str, int] = {}
    for (u, v) in data_edges:
        if nodes[u]["worker"] != nodes[v]["worker"]:
            handoffs[ents[v].job_name] = handoffs.get(ents[v].job_name, 0) + 1
    stats["cross_worker_data_edges_per_instance"] = dict(sorted(handoffs.items()))

    # ---- alternatives (S6) ---------------------------------------------------
    # independence is judged on the dependency-only closure (no order edges)
    dsucc: list[set[int]] = [set() for _ in range(n_nodes)]
    for u, v in dep_edges:
        dsucc[u].add(v)
    ddesc = _reach_bitsets(n_nodes, dsucc, topo)
    is_root = [not e.deps_entry_ids for e in ents]

    workers_out = []
    for w, (kind, hart) in enumerate(wkeys):
        out_items = []
        lst = items[w]
        for p, it in enumerate(lst):
            e = ents[it["entry"]]
            d = dict(entry=it["entry"], kind=it["kind"], gang=it["gang"], slice=it["slice"],
                     network=e.network, impl=e.impl, gated=0, wait=[], full=[], alts=[])
            if it["kind"] != K_GANG_SLICE:
                d["gated"] = int(is_root[it["entry"]])
                d["wait"] = squash(wait[it["entry"]])
                d["full"] = sorted(full[it["entry"]], key=lambda x: (x["eoff"], x["worker"], x["pos"]))
                if it["kind"] == K_NORMAL and alt_window > 0:
                    for q in range(p + 1, len(lst)):
                        c = lst[q]
                        if c["kind"] != K_NORMAL:
                            continue
                        if is_root[c["entry"]]:
                            continue            # gated: released by time, never run early
                        if (ddesc[it["entry"]] >> c["entry"]) & 1:
                            continue            # depends on the blocked item
                        d["alts"].append(q)
                        if len(d["alts"]) >= alt_window:
                            break
            out_items.append(d)
        workers_out.append(dict(kind=kind, hart=hart, items=out_items))
    stats["alt_candidates"] = sum(len(i["alts"]) for w in workers_out for i in w["items"])
    stats["gated_items"] = sum(i["gated"] for w in workers_out for i in w["items"])
    return dict(n_entries=n, gang=bool(gang), cyclic=bool(cyclic), alt_window=alt_window, spin_ms=spin_ms,
                workers=workers_out, gangs=gangs, stats=stats,
                needs_locks=bool((not gang) and any(len(e.harts) > 1 for e in ents)))


def format_stats(prog: dict) -> str:
    s = prog["stats"]
    return (f"static program: {s['entries']} entries on {s['workers']} workers "
            f"({s['gang_entries']} gang entries, {s['slice_items']} slice items); "
            f"edges total {s['edges_total']} (data {s['edges_data']}, time {s['edges_time']}, "
            f"instance {s['edges_instance']}, epoch {s['edges_epoch']}) -> elided same-worker "
            f"{s['elided_same_worker']}, removed by reduction {s['removed_by_reduction']}, remaining "
            f"cross-worker {s['remaining_cross_worker']} (data {s['remaining_cross_worker_data']}; "
            f"spin {s['spin_edges']}, sleep {s['sleep_edges']}) + {s['epoch_cross_worker']} epoch; "
            f"gated items {s['gated_items']}; alt candidates {s['alt_candidates']}")
