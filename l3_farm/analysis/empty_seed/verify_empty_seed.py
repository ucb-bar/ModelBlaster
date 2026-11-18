#!/usr/bin/env python
"""Offline check of the EMPTY knowledge seed (no LLM, no GPU, no network).

    python verify_empty_seed.py KB_SRC_DIR [--flag]

KB_SRC_DIR is a KernelBlaster ``src`` dir (live tree read-only, or a patched copy). The seed is always
``./seed`` (this directory). ``--flag`` sets KB_EMPTY_SEED=1 (needs the patched tree). Every path is run in its
own try/except and reported as OK / CRASH (traceback) / NOTE. Writes only under a fresh tmp dir.
"""
import asyncio
import json
import os
import random
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = sys.argv[1]
use_flag = "--flag" in sys.argv
os.environ["KB_KNOWLEDGE_SEED_DIR"] = str(HERE / "seed")
os.environ["KERNELBLASTER_SEED_MINED"] = "0"
if use_flag:
    os.environ["KB_EMPTY_SEED"] = "1"
else:
    os.environ.pop("KB_EMPTY_SEED", None)
sys.path.insert(0, src)

from kernelblaster.knowledge.outcomes import OutcomeRecord  # noqa: E402
from kernelblaster.knowledge.v2 import KnowledgeV2  # noqa: E402
from kernelblaster.knowledge.v2 import metrics as M  # noqa: E402
from kernelblaster.knowledge.v2.hypotheses import Hypothesis  # noqa: E402

RES: list = []


def step(name):
    def deco(fn):
        try:
            out = fn()
            RES.append((name, "OK", out))
            print(f"[OK   ] {name}: {out}")
        except Exception:
            tb = traceback.format_exc(limit=4)
            RES.append((name, "CRASH", tb))
            print(f"[CRASH] {name}\n{tb}")
        return fn
    return deco


tmp = Path(tempfile.mkdtemp(prefix="empty_seed_"))
print(f"mode: flag={'KB_EMPTY_SEED=1' if use_flag else 'unset'}  src={src}  tmp={tmp}")
kv = KnowledgeV2(tmp / "k", explore_frac=1.0)
TABLES = ["techniques", "states", "claims", "stats", "hypotheses", "outcomes", "scopes", "targets", "proposals",
          "realizations", "review"]


def counts(k):
    return {t: k.db.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}


print("row counts after fresh open:", json.dumps(counts(kv)))
srcs = {}
for r in kv.db.conn.execute("SELECT source FROM techniques"):
    srcs[r[0]] = srcs.get(r[0], 0) + 1
print("technique sources:", srcs, "| states:", len(kv.hierarchy.states), "| seed_version:", kv.db.get_meta("seed_version"))
print("registry targets:", len(kv.registry.targets), "scopes:", len(kv.registry.nodes))

TK = "cuda:sm_89"
est = M.RooflineInput(bytes=4e8, flops=1e9, time_us=900.0, work_items=1e6, launches=4)
box: dict = {}


@step("vector (roofline estimate)")
def _():
    box["vec"] = kv.vector(TK, estimate=est)
    return sorted(box["vec"])


@step("classify synthetic profile")
def _():
    box["cls"] = kv.classify(box["vec"], target_key=TK, op_class="gemm", dtype="fp32")
    return f"chain={box['cls'].chain} status={box['cls'].status} label={box['cls'].label()} leaf={box['cls'].leaf}"


@step("classify with empty vector")
def _():
    c = kv.classify({}, target_key=TK, op_class="gemm", dtype="fp32")
    return f"chain={c.chain} status={c.status}"


@step("candidates()")
def _():
    t, h = kv.candidates(target_key=TK, op_class="gemm", dtype="fp32", chain=[], vec=box["vec"], op_kind="matmul")
    return f"techniques={len(t)} hypotheses={len(h)}"


@step("select() with nothing catalogued, no hypotheses")
def _():
    ch = kv.select(target_key=TK, op_class="gemm", dtype="fp32", cls=box["cls"], vec=box["vec"],
                   rng=random.Random(0), op_kind="matmul")
    box["free"] = ch
    return f"Choice(item={ch.item_id!r}, kind={ch.kind}, evidence={ch.evidence}, desc={ch.desc[:70]!r}...)"


@step("retrieve() prompt section (no techniques, no state)")
def _():
    s = kv.retrieve(target_key=TK, op_class="gemm", dtype="fp32", cls=box["cls"], vec=box["vec"],
                    problem_id="p1", slot="s0", technique="free_form_optimization", op_kind="matmul")
    return f"len={len(s)} text={s!r}"


@step("already_tried() before any outcome")
def _():
    return repr(kv.already_tried("p1", "s0"))


@step("technique_tier / router hook for unknown id")
def _():
    import importlib.util
    spec = importlib.util.spec_from_file_location("model_router_alone", Path(src) / "kernelblaster/agents/model_router.py")
    mod = importlib.util.module_from_spec(spec)                      # file has no package-relative imports (agents/__init__ needs pandas, loguru)
    sys.modules["model_router_alone"] = mod                          # dataclasses needs it registered
    spec.loader.exec_module(mod)
    technique_tier = mod.technique_tier
    return f"tier(free_form_optimization, None)={technique_tier('free_form_optimization', None)!r}"


def rec(technique, status="improved", op_class="gemm", pid="p1", slot="s0"):
    return OutcomeRecord(technique=technique, status=status, target="cuda", arch="sm_89", problem_id=pid, slot=slot,
                         op_class=op_class, dtype="fp32", metric_before=100.0, metric_after=80.0, speedup=1.25)


after = {"bw_util": M.metric(55.0, "measured", "ncu"), "compute_util": M.metric(20.0, "measured", "ncu")}


@step("record(): free-form choice, state None")
def _():
    ch = box["free"]
    v2 = kv.record(rec("free_form_optimization"), choice=ch, reward=0.2, cls=box["cls"], before=box["vec"],
                   after=after, op_class="gemm", dtype="fp32")
    return f"state_before={None} classification={v2['classification']} chain={v2['chain']} mech={v2['mechanism']['label']}"


@step("record(): unknown free-form technique name (not in ontology), state None, no after-vector")
def _():
    from kernelblaster.knowledge.v2.knowledge import Choice
    ch = Choice("my_custom_register_tiling", "technique", "free text", 0.0, 0.0, "untested")
    kv.record(rec("my_custom_register_tiling", pid="p2"), choice=ch, reward=-0.05, cls=box["cls"], before=box["vec"],
              after=None, op_class="gemm", dtype="fp32")
    return f"stats rows={len(kv.stats.data)} outcomes={len(kv.outcomes)}"


@step("hypothesis add -> select(explore) -> record (hypothesis id)")
def _():
    h = Hypothesis(name="persistent_blocks", mechanism="Keep one persistent grid resident and loop over tiles to remove "
                   "launch tail effects", source="gap", how="Launch one block per SM and grid-stride over the tile queue.",
                   target_key=TK, op_classes=["gemm"], predicts={"launch_frac": "down"})
    kv.hypotheses.add(h, kv.review_path)
    ch = kv.select(target_key=TK, op_class="gemm", dtype="fp32", cls=box["cls"], vec=box["vec"],
                   rng=random.Random(1), op_kind="matmul")
    assert ch.kind == "hypothesis", ch
    kv.record(rec(ch.item_id), choice=ch, reward=0.05, cls=box["cls"], before=box["vec"], after=after,
              op_class="gemm", dtype="fp32")
    box["hid"] = ch.item_id
    return f"selected={ch.item_id} attempts={len(kv.hypotheses.items[ch.item_id].attempts)} status={kv.hypotheses.items[ch.item_id].status}"


@step("retrieve() with an open hypothesis")
def _():
    return kv.retrieve(target_key=TK, op_class="gemm", dtype="fp32", cls=box["cls"], vec=box["vec"],
                       problem_id="p1", slot="s0")[:160]


@step("maybe_upkeep(force) without upkeep model")
def _():
    evs = asyncio.run(kv.maybe_upkeep(force=True))
    return f"events={[e.get('event') for e in evs]}"


# ---- LLM-driven growth paths, with a canned fake model (no network)
def fake_llm_factory():
    calls = []

    async def fake(messages):
        sysm = messages[0]["content"]
        calls.append("discovery" if "STATES" in sysm else "generation")
        if "STATES" in sysm:
            return json.dumps({"proposal": {"id": "low_bw_gemm", "parent": None,
                                            "predicate": {"arith_intensity": ">", "v": 0.0},
                                            "mechanism": "kernel streams far below DRAM peak", "predicted_effects": {}}})
        return json.dumps({"hypotheses": [{"trigger": 0, "source": "profiler", "name": "swizzle_smem_layout",
                                           "mechanism": "XOR-swizzle the shared-memory tile layout to remove bank conflicts",
                                           "how": "Index smem as tile[row][col ^ (row & 7)] on stores and loads.",
                                           "predicts": {"bank_conflicts": "down"}, "op_classes": ["gemm"]}]})
    return fake, calls


@step("growth: >=5 'unexplained' outcomes + fake LLM -> state discovery + hypothesis generation")
def _():
    kv.upkeep_model = "fake"
    fake, calls = fake_llm_factory()
    kv._llm = fake
    rng = random.Random(3)
    from kernelblaster.knowledge.v2.knowledge import Choice
    for i in range(7):
        vec = kv.vector(TK, estimate=M.RooflineInput(bytes=4e8 * (1 + i), flops=1e9, time_us=900 + 50 * i,
                                                    work_items=1e6, launches=4))
        c = kv.classify(vec, target_key=TK, op_class="gemm", dtype="fp32")
        ch = Choice("free_form_optimization", "technique", "x", 0.0, 0.0, "untested")
        kv.record(rec("free_form_optimization", pid=f"q{i}"), choice=ch, reward=0.05 * rng.random(), cls=c, before=vec,
                  after=after, op_class="gemm", dtype="fp32",
                  extra={"ncu_rules": [{"rule": "UncoalescedGlobalAccess", "est_speedup_pct": 40}]} if i == 6 else None)
    evs = asyncio.run(kv.maybe_upkeep(force=True))
    return (f"llm calls={calls} events={[(e.get('event'), e.get('state') or e.get('name') or e.get('reason')) for e in evs]}"
            f" states_now={list(kv.hierarchy.states)} hypotheses_now={[h.name for h in kv.hypotheses.items.values()]}")


@step("classify after a discovered root state")
def _():
    c = kv.classify(box["vec"], target_key=TK, op_class="gemm", dtype="fp32")
    return f"chain={c.chain} status={c.status}"


@step("select() after discovery (still 0 techniques)")
def _():
    kv.explore_frac = 0.0
    c = kv.classify(box["vec"], target_key=TK, op_class="gemm", dtype="fp32")
    ch = kv.select(target_key=TK, op_class="gemm", dtype="fp32", cls=c, vec=box["vec"], rng=random.Random(5))
    return f"item={ch.item_id} kind={ch.kind}"


@step("which hypothesis triggers can fire in an empty run? (stalled slot after a discovered state; combine/port/gap-per-state need techniques)")
def _():
    from kernelblaster.knowledge.v2.knowledge import Choice
    c = kv.classify(box["vec"], target_key=TK, op_class="gemm", dtype="fp32")
    assert c.chain, "needs the discovered root state from the growth step"
    for i in range(5):
        ch = Choice("free_form_optimization", "technique", "x", 0.0, 0.0, "untested")
        kv.record(rec("free_form_optimization", status="neutral",
                      pid="stall", slot="s9"), choice=ch, reward=0.0, cls=c, before=box["vec"], after=after, op_class="gemm", dtype="fp32")
    trig = kv.hypothesis_triggers(kv.outcomes.rows())
    return f"trigger sources={sorted({t['source'] for t in trig})} keys={[t['key'][:60] for t in trig]}"


@step("growth: hypothesis wins on 2 problems -> automatic promotion to a technique -> select() offers it")
def _():
    from kernelblaster.knowledge.v2.knowledge import Choice
    h = Hypothesis(name="tile_queue_persistent", mechanism="Persistent blocks pull output tiles from an atomic queue to hide the "
                   "tail wave", source="gap", how="Launch #SM blocks; each loops atomicAdd(queue) over tiles.",
                   target_key=TK, op_classes=["gemm"], predicts={"bw_util": "up"})
    kv.hypotheses.add(h, kv.review_path)
    kv.explore_frac = 0.0
    before = {"bw_util": M.metric(30.0, "measured", "ncu")}
    aft = {"bw_util": M.metric(60.0, "measured", "ncu")}
    c = kv.classify(box["vec"], target_key=TK, op_class="gemm", dtype="fp32")
    for i, (pid, rw) in enumerate([("pa", .05), ("pa", .06), ("pb", .05), ("pb", .07)]):
        ch = Choice(h.id, "hypothesis", h.as_technique_desc(), 0.0, 0.0, "hypothesis under test", explore=True)
        kv.record(rec(h.id, pid=pid), choice=ch, reward=rw, cls=c, before=before, after=aft, op_class="gemm", dtype="fp32")
    evs = asyncio.run(kv.maybe_upkeep(force=True))
    names = list(kv.ontology.techniques)
    kv.ontology = kv._load_ontology()
    ch = kv.select(target_key=TK, op_class="gemm", dtype="fp32", cls=c, vec=box["vec"], rng=random.Random(9))
    return (f"events={[(e.get('event'), e.get('technique') or e.get('id')) for e in evs if 'hypothes' in e.get('event','') or e.get('event')=='promoted']} "
            f"techniques_now={list(kv.ontology.techniques)} select -> {ch.item_id} kind={ch.kind} evidence={ch.evidence}")


@step("sync_graph_hints (ModelBlaster graph_hints/v1)")
def _():
    p = tmp / "gh.json"
    p.write_text(json.dumps({"schema": "modelblaster.graph_hints/v1", "hints": [
        {"id": "fuse_conv_relu_1", "kind": "fuse", "status": "accepted", "target": "cuda", "evidence": [1, 2]},
        {"id": "unfuse_x", "kind": "unfuse", "status": "rejected", "target": "cuda", "evidence": []}]}))
    return f"synced={kv.sync_graph_hints(p, target_key=TK)}"


@step("summary()")
def _():
    return json.dumps({k: v for k, v in kv.summary().items() if k != "events"})


@step("reopen existing DB (idempotent seeding, resume path)")
def _():
    kv.db.conn.close()
    k2 = KnowledgeV2(tmp / "k")
    return json.dumps(counts(k2))


@step("merge: build_base + merge_knowledge_dir into a tmp GlobalStore, then materialize (read mode)")
def _():
    from kernelblaster.knowledge.v2 import merge as G
    store = G.GlobalStore(tmp / "gk").ensure()
    base = G.build_base(tmp / "basework")
    info = G.RunInfo(run_uid="t1", problem="p1", part="cuda:sm_89|fp32")
    res = G.merge_knowledge_dir(store, tmp / "k", seed_dir=base, info=info)
    out = tmp / "mat"
    m = G.materialize(store, "cuda:sm_89|fp32", out)
    k3 = KnowledgeV2(out)
    return f"merge keys={sorted(res)[:6]} materialized counts={json.dumps(counts(k3))}"


@step("seed_knowledge.py equivalent (KnowledgeV2 on fresh dir twice, then counts)")
def _():
    k = KnowledgeV2(tmp / "fresh2")
    return json.dumps(counts(k))


print("\nSUMMARY:", {n: s for n, s, _ in RES})
