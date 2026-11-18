"""X-147 GPU test of the unapply audit (runs on my L40S node under the frozen tree; zero LLM calls).

Real graph (8_ResNetBasicBlock fp16 IR re-extracted on the node, or 19_MobileNetV1), real flow code paths:
``GraphSearch.preapply`` (class table, ``--graph-preapply-classes``), ``GraphSearch.audit`` -> ``preapply_audit.audit_hook``
(alternative graphs through the ``unfuse`` hint, cache_plan stage kernels, A3 routing, verification + paired interleaved
timing through ``kb_graph_opt._remeasure_once``, decision, doc / events / state). Scenarios:

  classes   --graph-preapply-classes high pre-applies only the elementwise tail; all also pre-applies conv+bn.
  natural   conv+bn + add+relu pre-applied, kernels as the flow has them (composite seed = parity): both graphs measured.
  slowed    the same, with a deliberately slowed fused conv+bn kernel in the kernel cache: the audit must revert it.
  routed    the cache holds a faster verified conv kernel of ANOTHER shape (the seed with its fp64 accumulator made fp32):
            A3's route_ops must find it for the unfused stage ops, check it on their shapes and write a reuse entry.
  after     the graph the final model run would deploy after the revert (doc re-applied) is verified and timed against
            the incumbent: the revert is real (part of "slowed").

  python gpu_audit_test.py IR_DIR OUT_JSON WORK_DIR [delay_cycles]
"""
import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path

from modelblaster.pipeline import graph_hints as gh
from modelblaster.pipeline import graph_search as gsm
from modelblaster.pipeline import kb_graph_opt as go
from modelblaster.pipeline import preapply_audit as pa
from modelblaster.pipeline.kb_ingest import CudaCache
from modelblaster.pipeline.kb_proposals import _all_specs
from modelblaster.pipeline.kernel_abi import parse_signature

IR = Path(sys.argv[1])
OUT = Path(sys.argv[2])
BASE = Path(sys.argv[3])
DELAY = int(sys.argv[4]) if len(sys.argv) > 4 else 4_000_000          # clock cycles of the injected spin (~1.6 ms at 2.5 GHz)
KB = Path(os.environ.get("MB_KB_ROOT") or "/tmp/krisd/pa/code/ModelBlaster/third_party/KernelBlaster")
NAME = IR.name
ARCH = "89"
res = {"ir": NAME, "delay_cycles": DELAY, "t0": time.time(), "scenarios": {}}


def log(*a):
    print("[gpu_audit_test]", *a, flush=True)


def smi():
    import subprocess
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.max.mem,clocks.mem,driver_version", "--format=csv,noheader"],
                       capture_output=True, text=True)
    return r.stdout.strip()


res["gpu"] = smi()
g0 = json.loads((IR / "graph.json").read_text())
specs = _all_specs()
res["ops"] = [(o["name"], o["op"]) for o in g0["ops"]]


def config(classes, audit="final", route="on", **kw):
    return go.Config(kb=KB, arch=ARCH, remeasure=5, mb_evaluator="builtin", gpu_model="NVIDIA L40S",
                     graph_search="interleaved", graph_priors="standard", graph_explore=0.0,
                     graph_preapply_classes=classes, graph_preapply_audit=audit, graph_preapply_route=route, **kw)


def find_hint(kind_sub):
    """A fuse hint (all chains) for the pattern named by the stage list, from the real graph (scan of every pair)."""
    fake = {"ops": [{"name": o["name"], "us": 1.0} for o in gh.unique_names(copy.deepcopy(g0))["ops"]], "e2e_us": 100.0}
    for h in gsm.scan_pairs(g0, fake, specs, gh.new_doc(), ir=NAME):
        if h["fused_kind"] == kind_sub:
            return h
    raise SystemExit(f"no chain of kind {kind_sub} in {NAME}: {[o for o in res['ops']][:20]}")


# --------------------------------------------------------------------------------- hints of this graph
KINDS = {"resnet": ("fused__conv2d_f16__batchnorm2d_f16", "fused__add_f16__relu_f16"),
         "mobilenet": ("fused__conv2d_dw_f16__batchnorm2d_f16", "fused__batchnorm2d_f16__relu_f16")}
conv_kind, tail_kind = KINDS["resnet" if NAME.startswith("8_") else "mobilenet"]
H_CONV, H_TAIL = find_hint(conv_kind), find_hint(tail_kind)
res["hints"] = {"conv_head": H_CONV["id"], "tail": H_TAIL["id"], "conv_chains": H_CONV["chains"][:4],
                "tail_chains": H_TAIL["chains"][:4]}


OVERLAP = not NAME.startswith("8_")        # MobileNet: bn+relu and dwconv+bn share the bn ops (A4 pre-applies the more confident one)


def fresh_doc(both=True):
    doc = gh.new_doc()
    gh.merge(doc, [copy.deepcopy(H_CONV)] + ([copy.deepcopy(H_TAIL)] if both and not OVERLAP else []))
    return doc


# --------------------------------------------------------------------------------- scenario 0: class filtering
def both_hints():
    doc = gh.new_doc()
    gh.merge(doc, [copy.deepcopy(H_CONV), copy.deepcopy(H_TAIL)])
    return doc


w = BASE / "classes_high"
shutil.rmtree(w, ignore_errors=True)
gs = gsm.GraphSearch(config("high"), w, [IR])
done_high = gs.preapply(0, both_hints(), {NAME: g0}, [], specs)
w = BASE / "classes_all"
shutil.rmtree(w, ignore_errors=True)
gs = gsm.GraphSearch(config("all"), w, [IR])
done_all = gs.preapply(0, both_hints(), {NAME: g0}, [], specs)
w2 = BASE / "classes_all_conv_only"
shutil.rmtree(w2, ignore_errors=True)
gs2 = gsm.GraphSearch(config("all"), w2, [IR])
done_conv = gs2.preapply(0, fresh_doc(both=False), {NAME: g0}, [], specs)
w3 = BASE / "classes_high_conv_only"
shutil.rmtree(w3, ignore_errors=True)
gs3 = gsm.GraphSearch(config("high"), w3, [IR])
done_conv_high = gs3.preapply(0, fresh_doc(both=False), {NAME: g0}, [], specs)
res["classes"] = {"high": done_high, "all": done_all, "conv_only_all": done_conv, "conv_only_high": done_conv_high,
                  "rec_all": list(pa.load_records(w)),
                  "stage_ops_conv_only": {k: v["stage_ops"] for k, v in pa.load_records(w2).items()}}
assert done_high == [H_TAIL["id"]], done_high
assert done_conv_high == [], done_conv_high                 # conv head: never under high
assert done_conv == [H_CONV["id"]], done_conv                # forced by all
assert sorted(done_all) == sorted([H_TAIL["id"]] if OVERLAP else [H_CONV["id"], H_TAIL["id"]]), done_all
log("classes ok", res["classes"]["high"], res["classes"]["all"])


# --------------------------------------------------------------------------------- scenario helpers
def slow_source(code: str, kind: str) -> str:
    """The fused slot's source with a spin kernel launched first by the fused host function (correct output, slower)."""
    host = f"void kernel_{kind}("
    assert host in code, "fused host function not found"
    head, tail = code.split(host, 1)
    sig_end = tail.index("{") + 1
    delay = ("__global__ void mb_audit_delay(long long cyc) { long long t0 = clock64(); while (clock64() - t0 < cyc) {} }\n")
    return head + delay + host + tail[:sig_end] + f"\n    mb_audit_delay<<<1, 1>>>({DELAY}LL);" + tail[sig_end:]


def plant_slow_fused(cache: CudaCache, inc_graph: dict, sp: dict, kind: str, work: Path):
    """Put a deliberately slowed fused kernel into the cache for every instance of ``kind`` of the incumbent."""
    import json as _j
    p = pa.compose_model(IR, inc_graph, sp, cache, work / "seedprobe", ARCH)
    man = _j.loads((Path(p) / "manifest.json").read_text())
    slot = next(s for s in man["slots"] if s["op"] == kind)
    code = (Path(p) / slot["file"]).read_text()
    shapes = [o.get("shape", {}) for o in inc_graph["ops"] if o["op"] == kind]
    entry, _ = parse_signature(sp[kind].signature)
    e = cache.add_reuse(op=kind, entry=entry, signature=sp[kind].signature, arch=f"sm_{ARCH}", shapes=shapes,
                        code=slow_source(code, kind), slot_us=1.0, impl_id="test:slowed-fused",
                        source={"by": "x147-gpu-test", "note": f"spin {DELAY} cycles before the fused kernel"})
    cache.save()
    return e["impl_id"]


def plant_fast_stage(cache: CudaCache, g: dict, sp: dict, kind: str, work: Path):
    """A faster verified kernel of ``kind`` for ANOTHER shape (guard = a shape no instance has): the hand seed with the
    fp64 accumulator made fp32. Only nearest-shape routing can find it."""
    import json as _j
    plain, pa_added, _ = gh.apply_hints(g0, [], dict(sp))
    p = pa.compose_model(IR, plain, sp, cache, work / "fastprobe", ARCH)
    man = _j.loads((Path(p) / "manifest.json").read_text())
    slot = next(s for s in man["slots"] if s["op"] == kind)
    code = (Path(p) / slot["file"]).read_text()
    if "double acc" not in code:
        return None
    other = [dict(o.get("shape", {}), IC=o["shape"]["IC"] + 1) for o in g["ops"] if o["op"] == kind][:1]
    entry, _ = parse_signature(sp[kind].signature)
    e = cache.add_reuse(op=kind, entry=entry, signature=sp[kind].signature, arch=f"sm_{ARCH}", shapes=other,
                        code=code.replace("double acc", "float acc"), slot_us=1.0, impl_id="test:fast-other-shape",
                        source={"by": "x147-gpu-test", "note": "fp32 accumulator variant of the seed, guarded on another shape"})
    cache.save()
    return e["impl_id"]


def scenario(name: str, slow: bool, classes="all", route="on", fast_stage: str | None = None):
    log("scenario", name, "slow" if slow else "natural")
    w = BASE / name
    shutil.rmtree(w, ignore_errors=True)
    w.mkdir(parents=True)
    cfg = config(classes, route=route)
    gs = gsm.GraphSearch(cfg, w, [IR])
    doc = fresh_doc()
    done = gs.preapply(0, doc, {NAME: g0}, [], specs)
    assert H_CONV["id"] in done, done
    hp = w / "graph_hints.json"
    gh.save(doc, hp)
    cache = CudaCache(w / "cache")
    inc, added, _ = gh.apply_hints(g0, doc["hints"], dict(specs))
    sp = {**specs, **added}
    out = {"preapplied": done, "incumbent_ops": [o["name"] for o in inc["ops"]]}
    if slow:
        out["slowed_entry"] = plant_slow_fused(cache, inc, sp, conv_kind, w)
    if fast_stage:
        out["fast_entry"] = plant_fast_stage(cache, inc, sp, fast_stage, w)
    gpus = go.detect_gpus(cfg)
    leases = go.GpuLeases(gpus["local"])
    state = {"rounds": [{"round": 0, "decisions": []}]}
    sp_ = w / "state.json"
    sp_.write_text(json.dumps(state))
    t0 = time.time()
    rep = gs.audit("final", irs=[IR], doc=doc, hints_path=hp, cache=cache, specs=specs, cfg=cfg, leases=leases,
                   recorded=[], state=state, state_path=sp_)
    out["audit_seconds"] = round(time.time() - t0, 1)
    out["audit"] = rep
    out["statuses"] = {h["id"]: h["status"] for h in doc["hints"]}
    out["doc_saved"] = {h["id"]: h["status"] for h in gh.load(hp)["hints"]}
    out["state_rows"] = [{k: d.get(k) for k in ("hint", "status", "hint_status", "gain", "original_us", "variant_us", "phase")}
                         for d in state["rounds"][-1]["decisions"]]
    out["decision_events"] = [{k: e.get(k) for k in ("hint", "phase", "status", "measured", "gain", "original_us", "variant_us")}
                              for e in map(json.loads, gs.log_path.read_text().splitlines())
                              if e["event"] == "decision" and e.get("phase") == "preapply_audit"]
    out["exec_mode"] = gs.exec_mode("whole")
    # the graph the final model run deploys now (doc re-applied), verified and timed against the incumbent
    final, fadded, _ = gh.apply_hints(g0, doc["hints"], dict(specs))
    out["final_ops"] = [o["name"] for o in final["ops"]]
    if out["final_ops"] != out["incumbent_ops"]:
        sp2 = {**specs, **fadded}
        pf = pa.compose_model(IR, final, sp2, cache, w / "final_check" / "final", ARCH)
        pi = pa.compose_model(IR, inc, sp, cache, w / "final_check" / "inc", ARCH)
        times, info = pa.default_timer(cfg, leases, [])([pi, pf], w / "final_check" / "time")
        out["final_check"] = {"incumbent_us": pa._median(times[0]), "final_us": pa._median(times[1]),
                              "incumbent_runs": times[0], "final_runs": times[1],
                              "final_faster_by": (pa._median(times[0]) - pa._median(times[1])) / pa._median(times[0])
                              if times[0] and times[1] else None}
    return out


res["scenarios"]["natural"] = scenario("natural", slow=False)
res["scenarios"]["slowed"] = scenario("slowed", slow=True)
STAGE = "conv2d_f16" if NAME.startswith("8_") else "conv2d_dw_f16"
res["scenarios"]["routed"] = scenario("routed", slow=False, fast_stage=STAGE)
res["seconds"] = round(time.time() - res["t0"], 1)

# acceptance
nat, slo = res["scenarios"]["natural"]["audit"], res["scenarios"]["slowed"]["audit"]
checks = {}
for tag, rep in (("natural", nat), ("slowed", slo)):
    for hid, hr in rep["hints"].items():
        irs = hr["irs"].get(NAME) or {}
        checks[f"{tag}:{hid}:both_graphs_measured"] = bool(irs.get("inc_us") and irs.get("alt_us"))
routes = [r for hr in res["scenarios"]["routed"]["audit"]["hints"].values()
          for r in ((hr["irs"].get(NAME) or {}).get("route") or [])]
if res["scenarios"]["routed"].get("fast_entry"):
    checks["routed:route_written"] = any(r.get("result") == "routed" and r.get("written") for r in routes)
checks["routed:both_graphs_measured"] = all(bool((hr["irs"].get(NAME) or {}).get("inc_us") and (hr["irs"].get(NAME) or {}).get("alt_us"))
                                            for hr in res["scenarios"]["routed"]["audit"]["hints"].values())
checks["slowed:conv_reverted"] = H_CONV["id"] in slo["reverted"]
checks["slowed:doc_rejected"] = res["scenarios"]["slowed"]["statuses"][H_CONV["id"]] == "rejected"
checks["slowed:final_faster"] = (res["scenarios"]["slowed"].get("final_check") or {}).get("final_faster_by", 0) > 0.03
checks["natural:conv_not_reverted_by_noise"] = H_CONV["id"] not in nat["reverted"] or \
    nat["hints"][H_CONV["id"]]["gain_revert"] >= nat["margin"]
res["checks"] = checks
res["ok"] = all(checks.values())
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=1, default=str))
print(json.dumps({"checks": checks, "ok": res["ok"]}, indent=1))
sys.exit(0 if res["ok"] else 1)
