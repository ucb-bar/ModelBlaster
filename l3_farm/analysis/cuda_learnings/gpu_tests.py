#!/usr/bin/env python3
"""On-node GPU tests of the concurrent walker (run after recompose.py 41_GRUBidirectional ... siblings).

  gpu_tests.py RUN_DIR   (RUN_DIR = runs/41_GRUBidirectional.siblings: its ir/ and cache/ after the reuse pass)

1. deadlock watchdog: the GRU slot wrapped so that it spins forever when it runs with a reduced SM budget (i.e. as a
   concurrent branch); the driver's watchdog must end the replay (MB_CONC_DEADLOCK), build_and_run must rerun it
   with MB_CONCURRENCY=0 (sequential), pass, and record ``concurrency_fallback``;
2. policies: split vs none vs strict (strict: the GRU kernel's one-wave grid exceeds half the SMs -> exclusive);
3. sibling fusion composite seed: fused__par__ GRU fwd+rev ops seeded with the stage winners, stages forked onto
   helper threads' streams; must pass and replay as concurrent branches inside one slot;
4. eager vs graph output equality is checked by the driver itself (MB_GRAPH_CHECK) in every run.
-> RUN_DIR/gpu_tests.json
"""
import json
import os
import sys
import time
from pathlib import Path

from modelblaster.pipeline import fusion_ext
from modelblaster.pipeline import graph_hints as gh
from modelblaster.pipeline.kb_export import build_and_run, export_problem
from modelblaster.pipeline.kb_ingest import CudaCache, cache_plan
from modelblaster.pipeline.kb_proposals import _all_specs, with_fused_specs, write_variant_ir

run = Path(sys.argv[1]).resolve()
ONLY = sys.argv[2] if len(sys.argv) > 2 else "all"          # all | watchdog
ir, cache = run / "ir", CudaCache(run / "cache")
out = run / "gpu_tests"
out.mkdir(exist_ok=True)
specs = dict(_all_specs())
g = json.loads((ir / "graph.json").read_text())
res = {}


def export(tag, graph, plan, conc="on", policy="split", sources=None):
    d = write_variant_ir(ir, plan["graph"], out / f"{tag}_ir")
    src = dict(plan["sources"])
    src.update(sources or {})
    return export_problem(d, out / tag, slot_sources=src, slot_source_tags=plan["tags"], sub_sources=plan["sub_sources"],
                          specs=with_fused_specs(plan["graph"], specs), strategy="whole", concurrency=conc,
                          sm_budget_policy=policy)


def info(r):
    return {k: r.get(k) for k in ("passed", "e2e_us", "e2e_basis", "graph_nodes", "graph_skip", "concurrency_fallback",
                                  "partition_us", "stage")} | {"log_tail": (r.get("log") or "")[-600:]}


plan = cache_plan(g, cache, "sm_89", specs)
gru_sid = [s for s in plan["sources"] if s.startswith("gru_seq_f16")]
res["gru_slots"] = gru_sid
# ---- 1. watchdog
base = plan["sources"][gru_sid[0]]
spin = """
namespace mb_inner {
%s
}
__global__ void mb_spin_forever(volatile int *f) { while (*f == 0) { } }
void kernel_gru_seq_f16(const __half *x, const __half *h0, const __half *w_ih, const __half *w_hh, const __half *b_ih,
                        const __half *b_hh, __half *y, __half *hn, int B, int T, int I, int H, int reverse) {
    static int *flag = nullptr;
    if (!flag) { cudaMalloc(&flag, 4); cudaMemset(flag, 0, 4); }
    int d = 0, sms = 0; cudaGetDevice(&d); cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, d);
    if (mb_sm_budget() < sms) mb_spin_forever<<<1, 1>>>(flag);     // only as a concurrent branch
    mb_inner::kernel_gru_seq_f16(x, h0, w_ih, w_hh, b_ih, b_hh, y, hn, B, T, I, H, reverse);
}
""" % base
p1 = export("wd", g, plan, sources={sid: spin for sid in gru_sid})
os.environ["MB_CONC_TIMEOUT_S"] = "10"
t = time.time()
r1 = build_and_run(p1, arch="89", runs=1, graph=True)
res["watchdog"] = info(r1) | {"seconds": round(time.time() - t, 1)}
os.environ.pop("MB_CONC_TIMEOUT_S")
if ONLY == "watchdog":
    (run / "gpu_tests_watchdog.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
    sys.exit(0)
# ---- 2. policies
for pol in ("split", "none", "strict"):
    p = export(f"pol_{pol}", g, plan, policy=pol)
    r = build_and_run(p, arch="89", runs=1, graph=True)
    res[f"policy_{pol}"] = info(r) | {"exclusive": [ln for ln in (r.get("log") or "").splitlines()
                                                    if "MB_CONC_EXCLUSIVE" in ln][:4]}
p = export("seq", g, plan, conc="off")
res["sequential"] = info(build_and_run(p, arch="89", runs=1, graph=True))
# ---- 3. sibling fusion composite seed (fwd+rev of every layer as one fused__par__ op)
h = fusion_ext.siblings_hint("fused__par__gru_seq_f16__gru_seq_f16", None, {"by": "test"})
h.pop("groups")
gs, added, prov = gh.apply_hints(g, [h], dict(specs), statuses=None)
sp = dict(specs)
sp.update(added)
vplan = cache_plan(gs, cache, "sm_89", sp, split=True)
res["siblings_applied"] = len([a for a in prov["applied"] if a["kind"] == "fuse_siblings"])
for conc in ("off", "on"):
    d = write_variant_ir(ir, vplan["graph"], out / f"par_{conc}_ir")
    pp = export_problem(d, out / f"par_{conc}", slot_sources=vplan["sources"], slot_source_tags=vplan["tags"],
                        sub_sources=vplan["sub_sources"], specs=with_fused_specs(vplan["graph"], sp), strategy="whole",
                        concurrency=conc)
    man = json.loads((pp / "manifest.json").read_text())
    res[f"par_{conc}"] = info(build_and_run(pp, arch="89", runs=1, graph=True)) | {
        "seeds": {s["id"]: s["seed_source"] for s in man["slots"] if "par" in s["id"]}}
(run / "gpu_tests.json").write_text(json.dumps(res, indent=1, default=str))
print(json.dumps(res, indent=1, default=str))
