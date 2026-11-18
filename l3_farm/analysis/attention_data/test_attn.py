import json, sys, copy, pathlib, traceback
import numpy as np
from modelblaster.pipeline import graph_hints as gh, attention_fuse as af
from modelblaster.pipeline.kb_proposals import _all_specs
W = pathlib.Path(sys.argv[1])
res = {}
for name in sys.argv[2:]:
    ir = W / "irs" / name
    g = json.loads((ir / "graph.json").read_text())
    specs = _all_specs()
    cl = af.const_lookup_for(ir)
    r = res[name] = {"ops": len(g["ops"])}
    try:
        variants = {}
        att = [af.attention_hint({"by": "test"}, status="accepted")]
        variants["attn_only"] = gh.apply_hints(g, att, specs, const_lookup=cl)
        # the fused-kind spellings the flow produces: every MB fusion proposal accepted first
        props = gh.propose_from_profile(g, {o["name"]: 1.0 for o in g["ops"]}, specs, min_gain_pct=0.0)
        fz = [{**h, "status": "accepted"} for h in props]
        r["fusions"] = [h["fused_kind"] for h in fz]
        gf, addf, _ = gh.apply_hints(g, fz, specs)
        r["fused_kinds_present"] = sorted({o["op"] for o in gf["ops"] if o["op"].startswith("fused__")})
        variants["fusions+attn"] = gh.apply_hints(g, fz + att, specs, const_lookup=cl)
        # proposer path: verified_masks carried in the hint, no const lookup at apply time
        hp = af.propose(g, {o["name"]: 1.0 for o in g["ops"]}, const_lookup=cl, min_gain_pct=0.0)
        r["proposed_hints"] = [{k: h.get(k) for k in ("id", "kind", "verified_masks", "ops")} for h in hp]
        variants["proposed"] = gh.apply_hints(g, [{**h, "status": "accepted"} for h in hp], specs)
        # negative: a mask that is not triu(1) must not be fused as causal
        bad = lambda k: (lambda a: (a * 0 + 1) if a is not None else None)(cl(k))
        _, rep_bad = af.fuse(g, const_lookup=bad)
        r["nonCausalMask_fused"] = len(rep_bad["fused"])
        for vn, (g2, added, prov) in variants.items():
            n_att = sum(o["op"] == "attention_f16" for o in g2["ops"])
            v = r[vn] = {"ops": len(g2["ops"]), "attention_ops": n_att,
                         "kinds": sorted({o["op"] for o in g2["ops"]}),
                         "applied": [a["id"] for a in prov["applied"]], "skipped": prov["skipped"][:3]}
            vr = gh.verify_rewrite(ir, g2, specs={**specs, **added}, build=True, arch="89", work=W / "verify" / name / vn)
            v["verify"] = vr
            print(name, vn, "attention_ops", n_att, "ok", vr.get("ok"), "golden", vr["golden"].get("ok"),
                  "vs_orig", vr.get("vs_original_walk"), "rel", vr.get("vs_original_rel"), "driver", vr.get("driver", {}).get("passed"), flush=True)
    except Exception as e:
        r["error"] = traceback.format_exc()[-2000:]
        print(name, "ERROR", r["error"], flush=True)
print("@@T@@" + json.dumps(res, default=str))
