#!/usr/bin/env python3
"""Export the KernelBlaster slot problems for A2's LLM runs from 41's kept final program (zero LLM).
  mkprob.py rev   BUNDLE IRFINAL OUT   41's l1-5 reverse GRU ops, seeded with the shipped v1 kernel (998 us/op;
                                       the forward winner does 557 us/op on them: known headroom 1.8x)
  mkprob.py region BUNDLE IRFINAL OUT  41's whole model with the rnn_stack region, every GRU on the forward winner
                                       (A1's reuse), level-parallel composite seed
Class: $MB_RECURRENT_CLASS at export decides gemm vs recurrent."""
import json, os, sys
from pathlib import Path
from modelblaster.pipeline.kb_export import export_problem
from modelblaster.pipeline.kb_proposals import _all_specs, with_fused_specs, write_variant_ir
from modelblaster.pipeline import region_slots as rs

mode, bundle, fin, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
g = json.loads((fin / "graph.json").read_text())
specs = with_fused_specs(g, dict(_all_specs()))
kman = json.loads((bundle / "problem" / "manifest.json").read_text())
src = {s["id"]: (bundle / "problem" / s["file"]).read_text() for s in kman["slots"]}
if mode == "rev":
    from modelblaster.pipeline.graph_hints import unique_names
    gu = unique_names(g)
    idx = [i for i, o in enumerate(gu["ops"]) if o.get("slot_variant") == "v1" and o["op"] == "gru_seq_f16"]
    p = export_problem(fin, out, op_indices=idx, slot_sources=src, specs=specs, strategy="plan",
                       name="a2_41_gru_rev_l1_5", concurrency="off")
else:
    v0 = {**src, **{k: src["gru_seq_f16__v0"] for k in src if k.startswith("gru_seq_f16__v")}}
    names = rs.find_regions(g, "rnn_stack", specs)[0]
    gr, spec = rs.fuse_region(g, names, specs, "rnn_stack")
    rir = write_variant_ir(fin, gr, out.parent / (out.name + "_ir"))
    p = export_problem(rir, out, slot_sources=v0, specs={**specs, spec.op: spec}, strategy="whole",
                       name="a2_41_region", concurrency="off")
m = json.loads((p / "manifest.json").read_text())
print(json.dumps({"problem": str(p), "slots": [(s["id"], s["op_class"], s["seed_source"], len(s["instances"]))
                                                for s in m["slots"]]}))
