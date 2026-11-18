import json, sys
sys.path.insert(0, "/tmp/krisd/family/pyn")
import famlib as F, model as M
d = M.build()
r = F.time_problem(d, sources=M.variant_sources(), reps=5, name="A")
print("A model", {k: r.get(k) for k in ("passed", "e2e_us", "e2e_basis", "partition_us", "attr_sum_us")}, flush=True)
conv = M.conv_slots()
att = r.get("attr_slot_us") or {}
print("conv slots", len(conv), "conv attr us", round(sum(att.get(s, 0) for s in conv), 1), "of", r.get("attr_sum_us"))
json.dump({"A": {k: r.get(k) for k in ("passed", "e2e_us", "e2e_basis", "partition_us", "attr_sum_us", "attr_op_us", "attr_slot_us")}},
          open(F.W / "pilot" / "A_model.json", "w"), indent=1)
