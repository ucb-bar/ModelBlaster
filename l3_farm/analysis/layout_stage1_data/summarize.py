#!/usr/bin/env python3
"""markdown tables from walker_out/*.json and kernel_out/*.json"""
import json, glob
rows = []
print("| problem | folds | copies | walk vs original | golden (orig/fold) | driver | seed e2e us orig -> folded | folded ops: copies+gemm CUPTI us -> folded gemm us |")
print("|---|---|---|---|---|---|---|---|")
for f in sorted(glob.glob("walker_out/*.json"), key=lambda x: int(x.split("/")[-1].split("_")[0])):
    r = json.load(open(f)); o, n = r["orig"], r["fold"]
    app = r["prov"]["applied"][0]["folded"]
    det = []
    for gem, copies in app.items():
        before = sum(o["attr_op_us"].get(c, 0) for c in copies) + o["attr_op_us"].get(gem.split("+")[0] if "+" in gem else gem, 0)
        # fused head: its stages carry the names 'matmul', 'mul', ...; the original's attr names are the fused op name
        before = sum(o["attr_op_us"].get(c, 0) for c in copies) + o["attr_op_us"].get(gem, o["attr_op_us"].get(gem.split("+")[0], 0))
        det.append(f"{gem}: {before:.0f} -> {n['attr_op_us'].get(gem, float('nan')):.0f}")
    ident = "bit-identical" if r["bit_identical"] else ("identical up to NaN payload" if r.get("nan_aware_mismatch") == 0 else f"max diff {r['vs_original_walk']}")
    print(f"| {r['problem']} | {len(app)} | {r['copies_before']} -> {r['copies_after']} | {ident} | {r['golden_orig'].get('ok')}/{r['golden_fold'].get('ok')} | "
          f"{'pass' if o['passed'] else 'FAIL'}/{'pass' if n['passed'] else 'FAIL'} | {o['e2e_us']:.0f} -> {n['e2e_us']:.0f} | " + "; ".join(det) + " |")
b = json.load(open("kernel_out/bench.json"))
print()
print("| problem / op | GFLOP | tiled seed us | naive seed us | torch copy+bmm us (copies only / bmm only) | tiled / copy+bmm | TFLOP/s |")
print("|---|---|---|---|---|---|---|")
for x in b:
    print(f"| {x['problem'][:2]} {x['op'][:30]} | {x['gflop']:.1f} | {x['tiled_us']:.0f} | {x['naive_us']:.0f} | {x['copy_bmm_us']:.0f} ({x['copy_only_us']:.0f} / {x['bmm_only_us']:.0f}) | {x['tiled_over_copy_bmm']:.2f} | {x['tiled_tflops']:.1f} |")
