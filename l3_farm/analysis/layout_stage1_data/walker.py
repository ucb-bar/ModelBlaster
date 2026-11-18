#!/usr/bin/env python3
"""Walker-level check of the stage-1 fold on a REAL final IR at full size (X-142). Node-side.

  walker.py PROBLEM [--dev cpu|cuda] [--no-build] [--runs N]

IR = the problem's real final graph (l3_farm/results/<p>/work/final/<p>/ir/graph.json, copied to
/tmp/krisd/x142/final/<p>.graph.json) over weights/io re-extracted with the farm's own command (extract_graph --quant fp16
--bench-target-mb 0). The rewrite is layout_fold.fold_hint -> graph_hints.apply_hints (what the flow does for an accepted
fold_layout hint). Checks, as graph_hints.verify_rewrite: torch walk of the rewrite vs the golden (fp16 tolerance) and vs
the ORIGINAL walk (bit-identical); then export_problem(strategy="whole") + nvcc per-slot build + the driver (seed kernels
only) for original and folded: driver pass, e2e, per-op CUPTI. Writes out/<p>.json."""
import copy, json, sys, time, traceback
from pathlib import Path

W = Path("/tmp/krisd/x142")
sys.path.insert(0, "/tmp/krisd/x142/code")


def main():
    a = sys.argv[1:]
    p = a[0]
    dev = a[a.index("--dev") + 1] if "--dev" in a else "cpu"
    build = "--no-build" not in a
    runs = int(a[a.index("--runs") + 1]) if "--runs" in a else 1
    import torch
    from modelblaster.pipeline import graph_hints as gh, layout_fold as lf, kb_proposals as kp
    from modelblaster.pipeline.kb_export import _harness, export_problem, build_and_run
    mbir, walk_torch = _harness()
    ir_dir = W / "irs" / p
    final = json.loads((W / "final" / f"{p}.graph.json").read_text())
    res = {"problem": p, "dev": dev}
    work = W / "w" / p
    work.mkdir(parents=True, exist_ok=True)
    orig_ir = kp.write_variant_ir(ir_dir, final, work / "orig_ir")
    specs = kp._all_specs()
    hint = lf.fold_hint({"by": "x142", "ir": p, "ts": gh._now()})
    g2, added, prov = gh.apply_hints(final, [hint], specs, statuses=("proposed",))
    res["prov"] = {"applied": prov["applied"], "skipped": prov["skipped"]}
    specs = {**specs, **added}
    cnt = lambda g: {k: sum(o["op"] == k for o in g["ops"]) for k in sorted({o["op"] for o in g["ops"]})}
    res["ops_before"], res["ops_after"] = cnt(final), cnt(g2)
    res["copies_before"] = sum(o["op"] == "strided_copy_f16" for o in final["ops"])
    res["copies_after"] = sum(o["op"] == "strided_copy_f16" for o in g2["ops"])
    new_ir = kp.write_variant_ir(ir_dir, g2, work / "new_ir")
    print(p, "folded:", prov["applied"], "copies", res["copies_before"], "->", res["copies_after"], flush=True)
    # ---- torch walk (reference semantics): rewrite vs golden and vs the original walk
    t0 = time.time()
    with torch.no_grad():
        ir0 = mbir.load_ir(orig_ir)
        ref, _ = walk_torch.walk(ir0, dev=dev)
        t1 = time.time()
        ir1 = mbir.IR(ir0.name, g2, ir0.weights, ir0.golden_in, ir0.golden_out)
        out, _ = walk_torch.walk(ir1, dev=dev)
    res["walk_s"] = [round(t1 - t0, 1), round(time.time() - t1, 1)]
    res["golden_orig"] = mbir.compare(ref, ir0.golden_out, "fp16")
    res["golden_fold"] = mbir.compare(out, ir0.golden_out, "fp16")
    res["vs_original_walk"] = float((out.float().cpu() - ref.float().cpu()).abs().max()) if out.shape == ref.shape else None
    res["bit_identical"] = bool(torch.equal(out.cpu().contiguous().view(torch.int16), ref.cpu().contiguous().view(torch.int16))) \
        if out.dtype == torch.float16 else bool(torch.equal(out.cpu(), ref.cpu()))     # bitwise (nan/inf aware)
    o32, r32 = out.float().cpu(), ref.float().cpu()
    same = (o32 == r32) | (torch.isnan(o32) & torch.isnan(r32))          # nan payloads / signs ignored
    res["nan_aware_mismatch"] = int((~same).sum())
    gold = torch.from_numpy(ir0.golden_out.astype("float32")).reshape(-1)
    res["golden_nan"] = int(torch.isnan(gold).sum())
    res["n_inf_nan"] = {"inf": int(torch.isinf(r32).sum()), "nan": int(torch.isnan(r32).sum()), "numel": int(r32.numel())}
    print(p, "walk", res["walk_s"], "golden orig/fold", res["golden_orig"].get("ok"), res["golden_fold"].get("ok"),
          "vs_original_walk", res["vs_original_walk"], flush=True)
    del ref, out
    if build:
        for tag, ird, spc in (("orig", orig_ir, kp._all_specs()), ("fold", new_ir, specs)):
            try:
                t0 = time.time()
                prob = export_problem(ird, work / f"problem_{tag}", specs=spc, strategy="whole")
                te = time.time() - t0
                man = json.loads((prob / "manifest.json").read_text())
                slots = {s["id"]: {"op": s["op"], "seed": s.get("seed_source"), "n": len(s["instances"]),
                                   "shapes": [i.get("shape") for i in s["instances"]][:2]} for s in man["slots"]}
                t0 = time.time()
                r = build_and_run(prob, arch="89", work=work / f"build_{tag}", graph=True, runs=runs)
                keep = ("passed", "stage", "partition_us", "partition_graph_us", "e2e_us", "e2e_basis", "slot_us", "slots",
                        "attr_slot_us", "attr_op_us", "attr_sum_us", "outputs", "errors", "attr_skip", "graph_skip")
                res[tag] = {k: r.get(k) for k in keep}
                res[tag].update(export_s=round(te, 1), build_run_s=round(time.time() - t0, 1), manifest_slots=slots)
                if not r.get("passed"):
                    res[tag]["log"] = (r.get("log") or "")[-3000:]
                print(p, tag, "passed", r.get("passed"), "e2e_us", r.get("e2e_us"), flush=True)
            except Exception as e:   # noqa: BLE001
                res[tag] = {"error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-2500:]}
                print(p, tag, "ERROR", e, flush=True)
    (W / "out").mkdir(exist_ok=True)
    (W / "out" / f"{p}.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
