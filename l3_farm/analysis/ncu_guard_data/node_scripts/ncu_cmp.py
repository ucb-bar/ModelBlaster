"""X-149: what does `--clock-control none` change in the ncu reports the flow consumes?
Real slots (unfused ops of 8_ResNetBasicBlock / 19_MobileNetV1 fp16) through KernelBlaster's LocalEvaluator.profile
(the flow's own ncu path: ncu_command + parse + summarize + render). base vs none, N reps each, interleaved."""
import asyncio, json, os, sys, time
from pathlib import Path
sys.path.insert(0, "/tmp/krisd/ng/code/ModelBlaster/third_party/KernelBlaster/src")
from kernelblaster.knowledge import ncu as N
from kernelblaster.knowledge.slots import LocalEvaluator, SlotProblem
from modelblaster.pipeline import preapply_audit as pa
from modelblaster.pipeline.kb_ingest import CudaCache
from modelblaster.pipeline.kb_proposals import _all_specs

OUT = Path(sys.argv[1]); REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
names = sys.argv[3:] or ["8_ResNetBasicBlock", "19_MobileNetV1"]
specs = _all_specs()
res = {"t0": time.time(), "problems": {}}
for name in names:
    ir = Path("/tmp/krisd/ng/irs") / name
    g0 = json.loads((ir / "graph.json").read_text())
    cache = CudaCache(Path(f"/tmp/krisd/ng/cmp/{name}/cache"))
    pdir = pa.compose_model(ir, g0, specs, cache, Path(f"/tmp/krisd/ng/cmp/{name}/inc"), "89")
    prob = SlotProblem(Path(pdir))
    ev = LocalEvaluator("89", gpus="0", work_root=Path(f"/tmp/krisd/ng/cmp/{name}/work"))
    rows = {"base": [], "none": []}
    for rep in range(REPS):
        for mode in ("base", "none"):
            os.environ["KERNELBLASTER_NCU_CLOCK_CONTROL"] = mode
            cmd = N.ncu_command()
            t0 = time.time()
            by, err = asyncio.run(ev.profile(prob, prob.seed_sources(), f"{mode}{rep}"))
            rows[mode].append({"by": by, "err": err, "wall": round(time.time() - t0, 1), "cmd_tail": cmd[-40:]})
            print(name, mode, rep, "slots", len(by), "err", err, "wall", rows[mode][-1]["wall"], flush=True)
    res["problems"][name] = {"slots": {s: {"op": sl.op, "class": sl.op_class} for s, sl in prob.slots.items()}, "rows": rows}
    OUT.write_text(json.dumps(res, indent=1, default=str))
