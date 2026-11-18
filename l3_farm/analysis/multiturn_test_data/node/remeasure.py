"""Re-measure the final kernels of all arms (+ the seed) on ONE GPU, back to back, same graph execution (the arms'
own evaluator: LocalEvaluator arch 89, runtime auto). Each rep evaluates every candidate in a rotated order, paired
with the seed as incumbent (e2e_ratio) so the persistent host times both interleaved.
usage: remeasure.py PROBLEM_DIR OUTJSON REPS NAME=best_slots_dir ..."""
import asyncio, json, sys, time
from pathlib import Path
from kernelblaster.knowledge.slots import LocalEvaluator, SlotProblem
prob, outp, reps = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
cands = dict(a.split("=", 1) for a in sys.argv[4:])
p = SlotProblem(prob)
seeds = p.seed_sources()
cs = {"seed": seeds}
for n, d in cands.items():
    cs[n] = {**seeds, **{f.stem: f.read_text() for f in Path(d).glob("*.cu")}}
ev = LocalEvaluator("89", work_root=outp.parent / "rm_work", gpus="0")
async def main():
    res = {n: [] for n in cs}
    names = list(cs)
    for r in range(reps):
        order = names[r % len(names):] + names[:r % len(names)]
        for n in order:
            e = await ev.evaluate(p, cs[n], f"rm_{n}_{r}", **({"incumbent": seeds} if n != "seed" else {}))
            res[n].append({"ok": e.ok, "deployed_us": e.deployed_us, "partition_us": e.partition_us,
                           "ratio_vs_seed": e.e2e_ratio, "gpu": getattr(e, "gpu", None), "t": time.time()})
            print(n, r, res[n][-1], flush=True)
    outp.write_text(json.dumps(res, indent=1, default=str))
asyncio.run(main())
