"""X-149 GPU test (my L40S, frozen tree, no LLM): the guard on the reproduced locked state through the flow's own timing path.
phases: clean -> induce lock (ncu SIGTERM + target killed, as repro2 V4) -> evaluate (guard must flag) -> allowlisted reset
(sudo rusu reset-gpu -i 0, from THIS test script, never from the flow) -> evaluate (clean) -> flow ncu timeout + post-check."""
import asyncio, json, logging, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, "/tmp/krisd/ng/code/ModelBlaster/third_party/KernelBlaster/src")
os.environ["KERNELBLASTER_CLOCK_GUARD"] = "1"
os.environ["KERNELBLASTER_CLOCK_CONFIRM_S"] = "1.5"
from kernelblaster.knowledge import clock_guard as cg
from kernelblaster.knowledge.slots import LocalEvaluator, SlotProblem
from modelblaster.pipeline import preapply_audit as pa
from modelblaster.pipeline.kb_ingest import CudaCache
from modelblaster.pipeline.kb_proposals import _all_specs
from loguru import logger

OUT = Path(sys.argv[1]); NAME = sys.argv[2] if len(sys.argv) > 2 else "8_ResNetBasicBlock"
C = "/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04"
env = {**os.environ, "PATH": f"{C}/bin:" + os.environ["PATH"]}
res = {"name": NAME, "t0": time.time(), "phases": {}}
warn = []
logger.add(lambda m: warn.append(str(m)) if "CLOCK GUARD" in str(m) or "clock guard" in str(m) else None, level="WARNING")


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env).stdout.strip()


def smi():
    return sh(cg.query_command(None, 0))


def lock_gpu():
    """Reproduction of X-144: ncu on a kernel, SIGTERM to ncu, the (hung) target killed -> clocks stay locked."""
    cmd = (f"cd /tmp/krisd/ng/repro && (setsid ncu --kernel-name-base demangled -k regex:mbslot_a:: --launch-count 100000 "
           f"--section SpeedOfLight --csv ./t 3000 > v.out 2>&1 &) ; sleep 8; pkill -TERM ncu; sleep 2; pkill -9 -x t; pkill -9 ncu; sleep 6")
    sh(cmd)


ir = Path("/tmp/krisd/ng/irs") / NAME
g0 = json.loads((ir / "graph.json").read_text())
cache = CudaCache(Path(f"/tmp/krisd/ng/gt/{NAME}/cache"))
pdir = pa.compose_model(ir, g0, _all_specs(), cache, Path(f"/tmp/krisd/ng/gt/{NAME}/inc"), "89")
prob = SlotProblem(Path(pdir))
seeds = prob.seed_sources()


async def main():
    ev = LocalEvaluator("89", gpus="0", work_root=Path(f"/tmp/krisd/ng/gt/{NAME}/work"))
    orig = ev.run_timeout
    ev.run_timeout = lambda problem=None, profiling=False: 4.0 if profiling else orig(problem, profiling)   # ncu times out
    res["smi_start"] = smi()

    async def phase(name, tag):
        w0 = len(warn)
        r = await ev.evaluate(prob, seeds, tag)
        pool = ev._pool
        d = {"smi_after": smi(), "ok": r.ok, "e2e_us": r.deployed_us, "partition_us": r.partition_us, "gpu": r.gpu,
             "suspect": cg.clock_suspect(r), "warnings": warn[w0:], "clock_stats": ev.clock_stats(),
             "dev": [(x.label(), x.clock_suspect, round(max(0.0, x.quarantined_until - pool.clock()), 1), x.quarantines) for x in pool.devices]}
        res["phases"][name] = d
        print(name, {k: d[k] for k in ("ok", "e2e_us", "suspect", "dev")}, flush=True)
        OUT.write_text(json.dumps(res, indent=1, default=str))
        return r

    await phase("p0_clean", "p0")
    # induce the locked state the way X-144 got it: the flow's own ncu timeout (profile() -> killpg after the limit)
    for i in range(1, 5):
        w0 = len(warn)
        by, err = await ev.profile(prob, seeds, f"ncu_to{i}")
        await asyncio.sleep(3)
        d = {"err": err, "smi_after": smi(), "clock_stats": ev.clock_stats(), "warnings": warn[w0:],
             "dev": [(x.label(), x.clock_suspect, round(max(0.0, x.quarantined_until - ev._pool.clock()), 1)) for x in ev._pool.devices]}
        res["phases"][f"p1_flow_ncu_timeout_{i}"] = d
        print("p1", i, d, flush=True)
        if cg.classify(cg.parse_query(smi())[0])[0] != cg.OK:
            break
    res["smi_locked"] = smi()
    OUT.write_text(json.dumps(res, indent=1, default=str))
    if cg.classify(cg.parse_query(res["smi_locked"])[0])[0] == cg.OK:      # the flow's timeout did not lock it: shell reproduction
        lock_gpu(); res["smi_locked"] = smi(); res["lock_by"] = "shell (repro2 V4)"
    else:
        res["lock_by"] = "flow ncu timeout"
    print("locked state:", res["smi_locked"], res["lock_by"], flush=True)
    await phase("p2_locked_evaluate", "p2")
    await ev.close()                                  # stop the persistent hosts: nothing may hold the GPU for the reset
    await asyncio.sleep(2)
    res["reset_out"] = sh("sudo rusu reset-gpu -i 0 2>&1; sleep 8")      # documented allowlist; this test script only
    res["smi_reset"] = smi(); print("after reset:", res["smi_reset"], res["reset_out"][:200], flush=True)
    await phase("p3_after_reset", "p3")
    await ev.close()


asyncio.run(main())
