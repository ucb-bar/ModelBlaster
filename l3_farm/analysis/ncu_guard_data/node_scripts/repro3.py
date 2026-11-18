"""What does the flow's own ncu timeout (killpg of the shell's group) leave behind? processes + clocks at 1/3/10/30 s."""
import asyncio, json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, "/tmp/krisd/ng/code/ModelBlaster/third_party/KernelBlaster/src")
from kernelblaster.knowledge.slots import LocalEvaluator, SlotProblem
from kernelblaster.knowledge import clock_guard as cg
NAME = "8_ResNetBasicBlock"
C = "/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04"
env = {**os.environ, "PATH": f"{C}/bin:" + os.environ["PATH"]}
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True, env=env).stdout.strip()
prob = SlotProblem(Path(f"/tmp/krisd/ng/gt/{NAME}/inc/model"))
async def main():
    ev = LocalEvaluator("89", gpus="0", work_root=Path("/tmp/krisd/ng/gt/r3work"), keep=False)
    orig = ev.run_timeout
    ev.run_timeout = lambda problem=None, profiling=False: 4.0 if profiling else orig(problem, profiling)
    by, err = await ev.profile(prob, prob.seed_sources(), "r3")
    print("profile ->", err, flush=True)
    for t in (0, 2, 7, 20):
        await asyncio.sleep(t if t == 0 else (t - [0, 2, 7, 20][[0, 2, 7, 20].index(t) - 1]))
        print(f"+{t}s", sh(cg.query_command(None, 0)), "| procs:", sh("ps -eo pid,ppid,pgid,stat,etime,cmd | grep -E 'mbslot|/problem|ncu|nv-nsight' | grep -v grep | cut -c1-160"), flush=True)
        print("   compute apps:", sh("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader"), flush=True)
asyncio.run(main())
