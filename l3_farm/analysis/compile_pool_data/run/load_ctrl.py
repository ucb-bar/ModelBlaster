#!/usr/bin/env python3
"""One 'problem' of the compile-pool build-load replay (X-148): replays the recorded candidate sets of one problem
through LocalEvaluator/SSHEvaluator._build_so (no LLM, no GPU). usage: load_ctrl.py ARM PROBLEM OUT.json THINK_S
ARM: local (per-problem cache, driver builds) | shared (KERNELBLASTER_BUILD_CACHE) | remote (shared + remote builds)"""
import asyncio, json, os, resource, sys, time
from pathlib import Path

arm, prob_name, out, think = sys.argv[1], sys.argv[2], Path(sys.argv[3]), float(sys.argv[4])
CORPUS = Path("/tmp/krisd/cpool/corpus")
LT = Path(os.environ["LT_DIR"])
HOSTS = [h for h in os.environ.get("LT_HOSTS", "").split(",") if h]
from kernelblaster.knowledge import slots as S
from kernelblaster.knowledge.slots import SlotProblem

nv = {"n": 0, "s": 0.0, "compile": 0}
_orig = S.LocalEvaluator._nvcc
async def _nvcc(self, args):
    t = time.time(); r = await _orig(self, args)
    nv["n"] += 1; nv["s"] += time.time() - t; nv["compile"] += int("-c" in args)
    return r
S.LocalEvaluator._nvcc = _nvcc

class Sem:
    def __init__(self, inner): self.i, self.wait, self.n = inner, 0.0, 0
    async def __aenter__(self):
        t = time.time(); r = await self.i.__aenter__(); self.wait += time.time() - t; self.n += 1; return r
    async def __aexit__(self, *a): return await self.i.__aexit__(*a)

async def main():
    work = LT / "w" / prob_name
    if arm == "local":
        ev = S.LocalEvaluator(arch="89", cache_dir=work / "cache")
    elif arm == "shared":
        ev = S.LocalEvaluator(arch="89", work_root=work)
    else:
        ev = S.SSHEvaluator(HOSTS, arch="89", remote_dir="/tmp/krisd/cpool/rb", remote_path=os.environ["CUDA_HOME"] + "/bin",
                            work_root=work, remote_build=True,
                            remote_build_jobs=int(os.environ.get("LT_RB_JOBS", "3")))
    ev._unit_sem = ev._sem = sem = Sem(ev._unit_sem)
    sets = json.loads((CORPUS / "index.json").read_text())[prob_name]
    lat, ok, nunits = [], 0, 0
    t0 = time.time()
    for i, s in enumerate(sets):
        p = SlotProblem(CORPUS / prob_name / s)
        src = p.seed_sources()
        nunits += len(p.units(src))
        t = time.time()
        so, err = await ev._build_so(p, src, work / f"s{i}")
        lat.append(round(time.time() - t, 3)); ok += int(so is not None)
        if think: await asyncio.sleep(think)
    wall = time.time() - t0
    res = dict(arm=arm, problem=prob_name, sets=len(sets), units=nunits, ok=ok, wall=round(wall, 2), lat=lat,
               nvcc_calls=nv["n"], nvcc_compile_calls=nv["compile"], nvcc_s=round(nv["s"], 2),
               sem_wait_s=round(sem.wait, 2), sem_n=sem.n,
               cpu_self=round(resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime, 2),
               cpu_children=round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime, 2))
    if hasattr(ev, "remote_build_stats"):
        res["rb"] = ev.remote_build_stats()
    try:
        res["cache"] = ev.host_stats().get("build_cache")
    except Exception as e:
        res["cache_err"] = repr(e)
    try:
        ev.close()
    except Exception:
        pass
    out.write_text(json.dumps(res))

asyncio.run(main())
