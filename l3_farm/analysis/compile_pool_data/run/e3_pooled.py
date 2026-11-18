#!/usr/bin/env python3
"""X-148 E3: pooled functional check without an LLM. ModelBlaster kb_graph_opt (stub LLM plugin, real fused kernels) on the driver,
KernelBlaster runs on 2 pool hosts over ssh. usage: e3_pooled.py ARM(A|B) MODEL(tiny|branches) OUTDIR HOST1,HOST2
A = plain (driver builds, per-problem caches); B = KERNELBLASTER_BUILD_CACHE + KERNELBLASTER_REMOTE_BUILD=1."""
import json, os, shutil, subprocess, sys, time
from pathlib import Path
arm, model, out, hosts = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4].split(",")
sys.path.insert(0, "/tmp/krisd/cpool/code")
from modelblaster.pipeline.tests import test_kb_graph_opt_flow as T
from modelblaster.pipeline import graph_hints as gh
from modelblaster.pipeline.kb_graph_opt import Config, run
shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
(out / "m.py").write_text(T.MODEL if model == "tiny" else T.BRANCHES)
env = {**os.environ, "PYTHONPATH": "/tmp/krisd/cpool/code"}
cp = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.extract_graph", "--bench-file", str(out / "m.py"), "--out-dir", str(out / "ir"),
                     "--quant", "fp32"], capture_output=True, text=True, env=env, cwd="/tmp/krisd/cpool/code/ModelBlaster")
assert cp.returncode == 0, cp.stderr[-1500:]
fused = T.fused_sources()
# deterministic compile error on the first answer for every fused kind (a duplicate error text across sources), then the real kernel
plugin = T.PLUGIN.format(fused=json.dumps(fused)).replace(
    "def scripted(msgs):", "SEEN = set()\n\n\ndef scripted(msgs):\n    pass").replace(
    '            return "```cuda\\n" + FUSED[sid] + "```"',
    '            if sid not in SEEN and "failed" not in user:\n                SEEN.add(sid)\n                return "```cuda\\n" + FUSED[sid] + "\\n#error injected_compile_error\\n```"\n            return "```cuda\\n" + FUSED[sid] + "```"')
(out / "plugin.py").write_text(plugin)
for k in ("KERNELBLASTER_BUILD_CACHE", "KERNELBLASTER_REMOTE_BUILD"):
    os.environ.pop(k, None)
if arm == "B":
    os.environ["KERNELBLASTER_BUILD_CACHE"] = str(out / "buildcache")
    os.environ["KERNELBLASTER_REMOTE_BUILD"] = "1"
os.environ["KERNELBLASTER_SSH_CONTROL_DIR"] = "/tmp/krisd/cpool/mux"
os.environ["PATH"] = os.environ["CUDA_HOME"] + "/bin:" + os.environ["PATH"]
KB = Path("/tmp/krisd/cpool/code/ModelBlaster/third_party/KernelBlaster")
os.environ["PYTHONPATH"] = f"{KB}/src:/tmp/krisd/cpool/code"
cfg = Config(kb=KB, model="stub:graphopt", plugins=[str(out / "plugin.py")], python=sys.executable, arch="89", rounds=6, rollouts=1, steps=1,
             variant_steps=1, max_fix=1, top_k=4, max_variants=4, remeasure=5, profile_runs=1, accept_gain=-1.0,
             parallel_hints=4 if model == "branches" else 0, min_gain_pct=0.0, ssh_hosts=hosts,
             remote_dir="/tmp/krisd/cpool/e3remote", runtime="auto",
             kb_args=["--remote-cuda-bin=" + os.environ["CUDA_HOME"] + "/bin"])
t0 = time.time()
s = run([out / "ir"], out / "work", cfg)
wall = time.time() - t0
doc = gh.load(out / "work" / "graph_hints.json")
status = {h["id"]: h["status"] for h in doc["hints"]}
ir = next(iter(s["irs"].values()))
res = dict(arm=arm, model=model, wall=round(wall, 1), status=status, passed=ir.get("passed"), fused=sorted(ir.get("fused") or []),
           speedup=ir.get("speedup"), fixpoint=s.get("fixpoint"))
(out / "result.json").write_text(json.dumps(res, indent=1, default=str))
print(json.dumps(res, default=str))
