#!/usr/bin/env python3
"""E1 (X-120) orchestrator (local). Every remote step is a script fed to `srun --jobid=J --overlap bash -s` through
farm.ssh (the `computelab` alias, key auth), only on jobs recorded in e1_data/my_jobs.json by nodes.py.
Node files live under /tmp/krisd/e1 only.

  e1_run.py push JOB             frozen code + data tarballs + node scripts, sanity import, GPU facts -> /tmp/krisd/e1
  e1_run.py start JOB P [P ...]  detached worker: per problem e1_node.py then 3x torch baseline (same node, same session)
  e1_run.py status JOB           which problems are done / the worker log tail
  e1_run.py fetch JOB [P ...]    runs/<P>/* small files -> e1_data/runs/<P>/
  e1_run.py gpu JOB              hostname, memory clock, VBIOS, compute apps
"""
import json, os, shlex, sys, io, tarfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
L3 = HERE.parents[1]
sys.path.insert(0, str(L3)); sys.path.insert(0, str(HERE))
import farm  # noqa: E402
import nodes  # noqa: E402

DATA = L3 / "analysis" / "e1_data"
STAGE = DATA / "stage"
R = "/tmp/krisd/e1"
N = "/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04"
ENV = f"""set -o pipefail
export PATH={N}/bin:/usr/bin:/bin CUDA_HOME={N} TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=16
export MB_CUPTI_LIB={N}/extras/CUPTI/lib64/libcupti.so.12
. /tmp/krisd/venv/bin/activate
export PYTHONPATH={R}/code MB_PRUNE_DEAD=0 TMPDIR={R}/tmp
mkdir -p {R}/tmp
"""
AUDIT = {"15_DenseNet121", "17_SqueezeNetFireModule", "31_VisionAttention", "50_ReLUSelfAttention",
         "49_Mamba2ReturnFinalState", "6_GoogleNetInceptionModule", "14_DenseNet121DenseBlock", "8_ResNetBasicBlock"}

SMI_Q = ("nvidia-smi --query-gpu=name,uuid,pci.bus_id,driver_version,vbios_version,temperature.gpu,power.draw,power.limit,"
         "clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,clocks_throttle_reasons.active,utilization.gpu,memory.used --format=csv")

WORKER = r'''#!/bin/bash
# E1 node worker: for each problem e1_node.py, then the PyTorch baseline 3x on the same GPU in the same session
cd /tmp/krisd/e1
RUNS=${E1_RUNS:-runs}
FL=${E1_FLAGS:-}
for p in "$@"; do
  mkdir -p $RUNS/$p
  if [ -f $RUNS/$p/e1.json ] && grep -q '"t1"' $RUNS/$p/e1.json && [ -f $RUNS/$p/torch_3.txt ]; then echo "skip $p"; continue; fi
  echo "=== $p $(date +%T)"
  {SMI} > $RUNS/$p/smi_before.csv; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >> $RUNS/$p/smi_before.csv
  A=""; case " {AUDIT} " in *" $p "*) A="--audit";; esac
  rm -rf $RUNS/$p/ir $RUNS/$p/*_seq $RUNS/$p/*_conc $RUNS/$p/*_ir
  timeout 5400 python bin/e1_node.py $p data/$p $RUNS/$p --reps 6 $A $FL > $RUNS/$p/node.log 2>&1
  echo "e1_node rc=$? $(date +%T)"
  for i in 1 2 3; do
    timeout 900 python bin/torch_bench.py data/$p/problem.py $p 50 10 0 > $RUNS/$p/torch_$i.txt 2> $RUNS/$p/torch_$i.err
  done
  {SMI} > $RUNS/$p/smi_after.csv; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >> $RUNS/$p/smi_after.csv
  # keep the small files only
  rm -rf $RUNS/$p/ir $RUNS/$p/*_seq $RUNS/$p/*_conc $RUNS/$p/*_ir $RUNS/$p/cache $RUNS/$p/profile $RUNS/$p/reuse_*/m_* $RUNS/$p/reuse_*/hinted $RUNS/$p/reuse_*/base_* $RUNS/$p/reuse_*/guard_*
  echo "done $p $(date +%T)"
done
echo WORKER_DONE
'''


def run(job, script, timeout=600):
    return nodes.run(job, ENV + script, timeout)


def put(job, data, remote):
    if str(job) not in nodes.mine():
        raise SystemExit(f"job {job} is not mine")
    rc, out, err = farm.put(job, data, remote)
    if rc:
        raise RuntimeError(f"put {remote}: {err.decode()[-300:]}")


def push(job):
    sys.path.insert(0, str(L3))
    import torch_posthoc
    node_files = {"e1_node.py": (HERE / "e1_node.py").read_bytes(),
                  "torch_bench.py": torch_posthoc.BENCH.encode(),
                  "worker.sh": WORKER.replace("{SMI}", SMI_Q).replace("{AUDIT}", " ".join(sorted(AUDIT))).encode()}
    b = io.BytesIO()
    with tarfile.open(fileobj=b, mode="w:gz") as t:
        for k, v in node_files.items():
            ti = tarfile.TarInfo(f"bin/{k}"); ti.size = len(v); ti.mtime = time.time(); t.addfile(ti, io.BytesIO(v))
    put(job, b.getvalue(), f"{R}/bin.tar.gz")
    put(job, STAGE / "mb_frozen.tar.gz", f"{R}/mb_frozen.tar.gz")
    put(job, STAGE / "data.tar.gz", f"{R}/data.tar.gz")
    want = (STAGE / "mb_frozen.sha256").read_text().strip()
    sh = f"""set -e
cd {R}
echo "{want}  mb_frozen.tar.gz" | sha256sum -c -
[ -f code/.sha ] && [ "$(cat code/.sha)" = "{want}" ] || (rm -rf code && mkdir -p code && tar xzf mb_frozen.tar.gz -C code && find code -name '._*' -delete && ln -sfn {R}/code/ModelBlaster {R}/code/modelblaster && echo {want} > code/.sha)
[ -d data ] || tar xzf data.tar.gz
tar xzf bin.tar.gz
python -c "import torch, modelblaster.pipeline.kb_reuse, modelblaster.pipeline.kb_export; print('import ok', torch.__version__)"
nvcc --version | tail -1
{SMI_Q}
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
"""
    rc, out, err = run(job, sh, 900)
    print(out, err[-500:])


def pushbin(job):
    import torch_posthoc
    node_files = {"e1_node.py": (HERE / "e1_node.py").read_bytes(),
                  "torch_bench.py": torch_posthoc.BENCH.encode(),
                  "worker.sh": WORKER.replace("{SMI}", SMI_Q).replace("{AUDIT}", " ".join(sorted(AUDIT))).encode()}
    b = io.BytesIO()
    with tarfile.open(fileobj=b, mode="w:gz") as t:
        for k, v in node_files.items():
            ti = tarfile.TarInfo(f"bin/{k}"); ti.size = len(v); ti.mtime = time.time(); t.addfile(ti, io.BytesIO(v))
    put(job, b.getvalue(), f"{R}/bin.tar.gz")
    rc, out, err = run(job, f"cd {R} && tar xzf bin.tar.gz && echo binok", 120)
    print(job, out.strip(), err[-200:])


def start(job, probs):
    pre = f"mkdir -p {R}/logs; cd {R}\n"
    log = f"{R}/logs/worker_{int(time.time())}.log"
    mode = os.environ.get("E1_MODE", "")
    envs = "export E1_RUNS=runs_fixed E1_FLAGS=--only-fixed\n" if mode == "fixed" else ""
    body = ENV + envs + f"bash {R}/bin/worker.sh " + " ".join(shlex.quote(p) for p in probs) + f" > {log} 2>&1\n"
    body = body.replace(f"bash {R}/bin/worker.sh", f"flock {R}/gpu.lock bash {R}/bin/worker.sh")   # one worker per GPU
    cmd = (f"nohup setsid srun --jobid={job} --overlap bash -c {shlex.quote(pre + body)} "
           f"</dev/null >/dev/null 2>&1 & sleep 3; echo started {log}")
    if str(job) not in nodes.mine():
        raise SystemExit("not mine")
    rc, out, err = farm.ssh(cmd, b"", timeout=120)
    print(out.decode(), err.decode()[-300:], "rc", rc)


def status(job):
    sh = f"""cd {R}; ls logs 2>/dev/null | tail -3; for f in $(ls -t logs/worker_*.log | head -1); do tail -8 $f; done
ls runs 2>/dev/null | wc -l; for d in runs/*; do [ -f $d/torch_3.txt ] && echo done $(basename $d); done | tr '\\n' ' '
echo; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"""
    rc, out, err = run(job, sh, 120)
    print(out, err[-300:])


def fetch(job, probs):
    rd = {"fixed": "runs_fixed", "nearest": "runs_nearest"}.get(os.environ.get("E1_MODE"), "runs")
    ps = " ".join(shlex.quote(p) for p in probs) if probs else "*"
    sh = f"""cd {R}/{rd}; tar czf - $(for d in {ps}; do [ -d $d ] && find $d -maxdepth 2 \\( -name 'e1.json' -o -name 'node.log' -o -name 'torch_*.txt' -o -name 'torch_*.err' -o -name 'smi_*.csv' -o -name 'reuse.json' \\) ; done) 2>/dev/null | base64 -w0"""
    rc, out, err = run(job, sh, 600)
    import base64
    raw = base64.b64decode(out.strip() or b"")
    if not raw:
        print("nothing", err[-300:]); return False
    dest = DATA / rd
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as t:
        t.extractall(dest)
    print("fetched", len(raw), "bytes into", dest)
    return True


def gpu(job):
    rc, out, err = run(job, f"hostname; {SMI_Q}; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; echo apps-done", 120)
    print(job, out, err[-200:])


if __name__ == "__main__":
    cmd, job, *rest = sys.argv[1:]
    {"push": lambda: push(job), "start": lambda: start(job, rest), "status": lambda: status(job),
     "fetch": lambda: fetch(job, rest), "pushbin": lambda: pushbin(job), "gpu": lambda: gpu(job)}[cmd]()
