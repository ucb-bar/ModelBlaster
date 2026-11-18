#!/usr/bin/env python3
"""Reuse study orchestrator (local). Remote work only on jobs 5325505 / 5325510, only under /tmp/krisd/reuse_study.

  reuse_run.py prep P [P ...]       build stage/{data,npool,mb_frozen}.tar.gz (local, read-only on results)
  reuse_run.py push JOB             upload + unpack + import check
  reuse_run.py start JOB P [P ...]  detached worker (reuse_node then 3x torch graph baseline per problem)
  reuse_run.py status JOB
  reuse_run.py fetch JOB            small result files -> data/runs/
  reuse_run.py clean JOB            rm -rf /tmp/krisd/reuse_study
"""
import hashlib, io, json, os, shlex, subprocess, sys, tarfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rs  # noqa: E402
L3 = rs.L3
ROOT = L3.parent
STAGE = HERE / "stage"
R = rs.R
N = "/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04"
ENV = f"""set -o pipefail
export PATH={N}/bin:/usr/bin:/bin CUDA_HOME={N} TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=16
export MB_CUPTI_LIB={N}/extras/CUPTI/lib64/libcupti.so.12
. /tmp/krisd/venv/bin/activate
export PYTHONPATH={R}/code MB_PRUNE_DEAD=0 TMPDIR={R}/tmp
mkdir -p {R}/tmp
"""
SMI_Q = ("nvidia-smi --query-gpu=name,uuid,pci.bus_id,driver_version,vbios_version,temperature.gpu,power.draw,power.limit,"
         "clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,clocks_throttle_reasons.active,utilization.gpu,memory.used --format=csv")
WORKER = r'''#!/bin/bash
cd /tmp/krisd/reuse_study
RUNS=runs
for p in "$@"; do
  mkdir -p $RUNS/$p
  if [ -f $RUNS/$p/ru.json ] && grep -q '"t1"' $RUNS/$p/ru.json && [ -f $RUNS/$p/torch_3.txt ]; then echo "skip $p"; continue; fi
  echo "=== $p $(date +%T)"
  {SMI} > $RUNS/$p/smi_before.csv; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >> $RUNS/$p/smi_before.csv
  rm -rf $RUNS/$p/ir $RUNS/$p/orig $RUNS/$p/ex $RUNS/$p/nk4 $RUNS/$p/gbest $RUNS/$p/cold $RUNS/$p/*_ir
  timeout 5400 python bin/reuse_node.py $p data/$p $RUNS/$p --reps 6 --npool npool > $RUNS/$p/node.log 2>&1
  echo "reuse_node rc=$? $(date +%T)"
  for i in 1 2 3; do
    timeout 900 python bin/torch_bench.py data/$p/problem.py $p 50 10 0 > $RUNS/$p/torch_$i.txt 2> $RUNS/$p/torch_$i.err
  done
  {SMI} > $RUNS/$p/smi_after.csv; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >> $RUNS/$p/smi_after.csv
  rm -rf $RUNS/$p/ir $RUNS/$p/orig $RUNS/$p/ex $RUNS/$p/nk4 $RUNS/$p/gbest $RUNS/$p/cold $RUNS/$p/*_ir $RUNS/$p/cache $RUNS/$p/profile $RUNS/$p/profile_cold $RUNS/$p/probe_A* $RUNS/$p/reuse_*/m_* $RUNS/$p/reuse_*/hinted $RUNS/$p/reuse_*/base_* $RUNS/$p/reuse_*/guard_*
  echo "done $p $(date +%T)"
done
echo WORKER_DONE
'''


def add(tw, arc, data):
    ti = tarfile.TarInfo(arc)
    ti.size = len(data)
    ti.mtime = 1792900000
    tw.addfile(ti, io.BytesIO(data))


def src_dir(p):
    for b in ("results_headline_heavy", "results_headline"):
        d = L3 / b / p
        if (d / "work" / "model_runtime.json").exists():
            return d
    raise SystemExit(f"{p}: no finished dir")


def prep(ps):
    STAGE.mkdir(exist_ok=True)
    with tarfile.open(STAGE / "data.tar.gz", "w:gz") as tw:
        for p in ps:
            R_ = src_dir(p)
            add(tw, f"data/{p}/problem.py", (L3 / "problems" / f"{p}.py").read_bytes())
            for w in sorted((R_ / "work").glob("r[0-9][0-9]/winners")):
                for f in sorted(w.rglob("*")):
                    if f.is_file():
                        add(tw, f"data/{p}/work/{w.parent.name}/winners/{f.relative_to(w)}", f.read_bytes())
            gh = R_ / "work" / "graph_hints.json"
            if gh.exists():
                add(tw, f"data/{p}/work/graph_hints.json", gh.read_bytes())
            fin = R_ / "work" / "final" / p
            for f in sorted((fin / "problem").rglob("*")):
                if f.is_file() and "/build/" not in str(f) and f.name != "data.bin":
                    add(tw, f"data/{p}/work/final/{p}/problem/{f.relative_to(fin / 'problem')}", f.read_bytes())
            add(tw, f"data/{p}/work/final/{p}/ir/graph.json", (fin / "ir" / "graph.json").read_bytes())
            print("data", p, flush=True)
    pool = json.load(open(HERE / "data" / "pool.json"))
    rows, codes = [], {}
    for r in pool:
        if not r["in_win"] or not os.path.exists(r["path"]):
            continue
        rows.append({"id": r["ids"][0], "op": r["op"], "signature_sha": r["sig"], "dtype": r["dtype"], "arch": r["arch"],
                     "target": "cuda", "shapes": r["shapes"], "code_sha": r["code_sha"], "speedup_vs_seed": r["best_speedup"],
                     "slot_us": r["best_slot_us"], "_from_problem": r["origin"], "_create_ts": r["create_ts"],
                     "_store_ts": r["store_ts"], "_in_store": r["in_store"]})
        codes[r["code_sha"]] = Path(r["path"])
    with tarfile.open(STAGE / "npool.tar.gz", "w:gz") as tw:
        add(tw, "npool/index.jsonl", ("\n".join(json.dumps(x) for x in rows) + "\n").encode())
        for sha, c in codes.items():
            add(tw, f"npool/code/{sha}.cu", c.read_bytes())
    print("npool", len(rows), "records", len(codes), "codes")
    out = STAGE / "mb_frozen.tar.gz"
    env = dict(os.environ, COPYFILE_DISABLE="1")
    subprocess.run(["tar", "--no-xattrs", "-czf", str(out), "--exclude=.git", "--exclude=__pycache__", "--exclude=*.pyc",
                    "--exclude=ModelBlaster/gpu_eval/runs", "--exclude=ModelBlaster/gpu_eval/ablations",
                    "--exclude=ModelBlaster/experiments", "--exclude=ModelBlaster/third_party/KernelBlaster/runs",
                    "-C", str(ROOT), "ModelBlaster"], check=True, env=env)
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    (STAGE / "mb_frozen.sha256").write_text(h + "\n")
    print(out.stat().st_size, h)


def node_files():
    sys.path.insert(0, str(L3))
    import torch_posthoc
    return {"reuse_node.py": (HERE / "reuse_node.py").read_bytes(), "torch_bench.py": torch_posthoc.BENCH.encode(),
            "worker.sh": WORKER.replace("{SMI}", SMI_Q).encode()}


def push(job):
    b = io.BytesIO()
    with tarfile.open(fileobj=b, mode="w:gz") as t:
        for k, v in node_files().items():
            add(t, f"bin/{k}", v)
    run(job, f"mkdir -p {R}", 60)
    rs.put(job, b.getvalue(), f"{R}/bin.tar.gz")
    for f in ("mb_frozen.tar.gz", "data.tar.gz", "npool.tar.gz"):
        rs.put(job, STAGE / f, f"{R}/{f}")
    want = (STAGE / "mb_frozen.sha256").read_text().strip()
    sh = f"""set -e
cd {R}
echo "{want}  mb_frozen.tar.gz" | sha256sum -c -
rm -rf code && mkdir -p code && tar xzf mb_frozen.tar.gz -C code && find code -name '._*' -delete && ln -sfn {R}/code/ModelBlaster {R}/code/modelblaster
rm -rf data npool && tar xzf data.tar.gz && tar xzf npool.tar.gz && tar xzf bin.tar.gz
python -c "import torch, modelblaster.pipeline.kb_reuse, modelblaster.pipeline.kb_export; print('import ok', torch.__version__)"
nvcc --version | tail -1
ls npool/code | wc -l; ls data
{SMI_Q}
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
"""
    print(run(job, sh, 900))


def run(job, script, timeout=600):
    rc, o, e = rs.run(job, ENV + script, timeout)
    return o + e[-600:]


def start(job, probs):
    pre = f"mkdir -p {R}/logs; cd {R}\n"
    log = f"{R}/logs/worker_{int(time.time())}.log"
    body = ENV + f"flock {R}/gpu.lock bash {R}/bin/worker.sh " + " ".join(shlex.quote(p) for p in probs) + f" > {log} 2>&1\n"
    assert str(job) in rs.JOBS
    cmd = f"nohup setsid srun --jobid={job} --overlap bash -c {shlex.quote(pre + body)} </dev/null >/dev/null 2>&1 & sleep 3; echo started {log}"
    rc, o, e = rs.farm.ssh(cmd, b"", timeout=120)
    print(o.decode(), e.decode()[-300:], rc)


def status(job):
    sh = f"""cd {R}; for f in $(ls -t logs/worker_*.log | head -1); do tail -6 $f; done
for d in runs/*; do [ -f $d/torch_3.txt ] && echo done $(basename $d); done | tr '\\n' ' '
echo; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"""
    print(job, run(job, sh, 120))


def fetch(job, probs, rd='runs'):
    import base64
    ps = " ".join(shlex.quote(p) for p in probs) if probs else "*"
    sh = f"""cd {R}/{rd}; tar czf - $(for d in {ps}; do [ -d $d ] && find $d -maxdepth 2 \\( -name 'ru.json' -o -name 'node.log' -o -name 'torch_*.txt' -o -name 'torch_*.err' -o -name 'smi_*.csv' -o -name 'reuse.json' \\) ; done) 2>/dev/null | base64 -w0"""
    rc, o, e = rs.run(job, ENV + sh, 600)
    raw = base64.b64decode(o.strip() or b"")
    if not raw:
        print("nothing", e[-300:])
        return
    dest = HERE / "data" / "runs"
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as t:
        t.extractall(dest)
    print("fetched", len(raw), "into", dest)


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:]
    if cmd == "prep":
        prep(rest)
    elif cmd == "push":
        push(rest[0])
    elif cmd == "start":
        start(rest[0], rest[1:])
    elif cmd == "status":
        status(rest[0])
    elif cmd == "fetch":
        fetch(rest[0], rest[1:])
    elif cmd == "clean":
        print(rs.run(rest[0], f"rm -rf {R}; ls /tmp/krisd")[1])
