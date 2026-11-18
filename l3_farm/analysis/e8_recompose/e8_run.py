#!/usr/bin/env python3
"""E8 orchestrator (local). Every remote step is a script fed to `srun --jobid=J --overlap bash -s` through farm.ssh,
only on jobs listed in analysis/e8_data/my_jobs.json (nodes.py). Node files live under /tmp/krisd/e8 only.

  e8_run.py push JOB [P ...]          code (private ModelBlaster copy from state/mb.tar.gz, E8 scripts, shim, fixed
                                      kernels), bundles (recompose.bundle) and the two Claude solutions per problem
  e8_run.py rebuild JOB P [P ...]     node_side rebuild + buildso per problem -> e8_data/rebuild/<JOB>_<P>.json
  e8_run.py flowtime JOB P [P ...] [--tag T]  node_side flowtime (the flow's own driver, 3 requests) -> e8_data/flowtime/<JOB>_<P>.json
  e8_run.py bench JOB TAG ROUNDS P [P ...] [--procs K] [--k0 K0] [--impls a,b]
                                      e8_bench.py per problem, K fresh processes x ROUNDS interleaved rounds, nvidia-smi
                                      sampled every 1 s alongside -> e8_data/bench/<TAG>/<JOB>_<P>_<k>.json (+ _smi.csv)
  e8_run.py fixed JOB TAG             fixed_kernels (fma / copy / tiny200) + nvidia-smi -> e8_data/fixed/<TAG>_<JOB>.json
  e8_run.py smi JOB                   nvidia-smi -q (clocks, power, thermal, ECC, PCIe) -> e8_data/smi/<JOB>.txt
"""
import base64, io, json, shlex, sys, tarfile, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
L3 = HERE.parents[1]
sys.path.insert(0, str(L3)); sys.path.insert(0, str(HERE))
import farm  # noqa: E402
import nodes  # noqa: E402
import recompose  # noqa: E402

DATA = L3 / "analysis" / "e8_data"
R = "/tmp/krisd/e8"
N = "/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04"
ENV = f"""set -o pipefail
export PATH=/tmp/krisd/baseline_tools/py/bin:{N}/bin:/usr/bin:/bin CUDA_HOME={N} TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=16
export MB_CUPTI_LIB={N}/extras/CUPTI/lib64/libcupti.so.12
. /tmp/krisd/venv/bin/activate
export PYTHONPATH={R}/code TRITON_CACHE_DIR={R}/cache/triton TORCHINDUCTOR_CACHE_DIR={R}/cache/inductor TMPDIR={R}/cache/tmp
mkdir -p {R}/cache/triton {R}/cache/inductor {R}/cache/tmp {R}/cache/ext
"""
BASE = L3 / "baseline_cc" / "runs"


def solutions(p):
    out = {}
    for tag, sfx in (("freeform", "__T2"), ("cudaonly", "__CUDA")):
        c = sorted(BASE.glob(f"{p}__*{sfx}"))
        if c and (c[-1] / "workspace_final" / "solution.py").exists():
            out[tag] = c[-1] / "workspace_final" / "solution.py"
    return out


def run(job, script, timeout=3600):
    return nodes.run(job, ENV + script, timeout)


def put(job, data, remote):
    if str(job) not in nodes.mine():
        raise SystemExit(f"job {job} is not mine")
    rc, out, err = farm.put(job, data, remote)
    if rc:
        raise RuntimeError(f"put {remote}: {err.decode()[-300:]}")


def tarball(files):
    """{arcname: path or bytes} -> tar.gz bytes"""
    b = io.BytesIO()
    with tarfile.open(fileobj=b, mode="w:gz") as t:
        for arc, src in files.items():
            data = src if isinstance(src, bytes) else Path(src).read_bytes()
            ti = tarfile.TarInfo(arc); ti.size = len(data); ti.mtime = time.time()
            t.addfile(ti, io.BytesIO(data))
    return b.getvalue()


def push(job, probs):
    mb = L3 / "state" / "mb.tar.gz"
    stamp = recompose.sha1(mb.read_bytes())[:12]
    rc, out, err = run(job, f"cat {R}/code/.stamp 2>/dev/null || echo none\n", 120)
    if out.strip().splitlines()[-1:] != [stamp]:
        put(job, mb, f"{R}/mb.tar.gz")
        rc, out, err = run(job, f"rm -rf {R}/code && mkdir -p {R}/code && tar xzf {R}/mb.tar.gz -C {R}/code && "
                                f"ln -sfn {R}/code/ModelBlaster {R}/code/modelblaster && echo {stamp} > {R}/code/.stamp && echo ok\n", 600)
        print("code", out.strip()[-200:], err[-300:])
    files = {f"bin/{f}": HERE / f for f in ("node_side.py", "e8_bench.py", "e8_shim.cu", "fixed_kernels.cu")}
    with tempfile.TemporaryDirectory() as td:
        for p in probs:
            bd = recompose.bundle(p, Path(td) / p)
            for f in bd.rglob("*"):
                if f.is_file():
                    files[f"bundles/{p}/{f.relative_to(bd)}"] = f.read_bytes()
            for tag, s in solutions(p).items():
                files[f"sol/{p}/{tag}/solution.py"] = s
        put(job, tarball(files), f"{R}/push.tar.gz")
    rc, out, err = run(job, f"""cd {R} && tar xzf push.tar.gz && cd bin &&
[ e8_shim.so -nt e8_shim.cu ] || nvcc -O3 -std=c++17 -arch=sm_89 --default-stream per-thread -shared -Xcompiler -fPIC -o e8_shim.so e8_shim.cu &&
[ fixed_kernels -nt fixed_kernels.cu ] || nvcc -O3 -std=c++17 -arch=sm_89 -o fixed_kernels fixed_kernels.cu && ls {R}/bundles {R}/sol && echo PUSHED
""", 900)
    print(job, out[-500:], err[-500:])


def _grab(out, tag):
    line = next((l for l in out.splitlines() if l.startswith(tag)), None)
    return json.loads(line[len(tag):]) if line else None


def rebuild(job, probs):
    (DATA / "rebuild").mkdir(parents=True, exist_ok=True)
    for p in probs:
        t0 = time.time()
        rc, out, err = run(job, f"cd {R}/bin && python node_side.py rebuild {R}/bundles/{p} {R}/work/{p} && "
                                f"python node_side.py buildso {R}/work/{p}\n", 5400)
        d = _grab(out, "@@REBUILD@@") or {"ok": False, "log": (out + err)[-3000:]}
        so = next((l[6:] for l in out.splitlines() if l.startswith("@@SO@@")), None)
        d.update(so=so, job=job, wall_s=round(time.time() - t0, 1))
        if not so or so.startswith("FAILED"):
            d["so_log"] = (so or "") + err[-1500:]
        (DATA / "rebuild" / f"{job}_{p}.json").write_text(json.dumps(d, indent=1))
        print(p, "ok" if d.get("ok") else "FAIL", {k: v for k, v in (d.get("checks") or {}).items() if k.endswith("same")},
              d.get("t"), "so", (so or "")[-60:], flush=True)


def flowtime(job, probs, reps=3, tag=""):
    (DATA / "flowtime").mkdir(parents=True, exist_ok=True)
    for p in probs:
        rc, out, err = run(job, f"nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l; cd {R}/bin && "
                                f"python node_side.py flowtime {R}/work/{p} --reps {reps}\n", 5400)
        d = _grab(out, "@@FLOWTIME@@") or {"passed": False, "log": (out + err)[-3000:]}
        d.update(job=job, host=nodes.mine()[str(job)].get("host"), when=time.strftime("%Y-%m-%d %H:%M:%S"),
                 others_before=out.splitlines()[0].strip() if out else None)
        rec = recompose.artifacts(p)
        d["recorded_model_e2e_us"] = rec.get("model_e2e_us")
        d["recorded_runs"] = rec.get("model_e2e_runs")
        (DATA / "flowtime" / f"{job}_{p}{('_' + tag) if tag else ''}.json").write_text(json.dumps(d, indent=1, default=str))
        e = d.get("e2e_us")
        print(p, "passed" if d.get("passed") else "FAIL", "e2e", e, "recorded", rec.get("model_e2e_us"),
              f"{(e / rec['model_e2e_us'] - 1) * 100:+.2f}%" if e and rec.get("model_e2e_us") else "", d.get("e2e_runs"), flush=True)


def bench(job, tag, rounds, probs, procs=1, impls=None, k0=0):
    out_dir = DATA / "bench" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in probs:
        sol = solutions(p)
        for k in range(k0, k0 + procs):
            ic = {"torch": {"kind": "torch"},
                  "flow": {"kind": "flow", "so": f"{R}/work/{p}/build_so/flow.so", "shim": f"{R}/bin/e8_shim.so",
                           "manifest": f"{R}/work/{p}/problem/manifest.json", "data": f"{R}/work/{p}/problem/data.bin"}}
            for t in sol:
                ic[t] = {"kind": "solution", "path": f"{R}/sol/{p}/{t}/solution.py", "extdir": f"{R}/cache/ext/{p}_{t}"}
            if impls:
                ic = {n: v for n, v in ic.items() if n in impls}
            cfg = {"name": p, "problem": f"{R}/bundles/{p}/problem.py", "impls": ic, "rounds": rounds, "warm": 10,
                   "budget_ms": 200.0, "nmin": 20, "nmax": 200, "shuffle_seed": 1000 * k + 7, "proc": k, "tag": tag}
            cj = base64.b64encode(json.dumps(cfg).encode()).decode()
            smi = f"{R}/smi_{tag}_{p}_{k}.csv"
            t0 = time.time()
            rc, out, err = run(job, f"""cd {R}/bin && echo {cj} | base64 -d > cfg_{tag}_{p}_{k}.json
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active,utilization.gpu --format=csv,noheader -lms 1000 > {smi} 2>/dev/null &
SP=$!
python e8_bench.py cfg_{tag}_{p}_{k}.json 2> err_{tag}_{p}_{k}.log | grep '^@@E8@@'
kill $SP; sleep 0.2; echo @@SMI@@; cat {smi}; echo @@ERR@@; tail -c 2500 err_{tag}_{p}_{k}.log
""", 7200)
            d = _grab(out, "@@E8@@") or {"failed": True, "log": (out + err)[-4000:]}
            smi_txt = out.split("@@SMI@@", 1)[1].split("@@ERR@@")[0] if "@@SMI@@" in out else ""
            d.update(job=job, wall_s=round(time.time() - t0, 1), stderr_tail=out.split("@@ERR@@", 1)[1][-2500:] if "@@ERR@@" in out else err[-2500:])
            (out_dir / f"{job}_{p}_{k}.json").write_text(json.dumps(d))
            (out_dir / f"{job}_{p}_{k}_smi.csv").write_text(smi_txt)
            s = {n: round(v["summary"]["median"], 1) for n, v in d.get("impls", {}).items() if "summary" in v}
            ok = {n: v.get("check", {}).get("ok") for n, v in d.get("impls", {}).items()}
            print(tag, job, p, k, "wall", d["wall_s"], s, "checks", ok, "errors", list((d.get("errors") or {}).keys()),
                  "others", d.get("others_before_timing"), flush=True)


def fixed(job, tag):
    (DATA / "fixed").mkdir(parents=True, exist_ok=True)
    smi = f"{R}/smi_fixed_{tag}.csv"
    rc, out, err = run(job, f"""cd {R}/bin
nvidia-smi --query-gpu=uuid,serial,pci.bus_id,vbios_version,driver_version,clocks.max.sm,clocks.max.mem,power.limit,temperature.gpu,power.draw,clocks.sm,clocks.mem --format=csv,noheader; nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active,utilization.gpu --format=csv,noheader -lms 500 > {smi} 2>/dev/null &
SP=$!
./fixed_kernels 10 200
kill $SP; sleep 0.2; echo @@SMI@@; cat {smi}
""", 1800)
    d = _grab(out, "@@FIXED@@") or {"failed": True, "log": (out + err)[-2000:]}
    lines = out.splitlines()
    d.update(job=job, host=nodes.mine()[str(job)].get("host"), gpu_info=lines[0] if lines else None,
             others=lines[1] if len(lines) > 1 else None, when=time.strftime("%Y-%m-%d %H:%M:%S"),
             smi=out.split("@@SMI@@", 1)[1] if "@@SMI@@" in out else "")
    (DATA / "fixed" / f"{tag}_{job}.json").write_text(json.dumps(d, indent=1))
    ks = d.get("kernels", {})
    print(tag, job, d.get("host"), {k: sorted(b["med"] for b in v)[len(v) // 2] for k, v in ks.items()}, d.get("gpu_info"))


def smiq(job):
    (DATA / "smi").mkdir(parents=True, exist_ok=True)
    rc, out, err = run(job, "hostname; nvidia-smi -q -d CLOCK,POWER,TEMPERATURE,PERFORMANCE,ECC,MEMORY; nvidia-smi -q | grep -i -A12 'GPU Link Info'\nnvidia-smi --query-gpu=uuid,serial,vbios_version,inforom.img --format=csv\n", 120)
    (DATA / "smi" / f"{job}.txt").write_text(out)
    print(out[:300])


def main():
    a = sys.argv[1:]
    cmd, job = a[0], int(a[1])
    if cmd == "push":
        push(job, a[2:])
    elif cmd == "rebuild":
        rebuild(job, a[2:])
    elif cmd == "flowtime":
        tag = a[a.index("--tag") + 1] if "--tag" in a else ""
        flowtime(job, [x for x in a[2:] if x not in ("--tag", tag)], tag=tag)
    elif cmd == "bench":
        procs, impls, rest, k0 = 1, None, [], 0
        it = iter(a[4:])
        for x in it:
            if x == "--procs":
                procs = int(next(it))
            elif x == "--impls":
                impls = next(it).split(",")
            elif x == "--k0":
                k0 = int(next(it))
            else:
                rest.append(x)
        bench(job, a[2], int(a[3]), rest, procs, impls, k0)
    elif cmd == "fixed":
        fixed(job, a[2])
    elif cmd == "smi":
        smiq(job)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
