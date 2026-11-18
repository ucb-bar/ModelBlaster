#!/usr/bin/env python3
"""Rerun the old Sakana (robust-kbench) KernelBench L1/L2 kernels on Computelab L40S nodes with END-TO-END timing.

Per task it runs the (indentation-fixed) run_kernel.py of the old checkout, i.e. the command of run_levels.sh,
    python3 run_kernel_fixed.py --task_dir tasks/kernelbench/<level>/<task> --cuda_code_path <old kernel.cu> --warmup_time 25 --rep_time 100 --no-profile
(`--no-profile` = no NCU / torch profiler / clang-tidy; correctness + CUDA-event timing of torch native, torch.compile and the CUDA kernel).
Nodes are reached only through farm.ssh (`srun --jobid=J --overlap bash -s`, script on stdin); everything on a node lives under /tmp/krisd/sakana.

  FARM_CONFIG=analysis/l12_sweep/config_l12.json python3 analysis/sakana_rerun/sakana_run.py stage JOB [JOB ...]
  FARM_CONFIG=analysis/l12_sweep/config_l12.json python3 analysis/sakana_rerun/sakana_run.py run JOB [JOB ...] [--only level_1/task_3 ...]
Results: analysis/sakana_rerun/results/<level>/<task>/run.log (+ result.json parsed from the log). Resume-safe (tasks with result.json are skipped)."""
import base64, io, json, os, re, shlex, sys, tarfile, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import farm

REPO = HERE / "robust-kbench"
KERN = HERE / "old_highlighted"
RES = HERE / "results"
B = "/tmp/krisd/sakana"
CUDA = "/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04"
WARM, REP, TIMEOUT = 25, 100, 600


def sh(job, body, inp=b"", t=600):
    rc, out, err = farm.ssh(f"srun --jobid={job} --overlap bash -c {shlex.quote(body)}", inp, timeout=t)
    return rc, out.decode(errors="replace"), err.decode(errors="replace")[-300:]


def tasks():
    out = []
    for lv in ("level_1", "level_2"):
        for d in sorted((KERN / lv).glob("task_*"), key=lambda p: int(p.name.split("_")[1])):
            if (d / "kernel.cu").exists() and (REPO / "tasks/kernelbench" / lv / d.name).is_dir():
                out.append(f"{lv}/{d.name}")
    return out


def tarball():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for p in REPO.rglob("*"):
            r = p.relative_to(REPO)
            if p.is_file() and ".git" not in r.parts:
                tf.add(p, arcname=f"robust-kbench/{r}")
        for t in tasks():
            tf.add(KERN / t / "kernel.cu", arcname=f"kernels/{t}/kernel.cu")
        tf.add(HERE / "run_kernel_fixed.py", arcname="run_kernel_fixed.py")
        tf.add(HERE / "shims", arcname="shims")
    return buf.getvalue()


def stage(job):
    data = base64.encodebytes(tarball()).decode()
    body = f'''set -e
mkdir -p {B} && cd {B} && rm -rf robust-kbench kernels run_kernel_fixed.py
base64 -d <<'__T__' | tar xzf -
{data}
__T__
. /tmp/krisd/venv/bin/activate
pip install -q ninja dotmap numpy 2>&1 | tail -2
SP=$(python -c "import site;print(site.getsitepackages()[0])"); mkdir -p $SP/pycuda; cp {B}/shims/pycuda/*.py $SP/pycuda/   # pycuda does not build here; device-info shim (see shims/)
python3 -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
export PATH={CUDA}/bin:$PATH
nvcc --version | tail -1; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
ls {B}/kernels/level_1 | wc -l; ls {B}/kernels/level_2 | wc -l'''
    return sh(job, "bash -s", body.encode(), 900)


ONE = '''set -u
LV=$1; TK=$2
D={B}/ext/$LV/$TK; mkdir -p $D {B}/out/$LV/$TK; rm -rf {B}/kernels/$LV/$TK/eval_results   # no stale harness results
export TORCH_EXTENSIONS_DIR=$D PATH={CUDA}/bin:$PATH CUDA_HOME={CUDA} PYTHONPATH={B}/robust-kbench TORCH_CUDA_ARCH_LIST=8.9
. /tmp/krisd/venv/bin/activate
cd {B}/robust-kbench
echo "@@HOST@@ $(hostname) $(nvidia-smi --query-gpu=name,uuid --format=csv,noheader | head -1)"
echo "@@OTHERS@@ $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\\n' ';')"
timeout 1800 python3 {B}/run_kernel_fixed.py --task_dir tasks/kernelbench/$LV/$TK --cuda_code_path {B}/kernels/$LV/$TK/kernel.cu \\
  --warmup_time {W} --rep_time {R} --timeout {T} --no-profile --ext_dir $D 2>&1
echo "@@RC@@ $?"
rm -rf $D
'''.replace("{B}", B).replace("{CUDA}", CUDA).replace("{W}", str(WARM)).replace("{R}", str(REP)).replace("{T}", str(TIMEOUT))


def parse(log):
    r = {}
    m = re.search(r"==> Torch native avg_mean_time \(ms\): ([0-9.eE+-]+)", log); r["torch_native_ms"] = float(m.group(1)) if m else None
    m = re.search(r"==> Torch compile avg_mean_time \(ms\): ([0-9.eE+-]+)", log); r["torch_compile_ms"] = float(m.group(1)) if m else None
    m = re.search(r"==> CUDA avg_mean_time \(ms\): ([0-9.eE+-]+)", log); r["cuda_ms"] = float(m.group(1)) if m else None
    m = re.search(r"==> Speedup CUDA over Torch native: ([0-9.eE+-]+)x", log); r["speedup_native"] = float(m.group(1)) if m else None
    m = re.search(r"==> Speedup CUDA over Torch compile: ([0-9.eE+-]+)x", log); r["speedup_compile"] = float(m.group(1)) if m else None
    r["correct"] = bool(re.search(r'"correct": true', log))
    m = re.search(r"@@HOST@@ (.*)", log); r["host"] = m.group(1).strip() if m else None
    m = re.search(r"@@OTHERS@@ (.*)", log); r["others_before"] = m.group(1).strip() if m else None
    m = re.search(r"@@RC@@ (\d+)", log); r["rc"] = int(m.group(1)) if m else None
    if r["cuda_ms"] is None:
        errs = [l.strip() for l in log.splitlines() if re.search(r"error|Error|Traceback|timed out|Timeout", l)]
        r["error"] = (errs[-1] if errs else log.strip().splitlines()[-1] if log.strip() else "no output")[:300]
    return r


def worker(job, items, lock):
    for t in items:
        lv, tk = t.split("/")
        d = RES / lv / tk
        if (d / "result.json").exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for attempt in range(3):          # transient ssh/srun drops (rc 255) are retried here; a drop leaves no result.json
            try:
                rc, out, err = sh(job, f"bash -s {lv} {tk}", ONE.encode(), 2100)
            except Exception as e:
                print(t, "ssh error", repr(e)[:120], flush=True); rc, out, err = 255, "", ""
            if rc == 0 or "@@RC@@" in out:
                break
            print(t, "srun failed rc", rc, err[-100:].strip(), f"(attempt {attempt + 1}/3)", flush=True)
            time.sleep(30 * (attempt + 1))
        if rc != 0 and "@@RC@@" not in out:
            continue
        (d / "run.log").write_text(out)
        r = parse(out)
        r.update(task=t, level=lv, job=job, wall_s=round(time.time() - t0, 1))
        (d / "result.json").write_text(json.dumps(r, indent=1))
        with lock:
            print(f"{job} {t}: correct={r['correct']} native={r['torch_native_ms']} cuda={r['cuda_ms']} x{r['speedup_native']} ({r['wall_s']} s)", flush=True)


def main():
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "stage":
        for j in args:
            print(j, sh(j, "true")[0], stage(int(j))[1:], flush=True)
        return
    only = []
    if "--only" in args:
        i = args.index("--only"); only = args[i + 1:]; args = args[:i]
    jobs = [int(j) for j in args]
    todo = [t for t in tasks() if (not only or t in only) and not (RES / t / "result.json").exists()]
    print(len(todo), "tasks on", len(jobs), "nodes", flush=True)
    lock = threading.Lock()
    th = [threading.Thread(target=worker, args=(j, todo[i::len(jobs)], lock)) for i, j in enumerate(jobs)]
    [t.start() for t in th]; [t.join() for t in th]


main()
