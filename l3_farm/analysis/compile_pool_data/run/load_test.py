#!/usr/bin/env python3
"""Orchestrates one load-test run on the driver: load_test.py ARM N THINK LABEL ; env LT_HOSTS (remote hosts, also sampled)."""
import json, os, random, shutil, subprocess, sys, threading, time
from pathlib import Path

arm, N, think, label = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
ROOT = Path("/tmp/krisd/cpool"); LT = ROOT / "lt" / label
HOSTS = [h for h in os.environ.get("LT_HOSTS", "").split(",") if h]
index = json.loads((ROOT / "corpus/index.json").read_text())
names = sorted(index); random.Random(1).shuffle(names)
probs = names[:N]
shutil.rmtree(LT, ignore_errors=True); LT.mkdir(parents=True)
shared = LT / "shared_cache"
env = dict(os.environ, LT_DIR=str(LT), LT_HOSTS=",".join(HOSTS), PYTHONUNBUFFERED="1")
if arm in ("shared", "remote"):
    env["KERNELBLASTER_BUILD_CACHE"] = str(shared)
else:
    env.pop("KERNELBLASTER_BUILD_CACHE", None)
env["PATH"] = "/tmp/krisd/cpool/sshbin:" + env["PATH"]
env["KERNELBLASTER_SSH_CONTROL_DIR"] = str(ROOT / "mux")
Path(ROOT / "mux").mkdir(exist_ok=True)
if arm == "remote":      # cold host caches
    for h in HOSTS:
        subprocess.run(["/tmp/krisd/cpool/sshbin/ssh", h, "rm -rf /tmp/krisd/cpool/rb"], timeout=60)

def cpu_line(raw):
    v = [int(x) for x in raw.split()[1:9]]
    return sum(v) - v[3] - v[4], sum(v)          # busy (excl idle, iowait), total

samples = {"driver": []}
stop = threading.Event()
def drv():
    while not stop.is_set():
        samples["driver"].append((time.time(), *cpu_line(open("/proc/stat").readline()), float(open("/proc/loadavg").read().split()[0])))
        time.sleep(1)
procs = []
NC = {}
def hostsample(h):
    samples[h] = []
    NC[h] = int(subprocess.run(["/tmp/krisd/cpool/sshbin/ssh", h, "nproc"], capture_output=True, text=True, timeout=60).stdout.split()[0])
    p = subprocess.Popen(["/tmp/krisd/cpool/sshbin/ssh", h,
                          "bash -c 'while :; do echo $(date +%s.%N) $(head -1 /proc/stat); sleep 1; done'"],
                         stdout=subprocess.PIPE, text=True)
    procs.append(p)
    def rd():
        for ln in p.stdout:
            f = ln.split()
            try:
                samples[h].append((float(f[0]), *cpu_line(" ".join(f[1:])), 0.0))
            except Exception:
                pass
    threading.Thread(target=rd, daemon=True).start()
threading.Thread(target=drv, daemon=True).start()
for h in HOSTS:
    hostsample(h)
time.sleep(3)
t0 = time.time()
ps = [subprocess.Popen([sys.executable, str(ROOT / "load_ctrl.py"), arm, p, str(LT / f"{p}.json"), str(think)],
                       env=env, stdout=open(LT / f"{p}.log", "w"), stderr=subprocess.STDOUT) for p in probs]
for p in ps:
    p.wait()
wall = time.time() - t0
stop.set()
for p in procs:
    p.terminate()
res = [json.loads((LT / f"{p}.json").read_text()) for p in probs if (LT / f"{p}.json").exists()]
def cores(h):
    s = [x for x in samples[h] if t0 <= x[0] <= t0 + wall]
    if len(s) < 2: return None
    dcpu = [(b[1] - a[1]) / max(b[2] - a[2], 1) for a, b in zip(s, s[1:])]
    nc = os.cpu_count() if h == "driver" else NC[h]
    return dict(avg_cores=round(sum(dcpu) / len(dcpu) * nc, 2), peak_cores=round(max(dcpu) * nc, 2), busy_core_s=round(sum(dcpu) * nc, 1))
allat = sorted(l for r in res for l in r["lat"])
pct = lambda q: allat[min(len(allat) - 1, int(q * len(allat)))] if allat else None
rb = [r.get("rb") for r in res if r.get("rb")]
cs = [r.get("cache") for r in res if r.get("cache")]
summary = dict(label=label, arm=arm, N=N, think=think, problems=probs, wall=round(wall, 1),
               sets=sum(r["sets"] for r in res), failed_sets=sum(r["sets"] - r["ok"] for r in res), units=sum(r["units"] for r in res),
               local_nvcc_calls=sum(r["nvcc_calls"] for r in res), local_nvcc_compile=sum(r["nvcc_compile_calls"] for r in res),
               local_nvcc_s=round(sum(r["nvcc_s"] for r in res), 1), sem_wait_s=round(sum(r["sem_wait_s"] for r in res), 1),
               proc_cpu_s=round(sum(r["cpu_self"] + r["cpu_children"] for r in res), 1),
               remote_builds=sum(x["remote"] for x in rb), remote_cached=sum(x["remote_cached"] for x in rb),
               remote_fallback_local=sum(x["local"] for x in rb), remote_infra=sum(x["infra"] for x in rb), remote_full=sum(x["full"] for x in rb),
               remote_s=round(sum(x["remote_s"] for x in rb), 1), remote_queue_s=round(sum(x["queue_s"] for x in rb), 1),
               transfer_s=round(sum(x["transfer_s"] for x in rb), 1),
               set_lat_p50=pct(.5), set_lat_p90=pct(.9), set_lat_max=allat[-1] if allat else None,
               cpu={h: cores(h) for h in ["driver"] + HOSTS}, host_nproc=NC, driver_nproc=os.cpu_count())
summary["compiles_total"] = summary["local_nvcc_compile"] + summary["remote_builds"] - summary["remote_cached"]
summary["builds_per_min"] = round(summary["compiles_total"] / wall * 60, 1)
(LT / "summary.json").write_text(json.dumps(summary, indent=1))
(LT / "cpu_samples.json").write_text(json.dumps({k: [(round(a, 2), b, c) for a, b, c, _ in v] for k, v in samples.items()}))
print(json.dumps(summary))
