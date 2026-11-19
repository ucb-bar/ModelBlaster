"""Host build of the XPU-RT schedule walker (hosted POSIX, no Zephyr, no board).

Runs the real pipeline pieces end to end on a synthetic network:

    schedule.json --ingest_xpurt_schedule--> dispatch table (.c/.h)
                  --generate_xpurt_main-----> xpurt_main.c   (kind and hart modes)
    xpurt_main.c + table + fixture kernels + modelblaster_pool + mb_posix.h
                  --cc--> executable --run--> marker output

The fixture "model" has two ops per instance; op1 depends on op0 and each op
sleeps a fixed time, so the run shows (a) dependencies are honoured, (b) the
default kind-granular walker serialises everything on one worker, and (c) the
per-hart walker runs independent harts concurrently.

Needs only a C compiler. Works on Linux (the target) and on macOS (build-and-
test host: no thread affinity there, semaphores fall back to mutex+condvar).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")

pytestmark = pytest.mark.skipif(CC is None, reason="no C compiler on PATH")

N_JOBS = 4          # instances, one per hart
OP_US = 40_000      # sleep per op


def _write(p: Path, text: str) -> Path:
    p.write_text(text)
    return p


def _model_header() -> str:
    return """\
#ifndef SYNTH_MODEL_H
#define SYNTH_MODEL_H
#include <stdint.h>
#define MODEL_SYNTH_OUTPUT_SIZE 2
#define MODEL_SYNTH_OP_COUNT 2
typedef float model_synth_output_t;
typedef struct {
    const float *input;
    model_synth_output_t *output;
    void *pool;
    int buf_set;
} model_synth_state_t;
extern const int model_synth_n_buf_sets;
typedef void (*model_synth_dispatch_fn)(model_synth_state_t *);
typedef struct {
    int dispatch_id;
    const char *name;
    const char *op;
    const char *shape;
    unsigned long cycles;
} model_synth_op_record_t;
#endif
"""


def _kernels(backend: str) -> str:
    bs = backend
    BS = backend.upper()
    return f"""\
#include <stdatomic.h>
#include <stdio.h>
#include <unistd.h>
#include "synth_model.h"

static atomic_int done0, done1, running, maxpar, violations;

static void enter(void)
{{
    int r = atomic_fetch_add(&running, 1) + 1;
    int m = atomic_load(&maxpar);
    while (r > m && !atomic_compare_exchange_weak(&maxpar, &m, r)) {{}}
}}
static void leave(void) {{ atomic_fetch_sub(&running, 1); }}

static void op0(model_synth_state_t *s)
{{
    (void)s;
    enter();
    usleep({OP_US});
    atomic_fetch_add(&done0, 1);
    leave();
}}
static void op1(model_synth_state_t *s)
{{
    /* every executed op1 needs its own op0 already finished */
    if (atomic_load(&done0) <= atomic_load(&done1)) atomic_fetch_add(&violations, 1);
    enter();
    usleep({OP_US});
    atomic_fetch_add(&done1, 1);
    leave();
    s->output[0] = (float)atomic_load(&done0);
    s->output[1] = (float)atomic_load(&done1);
}}

const model_synth_dispatch_fn MODEL_SYNTH_DISPATCH_FNS_{BS}[MODEL_SYNTH_OP_COUNT] = {{ op0, op1 }};
void model_synth_reset_profile_{bs}(void) {{}}
/* buffers.c's count; weak so two backends' fixture files can both carry it */
__attribute__((weak)) const int model_synth_n_buf_sets = 1;
const model_synth_op_record_t *model_synth_profile_records_{bs}(int *count)
{{
    *count = 0;
    return NULL;
}}

__attribute__((destructor)) static void report(void)
{{
    printf("SYNTH_RESULT done0=%d done1=%d violations=%d maxpar=%d\\n",
           atomic_load(&done0), atomic_load(&done1),
           atomic_load(&violations), atomic_load(&maxpar));
}}
"""


def _schedule() -> dict:
    d: dict = {}
    for j in range(N_JOBS):
        for i in range(2):
            key = f"synth{j}_dispatch_{i}"
            d[key] = {
                "id": i, "ordinal": i + 1, "total": 2,
                "dependencies": [f"synth{j}_dispatch_0"] if i == 1 else [],
                "hardware_target": f"CPU_P#{j}",
                "start_time": 0.0 if i == 0 else 40.0,
                "duration": 40.0,
                "job_name": f"synth{j}",
                "module_name": "synth",
            }
    return {"dot_file": "synth.dot", "dispatches": d}


def _registry() -> dict:
    return {
        "system": "host4",
        "description": "four independent rvv harts",
        "default_dispatch_target": "any",
        "cores": [
            {"name": f"rvv{h}", "kind": "rvv", "harts": [h],
             "capabilities": ["synth_a", "synth_b"]}
            for h in range(N_JOBS)
        ],
    }


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    """Ingest + generate (both modes) + compile. Returns {mode: exe}."""
    w = tmp_path_factory.mktemp("walker")
    pkg = w / "pkg"
    pkg.mkdir()
    os.symlink(ROOT, pkg / "modelblaster")
    env = dict(os.environ, PYTHONPATH=str(pkg))

    _write(w / "schedule.json", json.dumps(_schedule()))
    _write(w / "registry.json", json.dumps(_registry()))
    _write(w / "graph.json", json.dumps({"ops": [
        {"dispatch_id": 0, "op": "synth_a", "name": "a"},
        {"dispatch_id": 1, "op": "synth_b", "name": "b"}]}))
    _write(w / "synth_model.h", _model_header())
    _write(w / "synth_test_io.h",
           "static const float model_synth_test_input[1] = { 0.0f };\n")
    _write(w / "kernels_rvv.c", _kernels("rvv"))

    def run(*a):
        r = subprocess.run([sys.executable, "-m", *a], cwd=w, env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, f"{a[0]} failed:\n{r.stdout}\n{r.stderr}"

    run("modelblaster.pipeline.ingest_xpurt_schedule",
        "--schedule", "schedule.json", "--registry", "registry.json",
        "--ir", "synth:graph.json", "--out", str(w / "demo_table.c"),
        "--name", "demo")

    exes = {}
    for mode in ("kind", "hart"):
        run("modelblaster.pipeline.generate_xpurt_main",
            "--schedule", "schedule.json", "--out", str(w / f"main_{mode}.c"),
            "--name", "demo", "--dispatch-table-header", "demo_table.h",
            "--core-kinds", "rvv", "--worker-granularity", mode)
        exe = w / f"walker_{mode}"
        cmd = [CC, "-O1", "-pthread", "-DMB_PLATFORM_POSIX", "-D_GNU_SOURCE",
               f"-I{ROOT / 'runtime'}", f"-I{ROOT / 'runtime' / 'modelblaster_pool'}",
               f"-I{w}", "-o", str(exe), f"main_{mode}.c", "demo_table.c",
               "kernels_rvv.c",
               str(ROOT / "runtime" / "modelblaster_pool" / "modelblaster_pool.c"),
               str(ROOT / "runtime" / "mb_placement.c")]
        r = subprocess.run(cmd, cwd=w, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]
        # Untimed warm-up: the first exec of a fresh binary can cost ~0.2 s
        # on some hosts (code-signing scan), which would swamp the timing.
        subprocess.run([str(exe)], capture_output=True, timeout=60)
        exes[mode] = exe
    return exes


#: These tests count kernel calls and time the run, so they start cold: the
#: walker's warm start (on by default on Linux) adds one untimed frame of
#: instance 0, which test_warm_start_runs_instance_zero_once_more checks.
COLD = dict(os.environ, MB_XPURT_WARM="0")


def _run(exe: Path, env=None) -> tuple[str, float]:
    t0 = time.monotonic()
    r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60,
                       env=env or COLD)
    dt = time.monotonic() - t0
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r.stdout, dt


def _result(out: str) -> dict:
    m = re.search(r"SYNTH_RESULT done0=(\d+) done1=(\d+) violations=(\d+) maxpar=(\d+)", out)
    assert m, out[-1500:]
    return dict(zip(("done0", "done1", "violations", "maxpar"), map(int, m.groups())))


def test_kind_mode_is_serial_and_correct(build):
    out, dt = _run(build["kind"])
    r = _result(out)
    assert r["done0"] == N_JOBS and r["done1"] == N_JOBS
    assert r["violations"] == 0
    assert r["maxpar"] == 1, "kind mode has one worker per kind: no concurrency"
    assert "worker[0] kind=rvv" in out and "worker[1]" not in out
    assert "=== MODELBLASTER_OUTPUT_BEGIN [synth] ===" in out
    assert dt >= 2 * N_JOBS * OP_US / 1e6 * 0.9


def test_warm_start_runs_instance_zero_once_more(build):
    """Default on Linux: one untimed frame of instance 0 before t=0, in
    dependency order, then the schedule itself."""
    for mode in ("kind", "hart"):
        out, _ = _run(build[mode], env=dict(os.environ))
        r = _result(out)
        assert r["done0"] == N_JOBS + 1 and r["done1"] == N_JOBS + 1, (mode, r)
        assert r["violations"] == 0, (mode, r)
        assert "xpurt: warm start:" in out, mode
    out, _ = _run(build["hart"])
    assert "xpurt: cold start" in out


def test_hart_mode_runs_harts_concurrently_and_correct(build):
    out, dt = _run(build["hart"])
    r = _result(out)
    assert r["done0"] == N_JOBS and r["done1"] == N_JOBS
    assert r["violations"] == 0, "op1 ran before its op0: dependency broken"
    assert r["maxpar"] >= 2, "independent harts should overlap"
    harts = sorted(int(h) for h in re.findall(r"pinned_hart=(\d+)", out))
    assert harts == list(range(N_JOBS)), out[-600:]
    assert dt < 2 * N_JOBS * OP_US / 1e6 * 0.8, f"no speedup over serial: {dt:.3f}s"


def test_default_output_unchanged_by_hart_option():
    """The kind-granular emission must be the original walker, byte for byte:
    only the opt-in hart mode may change the generated C."""
    sys.path.insert(0, str(ROOT.parent))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gxm", ROOT / "pipeline" / "generate_xpurt_main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    a = dict(networks=["n"], schedule_name="s", dispatch_table_header="s.h",
             core_kinds=["rvv"], backends=["rvv"], pool_sizes=[0],
             n_instances={"n": 1})
    base = mod._emit(**a)
    assert mod._emit(**a, per_hart_workers=False) == base
    assert mod._emit(**a, per_hart_workers=True) != base
    assert "XPURT_MAX_WORKERS" not in base


def test_a_core_kind_without_a_backend_stops_the_run(tmp_path):
    """--core-kinds rvv with --backends rvv_x60,scalar does not pair (the lists
    differ in length), so no backend runs kind rvv: the walker must stop with a
    non-zero exit rather than skip every dispatch and exit 0."""
    w = tmp_path
    pkg = w / "pkg"
    pkg.mkdir()
    os.symlink(ROOT, pkg / "modelblaster")
    env = dict(os.environ, PYTHONPATH=str(pkg))
    _write(w / "schedule.json", json.dumps(_schedule()))
    _write(w / "registry.json", json.dumps(_registry()))
    _write(w / "graph.json", json.dumps({"ops": [
        {"dispatch_id": 0, "op": "synth_a", "name": "a"},
        {"dispatch_id": 1, "op": "synth_b", "name": "b"}]}))
    _write(w / "synth_model.h", _model_header())
    _write(w / "synth_test_io.h",
           "static const float model_synth_test_input[1] = { 0.0f };\n")
    for bs in ("rvv_x60", "scalar"):
        _write(w / f"kernels_{bs}.c", _kernels(bs))
    for a in (("modelblaster.pipeline.ingest_xpurt_schedule",
               "--schedule", "schedule.json", "--registry", "registry.json",
               "--ir", "synth:graph.json", "--out", str(w / "demo_table.c"),
               "--name", "demo"),
              ("modelblaster.pipeline.generate_xpurt_main",
               "--schedule", "schedule.json", "--out", str(w / "main.c"),
               "--name", "demo", "--dispatch-table-header", "demo_table.h",
               "--core-kinds", "rvv", "--backends", "rvv_x60,scalar")):
        r = subprocess.run([sys.executable, "-m", *a], cwd=w, env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, f"{a[0]} failed:\n{r.stdout}\n{r.stderr}"
    exe = w / "walker"
    cmd = [CC, "-O1", "-pthread", "-DMB_PLATFORM_POSIX", "-D_GNU_SOURCE",
           f"-I{ROOT / 'runtime'}", f"-I{ROOT / 'runtime' / 'modelblaster_pool'}",
           f"-I{w}", "-o", str(exe), "main.c", "demo_table.c",
           "kernels_rvv_x60.c", "kernels_scalar.c",
           str(ROOT / "runtime" / "modelblaster_pool" / "modelblaster_pool.c"),
               str(ROOT / "runtime" / "mb_placement.c")]
    r = subprocess.run(cmd, cwd=w, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 2, r.stdout[-2000:]
    assert "asks for impl 'rvv'" in r.stdout


def test_hart_mode_worker_on_an_ai_cpu_registers_its_own_tid(tmp_path):
    """Simulate a K3 A100 CPU on the host: build the hart-mode walker with
    MODELBLASTER_AI_CPUS=1 and point the registration at a file. The worker on
    hart 1 must write its own TID there before pinning, and the run must still
    complete correctly."""
    if (os.cpu_count() or 1) < N_JOBS or sys.platform != "linux":
        pytest.skip("needs Linux and one CPU per hart in the fixture")
    w = tmp_path
    pkg = w / "pkg"
    pkg.mkdir()
    os.symlink(ROOT, pkg / "modelblaster")
    env = dict(os.environ, PYTHONPATH=str(pkg))
    _write(w / "schedule.json", json.dumps(_schedule()))
    _write(w / "registry.json", json.dumps(_registry()))
    _write(w / "graph.json", json.dumps({"ops": [
        {"dispatch_id": 0, "op": "synth_a", "name": "a"},
        {"dispatch_id": 1, "op": "synth_b", "name": "b"}]}))
    _write(w / "synth_model.h", _model_header())
    _write(w / "synth_test_io.h",
           "static const float model_synth_test_input[1] = { 0.0f };\n")
    _write(w / "kernels_rvv.c", _kernels("rvv"))
    for a in (("modelblaster.pipeline.ingest_xpurt_schedule",
               "--schedule", "schedule.json", "--registry", "registry.json",
               "--ir", "synth:graph.json", "--out", str(w / "demo_table.c"),
               "--name", "demo"),
              ("modelblaster.pipeline.generate_xpurt_main",
               "--schedule", "schedule.json", "--out", str(w / "main.c"),
               "--name", "demo", "--dispatch-table-header", "demo_table.h",
               "--core-kinds", "rvv", "--worker-granularity", "hart")):
        r = subprocess.run([sys.executable, "-m", *a], cwd=w, env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, f"{a[0]} failed:\n{r.stdout}\n{r.stderr}"
    exe = w / "walker"
    cmd = [CC, "-O1", "-pthread", "-DMB_PLATFORM_POSIX", "-D_GNU_SOURCE",
           '-DMODELBLASTER_AI_CPUS="1"',
           f"-I{ROOT / 'runtime'}", f"-I{ROOT / 'runtime' / 'modelblaster_pool'}",
           f"-I{w}", "-o", str(exe), "main.c", "demo_table.c", "kernels_rvv.c",
           str(ROOT / "runtime" / "modelblaster_pool" / "modelblaster_pool.c"),
           str(ROOT / "runtime" / "mb_placement.c")]
    r = subprocess.run(cmd, cwd=w, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    reg = w / "set_ai_thread"
    reg.write_text("")
    r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60,
                       env=dict(COLD, MODELBLASTER_AI_THREAD_PATH=str(reg)))
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    res = _result(r.stdout)
    assert res["done0"] == N_JOBS and res["done1"] == N_JOBS and res["violations"] == 0
    tids = [int(t) for t in reg.read_text().split()]
    assert len(tids) == 1, f"exactly the hart-1 worker registers: {tids}"
    assert tids[0] != os.getpid(), "a worker thread registers, never the process"


def _impl_kernels(backend: str) -> str:
    """Fixture backend that reports which backend ran each dispatch."""
    BS = backend.upper()
    return f"""\
#include <stdio.h>
#include "synth_model.h"
static void op0(model_synth_state_t *s) {{ (void)s; printf("RAN {backend} 0\\n"); fflush(stdout); }}
static void op1(model_synth_state_t *s) {{ (void)s; printf("RAN {backend} 1\\n"); fflush(stdout); }}
const model_synth_dispatch_fn MODEL_SYNTH_DISPATCH_FNS_{BS}[MODEL_SYNTH_OP_COUNT] = {{ op0, op1 }};
void model_synth_reset_profile_{backend}(void) {{}}
__attribute__((weak)) const int model_synth_n_buf_sets = 1;
const model_synth_op_record_t *model_synth_profile_records_{backend}(int *count)
{{ *count = 0; return NULL; }}
"""


def _build_impl_walker(w: Path, impls: dict[int, str], impl_backends: str) -> Path:
    pkg = w / "pkg"
    pkg.mkdir()
    os.symlink(ROOT, pkg / "modelblaster")
    env = dict(os.environ, PYTHONPATH=str(pkg))
    sched = _schedule()
    for d in sched["dispatches"].values():
        d["impl"] = impls[d["id"]]
    _write(w / "schedule.json", json.dumps(sched))
    _write(w / "registry.json", json.dumps(_registry()))
    _write(w / "graph.json", json.dumps({"ops": [
        {"dispatch_id": 0, "op": "synth_a", "name": "a"},
        {"dispatch_id": 1, "op": "synth_b", "name": "b"}]}))
    _write(w / "synth_model.h", _model_header())
    _write(w / "synth_test_io.h",
           "static const float model_synth_test_input[1] = { 0.0f };\n")
    for bs in ("rvv_vec", "rvv_mac"):
        _write(w / f"kernels_{bs}.c", _impl_kernels(bs))
    for a in (("modelblaster.pipeline.ingest_xpurt_schedule",
               "--schedule", "schedule.json", "--registry", "registry.json",
               "--ir", "synth:graph.json", "--out", str(w / "demo_table.c"),
               "--name", "demo"),
              ("modelblaster.pipeline.generate_xpurt_main",
               "--schedule", "schedule.json", "--out", str(w / "main.c"),
               "--name", "demo", "--dispatch-table-header", "demo_table.h",
               "--core-kinds", "rvv", "--backends", "rvv_vec",
               "--impl-backends", impl_backends,
               "--worker-granularity", "hart")):
        r = subprocess.run([sys.executable, "-m", *a], cwd=w, env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, f"{a[0]} failed:\n{r.stdout}\n{r.stderr}"
    exe = w / "walker"
    cmd = [CC, "-O1", "-pthread", "-DMB_PLATFORM_POSIX", "-D_GNU_SOURCE",
           f"-I{ROOT / 'runtime'}", f"-I{ROOT / 'runtime' / 'modelblaster_pool'}",
           f"-I{w}", "-o", str(exe), "main.c", "demo_table.c",
           "kernels_rvv_vec.c", "kernels_rvv_mac.c",
           str(ROOT / "runtime" / "modelblaster_pool" / "modelblaster_pool.c"),
           str(ROOT / "runtime" / "mb_placement.c")]
    r = subprocess.run(cmd, cwd=w, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    return exe


def test_one_core_runs_the_impl_each_dispatch_asks_for(tmp_path):
    """Per-dispatch implementation on one core kind (the K3 A100 case: IME for
    one GEMM, the vector unit for the next): every op0 asks for `ime`, every
    op1 for `rvv`, and each must run the backend --impl-backends maps it to."""
    exe = _build_impl_walker(tmp_path, {0: "ime", 1: "rvv"},
                             "rvv:ime=rvv_mac,rvv:rvv=rvv_vec")
    r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60, env=COLD)
    assert r.returncode == 0, r.stdout[-2000:]
    ran = re.findall(r"^RAN (\w+) (\d)$", r.stdout, re.M)
    assert sorted(ran) == sorted([("rvv_mac", "0")] * N_JOBS + [("rvv_vec", "1")] * N_JOBS), ran


def test_an_impl_the_build_lacks_stops_the_run(tmp_path):
    exe = _build_impl_walker(tmp_path, {0: "ime", 1: "npu"},
                             "rvv:ime=rvv_mac,rvv:rvv=rvv_vec")
    r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 2, r.stdout[-2000:]
    assert "asks for impl 'npu'" in r.stdout


# ---- Dynamic walker ---------------------------------------------------------
# A task whose inputs are ready but whose own core is busy may run on an idle
# core of the same kind, if safety_boundary x its planned duration ends before
# that idle core's own next task is due.

def _build_custom(w: Path, schedule: dict, n_harts: int, gen_extra: list[str],
                  cflags: list[str], name: str) -> Path:
    pkg = w / "pkg"
    if not pkg.exists():
        pkg.mkdir()
        os.symlink(ROOT, pkg / "modelblaster")
    env = dict(os.environ, PYTHONPATH=str(pkg))
    reg = {"system": f"host{n_harts}", "description": "independent rvv harts",
           "default_dispatch_target": "any",
           "cores": [{"name": f"rvv{h}", "kind": "rvv", "harts": [h],
                      "capabilities": ["synth_a", "synth_b"]} for h in range(n_harts)]}
    _write(w / "schedule.json", json.dumps(schedule))
    _write(w / "registry.json", json.dumps(reg))
    _write(w / "graph.json", json.dumps({"ops": [
        {"dispatch_id": 0, "op": "synth_a", "name": "a"},
        {"dispatch_id": 1, "op": "synth_b", "name": "b"}]}))
    _write(w / "synth_model.h", _model_header())
    _write(w / "synth_test_io.h",
           "static const float model_synth_test_input[1] = { 0.0f };\n")
    _write(w / "kernels_rvv.c", _kernels("rvv"))
    for a in (("modelblaster.pipeline.ingest_xpurt_schedule",
               "--schedule", "schedule.json", "--registry", "registry.json",
               "--ir", "synth:graph.json", "--out", str(w / "demo_table.c"),
               "--name", "demo"),
              ("modelblaster.pipeline.generate_xpurt_main",
               "--schedule", "schedule.json", "--out", str(w / f"main_{name}.c"),
               "--name", "demo", "--dispatch-table-header", "demo_table.h",
               "--core-kinds", "rvv", "--registry", "registry.json",
               "--worker-granularity", "hart", *gen_extra)):
        r = subprocess.run([sys.executable, "-m", *a], cwd=w, env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, f"{a[0]} failed:\n{r.stdout}\n{r.stderr}"
    exe = w / f"walker_{name}"
    cmd = [CC, "-O1", "-pthread", "-DMB_PLATFORM_POSIX", "-D_GNU_SOURCE",
           "-DMODELBLASTER_XPURT_TRACE", *cflags,
           f"-I{ROOT / 'runtime'}", f"-I{ROOT / 'runtime' / 'modelblaster_pool'}",
           f"-I{w}", "-o", str(exe), f"main_{name}.c", "demo_table.c", "kernels_rvv.c",
           str(ROOT / "runtime" / "modelblaster_pool" / "modelblaster_pool.c"),
           str(ROOT / "runtime" / "mb_placement.c")]
    r = subprocess.run(cmd, cwd=w, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    subprocess.run([str(exe)], capture_output=True, timeout=60, env=COLD)
    return exe


def _entry(job: int, op: int, hart: int, start: float) -> tuple[str, dict]:
    return f"synth{job}_dispatch_{op}", {
        "id": op, "ordinal": op + 1, "total": 2,
        "dependencies": [f"synth{job}_dispatch_0"] if op == 1 else [],
        "hardware_target": f"CPU_P#{hart}", "start_time": start,
        "duration": OP_US / 1000.0, "job_name": f"synth{job}", "module_name": "synth"}


def _dyn_stats(out: str) -> dict:
    m = re.search(r"dynamic walker safety_boundary=([\d.]+) moved=(\d+) refused_safety=(\d+)", out)
    assert m, out[-1500:]
    return {"safety": float(m.group(1)), "moved": int(m.group(2)), "refused": int(m.group(3))}


def _trace_rows(out: str) -> list[dict]:
    import csv
    import io
    body = out.split("=== MODELBLASTER_XPURT_TRACE_BEGIN ===\n", 1)[1].split(
        "=== MODELBLASTER_XPURT_TRACE_END ===", 1)[0]
    return list(csv.DictReader(io.StringIO(body)))


def test_dynamic_walker_moves_held_tasks_to_idle_cores(tmp_path):
    """Every task is planned on hart 0; harts 1-3 have nothing. The static walker
    runs all 8 ops one after another; the dynamic walker lets the idle harts take
    the op0s whose inputs are ready, and dependencies still hold."""
    d = dict(_entry(j, op, 0, 0.0 if op == 0 else 40.0 * (j + 1))
             for j in range(N_JOBS) for op in range(2))
    sched = {"dot_file": "synth.dot", "dispatches": d}
    static = _build_custom(tmp_path, sched, N_JOBS, [], [], "static")
    dyn = _build_custom(tmp_path, sched, N_JOBS, ["--dynamic-walker"], [], "dyn")

    out_s, dt_s = _run(static)
    out_d, dt_d = _run(dyn)
    for out in (out_s, out_d):
        r = _result(out)
        assert r["done0"] == N_JOBS and r["done1"] == N_JOBS, r
        assert r["violations"] == 0, "op1 ran before its op0"
    st = _dyn_stats(out_d)
    assert st["safety"] == 1.5 and st["moved"] >= 1, st
    assert dt_d < dt_s * 0.8, f"no gain: static {dt_s:.3f}s dynamic {dt_d:.3f}s"
    rows = _trace_rows(out_d)
    moved = [r for r in rows if r["moved"] == "1"]
    assert len(moved) == st["moved"]
    assert all(r["hart"] == "0" and r["ran_on_hart"] in ("1", "2", "3") for r in moved), moved
    assert "ran_on_hart" not in out_s, "static trace format must be unchanged"


def test_dynamic_walker_safety_boundary_refuses_a_move_that_would_overrun(tmp_path):
    """Hart 1's own next task is due at 50 ms. A held 40 ms task on hart 0 needs
    1.5 x 40 = 60 ms > 50 ms, so hart 1 refuses it; at 1.0 (40 ms) it fits."""
    d = dict([_entry(0, 0, 0, 0.0), _entry(1, 0, 0, 0.0), _entry(0, 1, 0, 40.0),
              _entry(2, 0, 1, 50.0), _entry(1, 1, 0, 80.0), _entry(2, 1, 1, 90.0)])
    sched = {"dot_file": "synth.dot", "dispatches": d}
    safe = _build_custom(tmp_path, sched, 2, ["--dynamic-walker"], [], "safe")
    loose = _build_custom(tmp_path, sched, 2, ["--dynamic-walker"],
                          ["-DXPURT_SAFETY_BOUNDARY=1.0"], "loose")

    out, _ = _run(safe)
    st = _dyn_stats(out)
    assert st["safety"] == 1.5 and st["moved"] == 0 and st["refused"] >= 1, st
    assert _result(out)["violations"] == 0

    out, _ = _run(loose)
    st = _dyn_stats(out)
    assert st["safety"] == 1.0 and st["moved"] == 1, st
    moved = [r for r in _trace_rows(out) if r["moved"] == "1"]
    assert [(r["name"], r["instance"], r["ran_on_hart"]) for r in moved] == [("a", "1", "1")], moved
    assert _result(out)["violations"] == 0


def test_dynamic_walker_needs_hart_mode():
    sys.path.insert(0, str(ROOT.parent))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gxm", ROOT / "pipeline" / "generate_xpurt_main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    a = dict(networks=["n"], schedule_name="s", dispatch_table_header="s.h",
             core_kinds=["rvv"], backends=["rvv"], pool_sizes=[0], n_instances={"n": 1})
    with pytest.raises(ValueError):
        mod._emit(**a, dynamic_walker=True)
    assert mod._emit(**a, per_hart_workers=True, dynamic_walker=True) != \
        mod._emit(**a, per_hart_workers=True)


# ---- Global pool (--pool-networks) -------------------------------------------
# Entries of pooled networks have no home core: any idle core of their kind takes
# the oldest ready one, released at instance * period, and data deps still hold.

def _pool_stats(out: str) -> dict:
    m = re.search(r"moved=(\d+) refused_safety=(\d+) pooled_taken=(\d+) steal_periodic=(\d)", out)
    assert m, out[-1500:]
    return dict(zip(("moved", "refused", "pooled", "steal_periodic"), map(int, m.groups())))


def test_pool_runs_soft_work_on_idle_cores_without_waiting_for_planned_starts(tmp_path):
    """Everything is planned on hart 0 with op1s spread out to 160 ms. Pooled, the
    four jobs run on four harts as soon as their inputs are ready."""
    d = dict(_entry(j, op, 0, 0.0 if op == 0 else 40.0 * (j + 1))
             for j in range(N_JOBS) for op in range(2))
    sched = {"dot_file": "synth.dot", "dispatches": d}
    static = _build_custom(tmp_path, sched, N_JOBS, [], [], "static")
    pool = _build_custom(tmp_path, sched, N_JOBS, ["--pool-networks", "synth"], [], "pool")

    out_s, dt_s = _run(static)
    out_p, dt_p = _run(pool)
    for out in (out_s, out_p):
        r = _result(out)
        assert r["done0"] == N_JOBS and r["done1"] == N_JOBS, r
        assert r["violations"] == 0, "op1 ran before its op0"
    st = _pool_stats(out_p)
    assert st["pooled"] == 2 * N_JOBS and st["steal_periodic"] == 0, st
    assert _result(out_p)["maxpar"] >= 2
    assert dt_p < dt_s * 0.6, f"no gain: static {dt_s:.3f}s pool {dt_p:.3f}s"
    rows = _trace_rows(out_p)
    assert all(r["pooled"] == "1" for r in rows), rows[:2]
    assert len({r["ran_on_hart"] for r in rows}) >= 2


def test_pool_honours_instance_release_from_the_period(tmp_path):
    """With a 100 ms period, instance j may not start before j * 100 ms even though
    every op0 is planned at t=0 and a core is free."""
    d = dict(_entry(j, op, 0, 0.0) for j in range(3) for op in range(2))
    sched = {"dot_file": "synth.dot", "dispatches": d}
    exe = _build_custom(tmp_path, sched, 2, ["--pool-networks", "synth:100"], [], "rel")
    out, _ = _run(exe)
    assert _result(out)["violations"] == 0
    first = {}
    for r in _trace_rows(out):
        if r["name"] == "a":
            first[int(r["instance"])] = int(r["actual_start_cycles"]) / 1e6   # ns -> ms
    assert sorted(first) == [0, 1, 2], first
    for j, t in first.items():
        assert t >= 100.0 * j - 1.0, f"instance {j} started at {t:.1f} ms, release {100 * j} ms"


def test_pool_networks_must_exist_and_need_hart_mode():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gxm2", ROOT / "pipeline" / "generate_xpurt_main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    a = dict(networks=["n"], schedule_name="s", dispatch_table_header="s.h",
             core_kinds=["rvv"], backends=["rvv"], pool_sizes=[0], n_instances={"n": 1})
    with pytest.raises(ValueError):
        mod._emit(**a, pool_networks={"n": 0.0})
    with pytest.raises(ValueError):
        mod._emit(**a, per_hart_workers=True, pool_networks={"missing": 0.0})
    src = mod._emit(**a, per_hart_workers=True, pool_networks={"n": 5.0})
    assert '{ "n", 5.0 }' in src and "#define XPURT_STEAL_PERIODIC 0" in src
