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


def _run(exe: Path) -> tuple[str, float]:
    t0 = time.monotonic()
    r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
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
    # The hosted-Linux opt-ins change nothing unless asked for.
    assert mod._emit(**a, warmup=False, placement=False) == base
    for hart in (False, True):
        assert "xpurt_warm" not in mod._emit(**a, per_hart_workers=hart)
        assert "XPURT_PLACEMENT" not in mod._emit(**a, per_hart_workers=hart)
    # --stamps off: the same text as before the option existed, whatever the
    # other options are; on: only then do the two extra stamps appear.
    for hart in (False, True):
        for warm in (False, True):
            for place in (False, True):
                kw = dict(per_hart_workers=hart, warmup=warm, placement=place)
                off = mod._emit(**a, **kw)
                assert mod._emit(**a, **kw, stamps=False) == off
                assert "got_cores_cycles" not in off
                assert "kernel_end_cycles" not in off
                on = mod._emit(**a, **kw, stamps=True)
                assert on != off and "got_cores_cycles" in on
    # --own-lines off: unchanged, whatever the other options are; on: the
    # worker visits its collected list instead of the whole table.
    for hart in (False, True):
        for warm in (False, True):
            for stamps in (False, True):
                kw = dict(per_hart_workers=hart, warmup=warm, placement=True, stamps=stamps)
                off = mod._emit(**a, **kw)
                assert mod._emit(**a, **kw, own_lines=False) == off
                assert "xpurt_own" not in off
                on = mod._emit(**a, **kw, own_lines=True)
                assert "for (int j_ = 0; j_ < xpurt_own_n; j_++)" in on
                assert on.count("strcmp(e_->core_kind, my_kind)") == 1
                # collected before the warm-up pass, never inside the timed one
                if warm:
                    assert on.index("xpurt_own[xpurt_own_n++]") < on.index("for (int pass_ = 0;")


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
                       env=dict(os.environ, MODELBLASTER_AI_THREAD_PATH=str(reg)))
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
    r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout[-2000:]
    ran = re.findall(r"^RAN (\w+) (\d)$", r.stdout, re.M)
    assert sorted(ran) == sorted([("rvv_mac", "0")] * N_JOBS + [("rvv_vec", "1")] * N_JOBS), ran


def test_an_impl_the_build_lacks_stops_the_run(tmp_path):
    exe = _build_impl_walker(tmp_path, {0: "ime", 1: "npu"},
                             "rvv:ime=rvv_mac,rvv:rvv=rvv_vec")
    r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 2, r.stdout[-2000:]
    assert "asks for impl 'npu'" in r.stdout


def test_warmup_pass_and_placement_record(tmp_path):
    """--warmup runs every entry twice (untimed pass, then the timed one) with
    dependencies honoured in both; --placement prints one row per entry with
    the cpu the kernel ran on (its pinned hart on Linux, -1 elsewhere)."""
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

    def run(*a):
        r = subprocess.run([sys.executable, "-m", *a], cwd=w, env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, f"{a[0]} failed:\n{r.stdout}\n{r.stderr}"

    run("modelblaster.pipeline.ingest_xpurt_schedule",
        "--schedule", "schedule.json", "--registry", "registry.json",
        "--ir", "synth:graph.json", "--out", str(w / "demo_table.c"),
        "--name", "demo")
    for mode in ("kind", "hart"):
        run("modelblaster.pipeline.generate_xpurt_main",
            "--schedule", "schedule.json", "--out", str(w / f"main_{mode}.c"),
            "--name", "demo", "--dispatch-table-header", "demo_table.h",
            "--core-kinds", "rvv", "--worker-granularity", mode,
            "--warmup", "--placement")
        exe = w / f"walker_{mode}"
        cmd = [CC, "-O1", "-pthread", "-DMB_PLATFORM_POSIX", "-D_GNU_SOURCE",
               f"-I{ROOT / 'runtime'}", f"-I{ROOT / 'runtime' / 'modelblaster_pool'}",
               f"-I{w}", "-o", str(exe), f"main_{mode}.c", "demo_table.c",
               "kernels_rvv.c",
               str(ROOT / "runtime" / "modelblaster_pool" / "modelblaster_pool.c"),
               str(ROOT / "runtime" / "mb_placement.c")]
        r = subprocess.run(cmd, cwd=w, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]
        out, _ = _run(exe)
        res = _result(out)
        assert res["done0"] == 2 * N_JOBS and res["done1"] == 2 * N_JOBS, res
        assert res["violations"] == 0, "dependency broken in a pass"
        assert "xpurt: warm-up pass on" in out
        m = re.search(r"=== MODELBLASTER_XPURT_PLACEMENT_BEGIN ===\n(.*?)"
                      r"=== MODELBLASTER_XPURT_PLACEMENT_END ===", out, re.S)
        assert m, out[-1500:]
        rows = [l.split(",") for l in m.group(1).strip().splitlines()[1:]]
        assert len(rows) == 2 * N_JOBS
        for row in rows:
            hart, before, after = int(row[6]), int(row[9]), int(row[10])
            if sys.platform.startswith("linux") and mode == "hart":
                assert before == after == hart, row
            else:
                assert before >= -1 and after >= -1, row


# --- two extra stamps per entry (generate_xpurt_main --stamps) ---------------
STAMP_SHORT_US = 10_000   # every op but the one below
STAMP_LONG_US = 60_000    # job 1's op0: holds hart 1 while job 0's op1 wants it


def _stamp_kernels(backend: str) -> str:
    """Fixture backend whose op0 of instance 1 runs long. Two buffer sets, so
    `buf_set` is the instance."""
    BS = backend.upper()
    return f"""\
#include <unistd.h>
#include "synth_model.h"
static void op0(model_synth_state_t *s) {{ usleep(s->buf_set == 1 ? {STAMP_LONG_US} : {STAMP_SHORT_US}); }}
static void op1(model_synth_state_t *s) {{ (void)s; usleep({STAMP_SHORT_US}); }}
const model_synth_dispatch_fn MODEL_SYNTH_DISPATCH_FNS_{BS}[MODEL_SYNTH_OP_COUNT] = {{ op0, op1 }};
void model_synth_reset_profile_{backend}(void) {{}}
const int model_synth_n_buf_sets = 2;
const model_synth_op_record_t *model_synth_profile_records_{backend}(int *count)
{{ *count = 0; return NULL; }}
"""


def _stamp_schedule() -> dict:
    """Job 0: op0 on hart 0, then op1 on harts 0+1 (multi-core). Job 1: both
    ops on hart 1, op0 long. Job 0's op1 has its input after one short op but
    hart 1 stays held by job 1's long op0."""
    def disp(job: int, i: int, target: str) -> dict:
        return {"id": i, "ordinal": i + 1, "total": 2,
                "dependencies": [f"synth{job}_dispatch_0"] if i == 1 else [],
                "hardware_target": target,
                "start_time": 0.0 if i == 0 else 10.0 + 50.0 * job,
                "duration": 10.0, "job_name": f"synth{job}", "module_name": "synth"}
    return {"dot_file": "synth.dot", "dispatches": {
        "synth0_dispatch_0": disp(0, 0, "CPU_P#0"),
        "synth1_dispatch_0": disp(1, 0, "CPU_P#1"),
        "synth0_dispatch_1": disp(0, 1, "CPU_P#0+CPU_P#1"),
        "synth1_dispatch_1": disp(1, 1, "CPU_P#1")}}


@pytest.mark.parametrize("extra", [
    pytest.param((), id="stamps"),
    pytest.param(("--placement",), id="stamps+placement"),
    # the warm-up pass uses pthread barriers, which macOS does not have
    pytest.param(("--warmup", "--placement"), id="stamps+warmup+placement",
                 marks=pytest.mark.skipif(sys.platform == "darwin",
                                          reason="--warmup needs pthread_barrier (Linux)")),
])
def test_stamps_are_ordered_and_the_core_wait_is_outside_the_run(tmp_path, extra):
    """--stamps adds got_cores (all cores locked) and kernel_end (dispatch
    returned) to each entry's own trace slot, printed as the LAST two trace
    columns. For every entry that ran a kernel: start <= got_cores <=
    kernel_end <= end. Known answer: the multi-core op1 of job 0 is ready
    after ~10 ms but one of its harts is held for ~60 ms, so its wait shows
    between start and got_cores, and got_cores -> kernel_end is its own
    ~10 ms."""
    w = tmp_path
    pkg = w / "pkg"
    pkg.mkdir()
    os.symlink(ROOT, pkg / "modelblaster")
    env = dict(os.environ, PYTHONPATH=str(pkg))
    _write(w / "schedule.json", json.dumps(_stamp_schedule()))
    _write(w / "registry.json", json.dumps(_registry()))
    _write(w / "graph.json", json.dumps({"ops": [
        {"dispatch_id": 0, "op": "synth_a", "name": "a"},
        {"dispatch_id": 1, "op": "synth_b", "name": "b"}]}))
    _write(w / "synth_model.h", _model_header())
    _write(w / "synth_test_io.h",
           "static const float model_synth_test_input[1] = { 0.0f };\n")
    _write(w / "kernels_rvv.c", _stamp_kernels("rvv"))
    for a in (("modelblaster.pipeline.ingest_xpurt_schedule",
               "--schedule", "schedule.json", "--registry", "registry.json",
               "--ir", "synth:graph.json", "--out", str(w / "demo_table.c"),
               "--name", "demo"),
              ("modelblaster.pipeline.generate_xpurt_main",
               "--schedule", "schedule.json", "--out", str(w / "main.c"),
               "--name", "demo", "--dispatch-table-header", "demo_table.h",
               "--core-kinds", "rvv", "--worker-granularity", "hart",
               "--stamps", *extra)):
        r = subprocess.run([sys.executable, "-m", *a], cwd=w, env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, f"{a[0]} failed:\n{r.stdout}\n{r.stderr}"
    exe = w / "walker"
    cmd = [CC, "-O1", "-pthread", "-DMB_PLATFORM_POSIX", "-D_GNU_SOURCE",
           "-DMODELBLASTER_XPURT_TRACE",
           f"-I{ROOT / 'runtime'}", f"-I{ROOT / 'runtime' / 'modelblaster_pool'}",
           f"-I{w}", "-o", str(exe), "main.c", "demo_table.c", "kernels_rvv.c",
           str(ROOT / "runtime" / "modelblaster_pool" / "modelblaster_pool.c"),
           str(ROOT / "runtime" / "mb_placement.c")]
    r = subprocess.run(cmd, cwd=w, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    out, _ = _run(exe)
    m = re.search(r"=== MODELBLASTER_XPURT_TRACE_BEGIN ===\n(.*?)"
                  r"=== MODELBLASTER_XPURT_TRACE_END ===", out, re.S)
    assert m, out[-1500:]
    lines = m.group(1).strip().splitlines()
    cols = lines[0].split(",")
    assert cols[-4:] == ["actual_start_cycles", "actual_end_cycles",
                         "got_cores_cycles", "kernel_end_cycles"], cols
    rows = [dict(zip(cols, l.split(","))) for l in lines[1:]]
    assert len(rows) == 4
    ns = {}
    for row in rows:
        s, g, k, e = (int(row[c]) for c in ("actual_start_cycles", "got_cores_cycles",
                                            "kernel_end_cycles", "actual_end_cycles"))
        assert 0 < s <= g <= k <= e, row
        ns[(int(row["instance"]), int(row["dispatch_id"]))] = (s, g, k, e)
    s, g, k, e = ns[(0, 1)]                       # the multi-core entry
    wait_ms, kernel_ms = (g - s) / 1e6, (k - g) / 1e6
    assert wait_ms >= 25.0, f"the wait for hart 1 must sit before got_cores: {wait_ms:.1f} ms"
    assert STAMP_SHORT_US / 1e3 * 0.9 <= kernel_ms < wait_ms, (kernel_ms, wait_ms)
    for key in ((0, 0), (1, 0)):                  # nothing to wait for
        s, g, k, e = ns[key]
        assert (g - s) / 1e6 < 5.0, (key, s, g)


# --- each worker visits only its own entries (generate_xpurt_main --own-lines) ---
@pytest.mark.parametrize("mode", ["kind", "hart"])
def test_own_lines_runs_the_same_entries_correctly(tmp_path, mode):
    """--own-lines changes how a worker finds its entries, not what runs: the
    same results, no dependency violation, and in hart mode the harts still
    overlap."""
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
               "--core-kinds", "rvv", "--worker-granularity", mode, "--own-lines")):
        r = subprocess.run([sys.executable, "-m", *a], cwd=w, env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, f"{a[0]} failed:\n{r.stdout}\n{r.stderr}"
    exe = w / "walker"
    cmd = [CC, "-O1", "-pthread", "-DMB_PLATFORM_POSIX", "-D_GNU_SOURCE",
           f"-I{ROOT / 'runtime'}", f"-I{ROOT / 'runtime' / 'modelblaster_pool'}",
           f"-I{w}", "-o", str(exe), "main.c", "demo_table.c", "kernels_rvv.c",
           str(ROOT / "runtime" / "modelblaster_pool" / "modelblaster_pool.c"),
           str(ROOT / "runtime" / "mb_placement.c")]
    r = subprocess.run(cmd, cwd=w, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    out, _ = _run(exe)
    res = _result(out)
    assert res["done0"] == N_JOBS and res["done1"] == N_JOBS
    assert res["violations"] == 0
    assert "=== MODELBLASTER_OUTPUT_BEGIN [synth] ===" in out
    if mode == "hart":
        assert res["maxpar"] >= 2, "independent harts should still overlap"
    else:
        assert res["maxpar"] == 1


@pytest.mark.parametrize("extra", [
    pytest.param((), id="own-lines+stamps"),
    pytest.param(("--warmup", "--placement"), id="own-lines+stamps+warmup+placement",
                 marks=pytest.mark.skipif(sys.platform == "darwin",
                                          reason="--warmup needs pthread_barrier (Linux)")),
])
def test_own_lines_keeps_the_multi_core_known_answer(tmp_path, extra, monkeypatch):
    """The --stamps known answer (a multi-core entry waiting ~60 ms for a hart
    another entry holds) holds unchanged with --own-lines: hart locks and
    dependency waits are untouched."""
    real_run = subprocess.run

    def run_with_own_lines(cmd, *a, **k):
        if isinstance(cmd, list) and "modelblaster.pipeline.generate_xpurt_main" in cmd:
            cmd = cmd + ["--own-lines"]
        return real_run(cmd, *a, **k)

    monkeypatch.setattr(subprocess, "run", run_with_own_lines)
    test_stamps_are_ordered_and_the_core_wait_is_outside_the_run(tmp_path, extra)
