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
