"""Host build of the walker's multi-tick mode (generate_xpurt_main --seq), hosted POSIX.

A stateful two-op int8 fixture: op0 adds the tick's input into a carried accumulator, op1 writes it out. With
--seq NET=SEQ.npz the generated main runs the schedule once per tick on that tick's input and prints one
MODELBLASTER_TICK line per tick with the count of output bytes that differ from the tick's golden. A correct
golden (the running sum) gives mismatch=0 on every tick; a golden that ignores the carried state does not.
Needs only a C compiler.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")

pytestmark = pytest.mark.skipif(CC is None, reason="no C compiler on PATH")

INPUTS = [[1], [2], [3], [4], [5]]

MODEL_H = """\
#ifndef ACC_MODEL_H
#define ACC_MODEL_H
#include <stdint.h>
#define MODEL_ACC_INPUT_SIZE 1
#define MODEL_ACC_OUTPUT_SIZE 1
#define MODEL_ACC_OP_COUNT 2
typedef int8_t model_acc_input_t;
typedef int8_t model_acc_output_t;
typedef struct {
    const model_acc_input_t *input;
    model_acc_output_t *output;
    void *pool;
    int buf_set;
} model_acc_state_t;
extern const int model_acc_n_buf_sets;
typedef void (*model_acc_dispatch_fn)(model_acc_state_t *);
typedef struct { int dispatch_id; const char *name, *op, *shape; unsigned long cycles; } model_acc_op_record_t;
#endif
"""

KERNELS = """\
#include "acc_model.h"
static int8_t state;
static void op0(model_acc_state_t *s) { state = (int8_t)(state + s->input[0]); }
static void op1(model_acc_state_t *s) { s->output[0] = state; }
const model_acc_dispatch_fn MODEL_ACC_DISPATCH_FNS_RVV[MODEL_ACC_OP_COUNT] = { op0, op1 };
void model_acc_reset_profile_rvv(void) {}
const int model_acc_n_buf_sets = 1;
const model_acc_op_record_t *model_acc_profile_records_rvv(int *count) { *count = 0; return 0; }
"""


def _schedule() -> dict:
    return {"dot_file": "acc.dot", "dispatches": {
        f"acc_dispatch_{i}": {
            "id": i, "ordinal": i + 1, "total": 2,
            "dependencies": ["acc_dispatch_0"] if i else [],
            "hardware_target": "CPU_P#0", "start_time": float(i), "duration": 1.0,
            "job_name": "acc", "module_name": "acc"}
        for i in (0, 1)}}


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    w = tmp_path_factory.mktemp("walker_seq")
    pkg = w / "pkg"
    pkg.mkdir()
    os.symlink(ROOT, pkg / "modelblaster")
    env = dict(os.environ, PYTHONPATH=str(pkg))
    (w / "schedule.json").write_text(json.dumps(_schedule()))
    (w / "registry.json").write_text(json.dumps({"system": "host1", "default_dispatch_target": "any", "cores": [
        {"name": "rvv0", "kind": "rvv", "harts": [0], "capabilities": ["acc_a", "acc_b"]}]}))
    (w / "graph.json").write_text(json.dumps({"ops": [
        {"dispatch_id": 0, "op": "acc_a", "name": "a"}, {"dispatch_id": 1, "op": "acc_b", "name": "b"}]}))
    (w / "acc_model.h").write_text(MODEL_H)
    (w / "acc_test_io.h").write_text("static const int8_t model_acc_test_input[1] = { 1 };\n")
    (w / "kernels.c").write_text(KERNELS)
    ins = np.array(INPUTS, np.int8)
    np.savez(w / "good.npz", inputs=ins, outputs=np.cumsum(ins, axis=0).astype(np.int8))
    np.savez(w / "stateless.npz", inputs=ins, outputs=ins)         # ignores the carried state

    def run(*a):
        r = subprocess.run([sys.executable, "-m", *a], cwd=w, env=env, capture_output=True, text=True)
        assert r.returncode == 0, f"{a[0]} failed:\n{r.stdout}\n{r.stderr}"

    run("modelblaster.pipeline.ingest_xpurt_schedule", "--schedule", "schedule.json", "--registry",
        "registry.json", "--ir", "acc:graph.json", "--out", str(w / "t.c"), "--name", "t")
    exes = {}
    for name in ("good", "stateless"):
        run("modelblaster.pipeline.generate_xpurt_main", "--schedule", "schedule.json",
            "--out", str(w / f"main_{name}.c"), "--name", "t", "--dispatch-table-header", "t.h",
            "--core-kinds", "rvv", "--networks", "acc", "--platform", "linux",
            "--model-gen-dir", f"acc={w}", "--seq", f"acc={name}.npz")
        exe = w / f"walker_{name}"
        inc = [f"-I{ROOT / 'runtime'}", f"-I{ROOT / 'runtime' / 'modelblaster_pool'}", f"-I{w}"]
        # main.c without -D: --platform linux selects hosted POSIX in the file itself
        r = subprocess.run([CC, "-O1", "-pthread", *inc, "-c", "-o", f"main_{name}.o", f"main_{name}.c"],
                           cwd=w, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]
        cmd = [CC, "-O1", "-pthread", "-DMB_PLATFORM_POSIX", "-D_GNU_SOURCE", *inc, "-o", str(exe),
               f"main_{name}.o", "t.c", "kernels.c",
               str(ROOT / "runtime" / "modelblaster_pool" / "modelblaster_pool.c"),
               str(ROOT / "runtime" / "mb_placement.c")]
        r = subprocess.run(cmd, cwd=w, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]
        exes[name] = exe
    return exes


def _ticks(exe: Path) -> list[tuple[int, int, int]]:
    # MB_XPURT_WARM=1 must not matter: --seq turns the warm pass off (it would advance the carried state)
    r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60,
                       env=dict(os.environ, MB_XPURT_WARM="1"))
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return [tuple(map(int, m)) for m in
            re.findall(r"=== MODELBLASTER_TICK \[acc#(\d+)\] === wall=(\d+) mismatch=(\d+)", r.stdout)]


def test_every_tick_runs_on_its_input_and_carries_state(build):
    ticks = _ticks(build["good"])
    assert [t for t, _, _ in ticks] == list(range(len(INPUTS)))
    assert all(m == 0 for _, _, m in ticks), ticks
    assert all(w > 0 for _, w, _ in ticks), ticks


def test_a_golden_without_the_carried_state_is_caught(build):
    ticks = _ticks(build["stateless"])
    assert ticks[0][2] == 0                       # tick 0: state 0 + input == input
    assert all(m == 1 for _, _, m in ticks[1:]), ticks
