"""Functions that move their own thread between K3 core kinds stay vector-free.

On the SpacemiT K3 a thread starts on an X100 (vlenb 32); labelling it an AI
task moves it onto an A100 (vlenb 128) inside the label write, and
sched_setaffinity can move it again. GCC sizes the vector part of a stack frame
from vlenb once, in the prologue, so a function that is on the stack while its
thread moves addresses its frame with the wrong vlenb afterwards. That zeroed
main's return address in the TASK-006 bench
(k3-work/research/k3-vlen-migration-hazard.md).

The harness, walker and pool are compiled without the vector extension today
(only kernels.c gets -march=...v), so they cannot hit it. MB_THREAD_MOVER
(runtime/mb_ai_task.h) keeps the moving functions vector-free even if that
changes; these tests pin that every moving function carries it and, where the
cross compiler is available, that it works.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from pipeline import generate_xpurt_main  # noqa: E402

HEADER = _ROOT / "runtime" / "mb_ai_task.h"
HARNESS = _ROOT / "harness_linux" / "src" / "main.c"
POOL = _ROOT / "runtime" / "modelblaster_pool" / "modelblaster_pool.c"
VFLAGS = ["-O2", "-march=rv64gcv_zvl256b_zfh_zvfh", "-mabi=lp64d", "-w"]
VECTOR_INSN = re.compile(r"\sv[a-z][a-z0-9.]*\s|vlenb")


def _cross() -> str | None:
    """The riscv64 cross gcc prefix, if this machine has it (the K3 builder does)."""
    prefix = os.environ.get("CROSS", "riscv64-unknown-linux-gnu-")
    return prefix if shutil.which(prefix + "gcc") and shutil.which(prefix + "objdump") else None


def _vector_insns(cross: str, obj: str, func: str) -> int | None:
    """Vector instructions in `func`, or None if the object has no such symbol."""
    out = subprocess.run([cross + "objdump", "-d", obj], capture_output=True, text=True,
                         check=True).stdout
    m = re.search(rf"^[0-9a-f]+ <{re.escape(func)}>:\n(.*?)(?=^[0-9a-f]+ <|\Z)", out,
                  re.M | re.S)
    if not m:
        return None
    return sum(1 for line in m.group(1).splitlines()
               if re.match(r"^\s+[0-9a-f]+:", line) and VECTOR_INSN.search(line))


class TheMovingFunctionsAreMarked(unittest.TestCase):
    def test_header_defines_the_guard_and_marks_the_label_call(self):
        s = HEADER.read_text()
        self.assertIn('#define MB_THREAD_MOVER __attribute__((target("arch=rv64gc")))', s)
        self.assertIn("defined(__riscv_vector)", s)   # inert in files compiled without V
        self.assertIn("static inline MB_THREAD_MOVER int mb_ai_task_label_self(void)", s)

    def test_single_model_harness(self):
        s = HARNESS.read_text()
        self.assertIn("static MB_THREAD_MOVER int pin_from_env(", s)
        self.assertIn("MB_THREAD_MOVER int main(void)", s)

    def test_pool_helper(self):
        s = POOL.read_text()
        self.assertIn("static MB_THREAD_MOVER void *modelblaster_pool_worker_fn(", s)
        self.assertIn("#ifndef MB_THREAD_MOVER", s)   # Zephyr builds do not include the header

    def test_generated_walker(self):
        sched = {"dot_file": "m.dot", "dispatches": {"m_dispatch_0": {
            "id": 0, "ordinal": 1, "total": 1, "dependencies": [], "hardware_target": "CPU_P#0",
            "start_time": 0.0, "duration": 1.0, "job_name": "m", "module_name": "m$dispatch_0"}}}
        with tempfile.TemporaryDirectory() as d:
            sp, out = Path(d) / "s.json", Path(d) / "s_main.c"
            sp.write_text(json.dumps(sched))
            argv = ["generate_xpurt_main", "--schedule", str(sp), "--out", str(out),
                    "--name", "s", "--dispatch-table-header", "s.h", "--platform", "linux",
                    "--core-kinds", "x100", "--backends", "rvv_x60", "--networks", "m"]
            old = sys.argv
            try:
                sys.argv = argv
                generate_xpurt_main.main()
            finally:
                sys.argv = old
            c = out.read_text()
        self.assertIn("static MB_THREAD_MOVER void *xpurt_worker(void *arg)", c)
        self.assertIn("#ifndef MB_THREAD_MOVER", c)


@unittest.skipUnless(_cross(), "riscv64 cross compiler not on PATH (run in the K3 builder)")
class TheGuardWorksUnderVectorFlags(unittest.TestCase):
    """Compiled WITH the vector extension, the moving functions still have none."""

    def _compile(self, src: Path, extra: list[str], d: str) -> str:
        obj = os.path.join(d, src.stem + ".o")
        subprocess.run([_cross() + "gcc", *VFLAGS, "-DMODELBLASTER_PLATFORM_LINUX",
                        "-DMODELBLASTER_USE_POOL", "-I", str(_ROOT / "runtime"),
                        "-I", str(_ROOT / "runtime" / "modelblaster_pool"), *extra,
                        "-c", str(src), "-o", obj], check=True, capture_output=True)
        return obj

    def test_guard_removes_vector_code(self):
        """The guard itself: a loop GCC vectorizes loses its vector code when
        marked. Without this, the checks below would pass with no guard at all,
        since nothing in today's moving functions attracts vector code."""
        body = "{ for (int i = 0; i < n; i++) a[i] = b[i] * 3 + c[i]; return a[n / 2]; }\n"
        src = ("#include \"mb_ai_task.h\"\n"
               "__attribute__((noinline)) int plain(int *a, const int *b, const int *c, int n)" + body +
               "MB_THREAD_MOVER __attribute__((noinline)) int marked(int *a, const int *b, const int *c, int n)" + body)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "loop.c"
            p.write_text(src)
            obj = self._compile(p, ["-O3"], d)
            self.assertGreater(_vector_insns(_cross(), obj, "plain"), 0)
            self.assertEqual(_vector_insns(_cross(), obj, "marked"), 0)

    def test_pool_worker(self):
        with tempfile.TemporaryDirectory() as d:
            obj = self._compile(POOL, [], d)
            self.assertEqual(_vector_insns(_cross(), obj, "modelblaster_pool_worker_fn"), 0)

    def test_label_call(self):
        src = ("#include \"mb_ai_task.h\"\n"
               "MB_THREAD_MOVER int use(void) { return mb_ai_task_label_self(); }\n")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "use.c"
            p.write_text(src)
            obj = self._compile(p, [], d)
            self.assertEqual(_vector_insns(_cross(), obj, "use"), 0)
            n = _vector_insns(_cross(), obj, "mb_ai_task_label_self")
            self.assertIn(n, (None, 0))   # None: inlined into use()


if __name__ == "__main__":
    unittest.main()
