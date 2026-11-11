"""Host (x86) tests for runtime/mb_placement.c and the Linux pool's self-placement.

No board needed: sched_setaffinity to a plain host CPU works, and the K3
/proc/set_ai_thread file is stood in for by a temp file via
MODELBLASTER_AI_THREAD_PATH, so we can check that the HELPER's TID (not the
creator's) is what gets written.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME = _ROOT / "runtime"
_CC = shutil.which("cc") or shutil.which("gcc")

pytestmark = pytest.mark.skipif(_CC is None, reason="no host C compiler")

_DRIVER = textwrap.dedent(r"""
    #define _GNU_SOURCE 1
    #include <sched.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    #include <sys/syscall.h>
    #include <unistd.h>
    #include "modelblaster_pool.h"

    static int cpus_seen[16];
    static long tids_seen[16];
    static void probe(void *ctx, size_t i)
    {
        (void)ctx;
        cpus_seen[i] = sched_getcpu();
        tids_seen[i] = (long)syscall(SYS_gettid);
    }

    int main(int argc, char **argv)
    {
        int harts[16], n = argc - 1;
        for (int i = 0; i < n; i++) harts[i] = atoi(argv[i + 1]);
        /* Master pins itself like a real caller would. */
        cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(harts[0], &cs);
        if (sched_setaffinity(0, sizeof cs, &cs) != 0) return 10;
        modelblaster_pool_t p = modelblaster_pool_create_on_harts(n, harts);
        if (!p) { puts("POOL_NULL"); return 3; }
        modelblaster_pool_parallelize_1d(p, probe, NULL, (size_t)n, 0);
        for (int i = 0; i < n; i++)
            printf("slice %d cpu %d tid %ld\n", i, cpus_seen[i], tids_seen[i]);
        printf("master_tid %ld\n", (long)syscall(SYS_gettid));
        modelblaster_pool_destroy(p);
        puts("DONE");
        return 0;
    }
""")


def _host_cpus():
    return sorted(os.sched_getaffinity(0))


def _build(tmp_path: Path, ai_cpus: str) -> Path:
    drv = tmp_path / "driver.c"
    drv.write_text(_DRIVER)
    exe = tmp_path / f"drv_{'ai' if ai_cpus else 'noai'}"
    cmd = [_CC, "-O1", "-DMODELBLASTER_PLATFORM_LINUX", "-pthread",
           f'-DMODELBLASTER_AI_CPUS="{ai_cpus}"',
           "-I", str(_RUNTIME), "-I", str(_RUNTIME / "modelblaster_pool"),
           str(drv), str(_RUNTIME / "mb_placement.c"),
           str(_RUNTIME / "modelblaster_pool" / "modelblaster_pool.c"),
           "-o", str(exe)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return exe


def _two_cpus():
    cpus = _host_cpus()
    if len(cpus) < 2:
        pytest.skip("host has fewer than 2 usable CPUs")
    return cpus[0], cpus[1]


def test_pool_helpers_land_on_requested_harts_without_ai_cpus(tmp_path):
    a, b = _two_cpus()
    exe = _build(tmp_path, "")
    r = subprocess.run([str(exe), str(a), str(b)], capture_output=True,
                       text=True, timeout=30)
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"slice 0 cpu {a}" in r.stdout
    assert f"slice 1 cpu {b}" in r.stdout
    assert "DONE" in r.stdout


def test_ai_cpu_helper_registers_its_own_tid(tmp_path):
    a, b = _two_cpus()
    exe = _build(tmp_path, str(b))
    proc = tmp_path / "set_ai_thread"
    proc.write_text("")
    env = dict(os.environ, MODELBLASTER_AI_THREAD_PATH=str(proc))
    r = subprocess.run([str(exe), str(a), str(b)], capture_output=True,
                       text=True, timeout=30, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    tids = {}
    for line in r.stdout.splitlines():
        parts = line.split()
        if parts[:1] == ["slice"]:
            tids[int(parts[1])] = int(parts[5])
    written = proc.read_text().split()
    # Exactly the AI helper's TID, and not the (non-AI) master's.
    assert written == [str(tids[1])]
    assert str(tids[0]) not in written
    assert f"slice 1 cpu {b}" in r.stdout


def test_placement_failure_makes_create_return_null_cleanly(tmp_path):
    a, b = _two_cpus()
    exe = _build(tmp_path, str(b))
    env = dict(os.environ,
               MODELBLASTER_AI_THREAD_PATH=str(tmp_path / "no_such_dir" / "x"))
    # A hang here (leaked/unjoined helper, or a creator waiting forever) shows
    # up as a timeout.
    r = subprocess.run([str(exe), str(a), str(b)], capture_output=True,
                       text=True, timeout=30, env=env)
    assert r.returncode == 3, r.stdout + r.stderr
    assert "POOL_NULL" in r.stdout
    assert "mb_placement: FAILED" in r.stderr
    assert "step=open-ai-thread-file" in r.stderr
