"""What the schedule walker guarantees about how a schedule is placed and run.

The walker is `generate_xpurt_main.py` plus `ingest_xpurt_schedule.py`; it runs
on Chipyard (Zephyr) and on Linux (host, QEMU, SpacemiT K1/K3). These tests pin:

  * a composite target (one dispatch over several harts) stays on one core
    kind, and runs on its first hart;
  * each model instance waits for its schedule-issued start time;
  * a core kind with no backend in the build stops the run (non-zero exit on
    Linux) instead of skipping its dispatches and reporting a wrong runtime;
  * core kinds that are not backend names pair with --backends by position.

Not provided by this walker (see harness_xpurt/LINUX.md): a per-dispatch
`impl` choice, reserving every hart of a composite target, start-gating every
DAG root, and an in-binary golden check (the runners compare the output
markers against the golden instead).
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))   # the `modelblaster` namespace shim

from pipeline import core_registry  # noqa: E402
from pipeline import generate_xpurt_main as gen  # noqa: E402
from pipeline import ingest_xpurt_schedule as ingest  # noqa: E402

#: The real board registry, so "CPU_P#n" resolves the way it does in a run.
_REG = core_registry.load(str(_ROOT / "cores" / "spacemit_k1.json"))


def _ir():
    def lin(did):
        return {"name": f"l{did}", "op": "linear_s8",
                "inputs": [f"t{did}"], "outputs": [f"t{did + 1}"],
                "shape": {"M": 8, "K": 256, "N": 256},
                "dispatch_id": did, "hardware_target": "any",
                "depends_on": [] if did == 0 else [did - 1]}
    return {"name": "m", "version": 1, "quant": "int8", "tensors": {},
            "ops": [lin(0), lin(1)]}


def _load(targets):
    sched = {"dot_file": "m.dot", "dispatches": {}}
    for did in (0, 1):
        sched["dispatches"][f"m_dispatch_{did}"] = {
            "id": did, "ordinal": 1, "total": 1,
            "dependencies": [] if did == 0 else ["m_dispatch_0"],
            "hardware_target": targets.get(did, "CPU_P#0"),
            "start_time": did * 1.0, "duration": 1.0,
            "job_name": "m", "module_name": f"m$dispatch_{did}"}
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "s.json"
        p.write_text(json.dumps(sched))
        return ingest.load(str(p), {"m": _ir()}, _REG,
                           cpu_p_kind="rvv", cpu_e_kind="rvv_c1")


def _emit(**kw):
    a = dict(networks=["m"], schedule_name="s", dispatch_table_header="t.h",
             core_kinds=["rvv"], backends=["rvv"], pool_sizes=[0],
             n_instances={"m": 1})
    a.update(kw)
    return gen._emit(**a)


class CompositeTargets(unittest.TestCase):
    def test_a_composite_target_cannot_mix_core_kinds(self):
        with self.assertRaisesRegex(ValueError, "mixes core kinds"):
            _load({0: "CPU_P#0+CPU_E#0"})

    def test_a_composite_target_runs_on_its_first_hart(self):
        single = _load({0: "CPU_P#1"})[0]
        composite = _load({0: "CPU_P#1+CPU_P#2"})[0]
        self.assertEqual(composite.core_kind, "rvv")
        self.assertEqual(composite.hart, single.hart)
        self.assertEqual(composite.core_name, single.core_name)


class StartGate(unittest.TestCase):
    def test_each_instance_waits_for_its_scheduled_start(self):
        src = _emit()
        self.assertEqual(src.count("uint64_t target_start = run_t0"), 1)
        gate = src.index("uint64_t target_start = run_t0")
        first = src.index("if (e_->dispatch_id == 0)")
        self.assertLess(first, gate, "the gate belongs to an instance's first dispatch")
        self.assertIn("e_->start_time_ms", src[gate:gate + 300])


class UnknownCoreKindIsFatal(unittest.TestCase):
    def test_a_kind_without_a_backend_stops_the_run(self):
        src = _emit()
        self.assertIn("FATAL: xpurt: no backend built for core_kind", src)
        self.assertNotIn("WARN unknown core_kind", src)
        tail = src[src.index("no backend built for core_kind"):]
        self.assertIn("XPURT_FATAL_EXIT();", tail[:300])

    def test_fatal_exits_non_zero_on_linux_and_reboots_on_zephyr(self):
        src = _emit()
        self.assertRegex(src, r"#ifdef MB_PLATFORM_POSIX\n#define XPURT_FATAL_EXIT\(\) "
                              r"do \{ fflush\(stdout\); exit\(2\); \} while \(0\)\n#else\n"
                              r"#define XPURT_FATAL_EXIT\(\) sys_reboot\(SYS_REBOOT_COLD\)")
        # every FATAL path goes through it
        for m in re.finditer(r'printf\("FATAL:[^;]*;\s*\n\s*(\S+)', src):
            self.assertEqual(m.group(1), "XPURT_FATAL_EXIT();", m.group(0))


class KindBackendPairing(unittest.TestCase):
    def test_kinds_that_are_not_backend_names_pair_by_position(self):
        src = _emit(core_kinds=["rvv", "rvv_c1"], backends=["rvv_x60", "rvv_x60"],
                    pool_sizes=[0, 0])
        self.assertIn('strcmp(e_->core_kind, "rvv") == 0', src)
        self.assertIn('strcmp(e_->core_kind, "rvv_c1") == 0', src)
        self.assertIn("MODEL_M_DISPATCH_FNS_RVV_X60[e_->dispatch_id](&s_m_rvv_c1)", src)
        self.assertEqual(src.count("extern const model_m_dispatch_fn MODEL_M_DISPATCH_FNS_RVV_X60"), 1)


if __name__ == "__main__":
    unittest.main()


class K3Placement(unittest.TestCase):
    """On hosted Linux every worker places itself (a K3 A100 thread must
    register its own TID before it may pin), and main stays off the AI CPUs."""

    def test_workers_place_themselves_and_main_stays_off_the_ai_cpus(self):
        src = _emit()
        self.assertIn('#include "mb_placement.h"', src)
        worker = src[src.index("static void *xpurt_worker(void *arg)"):]
        self.assertIn("mb_place_current_thread(wa->hart)", worker[:1500])
        main = src[src.index("int main(void)"):]
        self.assertLess(main.index("mb_pin_self_to_first_non_ai_cpu()"),
                        main.index("modelblaster_pool_create"),
                        "main must leave the AI CPUs before any pool helper exists")
        # attribute affinity cannot reach an A100 CPU: Zephyr only
        self.assertIn("#if defined(CONFIG_POSIX_THREADS_AFFINITY) && !MB_PLACEMENT_LINUX", src)

    def test_a_kinds_pool_runs_on_that_kinds_harts(self):
        src = _emit(core_kinds=["x100", "a100"], backends=["rvv_k3", "rvv_k3_a100"],
                    pool_sizes=[8, 8], pool_harts=[list(range(8)), list(range(8, 16))])
        self.assertIn("static const int xpurt_pool_harts_1[] = { 8, 9, 10, 11, 12, 13, 14, 15 };", src)
        self.assertIn("modelblaster_pool_create_on_harts(nh_, hs_)", src)
        # without hart lists the pool keeps the 0..n-1 default
        self.assertIn("modelblaster_pool_create(pool_sizes[k])",
                      _emit(pool_sizes=[2]))
