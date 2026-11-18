"""What the schedule walker guarantees about how a schedule is placed and run.

The walker is `generate_xpurt_main.py` plus `ingest_xpurt_schedule.py`; it runs
on Chipyard (Zephyr) and on Linux (host, QEMU, SpacemiT K1/K3). These tests pin:

  * a composite target (one dispatch over several harts) stays on one core
    kind, keeps its first hart as its home, and in hart mode reserves every
    hart of its set and runs on a pool over them;
  * each instance's dispatches use that instance's buffer set;
  * each model instance waits for its schedule-issued start time;
  * a core kind with no backend in the build stops the run (non-zero exit on
    Linux) instead of skipping its dispatches and reporting a wrong runtime;
  * core kinds that are not backend names pair with --backends by position.

Not provided by this walker (see harness_xpurt/LINUX.md): a per-dispatch
`impl` choice, start-gating every DAG root, and an in-binary golden check (the
runners compare the output markers against the golden instead).
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

    def test_a_composite_target_is_homed_on_its_first_hart(self):
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
        first = src.index("if (e_->dispatch_id == 0 && !warm_)")
        self.assertLess(first, gate, "the gate belongs to an instance's first dispatch, "
                                     "and only to the timed pass")
        self.assertIn("e_->start_time_ms", src[gate:gate + 300])


class UnknownCoreKindIsFatal(unittest.TestCase):
    def test_a_kind_without_a_backend_stops_the_run(self):
        src = _emit()
        self.assertIn("asks for impl", src)
        self.assertNotIn("WARN unknown core_kind", src)
        tail = src[src.index("asks for impl"):]
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
        self.assertIn('strcmp(e_->impl, "rvv") == 0', src)
        self.assertIn('strcmp(e_->impl, "rvv_c1") == 0', src)
        self.assertIn("model_m_state_t st_ = s_m_rvv_c1;", src)
        self.assertIn("MODEL_M_DISPATCH_FNS_RVV_X60[e_->dispatch_id](&st_)", src)
        self.assertEqual(src.count("extern const model_m_dispatch_fn MODEL_M_DISPATCH_FNS_RVV_X60"), 1)


class JobNames(unittest.TestCase):
    """A network whose name ends in digits keeps its name."""

    def test_longest_known_name_wins(self):
        known = {"yolov8_nano", "yolov8_nano_64x96"}
        self.assertEqual(gen._split_job_name("yolov8_nano_64x96", known), ("yolov8_nano_64x96", 0))
        self.assertEqual(gen._split_job_name("yolov8_nano_64x963", known), ("yolov8_nano_64x96", 3))
        self.assertEqual(gen._split_job_name("yolov8_nano7", known), ("yolov8_nano", 7))

    def test_without_names_the_trailing_digits_are_the_instance(self):
        self.assertEqual(gen._split_job_name("dronet12"), ("dronet", 12))
        self.assertEqual(gen._split_job_name("dronet"), ("dronet", 0))

    def test_the_schedule_names_its_networks(self):
        ds = [{"module_name": "yolov8_nano_64x96$dispatch_0_rvv"}, {"module_name": "m$dispatch_1"},
              {"module_name": ""}]
        self.assertEqual(gen.networks_of(ds), {"yolov8_nano_64x96", "m"})


class WarmStart(unittest.TestCase):
    def test_the_warm_pass_runs_before_t0_and_the_timed_pass_after(self):
        src = _emit()
        worker = src[src.index("static void *xpurt_worker(void *arg)"):]
        self.assertLess(worker.index("xpurt_walk(wa, 1);"), worker.index("xpurt_walk(wa, 0);"))
        main = src[src.index("int main(void)"):]
        self.assertLess(main.index("k_sem_take(&xpurt_warm_done"), main.index("k_sem_give(&xpurt_warm_go"))
        between = main[main.index("k_sem_take(&xpurt_warm_done"):main.index("k_sem_give(&xpurt_warm_go")]
        self.assertIn("run_t0 = (uint64_t)k_cycle_get_64();", between)
        self.assertIn('getenv("MB_XPURT_WARM")', main)

    def test_the_warm_pass_is_instance_zero_data_deps_only_and_untimed(self):
        src = _emit()
        walk = src[src.index("static void xpurt_walk("):src.index("static void *xpurt_worker(")]
        self.assertIn("if (warm_ && e_->instance != 0) continue;", walk)
        self.assertIn("if (!warm_ && e_->time_dep_entry_id >= 0)", walk)
        self.assertIn("struct k_sem *sems_ = warm_ ? warm_sems : completion_sems;", walk)
        self.assertNotIn("completion_sems[", walk.replace("warm_sems : completion_sems", ""))
        timed = walk[walk.index("if (!warm_) {"):]
        for what in ("xpurt_trace[i_].start_cycles", "XPURT_STREAM_EVENT", "wall_cycles_m["):
            self.assertIn(what, timed[:900])


class GateAndStream(unittest.TestCase):
    def test_the_gate_can_sleep_on_linux(self):
        src = _emit()
        self.assertIn("XPURT_GATE_WAIT(target_start);", src)
        self.assertIn('getenv("MB_XPURT_GATE_SLEEP")', src)
        self.assertIn("nanosleep(", src)

    def test_the_stream_line_carries_what_xpurt_reads(self):
        src = _emit()
        block = src[src.index("#ifdef MODELBLASTER_XPURT_STREAM"):]
        block = block[:block.index("#endif")]
        for field in ("entry_id", "network", "instance", "dispatch_id", "impl", "hart",
                      "predicted_start_ms", "predicted_duration_ms", "start_ticks",
                      "end_ticks", "ticks_per_s"):
            self.assertIn(f'\\"{field}\\"', block)


class ImplLegality(unittest.TestCase):
    _K3 = core_registry.load(str(_ROOT / "cores" / "spacemit_k3.json"))

    def test_an_impl_the_core_does_not_list_is_refused(self):
        with self.assertRaisesRegex(ValueError, "not legal on x100_core0"):
            ingest._check_impl_legal("k", "ime", "x100", (0,), self._K3)
        with self.assertRaisesRegex(ValueError, "not legal"):
            ingest._check_impl_legal("k", "ime", "x100", (0, 1), self._K3)

    def test_listed_impls_and_the_default_pass(self):
        ingest._check_impl_legal("k", "ime", "a100", (8, 9), self._K3)
        ingest._check_impl_legal("k", "rvv", "x100", (0,), self._K3)
        ingest._check_impl_legal("k", "x100", "x100", (0,), self._K3)
        ingest._check_impl_legal("k", "ime", "rvv", (0,), _REG)   # K1: no impls, unchecked


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


class PerInstanceBuffers(unittest.TestCase):
    def test_each_dispatch_uses_its_instances_buffer_set(self):
        src = _emit()
        self.assertIn("st_.buf_set = e_->instance % model_m_n_buf_sets;", src)
        self.assertIn("if (xpurt_entry_pool[i_] != NULL) st_.pool = xpurt_entry_pool[i_];", src)

    def test_ingest_records_instances_per_network(self):
        entries = _load({})
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "t.c"
            ingest.emit_table(entries, str(out), schedule_name="s")
            h = out.with_suffix(".h").read_text()
        self.assertIn("#define XPURT_N_INSTANCES_M 1", h)
        self.assertIn("int            n_harts;", h)


class CompositeAcrossHarts(unittest.TestCase):
    def test_ingest_keeps_every_hart_of_a_composite_target_first_hart_first(self):
        e = _load({0: "CPU_P#2+CPU_P#0+CPU_P#1"})[0]
        single = _load({0: "CPU_P#2"})[0]
        self.assertEqual(e.hart, single.hart)
        self.assertEqual(len(e.harts), 3)
        self.assertEqual(e.harts[0], e.hart)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "t.c"
            ingest.emit_table(_load({0: "CPU_P#2+CPU_P#0+CPU_P#1"}), str(out), schedule_name="s")
            c = out.read_text()
        self.assertIn(f"sched_0_harts[] = {{ {', '.join(str(h) for h in e.harts)} }};", c)
        self.assertIn(".n_harts = 3, .harts = sched_0_harts", c)
        self.assertIn(".n_harts = 1, .harts = NULL", c)

    def test_hart_mode_reserves_harts_and_builds_composite_pools(self):
        src = _emit(per_hart_workers=True)
        body = src[src.index("static void xpurt_walk("):]
        lock, call, unlock = (body.index("XPURT_LOCK_ENTRY(e_);"),
                              body.index("MODEL_M_DISPATCH_FNS_RVV[e_->dispatch_id](&st_);"),
                              body.index("XPURT_UNLOCK_ENTRY(e_);"))
        self.assertLess(lock, call)
        self.assertLess(call, unlock)
        self.assertIn("#define XPURT_LOCK_ENTRY(e)   xpurt_lock_entry(e)", src)
        self.assertIn("modelblaster_pool_create_on_harts(\n                    e_->n_harts, e_->harts);", src)
        main = src[src.index("int main(void)"):]
        self.assertLess(main.index("xpurt_entry_pool[i_] = cpools_[j_];"),
                        main.index("pthread_create(&tids[k]"))

    def test_kind_mode_takes_no_hart_locks(self):
        src = _emit()
        self.assertIn("#define XPURT_LOCK_ENTRY(e)   ((void)0)", src)
        self.assertNotIn("xpurt_lock_entry", src)
