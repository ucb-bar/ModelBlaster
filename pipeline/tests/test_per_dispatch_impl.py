"""The schedule can name an implementation per dispatch. The binary obeys it.

WHAT WAS BROKEN, and it was broken silently. XPU-RT has been able to solve
per-dispatch implementation choice for a while: `run_xpurt_schedule.py` builds
impl-tagged core-group combinations behind `scheduler.enable_impls`, and
`postprocessing.py` writes `impl` onto every dispatch plus `combo_impls` into
the metadata. So a schedule could say "this GEMM on the MAC unit, the next one
on the vector unit" -- on the same core, because `hardware_target` names WHERE
and `impl` names WITH WHAT.

ModelBlaster then ignored the field entirely. `ingest_xpurt_schedule.py` and
`generate_xpurt_main.py` contained no reference to `impl`; the walker selected
its per-backend dispatch table by `core_kind`, so every dispatch on a core ran
that core's single backend. A heterogeneous schedule produced a binary that
quietly ran one implementation everywhere, reported the runtime it got, and
nothing anywhere said the placement had not happened.

That is the worst shape a bug can have here: the measurement comes back, it is
plausible, and it is a measurement of something else.

These tests pin the three things that make it real:

  * an `impl` in the schedule reaches the emitted table,
  * a schedule WITHOUT `impl` is unchanged -- it defaults to `core_kind`, so
    every schedule solved before `enable_impls` existed still produces the
    table it always did,
  * asking for an implementation the binary was not built with is FATAL at
    run time rather than a silent fallback.
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
from pipeline import ingest_xpurt_schedule as ingest  # noqa: E402

#: The real board registry, so "CPU_P#0" resolves the way it does in a run.
_REG = core_registry.load(str(_ROOT / "cores" / "spacemit_k1.json"))


def _ir():
    """Two linears, so a schedule can give them different implementations."""
    def lin(did, n):
        return {"name": f"l{did}", "op": "linear_s8",
                "inputs": [f"t{did}"], "outputs": [f"t{did + 1}"],
                "shape": {"M": 8, "K": 256, "N": n},
                "dispatch_id": did, "hardware_target": "any",
                "depends_on": [] if did == 0 else [did - 1]}
    return {"name": "m", "version": 1, "quant": "int8", "tensors": {},
            "ops": [lin(0, 256), lin(1, 256)]}


def _schedule(impls=None):
    """`impls` is {dispatch_id: impl}; omit for a pre-enable_impls schedule."""
    out = {"dot_file": "m.dot", "dispatches": {}}
    for did in (0, 1):
        d = {"id": did, "ordinal": 1, "total": 1,
             "dependencies": [] if did == 0 else ["m_dispatch_0"],
             "hardware_target": "CPU_P#0",
             "start_time": did * 1.0, "duration": 1.0,
             "job_name": "m", "module_name": f"m$dispatch_{did}"}
        if impls and did in impls:
            d["impl"] = impls[did]
        out["dispatches"][f"m_dispatch_{did}"] = d
    return out


class TheScheduleNamesAnImplementationPerDispatch(unittest.TestCase):

    def _load(self, impls=None):
        with tempfile.TemporaryDirectory() as d:
            sched = Path(d) / "s.json"
            sched.write_text(json.dumps(_schedule(impls)))
            return ingest.load(str(sched), {"m": _ir()}, _REG,
                               cpu_p_kind="rvv", cpu_e_kind="rvv_c1")

    def test_impl_defaults_to_core_kind_when_the_schedule_is_silent(self):
        """Every schedule solved before `enable_impls` existed means this."""
        entries = self._load()
        self.assertEqual([e.impl for e in entries], ["rvv", "rvv"])
        self.assertEqual([e.core_kind for e in entries], ["rvv", "rvv"])

    def test_a_per_dispatch_impl_survives_into_the_entries(self):
        entries = self._load({1: "ime"})
        self.assertEqual([e.impl for e in entries], ["rvv", "ime"])

    def test_impl_is_independent_of_where_the_dispatch_RUNS(self):
        """The point of the field.

        Both dispatches are on CPU_P#0 -- the same core, the same hart, the
        same pool. Only the implementation differs. If `impl` were derived
        from `hardware_target` this could not be expressed at all.
        """
        entries = self._load({1: "ime"})
        self.assertEqual({e.core_name for e in entries}, {entries[0].core_name})
        self.assertEqual({e.hart for e in entries}, {entries[0].hart})
        self.assertNotEqual(entries[0].impl, entries[1].impl)

    def test_the_emitted_table_carries_it(self):
        entries = self._load({1: "ime"})
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "t.c"
            ingest.emit_table(entries, str(out), schedule_name="s")
            c = out.read_text()
        self.assertIn('.impl = "ime"', c)
        self.assertIn('.impl = "rvv"', c)


class AskingForAnImplTheBuildLacksIsFatal(unittest.TestCase):
    """A silent fallback here is a wrong measurement, not a slow one."""

    def test_the_walker_selects_on_impl_not_core_kind(self):
        from pipeline import generate_xpurt_main as gen
        src = gen._emit(
            networks=["m"], schedule_name="s",
            dispatch_table_header="t.h", core_kinds=["rvv"],
            backends=["rvv_x60"], pool_sizes=[1], n_instances={"m": 1},
            platform="linux")
        self.assertIn('strcmp(e_->impl, "rvv")', src,
                      "the dispatch branch must select on the entry's "
                      "implementation, not on the kind of core it sits on")
        # core_kind is still compared -- by the WORKER FILTER, which is
        # right: a worker owns a (core_kind, hart) pair and takes the entries
        # placed on it. What must not happen is the DISPATCH BRANCH choosing
        # a kernel by the core it happens to be sitting on.
        # core_kind is still compared -- by the WORKER FILTER, which is
        # right: a worker owns a (core_kind, hart) pair and takes the entries
        # placed on it. What must not happen is a kernel CHOICE made by the
        # core a dispatch happens to sit on, so check the guard that actually
        # wraps the dispatch-fn call.
        for line in src.splitlines():
            if "DISPATCH_FNS" in line and "strcmp" in line:
                self.assertIn("e_->impl", line)
        self.assertNotIn('strcmp(e_->core_kind, "rvv") == 0) {\n'
                         '                __asm__', src)

    def test_an_unknown_impl_reboots_rather_than_falling_back(self):
        from pipeline import generate_xpurt_main as gen
        src = gen._emit(
            networks=["m"], schedule_name="s",
            dispatch_table_header="t.h", core_kinds=["rvv"],
            backends=["rvv_x60"], pool_sizes=[1], n_instances={"m": 1},
            platform="linux")
        self.assertIn("FATAL", src)
        self.assertIn("sys_reboot", src)
        self.assertIn("e_->impl", src)


if __name__ == "__main__":
    unittest.main()
