"""K3 ingest: which implementation may run on which core, and where the ops
ingest adds go.

The K3's two matrix engines share one instruction encoding with different
tiles (X100 `ime` = the K1's 4x8x4, A100 `ime2` = 8x16x8). An `ime` kernel on
an A100 produces wrong answers rather than trapping (measured max_abs_err=48),
and an `ime2` kernel traps on an X100. The schedule's `impl` picks the kernel
table and `hardware_target` picks the cpu, so nothing stopped a schedule from
pairing them wrongly; the registry's per-core `impls` now does, at ingest.

Ingest also adds the scalar-FP ops the schedule never placed (activations
between layers). The K1 pins them to CPU_E#0; on the K3 CPU_E is the A100 kind,
and the Director's decision (2026-10-28) is to put them on an X100 instead,
which `added_op_target` expresses without changing the K1 default.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from pipeline import core_registry  # noqa: E402
from pipeline import ingest_xpurt_schedule as ingest  # noqa: E402

K3 = core_registry.load(str(_ROOT / "cores" / "spacemit_k3.json"))
K1 = core_registry.load(str(_ROOT / "cores" / "spacemit_k1.json"))


def _ir(ops=("linear_s8", "linear_s8")):
    out = []
    for did, op in enumerate(ops):
        out.append({"name": f"n{did}", "op": op, "inputs": [f"t{did}"],
                    "outputs": [f"t{did + 1}"], "shape": {"M": 8, "K": 256, "N": 256},
                    "dispatch_id": did, "hardware_target": "any",
                    "depends_on": [] if did == 0 else [did - 1]})
    return {"name": "m", "version": 1, "quant": "int8", "tensors": {}, "ops": out}


def _schedule(placed: dict[int, tuple[str, str | None]]):
    """placed = {dispatch_id: (hardware_target, impl or None)}."""
    out = {"dot_file": "m.dot", "dispatches": {}}
    prev = None
    for did, (target, impl) in sorted(placed.items()):
        d = {"id": did, "ordinal": 1, "total": 1,
             "dependencies": [prev] if prev else [], "hardware_target": target,
             "start_time": did * 1.0, "duration": 1.0, "job_name": "m",
             "module_name": f"m$dispatch_{did}"}
        if impl:
            d["impl"] = impl
        key = f"m_dispatch_{did}"
        out["dispatches"][key] = d
        prev = key
    return out


def _load(placed, ir=None, reg=K3, **kw):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "s.json"
        p.write_text(json.dumps(_schedule(placed)))
        return ingest.load(str(p), {"m": ir or _ir()}, reg,
                           cpu_p_kind="x100" if reg is K3 else "rvv",
                           cpu_e_kind="a100" if reg is K3 else "rvv_c1", **kw)


class RegistryImpls(unittest.TestCase):
    def test_k3_cores_declare_their_impls(self):
        self.assertEqual(K3.by_name["x100_core0"].impls, ("rvv", "ime"))
        self.assertEqual(K3.by_name["a100_core15"].impls, ("rvv", "ime2"))

    def test_k1_cores_declare_none(self):
        self.assertTrue(all(c.impls == () for c in K1.cores))


class ImplLegality(unittest.TestCase):
    def test_each_matrix_engine_on_its_own_kind(self):
        e = _load({0: ("CPU_P#0", "ime"), 1: ("CPU_E#0", "ime2")})
        self.assertEqual([(x.core_kind, x.hart, x.impl) for x in e],
                         [("x100", 0, "ime"), ("a100", 8, "ime2")])

    def test_x100_ime_on_an_a100_is_refused(self):
        with self.assertRaisesRegex(ValueError, "'ime' is not legal on a100_core8"):
            _load({0: ("CPU_E#0", "ime"), 1: ("CPU_E#0", "rvv")})

    def test_a100_ime2_on_an_x100_is_refused(self):
        with self.assertRaisesRegex(ValueError, "'ime2' is not legal on x100_core1"):
            _load({0: ("CPU_P#1", "ime2"), 1: ("CPU_P#1", "rvv")})

    def test_every_hart_of_a_composite_target_is_checked(self):
        e = _load({0: ("CPU_E#0+CPU_E#1", "ime2"), 1: ("CPU_P#0", None)})
        self.assertEqual(e[0].harts, (8, 9))

    def test_no_impl_means_the_kinds_own_backend(self):
        e = _load({0: ("CPU_E#2", None), 1: ("CPU_P#3", None)})
        self.assertEqual([(x.impl, x.hart) for x in e], [("a100", 10), ("x100", 3)])

    def test_k1_is_not_checked(self):
        e = _load({0: ("CPU_E#0", "ime"), 1: ("CPU_P#0", "ime")}, reg=K1)
        self.assertEqual([x.impl for x in e], ["ime", "ime"])


class AddedOps(unittest.TestCase):
    """linear -> gelu -> linear, with the gelu left out of the schedule."""

    IR = _ir(("linear_s8", "gelu_s8", "linear_s8"))
    PLACED = {0: ("CPU_E#0", "ime2"), 2: ("CPU_E#0", "ime2")}

    def _gelu(self, **kw):
        e = _load(self.PLACED, ir=self.IR, **kw)
        return next(x for x in e if x.dispatch_id == 1)

    def test_the_k1_default_is_unchanged(self):
        g = self._gelu()
        self.assertEqual((g.core_kind, g.hart), ("a100", 8))

    def test_on_the_k3_they_go_to_an_x100(self):
        g = self._gelu(added_op_target="CPU_P#0")
        self.assertEqual((g.core_kind, g.hart, g.impl), ("x100", 0, "x100"))


if __name__ == "__main__":
    unittest.main()
