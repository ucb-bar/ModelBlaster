"""Slots built as KernelBlaster validated them, and launch errors blamed on their slot.

KernelBench L3 #15 (DenseNet121, fp16, L40S) failed its round-2 profile with only
``MB_ERROR partition: invalid argument``. Five conv winners, each tuned and validated in its own
one-slot problem (KernelBlaster builds one TU per slot), failed once linked into the whole model:

- ModelBlaster's builtin build compiled every slot into ONE TU. Two other conv winners declare
  ``extern __shared__ __align__(128)``; nvcc then aligns the dynamic shared memory of every kernel
  in the TU to 128 B, so a kernel with a 4-byte static ``__shared__ int`` grows from 16 to 128 B of
  static smem, and its launch with 49040 B of dynamic smem (no opt-in needed alone: 16 + 49040 <=
  48 KB) exceeds the 48 KB default -> ``invalid argument`` (4 slots). The builtin builds now use the
  same per-slot TUs (``kb_export.units`` / ``build_units``), the assembled TU only as the fallback.
- One winner asks ``cudaFuncSetAttribute(k, MaxDynamicSharedMemorySize, 99 KB)`` for a kernel with
  static smem: always ``invalid argument`` on sm_89 (static + 99 KB > the opt-in limit), harmless
  for its own launch but left as the runtime's last error. KernelBlaster's persistent host never saw
  it: the slot library links its own CUDA runtime, whose last error the host did not read. The host
  now reads it (``mb_so_last_error``), so both runtimes judge a slot alike.
- The driver blamed the whole partition and stopped. It now checks each op's launch as it is made
  (``MB_ERROR <op> (slot <id>): ...``), fails that slot's ``MB_SLOT_CHECK`` and still checks every
  other slot; ``kb_plan.profile_ir`` rejects the failing cached winners for the IR and profiles
  again on the next source (the seed), instead of ending the run.

Pure-python tests run everywhere; the profile fallback needs torch (CPU); the build tests need
torch, nvcc and a CUDA GPU.
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
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modelblaster.pipeline import kb_export  # noqa: E402
from modelblaster.pipeline.kb_plan import failed_cached_slots  # noqa: E402

SIG_A = "void kernel_a(const float *x, float *y, int n)"
SIG_B = "void kernel_b(const float *x, float *y, int n)"


def _fake_problem(d: Path) -> Path:
    """A two-slot problem directory as export_problem writes it (manifest, head, slots, tail)."""
    (d / "slots").mkdir(parents=True)
    (d / "walker_head.cu").write_text("#include <cuda_runtime.h>\n// HEAD\n")
    (d / "walker_tail.cu").write_text("void mb_run_op(int i, void **b) {}\n// TAIL\n")
    (d / "slots" / "a.cu").write_text('#include <stdint.h>\n#define BLK 128\nextern "C" ' + SIG_A + " { /* A */ }\n")
    (d / "slots" / "b.cu").write_text(SIG_B + " { /* B */ }\n")
    (d / "driver.cpp").write_text("// driver\n")
    man = {"assembly": {"head": "walker_head.cu", "tail": "walker_tail.cu", "namespace_prefix": "mbslot_",
                        "order": ["a", "b"]},
           "slots": [{"id": "a", "file": "slots/a.cu", "signature": SIG_A},
                     {"id": "b", "file": "slots/b.cu", "signature": SIG_B}]}
    (d / "manifest.json").write_text(json.dumps(man))
    return d


class UnitsTest(unittest.TestCase):
    def test_one_tu_per_slot_and_a_declaring_tail(self):
        with tempfile.TemporaryDirectory() as t:
            p = _fake_problem(Path(t) / "p")
            u = dict(kb_export.units(p))
            self.assertEqual(list(u), ["slot_a.cu", "slot_b.cu", "walker_tail.cu"])
            # each slot alone with the head: nothing of another slot in its TU
            self.assertIn("// HEAD", u["slot_a.cu"])
            self.assertIn("namespace mbslot_a {", u["slot_a.cu"])
            self.assertIn("/* A */", u["slot_a.cu"])
            self.assertNotIn("/* B */", u["slot_a.cu"])
            self.assertNotIn("/* A */", u["slot_b.cu"])
            self.assertNotIn('extern "C"', u["slot_a.cu"])          # namespaced entry, as in assemble
            tail = u["walker_tail.cu"]
            self.assertIn(f"namespace mbslot_a {{ {SIG_A}; }}", tail)
            self.assertIn(f"namespace mbslot_b {{ {SIG_B}; }}", tail)
            self.assertTrue(tail.rstrip().endswith("// TAIL"))
            # the same slot code as the assembled TU, overrides included
            u2 = dict(kb_export.units(p, {"b": SIG_B + " { /* B2 */ }\n"}))
            self.assertIn("/* B2 */", u2["slot_b.cu"])
            self.assertIn("/* B2 */", kb_export.assemble(p, {"b": SIG_B + " { /* B2 */ }\n"}))

    def test_same_units_as_kernelblaster(self):
        kb_src = Path(__file__).resolve().parents[2] / "third_party" / "KernelBlaster" / "src"
        if not (kb_src / "kernelblaster" / "knowledge" / "slots.py").exists():
            self.skipTest("no KernelBlaster checkout")
        try:
            sys.path.insert(0, str(kb_src))
            from kernelblaster.knowledge.slots import wrap_slot as kb_wrap
        except Exception as e:  # noqa: BLE001 -- KernelBlaster's own dependencies
            self.skipTest(f"KernelBlaster not importable: {e}")
        finally:
            sys.path.remove(str(kb_src))
        code = '#include <stdint.h>\n#define X 1\nextern "C" ' + SIG_A + " {}\n"
        self.assertEqual(kb_export.wrap_slot("a", code), kb_wrap("a", code, "mbslot_"))


class BuildUnitsTest(unittest.TestCase):
    """The nvcc steps (subprocess mocked): -c per unit (and the driver), one link; any failure falls back
    to the assembled TU."""

    def _run(self, fail=(), **kw):
        calls = []

        def fake(cmd, **_):
            calls.append(cmd)
            rc = 1 if any(f in " ".join(map(str, cmd)) for f in fail) else 0
            if rc == 0 and "-o" in cmd:
                Path(cmd[cmd.index("-o") + 1]).write_text("bin")
            return subprocess.CompletedProcess(cmd, rc, "", "error" if rc else "")
        with tempfile.TemporaryDirectory() as t, mock.patch.object(kb_export.subprocess, "run", fake), \
                mock.patch.object(kb_export, "cupti_flags", lambda nvcc="nvcc": []):
            p = _fake_problem(Path(t) / "p")
            r = kb_export.build_and_run(p, arch="89", runs=0, graph=True, **kw)
        return r, calls

    def test_process_build_is_per_slot(self):
        r, calls = self._run()
        self.assertEqual(r["stage"], "built")
        compiles = [c for c in calls if "-c" in c]
        self.assertEqual(sorted(Path(c[c.index("-c") + 1]).name for c in compiles),
                         ["driver.cpp", "slot_a.cu", "slot_b.cu", "walker_tail.cu"])
        for c in compiles:
            self.assertIn("-arch=sm_89", c)
            self.assertIn("per-thread", c)                    # graph=True: PTDS in every unit
        link = [c for c in calls if "-c" not in c]
        self.assertEqual(len(link), 1)
        self.assertEqual(sum(str(x).endswith(".o") for x in link[0]), 4)
        self.assertFalse(any("kernels.cu" in " ".join(map(str, c)) for c in calls))

    def test_falls_back_to_the_assembled_tu(self):
        r, calls = self._run(fail=("slot_b.cu",))
        self.assertEqual(r["stage"], "built")
        self.assertTrue(any(str(c[-2]).endswith("kernels.cu") for c in calls))
        with mock.patch.dict(os.environ, {"MB_BUILD_UNITS": "0"}):
            r, calls = self._run()
        self.assertEqual(len(calls), 1)                        # just the assembled build
        self.assertTrue(str(calls[0][-2]).endswith("kernels.cu"))

    def test_shared_object_is_per_slot(self):
        calls = []

        def fake(cmd, **_):
            calls.append(cmd)
            Path(cmd[cmd.index("-o") + 1]).write_text("bin")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        with tempfile.TemporaryDirectory() as t, mock.patch.object(kb_export.subprocess, "run", fake):
            p = _fake_problem(Path(t) / "p")
            so, _ = kb_export.build_so(p, arch="89")
            self.assertEqual(so, p / "build" / "kernels.so")
        compiles = [c for c in calls if "-c" in c]
        self.assertEqual(len(compiles), 3)                     # no driver in the library
        for c in compiles:
            self.assertIn("-fPIC", c)
        self.assertIn("-shared", calls[-1])


class DriverSourceTest(unittest.TestCase):
    def test_launch_errors_are_per_op_and_seen_in_the_host(self):
        self.assertTrue(any("mb_so_last_error" in s and "cudaGetLastError" in s for s in kb_export.SO_ABI))
        src = kb_export._DRIVER_TEMPLATE
        self.assertIn('dlsym(h, "mb_so_last_error")', src)
        self.assertIn("K.last_error()", src)
        # the partition check goes on to the per-op checks after a launch error
        i = src.index('MB_RANGE("check_partition")')
        part = src[i:src.index('MB_RANGE("check_ops")')]
        self.assertIn("run_ops_checked(launch_bad)", part)
        self.assertNotIn("K.launch(dev)", part)
        self.assertIn("std::vector<int> slot_bad(launch_bad)", src)


class FailedCachedSlotsTest(unittest.TestCase):
    def test_blames_the_failed_cached_slots(self):
        e1, e2, e3 = {"impl_id": "1"}, {"impl_id": "2"}, {"impl_id": "3"}
        plan = {"used": {"conv__v1": e1, "conv__v2": e2, "fused__x/conv": e3}}
        res = {"passed": False, "slots": {"conv__v1": {"ok": False}, "conv__v2": {"ok": True},
                                          "fused__x": {"ok": False}, "relu": {"ok": False}}}
        self.assertEqual(failed_cached_slots(res, plan), {"conv__v1": e1, "fused__x/conv": e3})
        self.assertEqual(failed_cached_slots({"passed": True, "slots": res["slots"]}, plan), {})
        self.assertEqual(failed_cached_slots({"passed": False, "slots": {}}, plan), {})     # a fault: no verdicts
        self.assertEqual(failed_cached_slots(res, {"used": {}}), {})


def _have_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


@unittest.skipUnless(_have_torch(), "needs torch (CPU) for the export")
class ProfileFallbackTest(unittest.TestCase):
    def test_failing_cached_winner_is_rejected_and_the_profile_redone(self):
        from modelblaster.pipeline.kb_plan import profile_ir
        from modelblaster.pipeline.tests.test_graphopt_remote import FakeEval, _make_cache
        from modelblaster.pipeline.tests.test_kb_model_run_eval import _tiny_ir

        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            ir = _tiny_ir(d)
            cache, specs, sid = _make_cache(ir, d, "c")

            class FailFirst(FakeEval):
                def run(self, jobs, **kw):
                    self.slots_bad = (sid,) if not self.calls else ()
                    return super().run(jobs, **kw)
            ev = FailFirst()
            p = profile_ir(ir, d / "prof", cache=cache, arch="89", runs=1, kbeval=ev, specs=specs)
            self.assertEqual(len(ev.calls), 2)
            self.assertTrue(p["passed"])
            self.assertEqual([list(f["rejected"]) for f in p["cache_fallback"]], [[sid]])
            self.assertEqual(cache.entries[0]["rejected"], [p["ir"]])
            seeds = {r["slot"]: r["seed"] for r in p["ops"]}
            self.assertFalse(str(seeds[sid]).startswith("cache:"))      # the seed now
            # rejected for this IR: the next profile does not pick it again
            ev2 = FakeEval()
            p2 = profile_ir(ir, d / "prof2", cache=cache, arch="89", runs=1, kbeval=ev2, specs=specs)
            self.assertEqual(len(ev2.calls), 1)
            self.assertNotIn("cache_fallback", p2)

    def test_no_fallback_without_a_cache_or_a_blamed_slot(self):
        from modelblaster.pipeline.kb_plan import profile_ir
        from modelblaster.pipeline.tests.test_graphopt_remote import FakeEval
        from modelblaster.pipeline.tests.test_kb_model_run_eval import _tiny_ir

        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            ir = _tiny_ir(d)
            ev = FakeEval(ok=False)
            p = profile_ir(ir, d / "prof", arch="89", runs=1, kbeval=ev)
            self.assertEqual(len(ev.calls), 1)
            self.assertFalse(p["passed"])


# ---------------------------------------------------------------- on a GPU

# 128-byte aligned dynamic shared memory somewhere in the slot (as the DenseNet conv winners)
ALIGN128 = '''
__device__ float a128_sink[32];
__global__ void k_align128() {
    extern __shared__ __align__(128) unsigned char a128_smem[];
    a128_smem[threadIdx.x] = (unsigned char)threadIdx.x; __syncthreads();
    a128_sink[threadIdx.x] = a128_smem[(threadIdx.x + 1) % 32];
}
'''
# a static __shared__ int (16 B of static smem alone) and 49040 B of dynamic smem, without opting
# in: fits the 48 KB default alone (16 + 49040), not next to a 128-aligned declaration (128 + 49040)
BIG = '''
__device__ float big_sink[32];
__global__ void k_big() {
    extern __shared__ __align__(16) unsigned char big_smem[];
    __shared__ int s_last;
    if (threadIdx.x == 0) s_last = blockIdx.x + 1;
    big_smem[threadIdx.x] = (unsigned char)threadIdx.x; __syncthreads();
    big_sink[threadIdx.x] = big_smem[(threadIdx.x + 1) % 32] + s_last;
}
'''
# asks for the whole opt-in limit (99 KB on sm_89) for a kernel with static smem: always invalid argument
OPTIN = BIG.replace("big", "opt")


def _with_prelude(seed: str, sig: str, extra: str, prelude: str) -> str:
    """The seed with its entry renamed and a new entry (same signature) that runs ``prelude`` first."""
    from modelblaster.pipeline.kernel_abi import parse_signature

    entry, params = parse_signature(sig)
    body = re.sub(rf"\b{entry}\s*\(", f"{entry}_seed(", seed.replace('extern "C"', ""), count=1)
    args = ", ".join(p.name for p in params)
    return f"{body}\n{extra}\n{sig} {{\n{prelude}\n    {entry}_seed({args});\n}}\n"


def _have_gpu() -> bool:
    if shutil.which("nvcc") is None or shutil.which("nvidia-smi") is None or not _have_torch():
        return False
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True, timeout=30).returncode == 0
    except Exception:  # noqa: BLE001
        return False


@unittest.skipUnless(_have_gpu(), "needs torch, nvcc and a CUDA GPU")
class SlotIsolationGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from modelblaster.pipeline.kb_export import export_problem
        from modelblaster.pipeline.tests.test_kb_model_run_eval import _tiny_ir

        cls.tmp = tempfile.TemporaryDirectory()
        d = cls.d = Path(cls.tmp.name)
        cls.prob = export_problem(_tiny_ir(d), d / "prob")
        man = json.loads((cls.prob / "manifest.json").read_text())
        cls.sl = {s["id"]: s for s in man["slots"]}
        assert set(cls.sl) == {"linear", "relu"}, cls.sl
        cls.arch = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                                  capture_output=True, text=True).stdout.split()[0].replace(".", "")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def seed(self, sid):
        return (self.prob / self.sl[sid]["file"]).read_text()

    def build_run(self, sources, tag, units=True, runtime="process"):
        with mock.patch.dict(os.environ, {"MB_BUILD_UNITS": "1" if units else "0"}):
            return kb_export.build_and_run(self.prob, arch=self.arch, sources=sources, work=self.d / tag,
                                           graph=True, runtime=runtime)

    def align_sources(self):
        return {"linear": _with_prelude(self.seed("linear"), self.sl["linear"]["signature"], ALIGN128,
                                        "    k_align128<<<1, 32, 256>>>();"),
                "relu": _with_prelude(self.seed("relu"), self.sl["relu"]["signature"], BIG,
                                      "    k_big<<<1, 32, 49040>>>();")}

    def test_assembled_tu_breaks_a_slot_that_passes_alone(self):
        src = self.align_sources()
        r = self.build_run(src, "one_tu", units=False)
        self.assertFalse(r["passed"], r["log"])
        self.assertTrue(any("(slot relu): invalid argument" in e for e in r["errors"]), r["errors"])
        self.assertFalse(r["slots"]["relu"]["ok"])
        self.assertTrue(r["slots"]["linear"]["ok"])            # the other slot still gets its verdict
        r = self.build_run(src, "units", units=True)
        self.assertTrue(r["passed"], r["log"])

    def test_failed_opt_in_is_an_error_in_both_runtimes(self):
        src = {"relu": _with_prelude(self.seed("relu"), self.sl["relu"]["signature"], OPTIN,
                                     "    int o = 0; cudaDeviceGetAttribute(&o, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);\n"
                                     "    cudaFuncSetAttribute(k_opt, cudaFuncAttributeMaxDynamicSharedMemorySize, o);\n"
                                     "    k_opt<<<1, 32, 1024>>>();")}
        for runtime in ("process", "host"):
            r = self.build_run(src, f"optin_{runtime}", runtime=runtime)
            self.assertFalse(r["passed"], f"{runtime}: {r.get('log')}")
            self.assertTrue(any("(slot relu): invalid argument" in e for e in r["errors"]), (runtime, r["errors"]))
            self.assertFalse(r["slots"]["relu"]["ok"])
            self.assertTrue(r["slots"]["linear"]["ok"])


if __name__ == "__main__":
    unittest.main()
