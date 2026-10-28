"""SOL-ExecBench's tolerance as an acceptance gate (sol_tolerance.py; docs/sol_gate.md), CPU only.

* ``sol_compare`` semantics: matched ratio edges, inf / NaN, all-zero, max_error_cap, allow_negative_inf, the ignored
  ``required_match_ratio`` spelling, torch bf16 inputs, chunking;
* the C++ twin the driver embeds (``DRIVER_SNIPPET`` / ``DRIVER_CHECK`` + the generated tables) compiled with the host
  compiler and compared with ``sol_compare`` on random and special-value data;
* where the tolerance comes from (bench file parsed, recorded at extraction, env override order) and that extraction of a
  non-SOL bench is untouched;
* export: gate off / final = byte-identical outputs, accept = the manifest ``sol`` block, the tables, SOL's reference
  appended to data.bin, the slot bound derived from SOL's tolerance;
* MB_SOL lines parsed (kb_export.parse_driver_output) and summarised.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))

from modelblaster.pipeline import sol_tolerance as st  # noqa: E402

TOL = {"max_atol": 0.1, "max_rtol": 0.1}


def have_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class CompareTest(unittest.TestCase):
    def test_normalize_defaults_and_the_ignored_spelling(self):
        t = st.normalize({"max_atol": 0.003, "max_rtol": 1e-5, "required_match_ratio": 0.9})
        self.assertEqual(t["required_matched_ratio"], 0.99)             # SOL's pydantic model ignores the other spelling
        self.assertEqual(t["ignored_keys"], ["required_match_ratio"])
        self.assertIsNone(t["max_error_cap"])
        self.assertFalse(t["allow_negative_inf"])
        self.assertEqual(st.normalize({})["max_atol"], 1e-2)
        self.assertEqual(st.normalize({"required_matched_ratio": 0.5})["required_matched_ratio"], 0.5)
        self.assertEqual(st.normalize({"required_matched_ratio": None})["required_matched_ratio"], 0.99)
        self.assertEqual(st.normalize({"required_matched_ratio": 0.0})["required_matched_ratio"], 0.0)

    def test_matched_ratio_edges(self):
        ref = np.ones(1000, np.float32)
        out = ref.copy()
        out[:10] += 1.0                                                  # exactly 1% off: 990 / 1000 = the required 0.99
        r = st.sol_compare(out, ref, TOL)
        self.assertTrue(r["ok"])
        self.assertEqual(r["matched"], 990)
        self.assertAlmostEqual(r["matched_ratio"], 0.99)
        out[10] += 1.0                                                   # 989 / 1000
        r = st.sol_compare(out, ref, TOL)
        self.assertFalse(r["ok"])
        self.assertEqual(r["matched"], 989)
        self.assertFalse(r["nonfinite_output"])
        self.assertAlmostEqual(r["max_abs_err"], 1.0)
        self.assertAlmostEqual(r["max_excess"], 1.0 - (0.1 + 0.1 * 1.0))
        # the allowance is atol + rtol*|ref|: an element at the edge matches
        self.assertTrue(st.sol_compare(np.array([1.2], np.float32), np.array([1.0], np.float32), {"max_atol": 0.1, "max_rtol": 0.1})["ok"] is False)
        self.assertTrue(st.sol_compare(np.array([1.19], np.float32), np.array([1.0], np.float32), {"max_atol": 0.1, "max_rtol": 0.1})["ok"])

    def test_required_ratio_from_the_tolerance(self):
        ref = np.ones(100, np.float32)
        out = ref.copy()
        out[:20] += 1
        self.assertFalse(st.sol_compare(out, ref, TOL)["ok"])
        self.assertTrue(st.sol_compare(out, ref, {**TOL, "required_matched_ratio": 0.8})["ok"])
        # L2's spelling is ignored: the default 0.99 applies
        self.assertFalse(st.sol_compare(out, ref, {**TOL, "required_match_ratio": 0.8})["ok"])

    def test_inf_nan(self):
        ref = np.ones(1000, np.float32)
        out = ref.copy()
        out[3] = np.inf
        r = st.sol_compare(out, ref, {**TOL, "required_matched_ratio": 0.5})
        self.assertFalse(r["ok"])                                        # any non-finite output fails, whatever the ratio
        self.assertTrue(r["nonfinite_output"])
        out = ref.copy()
        out[3] = np.nan
        self.assertFalse(st.sol_compare(out, ref, TOL)["ok"])
        # a NaN in the reference with a finite output: unmatched (NaN compares false), not a non-finite *output*
        r = st.sol_compare(ref.copy(), np.where(np.arange(1000) == 3, np.nan, 1.0).astype(np.float32), TOL)
        self.assertFalse(r["nonfinite_output"])
        self.assertEqual(r["matched"], 999)
        self.assertTrue(r["ok"])                                         # 99.9% matched, as SOL's check counts it
        # an inf reference with a finite output matches (diff inf <= atol + rtol*inf = inf): SOL's quirk, reproduced
        r = st.sol_compare(np.array([1.0], np.float32), np.array([np.inf], np.float32), TOL)
        self.assertEqual(r["matched"], 1)

    def test_allow_negative_inf(self):
        ref = np.array([1.0, -np.inf, 2.0, -np.inf], np.float32)
        out = np.array([1.0, -np.inf, 2.0, -np.inf], np.float32)
        self.assertFalse(st.sol_compare(out, ref, TOL)["ok"])           # not allowed: -inf in the output fails
        r = st.sol_compare(out, ref, {**TOL, "allow_negative_inf": True})
        self.assertTrue(r["ok"])
        self.assertEqual(r["matched"], 4)
        out2 = out.copy()
        out2[1] = -5.0                                                   # reference -inf, output finite: matches (inf allowance)...
        self.assertEqual(st.sol_compare(out2, ref, {**TOL, "allow_negative_inf": True})["matched"], 4)
        out3 = out.copy()
        out3[0] = np.inf                                                 # +inf is never allowed
        self.assertFalse(st.sol_compare(out3, ref, {**TOL, "allow_negative_inf": True})["ok"])

    def test_all_zero_vs_nonzero(self):
        ref = np.full(100, 0.001, np.float32)
        r = st.sol_compare(np.zeros(100, np.float32), ref, {"max_atol": 1.0, "max_rtol": 1.0})
        self.assertTrue(r["all_zero_vs_nonzero_ref"])
        self.assertFalse(r["ok"])                                        # within tolerance everywhere, still fails
        self.assertTrue(st.sol_compare(np.zeros(100, np.float32), np.zeros(100, np.float32), TOL)["ok"])
        self.assertFalse(st.sol_compare(np.zeros(100, np.float32), ref, {"max_atol": 1.0, "max_rtol": 1.0})["ok"])

    def test_max_error_cap(self):
        ref = np.ones(1000, np.float32)
        out = ref.copy()
        out[0] += 0.5                                                    # one element off: matched ratio 0.999 passes
        self.assertTrue(st.sol_compare(out, ref, TOL)["ok"])
        r = st.sol_compare(out, ref, {**TOL, "max_error_cap": 0.3})
        self.assertTrue(r["max_error_cap_exceeded"])
        self.assertFalse(r["ok"])
        self.assertTrue(st.sol_compare(out, ref, {**TOL, "max_error_cap": 0.6})["ok"])

    def test_shape_and_empty(self):
        self.assertEqual(st.sol_compare(np.zeros(3), np.zeros(4), TOL)["ok"], False)
        self.assertIn("shape", st.sol_compare(np.zeros(3), np.zeros(4), TOL)["why"])
        self.assertTrue(st.sol_compare(np.zeros(0), np.zeros(0), TOL)["ok"])
        res = st.sol_compare_all([np.ones(3)], [np.ones(3), np.ones(3)], TOL)
        self.assertFalse(res[0]["ok"])

    def test_chunking_does_not_change_the_result(self):
        rng = np.random.default_rng(0)
        ref = rng.normal(size=10007).astype(np.float32)
        out = ref + rng.normal(scale=0.08, size=ref.shape).astype(np.float32)
        a, b = st.sol_compare(out, ref, TOL), st.sol_compare(out, ref, TOL, chunk=97)
        self.assertEqual(a, b)

    @unittest.skipUnless(have_torch(), "needs torch")
    def test_torch_inputs_bf16_exact(self):
        import torch
        ref = torch.randn(64, dtype=torch.float32).to(torch.bfloat16)
        out = ref.clone()
        r = st.sol_compare(out, ref, {"max_atol": 0.0, "max_rtol": 0.0})
        self.assertTrue(r["ok"])
        self.assertEqual(r["max_abs_err"], 0.0)
        from modelblaster.pipeline import kb_export
        mbir, _ = kb_export._harness()
        self.assertTrue(mbir.compare_sol(out, ref, TOL)["ok"])
        res = mbir.compare(out.float(), ref.float().numpy(), "fp32", sol=TOL)
        self.assertTrue(res["ok"])
        self.assertIn("sol", res)
        bad = mbir.compare(out.float() + 1.0, ref.float().numpy(), "fp32", sol={"max_atol": 1e9, "max_rtol": 0})
        self.assertFalse(bad["ok"])                                      # the flow's bound fails, SOL's would pass: AND


# ---------------------------------------------------------------- the C++ twin

HARNESS = r'''
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
static double env_num(const char *k, double d) { const char *v = getenv(k); return v && *v ? atof(v) : d; }
static std::vector<unsigned char> blob_v;
static const unsigned char *mb_blob;
@TABLES@
static const long B_NUMEL[] = {@N@};
static const char *B_NAME[] = {"out"};
@CODE@
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb");
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    blob_v.resize(sz); if (fread(blob_v.data(), 1, sz, f) != (size_t)sz) return 2;
    fclose(f);
    mb_blob = blob_v.data();
    const float *a = (const float *)mb_blob, *g = (const float *)(mb_blob + 4 * B_NUMEL[0]);
    bool ok = mb_sol_buffer("T", 0, a, g, true);
    printf("%s\n", ok ? "RESULT 1" : "RESULT 0");
    return 0;
}
'''


def _cxx() -> str | None:
    for c in ("c++", "g++", "clang++"):
        if shutil.which(c):
            return c
    return None


@unittest.skipUnless(_cxx(), "needs a C++ compiler")
class TwinTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="mb_sol_cxx_"))
        cls.n = 1000

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def build(self, name, tol, *, use="sol", margin=1.0, required=None, enforce=True, replace=False, n=None):
        n = n or self.n
        t = st.normalize(tol)
        spec = {"enforce": enforce, "bound": "replace" if replace else "and",
                "outputs": {"out": {"index": 0, "tolerance": {k: t[k] for k in st.KEYS}, "use": use, "margin": margin,
                                    "required": required if required is not None else t["required_matched_ratio"]}}}
        bufs = [{"name": "out", "kind": "output"}]
        tables, code = st.driver_parts(spec, bufs, {"out": 8 * n})        # [a][golden][sol ref]
        src = HARNESS.replace("@TABLES@", tables).replace("@CODE@", code).replace("@N@", str(n))
        cc, exe = self.tmp / f"{name}.cpp", self.tmp / name
        cc.write_text(src)
        cp = subprocess.run([_cxx(), "-std=c++17", "-O1", "-pthread", "-o", str(exe), str(cc)], capture_output=True, text=True)
        self.assertEqual(cp.returncode, 0, cp.stderr[-3000:])
        return exe

    def run_case(self, exe, a, golden, sref, env=None):
        p = self.tmp / "case.bin"
        p.write_bytes(np.asarray(a, np.float32).tobytes() + np.asarray(golden, np.float32).tobytes()
                      + np.asarray(sref, np.float32).tobytes())
        cp = subprocess.run([str(exe), str(p)], capture_output=True, text=True, env={**os.environ, **(env or {})})
        self.assertEqual(cp.returncode, 0, cp.stderr)
        return st.parse_sol_lines(cp.stdout), cp.stdout

    def cases(self):
        rng = np.random.default_rng(1)
        n = self.n
        ref = rng.normal(size=n).astype(np.float32)
        yield "exact", ref.copy(), ref
        yield "noisy", ref + rng.normal(scale=0.05, size=n).astype(np.float32), ref
        o = ref.copy(); o[:7] += 5
        yield "seven_bad", o, ref
        o = ref.copy(); o[:11] += 5
        yield "eleven_bad", o, ref
        o = ref.copy(); o[5] = np.inf
        yield "inf_out", o, ref
        o = ref.copy(); o[5] = -np.inf
        r2 = ref.copy(); r2[5] = -np.inf
        yield "neginf_both", o, r2
        r3 = ref.copy(); r3[9] = np.nan
        yield "nan_ref", ref.copy(), r3
        r4 = ref.copy(); r4[9] = np.inf
        yield "inf_ref", ref.copy(), r4
        yield "zeros", np.zeros(n, np.float32), ref
        yield "all_nan_out", np.full(n, np.nan, np.float32), ref

    def test_twin_matches_sol_compare(self):
        for k, tol in enumerate([TOL, {**TOL, "allow_negative_inf": True}, {**TOL, "max_error_cap": 3.0},
                                 {"max_atol": 1e-3, "max_rtol": 0.0, "required_matched_ratio": 0.5}]):
            exe = self.build(f"t{k}", tol)
            for name, a, ref in self.cases():
                with self.subTest(tol=k, case=name):
                    sol, _ = self.run_case(exe, a, np.zeros(self.n), ref)       # gate = SOL's reference (use=sol)
                    got = sol["out"]["sol"]
                    want = st.sol_compare(a, ref, tol)
                    self.assertEqual(got["ok"], want["ok"])
                    self.assertAlmostEqual(got["matched_ratio"], want["matched_ratio"], places=6)
                    self.assertEqual(got["nonfinite_output"], want["nonfinite_output"])
                    if np.isfinite(want["max_abs_err"]):
                        self.assertAlmostEqual(got["max_abs_err"], want["max_abs_err"], delta=1e-5 * max(1.0, want["max_abs_err"]))
                    else:
                        self.assertTrue(np.isinf(got["max_abs_err"]))
                    self.assertEqual(sol["out"]["gate"], "sol")

    def test_gate_reference_margin_and_enforcement(self):
        rng = np.random.default_rng(2)
        ref = rng.normal(size=self.n).astype(np.float32)
        a = ref + 0.06                                                    # |err| 0.06 against allowance 0.1 + 0.1|ref|
        tol = {"max_atol": 0.05, "max_rtol": 0.0}                        # atol 0.05 < 0.06: fails at margin 1; the golden is `ref`
        exe = self.build("m1", tol, use="golden", margin=1.0)
        sol, out = self.run_case(exe, a, ref, ref)
        self.assertFalse(sol["out"]["golden"]["ok"])
        self.assertEqual(sol["out"]["gate"], "golden")
        self.assertIn("RESULT 0", out)                                    # enforced: the buffer check fails
        # margin 2: the allowance doubles (a margin > 1 loosens); margin 0.5 tightens
        exe = self.build("m2", {"max_atol": 0.05, "max_rtol": 0.0}, use="golden", margin=2.0)
        _, out = self.run_case(exe, a, ref, ref)
        self.assertIn("RESULT 1", out)
        # margin != 1 also prints the golden at SOL's own tolerance (golden1), never gating
        sol, _ = self.run_case(exe, a, ref, ref)
        self.assertIn("golden1", sol["out"])
        self.assertFalse(sol["out"]["golden1"]["ok"])
        self.assertEqual(sol["out"]["gate"], "golden")
        # report only: the verdict is printed, the run passes
        exe = self.build("m3", tol, use="golden", margin=1.0, enforce=False)
        sol, out = self.run_case(exe, a, ref, ref)
        self.assertFalse(sol["out"]["golden"]["ok"])
        self.assertIn("RESULT 1", out)
        # ... unless the environment turns enforcement on at run time, or the gate off
        _, out = self.run_case(exe, a, ref, ref, env={"MB_SOL_ENFORCE": "1"})
        self.assertIn("RESULT 0", out)
        sol, out = self.run_case(self.build("m4", tol, use="golden", margin=1.0), a, ref, ref, env={"MB_SOL": "0"})
        self.assertEqual(sol, {})
        self.assertIn("RESULT 1", out)

    def test_flow_bound_and_replace_modes(self):
        ref = np.ones(self.n, np.float32)
        a = ref.copy()
        exe = self.build("a1", TOL, use="sol")
        # the flow's check failing (flow_ok False in the harness is always True here): AND keeps it, replace drops it
        sol, out = self.run_case(exe, a, ref, ref)
        self.assertIn("RESULT 1", out)
        exe = self.build("a2", TOL, use="sol", replace=True)
        _, out = self.run_case(exe, a, ref, ref)
        self.assertIn("RESULT 1", out)

    def test_required_ratio_override_of_the_gate(self):
        rng = np.random.default_rng(3)
        ref = rng.normal(size=self.n).astype(np.float32)
        a = ref.copy(); a[:30] += 5                                      # 97% matched
        exe = self.build("r1", TOL, use="golden", required=0.99)
        _, out = self.run_case(exe, a, ref, ref)
        self.assertIn("RESULT 0", out)
        exe = self.build("r2", TOL, use="golden", required=0.95)          # the no-regression policy's lowered floor
        sol, out = self.run_case(exe, a, ref, ref)
        self.assertIn("RESULT 1", out)
        self.assertEqual(sol["out"]["golden"]["required_ratio"], 0.95)

    def test_big_tensor_threads(self):
        n = (1 << 20) + 13
        rng = np.random.default_rng(4)
        ref = rng.normal(size=n).astype(np.float32)
        a = ref + rng.normal(scale=0.03, size=n).astype(np.float32)
        exe = self.build("big", TOL, use="sol", n=n)
        sol, _ = self.run_case(exe, a, np.zeros(n), ref)
        want = st.sol_compare(a, ref, TOL)
        self.assertEqual(sol["out"]["sol"]["ok"], want["ok"])
        self.assertAlmostEqual(sol["out"]["sol"]["matched_ratio"], want["matched_ratio"], places=6)


# ---------------------------------------------------------------- the tolerance's source

BENCH = '''import torch
import torch.nn as nn
SOL_TOLERANCE = {"max_atol": 0.02, "max_rtol": 1e-05, "required_match_ratio": 0.9, "max_error_cap": None, "allow_negative_inf": False}
class Model(nn.Module):
    def forward(self, x, w):
        return torch.relu(torch.matmul(x, w)) + 1.0
def get_inputs():
    torch.manual_seed(0)
    return [torch.randn(16, 32), torch.randn(32, 24) / 6]
def get_init_inputs():
    return []
def sol_reference_outputs(inputs=None, device="cpu"):
    x, w = get_inputs() if inputs is None else inputs
    with torch.no_grad():
        return [(torch.relu(x.to(device) @ w.to(device)) + 1.0).cpu()]
'''
PLAIN = "\n".join(ln for ln in BENCH.splitlines() if not ln.startswith("SOL_TOLERANCE")) + "\n"


class SourceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mb_sol_src_"))
        self.env = {k: os.environ.pop(k, None) for k in ("MB_SOL_TOLERANCE", "MB_SOL_BENCH", "MB_SOL_GATE")}
        st.STATE.reset()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        for k, v in self.env.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
        st.STATE.reset()

    def test_bench_tolerance_is_parsed_not_imported(self):
        (self.tmp / "b.py").write_text("raise SystemExit('imported!')\nSOL_TOLERANCE = {'max_atol': 0.5}\n")
        self.assertEqual(st.read_bench_tolerance(self.tmp / "b.py"), {"max_atol": 0.5})
        (self.tmp / "k.py").write_text(PLAIN)
        self.assertIsNone(st.read_bench_tolerance(self.tmp / "k.py"))           # a KernelBench file
        self.assertIsNone(st.read_bench_tolerance(self.tmp / "missing.py"))

    def test_resolution_order(self):
        g = {"output": {"tensors": ["a", "b"]}}
        self.assertIsNone(st.tolerances_for(g))
        (self.tmp / "b.py").write_text("SOL_TOLERANCE = {'max_atol': 0.5, 'max_rtol': 0.25}\n")
        os.environ["MB_SOL_BENCH"] = str(self.tmp / "b.py")                      # an IR that did not record it: the bench
        t = st.tolerances_for(g)
        self.assertEqual([x["max_atol"] for x in t], [0.5, 0.5])
        st.record_in_graph(g, {"max_atol": 0.003, "max_rtol": 1e-5})
        self.assertEqual(st.tolerances_for(g)[1]["max_atol"], 0.003)             # what the IR recorded beats the bench
        os.environ["MB_SOL_TOLERANCE"] = json.dumps({"max_atol": 7})
        self.assertEqual(st.tolerances_for(g)[0]["max_atol"], 7.0)               # an explicit override beats both
        self.assertEqual(len(st.tolerances_for(g)), 2)

    def test_modes(self):
        for v, want in (("off", "off"), ("final", "final"), ("accept", "accept"), ("1", "accept"), ("0", "off")):
            os.environ["MB_SOL_GATE"] = v
            self.assertEqual(st.mode(), want)
        os.environ["MB_SOL_GATE"] = "bogus"
        with self.assertRaises(ValueError):
            st.mode()
        os.environ.pop("MB_SOL_GATE")
        self.assertEqual(st.mode(), "off")
        self.assertEqual(st.STATE.frac, 0.5)                                     # the defaults of the settings
        self.assertEqual(st.STATE.margin, 0.5)
        self.assertEqual(st.STATE.bound_mode, "and")
        self.assertIsNone(st.effective("x"))
        st.STATE.mode_cfg = "final"
        self.assertIsNone(st.effective("x"))                                     # final: only the gate's own scopes carry tables
        with st.scope("enforce"):
            self.assertEqual(st.effective("x"), "enforce")
        st.STATE.mode_cfg = "accept"
        self.assertEqual(st.effective("x"), "report")                            # no policy yet: report only
        st.STATE.policy["x"] = {"status": "reachable"}
        self.assertEqual(st.effective("x"), "enforce")

    def test_slot_cap(self):
        tols = [st.normalize({"max_atol": 0.002, "max_rtol": 1e-5})]
        stats = [{"max": 5.1, "mean": 1.0}]
        cap = st.slot_cap(tols, stats, 0.25)
        self.assertAlmostEqual(cap, 0.25 * (0.002 + 1e-5) / 5.1, places=9)       # 9.8e-5: single-pass TF32 (~1e-3) fails it
        self.assertEqual(st.slot_cap([st.normalize({"max_atol": 1e-7, "max_rtol": 0})], stats, 0.25), 2e-5)   # the floor
        self.assertIsNone(st.slot_cap(tols, stats, 0.0))
        self.assertIsNone(st.slot_cap(tols, [{"max": 0.0, "mean": 0.0}], 0.25))

    def test_reference_file_roundtrip(self):
        d = self.tmp / "ir"
        d.mkdir()
        np.savez(d / st.REF_FILE, meta=np.array(json.dumps({"device": "cuda", "dtypes": ["float32"]})),
                 out0=np.arange(6, dtype=np.float32))
        r = st.load_reference(d)
        self.assertEqual(r["device"], "cuda")
        self.assertEqual(r["arrays"][0].tolist(), list(range(6)))
        self.assertIsNone(st.load_reference(self.tmp / "none"))
        st.STATE.refs["m"] = str(d / st.REF_FILE)
        self.assertIsNotNone(st.load_reference(self.tmp / "none", "m"))          # a variant dir finds it by the IR's name


# ---------------------------------------------------------------- extraction and export

@unittest.skipUnless(have_torch(), "needs torch")
class ExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="mb_sol_exp_"))
        d = cls.tmp
        for name, text in (("sol", BENCH), ("plain", PLAIN)):
            (d / name).mkdir()
            (d / name / "bench.py").write_text(text)                    # the IR's name comes from the file name
            (d / f"{name}.py").write_text(text)
            cp = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.extract_graph", "--bench-file",
                                 str(d / name / "bench.py"), "--out-dir", str(d / f"ir_{name}"), "--quant", "fp32"],
                                cwd=ROOT.parent, capture_output=True, text=True, timeout=900,
                                env={**os.environ, "MB_SOL_REF": ""})
            assert cp.returncode == 0, cp.stdout[-2000:] + cp.stderr[-2000:]
        cls.ir = d / "ir_sol"
        cls.irname = json.loads((cls.ir / "graph.json").read_text())["name"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.env = {k: os.environ.pop(k, None) for k in ("MB_SOL_TOLERANCE", "MB_SOL_BENCH", "MB_SOL_GATE",
                                                          "MB_SOL_SLOT_FRAC", "MB_SOL_GOLDEN_MARGIN", "MB_SOL_BOUND")}
        st.STATE.reset()
        self.n = 0

    def tearDown(self):
        for k, v in self.env.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
        st.STATE.reset()

    def export(self, ir=None, **kw):
        from modelblaster.pipeline import kb_export
        self.n += 1
        ExportTest.count = getattr(ExportTest, "count", 0) + 1
        out = self.tmp / f"run{ExportTest.count}" / "problem"       # driver.cpp .incbin's data.bin by path: texts() normalises it
        return kb_export.export_problem(ir or self.ir, out, strategy="whole", **kw)

    @staticmethod
    def texts(d):
        return {f: (Path(d) / f).read_bytes().replace(str(d).encode(), b"<D>") for f in
                ("driver.cpp", "manifest.json", "data.bin", "walker_head.cu", "walker_tail.cu")}

    def test_extraction_records_the_tolerance_and_nothing_else(self):
        g = json.loads((self.ir / "graph.json").read_text())
        rec = g["sol_tolerance"]
        self.assertEqual(rec["outputs"][0]["max_atol"], 0.02)
        self.assertEqual(rec["outputs"][0]["required_matched_ratio"], 0.99)       # `required_match_ratio` is not SOL's key
        self.assertEqual(rec["ignored_keys"], ["required_match_ratio"])
        g2 = json.loads((self.tmp / "ir_plain" / "graph.json").read_text())
        self.assertNotIn("sol_tolerance", g2)                                     # a KernelBench file: nothing recorded
        g.pop("sol_tolerance")
        self.assertEqual(g, g2)                                                   # ... and nothing else differs
        for f in ("io.npz", "weights.npz"):
            a, b = np.load(self.ir / f), np.load(self.tmp / "ir_plain" / f)
            self.assertEqual(sorted(a.files), sorted(b.files))
            for k in a.files:
                self.assertTrue(np.array_equal(a[k], b[k]))
        self.assertFalse((self.ir / st.REF_FILE).exists())                        # SOL's reference only with MB_SOL_REF

    def test_manifest_name_survives_the_reference_loop(self):
        """The reference loop reused the name `name` and overwrote export_problem's argument: a hint's orig / variant exports got the output
        op's name ("add") and KernelBlaster refused the pair (`problem names collide`) in 4 of the first 30 B200 pilot problems."""
        st.STATE.mode_cfg = "accept"
        np.savez(self.ir / st.REF_FILE, meta=np.array(json.dumps({"device": "cuda", "dtypes": ["float32"]})),
                 out0=np.full((16, 24), 1.5, np.float32))
        try:
            d = self.export(name="hint__orig")
        finally:
            os.unlink(self.ir / st.REF_FILE)
        man = json.loads((d / "manifest.json").read_text())
        self.assertIn("ref_offset", man["sol"]["outputs"]["add"])                 # the reference branch ran
        self.assertEqual(man["name"], "hint__orig")

    def test_record_and_reference_cli_helpers(self):
        d = self.tmp / "ir_copy"
        shutil.copytree(self.tmp / "ir_plain", d)
        self.assertIsNone(st.record_in_ir(d, self.tmp / "plain.py"))
        rec = st.record_in_ir(d, self.tmp / "sol.py")
        self.assertEqual(rec["bench"], "sol.py")
        p = st.ensure_reference(d, self.tmp / "sol.py", device="cpu")
        z = np.load(p)
        self.assertEqual(json.loads(str(z["meta"]))["device"], "cpu")
        self.assertEqual(z["out0"].shape, (16, 24))
        self.assertEqual(st.load_reference(d)["device"], "cpu")

    def test_off_and_final_are_byte_identical(self):
        base = self.texts(self.export(sol=False))
        self.assertEqual(self.texts(self.export()), base)                         # default: off
        st.STATE.mode_cfg = "final"
        self.assertEqual(self.texts(self.export()), base)                         # final: no tables outside the gate's scopes
        st.STATE.mode_cfg = "off"
        with st.scope("enforce"):
            self.assertEqual(self.texts(self.export()), base)                     # off is off, whatever scope
        d = self.export(ir=self.tmp / "ir_plain")
        st.STATE.mode_cfg = "accept"
        self.assertEqual(self.texts(self.export(ir=self.tmp / "ir_plain")), self.texts(d))   # no SOL tolerance: nothing

    def test_accept_adds_the_block_tables_and_reference(self):
        st.STATE.mode_cfg = "accept"
        # SOL's reference outputs as the GPU run would give them (device cuda): the relu(x @ w) + 1 of the bench, perturbed
        golden_like = np.full((16, 24), 1.5, np.float32)
        np.savez(self.ir / st.REF_FILE, meta=np.array(json.dumps({"device": "cuda", "dtypes": ["float32"]})), out0=golden_like)
        try:
            d = self.export()
        finally:
            os.unlink(self.ir / st.REF_FILE)
        man = json.loads((d / "manifest.json").read_text())
        sol = man["sol"]
        self.assertFalse(sol["enforce"])                                          # no policy yet: report only
        o = sol["outputs"]["add"]
        self.assertEqual(o["use"], "sol")
        self.assertEqual(o["margin"], 1.0)
        self.assertEqual(o["tolerance"]["max_atol"], 0.02)
        blob = (d / "data.bin").read_bytes()
        got = np.frombuffer(blob[o["ref_offset"]:o["ref_offset"] + 4 * 16 * 24], np.float32)
        self.assertTrue(np.array_equal(got, golden_like.reshape(-1)))             # SOL's reference rides in data.bin
        drv = (d / "driver.cpp").read_text()
        self.assertIn("MB_SOL_ENFORCE = 0", drv)
        self.assertIn("mb_sol_buffer(tag, b, a, r, ok)", drv)
        self.assertIn("static MbSolVerdict mb_sol_compare", drv)
        # with a policy the same export enforces
        st.STATE.policy[man["ir_name"]] = {"status": "reachable", "outputs": {"0": {"ref": "golden", "margin": 0.5, "required": 0.97}}}
        d2 = self.export()
        m2 = json.loads((d2 / "manifest.json").read_text())["sol"]
        self.assertTrue(m2["enforce"])
        self.assertEqual(m2["outputs"]["add"]["use"], "golden")
        self.assertEqual(m2["outputs"]["add"]["required"], 0.97)
        self.assertIn("MB_SOL_ENFORCE = 1", (d2 / "driver.cpp").read_text())
        self.assertIn("0.97", (d2 / "driver.cpp").read_text())

    def test_cpu_reference_is_the_goldens_twin_and_dtype_contract(self):
        st.STATE.mode_cfg = "accept"
        np.savez(self.ir / st.REF_FILE, meta=np.array(json.dumps({"device": "cpu", "dtypes": ["float32"]})),
                 out0=np.ones((16, 24), np.float32))
        try:
            sol = json.loads((self.export() / "manifest.json").read_text())["sol"]
            self.assertEqual(sol["outputs"]["add"]["use"], "golden")              # no GPU: not SOL's TF32 reference
            self.assertTrue(any("CPU" in w for w in sol["warnings"]))
            np.savez(self.ir / st.REF_FILE, meta=np.array(json.dumps({"device": "cuda", "dtypes": ["bfloat16"]})),
                     out0=np.ones((16, 24), np.float32))
            sol = json.loads((self.export() / "manifest.json").read_text())["sol"]
            self.assertIsNone(sol["outputs"]["add"]["ref"])                       # dtype contract: bf16 reference, fp32 buffer
            self.assertTrue(any("dtype contract" in w for w in sol["warnings"]))
        finally:
            os.unlink(self.ir / st.REF_FILE)

    def test_slot_bound_derived_from_the_tolerance(self):
        st.STATE.mode_cfg = "accept"
        st.STATE.policy[self.irname] = {"status": "reachable", "outputs": {}}          # the seed was measured: enforcing
        os.environ["MB_SOL_TOLERANCE"] = json.dumps({"max_atol": 1e-7, "max_rtol": 0.0})
        d = self.export()
        man = json.loads((d / "manifest.json").read_text())
        self.assertEqual(man["tolerance"]["slot_rel"], 2e-5)                      # the floor: fp32 kernels' own noise
        self.assertTrue(all(s["tolerance"]["slot_rel"] == 2e-5 for s in man["slots"]))
        self.assertTrue(all(o["tolerance"]["slot_rel"] == 2e-5 for o in man["ops"]))
        self.assertIn("2e-05", (d / "driver.cpp").read_text())
        st.STATE.slot_frac = 0.0                                                  # switched off: the flow's bound
        man = json.loads((self.export() / "manifest.json").read_text())
        self.assertEqual(man["tolerance"]["slot_rel"], 1e-3)
        self.assertNotIn("tolerance", man["ops"][0])
        # a generous tolerance does not loosen the flow's slot bound (min, never max)
        st.STATE.slot_frac = None
        os.environ["MB_SOL_TOLERANCE"] = json.dumps({"max_atol": 1.0, "max_rtol": 1.0})
        man = json.loads((self.export() / "manifest.json").read_text())
        self.assertEqual(man["tolerance"]["slot_rel"], 1e-3)

    def test_one_op_partition_gets_the_cap_but_no_sol_block(self):
        st.STATE.mode_cfg = "accept"
        st.STATE.policy[self.irname] = {"status": "reachable", "outputs": {}}
        os.environ["MB_SOL_TOLERANCE"] = json.dumps({"max_atol": 1e-7, "max_rtol": 0.0})
        d = self.export(op_indices=[0])                                           # a KernelBlaster problem: one op, no graph output
        man = json.loads((d / "manifest.json").read_text())
        self.assertNotIn("sol", man)
        self.assertEqual(man["sol_slot_cap"], 2e-5)
        self.assertEqual(man["tolerance"]["slot_rel"], 2e-5)
        self.assertNotIn("MB_SOL", (d / "driver.cpp").read_text())
        st.STATE.mode_cfg = "off"
        self.assertNotIn("sol_slot_cap", json.loads((self.export(op_indices=[0]) / "manifest.json").read_text()))
        st.STATE.mode_cfg = "accept"
        st.STATE.policy.clear()                                                   # before the seed's measured: no cap yet
        self.assertNotIn("sol_slot_cap", json.loads((self.export(op_indices=[0]) / "manifest.json").read_text()))

    def test_cli_check_export_only(self):
        pack = self.tmp / "pack"
        (pack / "ir").mkdir(parents=True, exist_ok=True)
        shutil.copy(self.ir / "graph.json", pack / "ir" / "graph.json")
        out = self.tmp / "check_work"
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = st.main(["check", str(pack), "--ir", str(self.ir), "--export-only", "--work", str(out)])
        self.assertEqual(rc, 0, buf.getvalue()[-2000:])
        j = json.loads(buf.getvalue())
        self.assertEqual(j["sol"]["outputs"]["add"]["tolerance"]["max_atol"], 0.02)
        self.assertFalse(j["sol"]["enforce"])                                     # report: it prints, the log is the verdict

    def test_walker_golden_checks_add_sol_only_when_the_gate_is_on(self):
        import types
        from modelblaster.pipeline import kb_export
        mbir, _ = kb_export._harness()
        ir = types.SimpleNamespace(graph=json.loads((self.ir / "graph.json").read_text()))
        self.assertIsNone(mbir.sol_for(ir))                                       # off: the golden checks are what they were
        os.environ["MB_SOL_GATE"] = "final"
        try:
            self.assertEqual(mbir.sol_for(ir)["max_atol"], 0.02)
            self.assertIsNone(mbir.sol_for(types.SimpleNamespace(graph={"output": {"tensors": ["a"]}})))   # no tolerance: nothing
        finally:
            os.environ.pop("MB_SOL_GATE")

    def test_final_mode_scope_carries_the_tables(self):
        st.STATE.mode_cfg = "final"
        with st.scope("report"):
            d = self.export()
        self.assertIn("sol", json.loads((d / "manifest.json").read_text()))


class ParseTest(unittest.TestCase):
    LOG = """MB_SOL add ref=sol matched=0.986120 required=0.99 max_abs_err=0.0487 max_rel_err=3.2 max_excess=0.027 nonfinite=0 allzero=0 cap_exceeded=0 n=4718592 atol=0.021 rtol=1e-05 ok=0 gate=1
MB_SOL add ref=golden matched=0.994800 required=0.99 max_abs_err=0.0416 max_rel_err=2.9 max_excess=0.02 nonfinite=0 allzero=0 cap_exceeded=0 n=4718592 atol=0.0105 rtol=5e-06 ok=1 gate=0
MB_CHECK output add max_abs_err=0.0487 bound=0.0521 ok=0
failed
"""

    def test_parse_and_verdict(self):
        sol = st.parse_sol_lines(self.LOG)
        self.assertEqual(sol["add"]["gate"], "sol")
        self.assertFalse(sol["add"]["sol"]["ok"])
        self.assertAlmostEqual(sol["add"]["sol"]["matched_ratio"], 0.98612)
        self.assertTrue(sol["add"]["golden"]["ok"])
        v = st.verdict(sol)
        self.assertFalse(v["ok"])
        o = v["outputs"]["add"]
        self.assertEqual(o["ref"], "sol")
        self.assertTrue(o["golden_ok"])
        self.assertEqual(st.parse_sol_lines("passed\n"), {})
        self.assertIsNone(st.verdict({}))

    def test_a_failure_in_any_check_of_the_run_stands(self):
        bad = self.LOG.replace("ok=0 gate=1", "ok=0 gate=1")
        good = self.LOG.split("\n")[0].replace("ok=0", "ok=1").replace("matched=0.986120", "matched=1.000000")
        sol = st.parse_sol_lines(bad.split("\n")[0] + "\n" + good + "\n")
        self.assertFalse(sol["add"]["sol"]["ok"])                                 # the eager check failed, the graph replay passed

    def test_driver_output_parser(self):
        from modelblaster.pipeline import kb_export
        res = kb_export.parse_driver_output(self.LOG)
        self.assertFalse(res["passed"])
        self.assertIn("sol", res)
        self.assertEqual(res["outputs"]["add"]["ok"], False)
        res2 = kb_export.parse_driver_output("MB_CHECK output add max_abs_err=0 bound=1 ok=1\npassed\n")
        self.assertNotIn("sol", res2)                                             # an ungated run: the dict is what it was

    def test_cli_parse(self):
        p = Path(tempfile.mkdtemp()) / "log.txt"
        p.write_text(self.LOG)
        self.assertEqual(st.main(["parse", str(p)]), 1)
        p.write_text(self.LOG.replace("ok=0 gate=1", "ok=1 gate=1"))
        self.assertEqual(st.main(["parse", str(p)]), 0)

    def test_cli_compare_and_tolerance(self):
        d = Path(tempfile.mkdtemp())
        ref = np.ones(100, np.float32)
        np.savez(d / "o.npz", out0=ref + 0.01)
        np.savez(d / "r.npz", out0=ref)
        self.assertEqual(st.main(["compare", str(d / "o.npz"), str(d / "r.npz"), "--tolerance", json.dumps(TOL)]), 0)
        self.assertEqual(st.main(["compare", str(d / "o.npz"), str(d / "r.npz"), "--tolerance",
                                  json.dumps({"max_atol": 1e-4, "max_rtol": 0})]), 1)
        (d / "b.py").write_text("SOL_TOLERANCE = {'max_atol': 0.5}\n")
        self.assertEqual(st.main(["tolerance", "--bench", str(d / "b.py")]), 0)
        self.assertEqual(st.main(["tolerance", "--bench", str(d / "none.py")]), 1)


if __name__ == "__main__":
    unittest.main()
