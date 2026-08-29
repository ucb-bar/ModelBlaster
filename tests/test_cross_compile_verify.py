"""An RVV kernel cannot be verified by compiling it for x86.

`rvv_x60` used to declare `verify_method=host_ctypes`. The host is x86, so the
verify step ran `cc` over a candidate full of `vint32m4_t` and
`__riscv_vle8_v_i8m1` and got "unknown type name" -- every time, for every
candidate, no matter how good it was. The generator reads a verify failure as
the model being wrong, so it retried four times and then fell back to the
algorithm's seed, writing the SCALAR REFERENCE into a build labelled
`target: rvv_x60` under `source: seed`.

That is the same silent-scalar regression `backends.py` opens by warning about
-- DroNet at 195 ms against RVV's 113 ms with `source=reference` for all eight
ops -- reached through the verify path rather than the curated-lookup path. It
is worse than a crash because the build succeeds and the number looks like an
RVV measurement.

So `rvv_x60` cross-compiles instead: the candidate must BUILD for the target
ISA, and numeric correctness is left to the on-board golden compare, which is
where this backend's docstring already said correctness is established. These
tests pin both halves -- that a real intrinsic error is still caught, and that
a correct RVV kernel is no longer rejected for the crime of not being x86.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modelblaster.pipeline.backends import (  # noqa: E402
    BACKENDS, VERIFY_CROSS_COMPILE,
)
from modelblaster.pipeline.generate_kernels import (  # noqa: E402
    cross_compile_verify,
)
from modelblaster.pipeline.reference_kernels import KERNEL_SPECS  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = KERNEL_SPECS["matmul_s8"]
BACKEND = BACKENDS["rvv_x60"]


def _cross_available() -> bool:
    cross = os.environ.get("CROSS", "")
    return bool(cross) and shutil.which(f"{cross}gcc") is not None


# A minimal but real RVV kernel: correct signature, genuine intrinsics, and
# the scalar requantize tail the algorithm requires for bit-exactness.
GOOD = """\
void kernel_matmul_s8(const int8_t *a, const int8_t *b, int8_t *output,
                      int M, int K, int N,
                      float scale_a, float scale_b, float scale_out,
                      int transpose_b, float scale_div,
                      int activation_min, int activation_max) {
    const float total = (scale_a * scale_b) / (scale_out * scale_div);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            if (transpose_b) {
                const int8_t *ar = a + (size_t)i * K, *br = b + (size_t)j * K;
                size_t k = 0;
                vint32m1_t red = __riscv_vmv_v_x_i32m1(0, 1);
                while (k < (size_t)K) {
                    size_t vl = __riscv_vsetvl_e8m1((size_t)K - k);
                    vint8m1_t va = __riscv_vle8_v_i8m1(ar + k, vl);
                    vint8m1_t vb = __riscv_vle8_v_i8m1(br + k, vl);
                    vint16m2_t p = __riscv_vwmul_vv_i16m2(va, vb, vl);
                    red = __riscv_vwredsum_vs_i16m2_i32m1(p, red, vl);
                    k += vl;
                }
                acc = __riscv_vmv_x_s_i32m1_i32(red);
            } else {
                for (int k = 0; k < K; k++)
                    acc += (int32_t)a[i*K + k] * (int32_t)b[k*N + j];
            }
            int32_t v = (int32_t)roundf((float)acc * total);
            if (v < activation_min) v = activation_min;
            if (v > activation_max) v = activation_max;
            output[i*N + j] = (int8_t)v;
        }
    }
}
"""

# Right signature, real-looking RVV -- but `vint8m9_t` is not a type and
# `__riscv_vle8_v_i8m9` is not an intrinsic. This is the shape of mistake the
# cross-compile is there to catch.
BAD = GOOD.replace("vint8m1_t va", "vint8m9_t va").replace(
    "__riscv_vle8_v_i8m1(ar + k, vl)", "__riscv_vle8_v_i8m9(ar + k, vl)")


class BackendDeclarationTests(unittest.TestCase):

    def test_rvv_x60_does_not_claim_host_verification(self):
        """The host cannot execute rv64gcv; it must not claim to verify it."""
        self.assertEqual(BACKEND.verify_method, VERIFY_CROSS_COMPILE)

    def test_host_cc_really_cannot_compile_the_good_kernel(self):
        """The premise, not an assumption: x86 `cc` rejects valid RVV.

        If this ever stops being true the cross-compile route is no longer
        load-bearing and someone should reconsider it -- so it is asserted
        rather than believed.
        """
        cc = shutil.which("cc") or shutil.which("gcc")
        if not cc:
            self.skipTest("no host compiler")
        src = "#include <stdint.h>\n#include <math.h>\n" + GOOD
        proc = subprocess.run([cc, "-x", "c", "-c", "-", "-o", os.devnull],
                              input=src, capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("vint32m1_t", proc.stderr)


class CrossCompileVerifyTests(unittest.TestCase):

    def setUp(self):
        if not _cross_available():
            self.skipTest('CROSS unset or toolchain missing; '
                          'eval "$(scripts/setup_spacemit_toolchain.sh)"')

    def test_accepts_a_real_rvv_kernel(self):
        r = cross_compile_verify(SPEC, GOOD, BACKEND, REPO)
        self.assertTrue(r.ok, r.message)

    def test_still_rejects_a_bogus_intrinsic(self):
        """Weakening verify must not mean accepting anything."""
        r = cross_compile_verify(SPEC, BAD, BACKEND, REPO)
        self.assertFalse(r.ok)

    def test_says_it_did_not_execute_the_kernel(self):
        """The result must not read as a numeric pass to whoever logs it."""
        r = cross_compile_verify(SPEC, GOOD, BACKEND, REPO)
        self.assertIn("NOT executed", r.message)


class NoToolchainTests(unittest.TestCase):

    def test_refuses_rather_than_passing_when_cross_is_unset(self):
        """No toolchain means UNVERIFIED, and unverified is not a pass.

        The failure mode being excluded: treat a missing compiler as "nothing
        to check" and return ok, which accepts every candidate unbuilt.
        """
        saved = os.environ.get("CROSS")
        os.environ["CROSS"] = ""
        try:
            r = cross_compile_verify(SPEC, GOOD, BACKEND, REPO)
            self.assertFalse(r.ok)
            self.assertIn("CROSS", r.message)
        finally:
            if saved is None:
                os.environ.pop("CROSS", None)
            else:
                os.environ["CROSS"] = saved


if __name__ == "__main__":
    unittest.main()
