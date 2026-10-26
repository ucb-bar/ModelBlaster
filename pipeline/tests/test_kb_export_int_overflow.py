"""Kernel calls whose C `int` size would overflow (>= 2**31 elements).

KernelBench L3 #31 (VisionAttention, fp16 native sizes) scales the
[8, 16384, 16384] attention scores -- 2**31 elements -- with one
`ewise_scalar_f16 n=2147483648`. The walker passed that literal to the
kernel's `int n`, which narrowed it to -2**31: the kernel wrote nothing and
the slot check saw rel_err = 1/sqrt(32) (the untouched zero output against
`c * x`). Flat elementwise calls are now split into int-sized chunks over
offset pointers; anything else that overflows fails the export.

The CUDA test needs nvcc and a GPU and is skipped otherwise; set
MB_BIG_GPU_TESTS=1 to also run the real 2**31-element call (8 GiB).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modelblaster.pipeline import kb_export  # noqa: E402
from modelblaster.pipeline.kb_export import (CHUNK_ELEMS, DEFAULT_HAND_CU_F16, MB_ANY_PTR,  # noqa: E402
                                             CudaLibrary, _call_stmt, wrap_slot)
from modelblaster.pipeline.kernel_abi import parse_signature  # noqa: E402

SIG = "void kernel_ewise_scalar_f16(const __half *input, __half *output, int n, int opc, float c)"
FN = "mbslot_ewise_scalar_f16::kernel_ewise_scalar_f16"
N31 = 2 ** 31


def _ewise_call(n, *, numel=None, elementwise=True, cuda=True, opc=3, c="0.17677669529663687f"):
    _, params = parse_signature(SIG)
    bufs = [{"dtype": "f16", "numel": numel or n}] * 24
    args = ["mb_any{b[23]}", "mb_any{b[10]}", str(n), str(opc), c]
    scalars = [(2, params[2], n), (3, params[3], opc), (4, params[4], 0.17677669529663687)]
    return _call_stmt(FN, args, scalars, [(0, 23), (1, 10)], n_ptr_params=2, bufs=bufs,
                      elementwise=elementwise, cuda=cuda, op_name="attn#12")


class CallStmtTest(unittest.TestCase):
    def test_in_range_call_unchanged(self):
        self.assertEqual(_ewise_call(4194304),
                         f"{FN}(mb_any{{b[23]}}, mb_any{{b[10]}}, 4194304, 3, 0.17677669529663687f);")
        self.assertEqual(_ewise_call(2 ** 31 - 1).count(FN), 1)

    def test_vision_attention_scale_is_chunked(self):
        s = _ewise_call(N31)
        self.assertNotIn(", 2147483648, ", s)                       # never the narrowed literal
        self.assertIn(f"for (long o_ = 0; o_ < {N31}L; o_ += {CHUNK_ELEMS}L)", s)
        self.assertIn("mb_any{(char *)b[23] + o_ * 2L}", s)          # fp16: 2-byte element offsets
        self.assertIn("mb_any{(char *)b[10] + o_ * 2L}", s)
        self.assertIn(f"(int)({N31}L - o_ < {CHUNK_ELEMS}L ? {N31}L - o_ : {CHUNK_ELEMS}L), 3, "
                      "0.17677669529663687f", s)
        self.assertLessEqual(CHUNK_ELEMS, 2 ** 31 - 1)

    def test_unchunkable_overflow_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "attn#12: int n=2147483648"):
            _ewise_call(N31, elementwise=False)                       # e.g. a flat reduction
        with self.assertRaises(ValueError):
            _ewise_call(N31, numel=1)                                 # a buffer is not n elements
        with self.assertRaises(ValueError):
            _ewise_call(N31, cuda=False)                              # OpenCL: no chunked form
        _, params = parse_signature("void kernel_bmm_f16(const __half *A, const __half *B, __half *C, "
                                    "int batch, int M, int K, int N)")
        with self.assertRaisesRegex(ValueError, "int M=4294967296"):
            _call_stmt("bmm", ["a", "b", "c", "1", "4294967296", "1", "1"],
                       [(3, params[3], 1), (4, params[4], 2 ** 32), (5, params[5], 1), (6, params[6], 1)],
                       [(0, 0), (1, 1), (2, 2)], n_ptr_params=3, bufs=[{"dtype": "f16", "numel": 2 ** 32}] * 3,
                       elementwise=False)


def _have_cuda() -> bool:
    if shutil.which("nvcc") is None or shutil.which("nvidia-smi") is None:
        return False
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True, timeout=30).returncode == 0
    except Exception:
        return False


MAIN = r'''
#include <cstdio>
#include <vector>
void run(void **b) { @STMT@ }
int main() {
    const long n = @N@;
    std::vector<__half> h(n);
    for (long i = 0; i < n; ++i) h[i] = __float2half((float)((i * 7919) % 2048) / 64.0f - 16.0f);
    void *b[24] = {};
    cudaMalloc(&b[23], n * 2); cudaMalloc(&b[10], n * 2); cudaMemset(b[10], 0, n * 2);
    cudaMemcpy(b[23], h.data(), n * 2, cudaMemcpyHostToDevice);
    run(b);
    std::vector<__half> y(n);
    if (cudaMemcpy(y.data(), b[10], n * 2, cudaMemcpyDeviceToHost) != cudaSuccess) { printf("cuda error\n"); return 1; }
    long bad = 0;
    for (long i = 0; i < n; ++i) {
        float r = __half2float(__float2half(__half2float(h[i]) * 0.17677669529663687f));
        if (__half2float(y[i]) != r) ++bad;
    }
    printf("bad=%ld\n", bad);
    return bad != 0;
}
'''


@unittest.skipUnless(_have_cuda(), "needs nvcc and a CUDA GPU")
class ChunkedHandKernelTest(unittest.TestCase):
    """The chunked call drives the real fp16 hand kernel over every element exactly once."""

    def _build_run(self, n):
        lib = CudaLibrary(DEFAULT_HAND_CU_F16.read_text())
        tu = "\n".join([lib.head_source(), wrap_slot("ewise_scalar_f16", lib.slot_source("kernel_ewise_scalar_f16")),
                        MB_ANY_PTR, MAIN.replace("@STMT@", _ewise_call(n)).replace("@N@", f"{n}L")])
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "t.cu").write_text(tu)
            cp = subprocess.run(["nvcc", "-O2", "-std=c++17", "-o", f"{d}/t", f"{d}/t.cu"],
                                capture_output=True, text=True, timeout=600)
            self.assertEqual(cp.returncode, 0, cp.stderr)
            rp = subprocess.run([f"{d}/t"], capture_output=True, text=True, timeout=900)
            self.assertEqual(rp.returncode, 0, rp.stdout + rp.stderr)
            self.assertIn("bad=0", rp.stdout)

    def test_small_chunks_cover_everything(self):
        # shrink the int limit and the chunk so a small buffer takes the chunked path, with a partial last chunk
        with mock.patch.object(kb_export, "INT32_MAX", 9999), mock.patch.object(kb_export, "CHUNK_ELEMS", 4096):
            self._build_run(10000 + 123)

    @unittest.skipUnless(os.environ.get("MB_BIG_GPU_TESTS"), "set MB_BIG_GPU_TESTS=1 (8 GiB of device memory)")
    def test_vision_attention_size(self):
        self._build_run(N31)


if __name__ == "__main__":
    unittest.main()
