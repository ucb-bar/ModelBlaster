"""BACKEND=llm on ime2_a100 reuses kernels that already exist.

ime2_a100 sets `llm_reuses_curated`: before calling the LLM, an op is looked
up in the backend's own folder and then its alias folders, as BACKEND=reference
does, so verified kernels (the RVV gelu, the A100 linear) are not regenerated.
Backends without the flag keep the old behaviour (own folder only), so other
LLM experiments are unchanged. Verification is stubbed: this tests the lookup.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline import backends as backends_mod  # noqa: E402
from pipeline import generate_kernels  # noqa: E402
from pipeline.reference_kernels import KERNEL_SPECS  # noqa: E402

KERNELS = str(Path(__file__).resolve().parents[2] / "kernels")


class _NoLLM:
    model_id = "none"

    def converse(self, *args, **kwargs):
        raise AssertionError("LLM was called")


def _pick(target: str, op: str, kernels: str = KERNELS):
    spec = KERNEL_SPECS[op]
    with mock.patch.object(generate_kernels, "_verify",
                           return_value=generate_kernels.VerifyResult(True, "stub")):
        return generate_kernels.generate_one_llm(
            spec, [], _NoLLM(), backends_mod.get(target), "", 1, lambda m: None,
            impls={op: spec.reference_impl}, specs=[spec], repo_root=None,
            model_dir=None, build_dir=None, harness_dir=None, io_path=None,
            cache_dir=None, global_curated_dir=kernels)


class TestLlmReusesCurated(unittest.TestCase):
    def test_ime2_a100_reuses_rvv_gelu_without_llm(self):
        code, algo = _pick("ime2_a100", "gelu_s8")
        self.assertEqual(algo, "rvv_memo_lut_gather")

    def test_ime2_a100_still_uses_its_own_linear(self):
        code, algo = _pick("ime2_a100", "linear_s8")
        self.assertEqual(algo, "ime2_vmadot_8x8x16")
        self.assertIn(".insn r 0x2b, 3, 0x71", code)   # smt.vmadot

    def test_ime2_a100_uses_its_own_kernel_for_every_matrix_op(self):
        for op in ("matmul_s8", "conv2d_s8", "conv2d_batchnorm2d_silu_s8"):
            with self.subTest(op=op):
                code, algo = _pick("ime2_a100", op)
                self.assertEqual(algo, "ime2_vmadot_8x8x16")
                self.assertIn(".insn r 0x2b, 3, 0x71", code)

    def test_ime2_a100_own_candidate_without_file_goes_to_llm(self):
        # An ime2_a100 candidate with no kernel file must be generated, not
        # replaced by the RVV kernel the alias folder already has. A tree with
        # the rvv folder and an empty ime2_a100 folder stands in for the state
        # before the A100 kernels were written.
        with tempfile.TemporaryDirectory() as tmp:
            os.symlink(os.path.join(KERNELS, "rvv"), os.path.join(tmp, "rvv"))
            os.mkdir(os.path.join(tmp, "ime2_a100"))
            for op in ("matmul_s8", "conv2d_batchnorm2d_silu_s8"):
                with self.subTest(op=op):
                    with self.assertRaisesRegex(AssertionError, "LLM was called"):
                        _pick("ime2_a100", op, kernels=tmp)

    def test_ime2_kernels_must_use_the_matrix_instruction(self):
        err = generate_kernels._check_algorithm_structure(
            "void kernel_matmul_s8(void) {}", "ime2_vmadot_8x8x16")
        self.assertIsNotNone(err)
        curated = Path(KERNELS, "ime2_a100",
                       "ime2_a100_linear_s8_ime2_vmadot_8x8x16.c").read_text()
        self.assertIsNone(generate_kernels._check_algorithm_structure(
            curated, "ime2_vmadot_8x8x16"))

    def test_backend_without_flag_still_calls_llm(self):
        with self.assertRaisesRegex(AssertionError, "LLM was called"):
            _pick("rvv_x60", "gelu_s8")


if __name__ == "__main__":
    unittest.main()
