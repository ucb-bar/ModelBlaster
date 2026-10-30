"""Smoke tests for the firesim_eval module.

These tests don't actually drive the FPGA — they exercise:
  - module imports + dataclass shapes
  - memory_model_stanza renders without crashing and includes the
    expected key markers (cache size, worked-example numbers)
  - evaluate_top_k correctly orders + dedupes the candidate list when
    given a stub evaluator that returns canned cycles
The real FPGA round-trip is exercised end-to-end by the optimize loop
when --firesim-eval is passed; that's covered by the design report's
results table, not by an automated unit test.

Run from the zephyr-chipyard-sw repo root:
    python -m modelblaster.optimize.firesim_eval.test_evaluator
"""

from __future__ import annotations

import sys

from modelblaster.optimize.firesim_eval import (
    FiresimEvalConfig, FiresimEvalResult,
    evaluate_top_k, memory_model_stanza, QUAD_ROCKET_SATURN_MEMORY_MODEL,
)
from modelblaster.pipeline.reference_kernels import KERNEL_SPECS


# ---------------------------------------------------------------------------

def test_memory_model_stanza_basic():
    s = memory_model_stanza()
    # Expected substrings — these MUST appear or the LLM has nothing to
    # reason from.
    must = [
        "32 KB",            # L1D
        "4 MB",             # LLC
        "576 KB",           # worked-example weight footprint
        "TILE_OC",          # the worked-example variable
        "OC=128",           # the worked-example shape
    ]
    missing = [m for m in must if m not in s]
    assert not missing, f"memory_model_stanza missing expected text: {missing}"
    # The model itself should be the quad-rocket-saturn one by default.
    assert QUAD_ROCKET_SATURN_MEMORY_MODEL.l1d_size_bytes == 32 * 1024
    assert QUAD_ROCKET_SATURN_MEMORY_MODEL.llc_size_bytes == 4 * 1024 * 1024
    print("ok: memory_model_stanza")


def test_evaluate_top_k_ordering():
    """Stub the evaluator's evaluate() method to return canned firesim
    cycles. Verify the function picks the lowest-firesim candidate and
    correctly walks the deduped candidate list."""

    spec = KERNEL_SPECS["conv2d"]

    # Three candidates: c1 is a duplicate of the baseline, c2 has lower
    # spike cycles than c3 but worse firesim, c3 wins on firesim.
    code_baseline = "void kernel_conv2d(/* baseline */) { /* baseline body */ }"
    code_2 = "void kernel_conv2d(/* opt2 */) { /* opt2 body */ }"
    code_3 = "void kernel_conv2d(/* opt3 */) { /* opt3 body */ }"

    spike_baseline = (code_baseline, 1000)
    candidates = [(code_2, 800), (code_3, 900)]

    # Canned firesim cycles per code-body.
    firesim_canned = {
        code_baseline: 5000,
        code_2: 4500,    # spike says better but firesim says worse
        code_3: 3000,    # firesim winner
    }

    class StubEvaluator:
        def __init__(self):
            self.calls = []
        def evaluate(self, spec, code):  # noqa: D401 - signature match
            self.calls.append(code)
            cyc = firesim_canned[code]
            return FiresimEvalResult(
                ok=True, cycles_for_op=cyc, cycles_by_op={spec.op: cyc},
                wall_cycles=cyc + 50000, golden_ok=True,
                golden_max_abs_err=0.0,
                diagnostic=f"stub firesim={cyc}",
            )

    log_lines: list[str] = []
    log = log_lines.append

    stub = StubEvaluator()
    best_code, best_cyc, history = evaluate_top_k(
        spec, candidates, evaluator=stub,
        spike_baseline=spike_baseline, log=log, k=3,
    )
    assert best_code is code_3, f"expected code_3 to win on firesim, got {best_code!r}"
    assert best_cyc == 3000, f"expected 3000 cycles, got {best_cyc}"
    # Should have called the evaluator exactly 3 times (baseline + 2
    # candidates, deduped).
    assert len(stub.calls) == 3, f"expected 3 evaluator calls, got {len(stub.calls)}"
    # History should record all 3 with cycles + labels.
    assert len(history) == 3
    labels = [h["label"] for h in history]
    assert labels[0] == "baseline"
    assert "top-1" in labels
    print("ok: evaluate_top_k_ordering")


def test_evaluate_top_k_dedupe_baseline():
    """If a candidate's source matches the spike-baseline byte-for-byte,
    don't double-evaluate it on firesim."""
    spec = KERNEL_SPECS["conv2d"]
    code_baseline = "void kernel_conv2d() { /* identical body */ }"
    spike_baseline = (code_baseline, 1000)
    # Candidate is the same code (modulo whitespace). Should be skipped.
    candidates = [(code_baseline + "\n", 1000), ("void kernel_conv2d() { /* alt */ }", 800)]

    class StubEvaluator:
        def __init__(self):
            self.calls = 0
        def evaluate(self, spec, code):
            self.calls += 1
            return FiresimEvalResult(
                ok=True, cycles_for_op=2000, cycles_by_op={spec.op: 2000},
                wall_cycles=50000, golden_ok=True,
            )

    stub = StubEvaluator()
    best_code, best_cyc, history = evaluate_top_k(
        spec, candidates, evaluator=stub,
        spike_baseline=spike_baseline, log=lambda m: None, k=3,
    )
    # Baseline + 1 unique alt = 2 calls. The duplicate `candidates[0]`
    # should be filtered (it matches baseline after _normalize).
    assert stub.calls == 2, f"expected 2 unique evaluator calls, got {stub.calls}"
    print("ok: evaluate_top_k_dedupe_baseline")


def test_evaluate_top_k_all_fail():
    spec = KERNEL_SPECS["conv2d"]
    spike_baseline = ("void kernel_conv2d() {}", 1000)
    candidates = [("void kernel_conv2d() { /* alt */ }", 800)]

    class StubEvaluator:
        def evaluate(self, spec, code):
            return FiresimEvalResult(ok=False, diagnostic="stub fail")

    best_code, best_cyc, history = evaluate_top_k(
        spec, candidates, evaluator=StubEvaluator(),
        spike_baseline=spike_baseline, log=lambda m: None, k=3,
    )
    assert best_code is None and best_cyc is None
    assert all(not h["firesim_ok"] for h in history)
    print("ok: evaluate_top_k_all_fail")


def test_config_defaults_present():
    cfg = FiresimEvalConfig()
    # These knobs are referenced by the evaluator; they must exist.
    assert cfg.firesim_root
    assert cfg.firesim_env
    assert cfg.firesim_timeout_sec > 0
    assert cfg.fpga_wait_timeout_sec > 0
    assert cfg.replacement_threshold_pct >= 0
    assert cfg.board_target == "chipyard_riscv64/rocketchip_virt_riscv64"
    print("ok: config_defaults_present")


def main() -> int:
    test_memory_model_stanza_basic()
    test_evaluate_top_k_ordering()
    test_evaluate_top_k_dedupe_baseline()
    test_evaluate_top_k_all_fail()
    test_config_defaults_present()
    print("\nALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
