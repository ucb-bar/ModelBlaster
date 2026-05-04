"""On-spike profile + verify helper.

Given a {op_kind: c_code} dict, this writes kernels.{c,h} into the model
directory, runs an incremental west build, runs spike (with backend-specific
args), parses the profile + output blocks, and returns a HarnessResult
covering cycles per op kind, the printed model output, and (optionally) a
numerical compare against a PyTorch golden.

Used in two roles:
  * the optimize beam search (correctness already done; just need cycles).
  * the correctness verify path for backends that can't be host-compiled
    (RVV intrinsics, custom accelerator extensions). In that case the
    "verify" is the model output matching the PyTorch golden inside the
    full Zephyr harness, run on spike.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from agents.pipeline.backends import Backend
from agents.pipeline.reference_kernels import KernelSpec
from agents.validation.spike_runner import (
    BEGIN, find_spike, parse_output, parse_profile,
)
from agents.validation.runner_common import parse_verify


@dataclass
class HarnessResult:
    cycles_by_op: dict[str, int] = field(default_factory=dict)
    output: Optional[np.ndarray] = None
    golden_ok: Optional[bool] = None
    golden_max_abs_err: float = 0.0
    golden_max_rel_err: float = 0.0
    raw_stdout: str = ""


class ProfileError(RuntimeError):
    pass


def _emit_kernels_files(
    impls: dict[str, str],
    specs: list[KernelSpec],
    backend: Backend,
    out_dir: str,
    model_name: Optional[str] = None,
) -> None:
    # Local import to avoid a cycle with generate_kernels.
    from agents.pipeline.generate_kernels import emit_kernels_h, emit_kernels_c
    emit_kernels_h(specs, out_dir, model_name=model_name)
    emit_kernels_c(impls, "optimize", out_dir, backend=backend, model_name=model_name)


def _west_build(
    *,
    harness_dir: str,
    build_dir: str,
    model_dir: str,
    backend: Backend,
    pristine: bool,
    repo_root: str,
) -> tuple[bool, str]:
    cmd = ["west", "build", "-b", "spike_riscv64", harness_dir,
           "--build-dir", build_dir]
    if pristine:
        cmd.insert(2, "-p")
    cmd += ["--",
            f"-DMODEL_DIR={model_dir}",
            f"-DAGENTS_BACKEND={backend.name}"]
    if backend.kernel_cflags:
        cmd.append(f"-DAGENTS_KERNEL_CFLAGS={';'.join(backend.kernel_cflags)}")
    env = os.environ.copy()
    env["PATH"] = "/usr/bin:" + env.get("PATH", "")
    proc = subprocess.run(
        cmd, cwd=repo_root, env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        # Where the actual cc1 diagnostic lands varies. west/ninja
        # sometimes route compile errors to stdout (when a python wrapper
        # buffers them through), but on incremental ninja rebuilds the
        # gcc output goes straight to stderr. Show both with generous
        # tails so the LLM retry sees the real error, not just the
        # "command exited with status 1" wrapper.
        return False, (
            f"west build failed (rc={proc.returncode}):\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stdout (tail):\n{proc.stdout[-3000:]}\n"
            f"  stderr (tail):\n{proc.stderr[-3000:]}"
        )
    return True, ""


def _spike_run(elf: str, backend: Backend, timeout: float) -> tuple[bool, str]:
    spike = find_spike()
    cmd = [spike, *backend.spike_args, elf]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        # subprocess.run() kills the child before raising, so no cleanup needed.
        # Convert to a retriable failure with actionable LLM feedback.
        return False, (
            f"spike timed out after {timeout:.0f} s — kernel is too slow to "
            f"verify at this input scale (likely scalar code). "
            f"For rvv_f16 at 2048×2048 use RVV widening intrinsics "
            f"(vfwmacc_vv_f32m*, vfwmul_vv_f32m*) with LMUL≥4 so the "
            f"inner loop processes ≥32 fp16 elements per vector instruction. "
            f"Scalar triple-loop: ~860 s on spike; vectorized: ~172 s. "
            f"cmd: {' '.join(cmd)}"
        )
    out = proc.stdout + proc.stderr
    # Accept either the legacy AGENTS_OUTPUT_BEGIN marker or the modern
    # AGENTS_VERIFY summary as proof-of-life. Both are emitted by the
    # harness — the verify replaces the per-element output dump for
    # speed but the legacy path is still present in older binaries.
    if (BEGIN not in out) and ("=== AGENTS_VERIFY " not in out):
        return False, (
            f"spike output missing markers. cmd: {' '.join(cmd)}\n"
            f"--- output (tail) ---\n{out[-2000:]}"
        )
    return True, out


def build_and_run(
    impls: dict[str, str],
    specs: list[KernelSpec],
    *,
    backend: Backend,
    model_dir: str,
    build_dir: str,
    repo_root: str,
    harness_dir: str,
    pristine: bool = False,
    timeout: float = 60.0,
    io_path: Optional[str] = None,
    # Tolerance for the end-to-end model-output check inside this builder.
    # None (default) → autoselect from the io.npz golden's dtype:
    #   * fp32 → atol=1e-5, rtol=1e-4 (microkernel sweep — same as
    #     spike_runner._check_one's default for f32)
    #   * fp16 → atol=1e-2, rtol=1e-2 (fp16 has ~3.3 sig digits; conv2d
    #     reduction-style kernels naturally produce ~1e-3 absolute error
    #     under valid fp32-accumulator + fp16-cast contracts)
    # Tightening these would risk failing valid candidates; loosening
    # would let drift accumulate across composed ops. The split mirrors
    # runner_common.check_one's autoselect — same logic, same place.
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    model_name: Optional[str] = None,
) -> HarnessResult:
    """Build the harness with `impls` substituted in, run spike, parse profile
    and (optionally) compare model output to a PyTorch golden.

    Raises ProfileError on build / spike / parse failure. Numerical mismatch
    against a provided golden does NOT raise — it's recorded on the returned
    HarnessResult so callers can decide what to do (verify-fail vs warn).
    """
    # If model_name not passed, peek at graph.json in the parent IR dir.
    # Path layout: model_dir = <ir_dir>/<target>/, graph.json lives at
    # <ir_dir>/graph.json. This makes every existing call site automatically
    # get the right kernel-mangling without having to thread model_name
    # explicitly.
    if model_name is None:
        for candidate in (
            os.path.join(model_dir, "graph.json"),
            os.path.join(os.path.dirname(model_dir), "graph.json"),
        ):
            if os.path.exists(candidate):
                import json as _json
                with open(candidate) as _f:
                    model_name = _json.load(_f).get("name")
                break
    _emit_kernels_files(impls, specs, backend, model_dir, model_name=model_name)

    ok, err = _west_build(
        harness_dir=harness_dir, build_dir=build_dir, model_dir=model_dir,
        backend=backend, pristine=pristine, repo_root=repo_root,
    )
    if not ok:
        raise ProfileError(err)

    elf = os.path.join(build_dir, "zephyr", "zephyr.elf")
    if not os.path.exists(elf):
        raise ProfileError(f"zephyr.elf not produced at {elf}")

    ok, out = _spike_run(elf, backend, timeout)
    if not ok:
        raise ProfileError(out)

    profile = parse_profile(out) or []
    cycles_by_op: dict[str, int] = {}
    for row in profile:
        cycles_by_op[row["op"]] = cycles_by_op.get(row["op"], 0) + int(row["cycles"])

    # Modern harness emits a single AGENTS_VERIFY summary line; legacy
    # binaries still ship the per-element AGENTS_OUTPUT block. Prefer
    # the summary when present (saves shipping the full tensor over
    # the slow HTIF UART on FireSim). `actual` is left unset when the
    # summary is used since callers don't need per-element values.
    verify = parse_verify(out)
    actual = None if verify is not None else parse_output(out)
    result = HarnessResult(
        cycles_by_op=cycles_by_op,
        output=actual if actual is not None else np.empty(0, dtype=np.float32),
        raw_stdout=out,
    )

    if io_path:
        import math as _math
        import json as _json
        raw_golden = np.load(io_path)["output"]
        # Autoselect tolerances from golden dtype when not explicitly set.
        # For fp32, scale atol with sqrt(max_K) from sibling graph.json so
        # large-N matmul/reduction kernels are not falsely rejected for
        # expected f32 accumulation error.
        is_fp16 = raw_golden.dtype == np.float16
        def _adaptive_atol(base: float, scale: float) -> float:
            _graph_path = os.path.join(os.path.dirname(io_path), "graph.json")
            if os.path.exists(_graph_path):
                try:
                    with open(_graph_path) as _gf:
                        _ir = _json.load(_gf)
                    _max_k = max(
                        (op.get("shape", {}).get("K", 1)
                         for op in _ir.get("ops", [])),
                        default=1,
                    )
                    return max(base, scale * _math.sqrt(_max_k))
                except Exception:
                    pass
            return base
        if atol is not None:
            eff_atol = atol
        elif is_fp16:
            eff_atol = _adaptive_atol(1e-2, 1e-3)
        else:
            eff_atol = _adaptive_atol(1e-5, 1e-4)
        eff_rtol = rtol if rtol is not None else (1e-2 if is_fp16 else 1e-4)
        n_golden = int(np.asarray(raw_golden).reshape(-1).size)

        if verify is not None:
            # In-binary summary path. Pass condition matches numpy's
            # allclose semantics conservatively: every element satisfies
            # at least one of (abs <= atol) or (rel <= rtol) iff the
            # global max of either bound holds.
            if verify["n"] != n_golden:
                result.golden_ok = False
                result.golden_max_abs_err = float("inf")
            else:
                result.golden_max_abs_err = verify["max_abs_err"]
                result.golden_max_rel_err = verify["max_rel_err"]
                result.golden_ok = bool(
                    verify["max_abs_err"] <= eff_atol
                    or verify["max_rel_err"] <= eff_rtol
                )
        else:
            # Legacy per-element path. Identical to before this branch
            # was added; kept for older harness binaries / int8 paths
            # that haven't moved to AGENTS_VERIFY yet.
            golden = raw_golden.astype(np.float32).reshape(-1)
            if actual is None or actual.shape != golden.shape:
                result.golden_ok = False
                result.golden_max_abs_err = float("inf")
            else:
                abs_err = np.abs(actual - golden)
                denom = np.maximum(np.abs(golden), 1e-12)
                rel_err = abs_err / denom
                result.golden_max_abs_err = float(abs_err.max())
                result.golden_max_rel_err = float(rel_err.max())
                result.golden_ok = bool(
                    np.all(abs_err <= eff_atol + eff_rtol * np.abs(golden))
                )

    return result


def build_and_profile(
    impls: dict[str, str],
    specs: list[KernelSpec],
    *,
    backend: Backend,
    model_dir: str,
    build_dir: str,
    repo_root: str,
    harness_dir: str,
    pristine: bool = False,
    timeout: float = 60.0,
) -> dict[str, int]:
    """Backwards-compatible wrapper: just return cycles_by_op."""
    res = build_and_run(
        impls, specs,
        backend=backend, model_dir=model_dir, build_dir=build_dir,
        repo_root=repo_root, harness_dir=harness_dir,
        pristine=pristine, timeout=timeout,
    )
    return res.cycles_by_op
