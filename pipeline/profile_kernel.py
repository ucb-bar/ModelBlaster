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
        # west/ninja put the actual gcc compile errors on stdout; only the
        # "FATAL ERROR: ..." final-status line lands on stderr. Show both
        # so the LLM retry sees the actual diagnostic, not just the
        # "command exited with status 1" wrapper.
        return False, (
            f"west build failed (rc={proc.returncode}):\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stdout (tail):\n{proc.stdout[-2500:]}\n"
            f"  stderr (tail):\n{proc.stderr[-1000:]}"
        )
    return True, ""


def _spike_run(elf: str, backend: Backend, timeout: float) -> tuple[bool, str]:
    spike = find_spike()
    cmd = [spike, *backend.spike_args, elf]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout
    )
    out = proc.stdout + proc.stderr
    if BEGIN not in out:
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
    # Match spike_runner._check_one's defaults (atol=1e-5, rtol=1e-4) so a
    # kernel that PASSes here is also safe to combine in run-level
    # validation. Looser tolerances here would let FP-reordering drift
    # accumulate across ops past the run-level gate.
    atol: float = 1e-5,
    rtol: float = 1e-4,
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

    actual = parse_output(out)
    result = HarnessResult(
        cycles_by_op=cycles_by_op,
        output=actual,
        raw_stdout=out,
    )

    if io_path:
        golden = np.load(io_path)["output"].astype(np.float32).reshape(-1)
        if actual.shape != golden.shape:
            result.golden_ok = False
            result.golden_max_abs_err = float("inf")
        else:
            abs_err = np.abs(actual - golden)
            denom = np.maximum(np.abs(golden), 1e-12)
            rel_err = abs_err / denom
            result.golden_max_abs_err = float(abs_err.max())
            result.golden_max_rel_err = float(rel_err.max())
            result.golden_ok = bool(
                np.all(abs_err <= atol + rtol * np.abs(golden))
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
