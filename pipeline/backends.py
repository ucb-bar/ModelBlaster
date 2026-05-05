"""HW-target backend registry.

Each Backend is a self-contained description of how to build, run, and
verify kernels for a particular RISC-V target variant. Adding a new target
(gemmini, rocc accelerator, custom ISA extension) means dropping a new
Backend entry here plus a `<name>.conf` and `optimization_guide_<name>.md`.

The Backend object intentionally stays declarative — the orchestrators
(generate_kernels, profile_kernel) read its fields and route accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# Verify routing.
#   "host_ctypes": compile candidate as a host .so (x86) and call via ctypes
#                  for per-shape numerical compare against the reference. Fast.
#   "spike_harness": cross-compile the candidate into the full Zephyr harness,
#                  run on spike, compare model output to the PyTorch golden.
#                  Required for backends that use target-specific intrinsics
#                  (RVV, custom extensions) which the host toolchain can't
#                  build or run.
VERIFY_HOST_CTYPES = "host_ctypes"
VERIFY_SPIKE_HARNESS = "spike_harness"


@dataclass(frozen=True)
class Backend:
    name: str
    description: str
    # Extra C compile flags applied ONLY to kernels.c (so we don't perturb
    # Zephyr's own build). Empty for scalar (matches the toolchain default).
    kernel_cflags: tuple[str, ...] = ()
    # Headers prepended to kernels.c above `#include "kernels.h"`.
    kernel_includes: tuple[str, ...] = ()
    # Per-backend prj.conf overlay file under agents/harness/backends/.
    # Conventionally named "<name>.conf". Empty string means "no overlay".
    prj_conf_overlay: str = ""
    # Args appended to the spike command line (e.g. --isa=rv64gcv_zicntr).
    spike_args: tuple[str, ...] = ()
    # Optimization guide markdown file under agents/pipeline/prompts/.
    optimization_guide: str = "optimization_guide_scalar.md"
    # How verify is performed.
    verify_method: str = VERIFY_HOST_CTYPES

    def resolved_kernel_cflags(self, repo_root: str) -> tuple[str, ...]:
        """kernel_cflags with `<repo_root>` placeholders substituted.

        Used for backends that need an -isystem / -I path into the
        vendored driver headers under agents/cores/<backend>/include/
        — gemmini's the first one. The placeholder is intentional
        (over hardcoding `${REPO_ROOT}`-prefixed paths) so the
        Backend definition stays repo-relative and serializable.
        """
        return tuple(f.replace("<repo_root>", repo_root)
                     for f in self.kernel_cflags)


SCALAR = Backend(
    name="scalar",
    description="rv64imafdc scalar baseline. Host-verifiable.",
    optimization_guide="optimization_guide_scalar.md",
    verify_method=VERIFY_HOST_CTYPES,
    prj_conf_overlay="scalar.conf",
)


RVV = Backend(
    name="rvv",
    description="rv64gcv with vector extension intrinsics (riscv_vector.h).",
    kernel_cflags=("-march=rv64gcv", "-mabi=lp64d"),
    kernel_includes=("<riscv_vector.h>",),
    prj_conf_overlay="rvv.conf",
    spike_args=("--isa=rv64gcv_zicntr",),
    optimization_guide="optimization_guide_rvv.md",
    verify_method=VERIFY_SPIKE_HARNESS,
)


# fp16 variants: same shape as scalar/rvv but with Zfh (and Zvfh on rvv)
# enabled. The Kconfig overlay (scalar_f16.conf / rvv_f16.conf) tells
# Zephyr to context-switch the half-precision register state and tells
# target_riscv.cmake to append _zfh / _zvfh to the global -march. For
# rvv we additionally pin kernel_cflags to a march that includes the
# vector half-prec extension — the per-source -march would otherwise
# preempt the Kconfig-driven global -march and kernels.c would lose
# Zvfh.
SCALAR_F16 = Backend(
    name="scalar_f16",
    description="rv64imafdc + Zfh scalar half-precision.",
    optimization_guide="optimization_guide_scalar.md",
    verify_method=VERIFY_HOST_CTYPES,
    prj_conf_overlay="scalar_f16.conf",
    # Default spike ISA is rv64gc (no Zfh) — fadd.h / fcvt.h.s would
    # fault as illegal instruction. Pass an explicit isa so the
    # simulator decodes Zfh.
    spike_args=("--isa=rv64gc_zicntr_zfh",),
)


RVV_F16 = Backend(
    name="rvv_f16",
    description="rv64gcv + Zfh + Zvfh (vector half-precision).",
    kernel_cflags=("-march=rv64gcv_zfh_zvfh", "-mabi=lp64d"),
    kernel_includes=("<riscv_vector.h>",),
    prj_conf_overlay="rvv_f16.conf",
    spike_args=("--isa=rv64gcv_zicntr_zfh_zvfh",),
    optimization_guide="optimization_guide_rvv.md",
    verify_method=VERIFY_SPIKE_HARNESS,
)


# Gemmini integer accelerator (chipyard's default int8 RoCC config —
# DIM=16, elem_t=int8, acc_t=int32). Stage 1 is RoCC-only; ReRoCC
# variants will be added later behind separate target names.
#
# kernel_cflags: rv64gc_zicntr (no V — gemmini ops are custom RoCC
# instructions, not vector). The -isystem points at the vendored
# headers under agents/cores/gemmini/include/. -DGEMMINI_ROCC tells
# the gemmini.h header to take the RoCC code paths (vs ReRoCC).
#
# CMake substitutes `<repo_root>` at build time when injecting these
# into the kernel TU's COMPILE_OPTIONS — see harness CMakeLists.txt
# AGENTS_KERNEL_CFLAGS handling.
GEMMINI = Backend(
    name="gemmini",
    description=(
        "rv64imafdc + Gemmini int8 RoCC accelerator (chipyard default "
        "config, DIM=16). Tiled int8 conv/matmul via tiled_conv_auto / "
        "tiled_matmul_auto."
    ),
    kernel_cflags=(
        "-march=rv64imafdc",
        "-mabi=lp64d",
        # Two include paths needed:
        #   .../include — so kernels.c's `#include "gemmini.h"` resolves
        #   .../        — so gemmini.h's `#include "include/gemmini_params.h"`
        #                 and `#include "rocc-software/src/xcustom.h"` resolve
        # The asymmetric layout is gemmini-rocc-tests' upstream convention.
        "-isystem<repo_root>/agents/cores/gemmini/include",
        "-isystem<repo_root>/agents/cores/gemmini",
        "-DGEMMINI_ROCC",
        "-DBAREMETAL",
    ),
    kernel_includes=("\"gemmini.h\"",),
    prj_conf_overlay="gemmini.conf",
    # Chipyard's spike with --extension=gemmini decodes the custom RoCC
    # opcode (XCUSTOM_ACC=3) and routes the instructions through
    # libgemmini.so. The agents-flow spike binary doesn't ship that
    # extension; spike_runner.py picks the chipyard spike via
    # AGENTS_GEMMINI_SPIKE env when this backend is selected.
    spike_args=("--extension=gemmini", "--isa=rv64gc_zicntr"),
    optimization_guide="optimization_guide_scalar.md",
    verify_method=VERIFY_SPIKE_HARNESS,
)


BACKENDS: dict[str, Backend] = {
    SCALAR.name: SCALAR,
    RVV.name: RVV,
    SCALAR_F16.name: SCALAR_F16,
    RVV_F16.name: RVV_F16,
    GEMMINI.name: GEMMINI,
}


def get(name: str) -> Backend:
    if name not in BACKENDS:
        raise SystemExit(
            f"unknown target backend: {name}. "
            f"Available: {sorted(BACKENDS)}"
        )
    return BACKENDS[name]
