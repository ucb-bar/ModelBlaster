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
    # Per-backend prj.conf overlay file under modelblaster/harness/backends/.
    # Conventionally named "<name>.conf". Empty string means "no overlay".
    prj_conf_overlay: str = ""
    # Args appended to the spike command line (e.g. --isa=rv64gcv_zicntr).
    spike_args: tuple[str, ...] = ()
    # Optimization guide markdown file under modelblaster/pipeline/prompts/.
    optimization_guide: str = "optimization_guide_scalar.md"
    # How verify is performed.
    verify_method: str = VERIFY_HOST_CTYPES
    # Per-backend verify-tolerance overrides for the spike-harness end-to-
    # end model golden compare. None means "use the dtype-derived default
    # from runner_common._select_tolerance". Set non-None when this backend
    # produces values that differ from the PyTorch golden by more than the
    # default tolerance for *known structural reasons* — e.g. gemmini's
    # float-scale requantize differs from the PyTorch Q0.31 reference by
    # ~1-3 LSBs per int8 output. Used by generate_kernels._verify and
    # build_and_run.
    atol_override: float | None = None
    rtol_override: float | None = None

    def resolved_kernel_cflags(self, repo_root: str) -> tuple[str, ...]:
        """kernel_cflags with `<repo_root>` placeholders substituted.

        Used for backends that need an -isystem / -I path into the
        vendored driver headers under modelblaster/cores/<backend>/include/
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
    kernel_cflags=(
        "-march=rv64gcv",
        "-mabi=lp64d",
        # Tells universal (non-target-affined) algorithms that conv2d
        # weights are IHWOC-packed. Target-specific RVV algorithms
        # unconditionally assume IHWOC (declared via weight_layout field).
        "-DMODELBLASTER_RVV_IHWOC_WEIGHTS=1",
    ),
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


# Saturn OPU (Outer Product Unit) — integer matrix MAC engine layered on
# top of the V opcode. Encoded as custom .insn r 0x57 ops (no extra
# -march extension needed); the actual decode happens in Saturn HW via
# the OuterProductSequencer / OuterProductUnit modules. See
#   modelblaster/cores/saturn_opu/include/saturn_opu.h
# for the asm-macro programming model and
#   modelblaster/notes/saturn_opu_backend.md
# for design notes and current status.
#
# Stage 1 is integer-only (i8 x i8 -> i32 accumulator via VOPACC). FP
# variants (opuMxParams / fp8 / fp16 OPU) will land behind separate
# backend names once that path is validated.
#
# Build requirements:
#   - Saturn bitstream / verilator with VectorParams.opuParams (e.g.
#     chipyard `REFV256D128DualRocketSaturnOPUGemmini32x32Q31WsConfig`
#     or one of the Shuttle-side `OPUV*ShuttleConfig` classes from
#     chipyard/OPUConfigs.scala on the saturn opu-fp8 branch).
#   - rv64gcv toolchain — no Zfh/Zvfh needed for integer OPU.
#
# Spike status: upstream spike does NOT decode the OPU custom
# instructions today. Building this backend's elf and running it on
# stock spike will trap as illegal instruction on the first VOPACC.
# The verify path is therefore FireSim-only (or a custom spike fork);
# the spike_args field stays empty as a load-bearing marker, and
# spike_runner.py / generate_kernels._verify treat empty spike_args +
# verify_method=VERIFY_SPIKE_HARNESS as "spike unsupported, skip verify"
# (TODO once that flag exists; until then, set BACKEND=reference + a
# curated kernel and run on FireSim directly via run.sh RUNNER=firesim).
RVV_OPU = Backend(
    name="rvv_opu",
    description=(
        "rv64gcv + Saturn OPU integer matrix engine (i8×i8→i32 via "
        "VOPACC custom .insn). Layered on the V extension; OPU custom "
        "instructions encoded as .insn r 0x57 + custom funct fields."
    ),
    kernel_cflags=(
        "-march=rv64gcv",
        "-mabi=lp64d",
        # Vendored OPU header location. Same `<repo_root>` placeholder
        # convention as gemmini; resolved_kernel_cflags() substitutes
        # at build time.
        "-isystem<repo_root>/modelblaster/cores/saturn_opu/include",
        # Marker for kernels that want to gate code on "OPU available".
        "-DMODELBLASTER_SATURN_OPU=1",
        # NOTE: deliberately does NOT carry MODELBLASTER_RVV_IHWOC_WEIGHTS
        # forward from the plain rvv backend. That flag tells
        # universal (non-target-affined) kernels to assume IHWOC
        # packing, but the skeleton only packs weights when an
        # rvv_opu-affined conv2d algorithm declares weight_layout=ihwoc.
        # Until that algorithm exists, weights stay OIHW and the
        # universal scalar fallback expects OIHW. When the first OPU
        # conv2d_s8 algorithm lands, decide whether OPU wants IHWOC
        # (probably yes, since OPU benefits from contiguous K-stride
        # input loads) and add the flag back then alongside the
        # AlgorithmCandidate.weight_layout = "ihwoc" declaration.
    ),
    kernel_includes=("<riscv_vector.h>", "\"saturn_opu.h\""),
    prj_conf_overlay="rvv_opu.conf",
    # Custom OPU-extension spike, built from
    # hw/chipyard/toolchains/riscv-tools/riscv-isa-sim/customext/saturn_opu.cc.
    # `--extension=saturn_opu` loads the libcustomext.so that registers
    # VOPACC / OPMVINBCAST / VMV_VR / VMV_RV decoders. _run_lib.sh
    # routes this backend at the OPU-built spike via MODELBLASTER_OPU_SPIKE
    # (mirror of MODELBLASTER_GEMMINI_SPIKE).
    spike_args=("--extension=saturn_opu", "--isa=rv64gcv_zicntr"),
    optimization_guide="optimization_guide_rvv.md",
    verify_method=VERIFY_SPIKE_HARNESS,
)


# Gemmini integer accelerator (chipyard's default int8 RoCC config —
# DIM=16, elem_t=int8, acc_t=int32). Stage 1 is RoCC-only; ReRoCC
# variants will be added later behind separate target names.
#
# kernel_cflags: rv64gc_zicntr (no V — gemmini ops are custom RoCC
# instructions, not vector). The -isystem points at the vendored
# headers under modelblaster/cores/gemmini/include/. -DGEMMINI_ROCC tells
# the gemmini.h header to take the RoCC code paths (vs ReRoCC).
#
# CMake substitutes `<repo_root>` at build time when injecting these
# into the kernel TU's COMPILE_OPTIONS — see harness CMakeLists.txt
# MODELBLASTER_KERNEL_CFLAGS handling.
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
        "-isystem<repo_root>/modelblaster/cores/gemmini/include",
        "-isystem<repo_root>/modelblaster/cores/gemmini",
        "-DGEMMINI_ROCC",
        "-DBAREMETAL",
        # Tells gemmini-target kernels — including the scalar reference
        # fallback — that conv2d weights have been pre-packed to flat
        # HWIO ([KH*KW*IC, OC]) at codegen time by
        # generate_skeleton.py::_backend_pack_weight. Without this
        # define, kernels read OIHW and produce garbage when handed an
        # HWIO blob. Other backends (scalar/rvv) leave weights in OIHW,
        # so the define is gemmini-only.
        "-DMODELBLASTER_GEMMINI_HWIO_WEIGHTS=1",
    ),
    kernel_includes=("\"gemmini.h\"",),
    prj_conf_overlay="gemmini.conf",
    # Chipyard's spike with --extension=gemmini decodes the custom RoCC
    # opcode (XCUSTOM_ACC=3) and routes the instructions through
    # libgemmini.so. The modelblaster-flow spike binary doesn't ship that
    # extension; spike_runner.py picks the chipyard spike via
    # MODELBLASTER_GEMMINI_SPIKE env when this backend is selected.
    spike_args=("--extension=gemmini", "--isa=rv64gc_zicntr"),
    optimization_guide="optimization_guide_scalar.md",
    verify_method=VERIFY_SPIKE_HARNESS,
    # The gemmini_im2col_full_C algorithm (validated May 2026) drains raw
    # int32 accumulators and applies Q0.31 requantize in scalar — bit-exact
    # with the PyTorch golden (max_abs_err=0 on Saturn RTL FireSim).
    # The legacy gemmini_tiled_conv algorithm uses float-scale mvout and
    # drifts ~6 LSBs through dronet's 9 conv layers; atol=8 covers that
    # path if it gets picked by the cache probe. See
    # modelblaster/notes/gemmini_extension_plan.md "Stage 1.5" section.
    atol_override=8.0,
    rtol_override=0.0,
)


# Variant of GEMMINI that targets the bit-exact Q0.31 mvout requantize path.
# Picked when running on a Q31GemminiRocketConfig bitstream / verilator OR on
# chipyard spike with libgemmini.so.q31 swapped in for libgemmini.so. Flips
# acc_scale_t from f32 to int32 in the codegen and tells gemmini-target
# kernels to fold (output_multiplier, output_shift) into a single Q0.31
# scale. See gemmini_q31_acc_scale_validated memory entry for status.
#
# To use:
#   1. Replace modelblaster/cores/gemmini/include/gemmini_params.h with the
#      Q31-emitted header (modelblaster/cores/gemmini/include/gemmini_params_q31.h
#      or copy from chipyard verilog gen output).
#   2. cp libgemmini.so.q31 → libgemmini.so in the spike lib dir
#      (or set MODELBLASTER_GEMMINI_LIB_DIR to a dir that has libgemmini.so.q31
#      renamed to libgemmini.so).
#   3. TARGET=gemmini_q31 modelblaster/examples/<m>/run.sh ...
GEMMINI_Q31 = Backend(
    name="gemmini_q31",
    description=(
        "Same as gemmini but for the Q0.31 acc_scale bitstream variant: "
        "tiled_conv_auto's mvout requantize is bit-exact integer instead "
        "of f32. Used after chipyard rebuilds with Q31GemminiRocketConfig."
    ),
    kernel_cflags=GEMMINI.kernel_cflags + (
        # Tells gemmini_conv2d_s8_gemmini_tiled_conv.c to fold (mult, shift)
        # into a single Q0.31 scale instead of computing a float scale via
        # ldexpf. Required when acc_scale_t is int32 (Q31 gemmini config).
        "-DMODELBLASTER_GEMMINI_Q31_ACC_SCALE=1",
    ),
    kernel_includes=GEMMINI.kernel_includes,
    prj_conf_overlay=GEMMINI.prj_conf_overlay,
    spike_args=GEMMINI.spike_args,
    optimization_guide=GEMMINI.optimization_guide,
    verify_method=GEMMINI.verify_method,
    # Q0.31 fold introduces ≤1 LSB drift per layer (one rounded multiply
    # vs the two-stage TFLite formula). dronet (~30 layers) stays well
    # inside atol; yolov8 (~200 layers) accumulates to ~80 LSB. Bumping
    # to atol=128 lets gemmini_tiled_conv (HW im2col + GEMM in one call)
    # win verify on yolov8 too instead of falling back to the slower
    # gemmini_im2col_full_C (CPU im2col + scalar Q0.31). The drift is a
    # known-bounded mode (single-stage vs two-stage Q0.31 fold), not a
    # kernel bug; downstream model accuracy on int8 detection is robust
    # to it.
    atol_override=128.0,
    rtol_override=0.0,
)


BACKENDS: dict[str, Backend] = {
    SCALAR.name: SCALAR,
    RVV.name: RVV,
    SCALAR_F16.name: SCALAR_F16,
    RVV_F16.name: RVV_F16,
    RVV_OPU.name: RVV_OPU,
    GEMMINI.name: GEMMINI,
    GEMMINI_Q31.name: GEMMINI_Q31,
}


def get(name: str) -> Backend:
    if name not in BACKENDS:
        raise SystemExit(
            f"unknown target backend: {name}. "
            f"Available: {sorted(BACKENDS)}"
        )
    return BACKENDS[name]
