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
    kernel_cflags=(
        "-march=rv64gcv_zfh_zvfh",
        "-mabi=lp64d",
        # Same contract as the plain rvv backend: conv2d_s8's
        # target-affined algorithms declare weight_layout="ihwoc", so the
        # skeleton packs conv2d_s8 weights IHWOC on this backend too and
        # the universal (non-affined) reference impls must be told. Note
        # the flag is only read by the ops whose reference_impl tests it
        # (conv2d_s8 / conv2d_silu_s8 / conv2d_pool_s8) -- the fp16 conv
        # ops declare weight_layout="oihw" and are packed OIHW, which is
        # why the packing is resolved per-op in generate_skeleton rather
        # than once per backend.
        "-DMODELBLASTER_RVV_IHWOC_WEIGHTS=1",
    ),
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
        "-isystem<repo_root>/cores/saturn_opu/include",
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
        "-isystem<repo_root>/cores/gemmini/include",
        "-isystem<repo_root>/cores/gemmini",
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
#
# SPLIT (2026-08-28, correctness campaign): this used to be ONE Backend
# that quietly turned into a fused Gemmini+Saturn(RVV) target the moment
# `v` was added to kernel_cflags — every "gemmini_q31" number on record
# up to that point (5.222x, 3,411,130 cycles, 155x-off-baseline) was
# really "Gemmini+RVV vs RVV", not a statement about Gemmini alone. It is
# now two backends:
#   - GEMMINI_Q31      — PURE Gemmini. rv64imafdc, no `v`. Gemmini kernels
#                        only for the ops Gemmini actually covers
#                        correctly (conv2d_s8, add_s8, linear_s8); every
#                        other op is scalar reference. This is the honest
#                        "what does Gemmini alone do" measurement,
#                        coverage gap included.
#   - GEMMINI_Q31_RVV  — the fused target. Same Gemmini kernels for
#                        conv2d_s8/add_s8/linear_s8 PLUS `v` + the RVV
#                        kernels for the ops Gemmini doesn't (efficiently)
#                        cover. All the previously-reported wins live
#                        here now, correctly labeled.
# Both targets pick ONLY bit-exact algorithms for conv2d_s8/maxpool2d_s8/
# relu_s8 — see kernels/gemmini_q31/archive/ for the two kernels
# (gemmini_tiled_conv_pool, gemmini_resadd_relu) that turned out to
# declare accuracy_class=bit_exact and are not (experiments/
# kernel_opt_log.jsonl ids 1100-1108).
GEMMINI_Q31 = Backend(
    name="gemmini_q31",
    description=(
        "PURE Gemmini on the Q0.31 acc_scale bitstream variant: no `v`, "
        "no RVV fallback kernels. Gemmini kernels for conv2d_s8 "
        "(gemmini_im2col_full_C — bit-exact; gemmini_tiled_conv is NOT "
        "offered here, see its header), add_s8 and linear_s8; every "
        "other op (batchnorm2d_s8, maxpool2d_s8, relu_s8, sigmoid_s8) is "
        "scalar reference_impl. Used after chipyard rebuilds with "
        "Q31GemminiRocketConfig. For the fused Gemmini+RVV target, see "
        "GEMMINI_Q31_RVV below."
    ),
    kernel_cflags=GEMMINI.kernel_cflags + (
        # Tells gemmini_conv2d_s8_gemmini_tiled_conv.c to fold (mult, shift)
        # into a single Q0.31 scale instead of computing a float scale via
        # ldexpf. Required when acc_scale_t is int32 (Q31 gemmini config).
        # (gemmini_tiled_conv isn't actually selected by this target's
        # curated dir any more — see its header for why — but the define
        # is harmless to carry and keeps the TU consistent if it's ever
        # re-offered.)
        "-DMODELBLASTER_GEMMINI_Q31_ACC_SCALE=1",
    ),
    kernel_includes=GEMMINI.kernel_includes,
    prj_conf_overlay=GEMMINI.prj_conf_overlay,
    spike_args=GEMMINI.spike_args,
    optimization_guide=GEMMINI.optimization_guide,
    verify_method=GEMMINI.verify_method,
    # Both conv2d_s8 (gemmini_im2col_full_C) and add_s8/linear_s8
    # (gemmini_resadd / gemmini_tiled_matmul, both verified bit-exact in
    # isolation, kernel_opt_log id 1102/1103) are now exact, and every
    # other op is scalar reference (exact by construction). Tight atol —
    # this target should verify at max_abs_err=0.
    atol_override=1.0,
    rtol_override=0.0,
)


# Fused Gemmini+Saturn(RVV) target: same acc_scale_t=int32 Q0.31 bitstream
# as GEMMINI_Q31, but with `v` added so ops Gemmini doesn't (efficiently)
# cover fall back to the proven RVV kernels instead of scalar. This is
# where every previously-reported "gemmini_q31" speed number now lives —
# see the SPLIT note above GEMMINI_Q31 for why it moved.
#
# kernel_cflags: unlike GEMMINI, this is NOT rv64imafdc — see exp
# 310/312 (kernel_opt_log.jsonl). Every op with no Gemmini kernel
# (batchnorm2d_s8, and — by triage — maxpool2d_s8/relu_s8/linear_s8,
# which measure SLOWER on Gemmini than on RVV) was silently falling
# back to SCALAR, not RVV, because GEMMINI.kernel_cflags carries no
# `v`. exp 310 proved hart 1 (where main.c pins execution once
# CONFIG_RISCV_ISA_EXT_V is defined — see harness/src/main.c) still
# has a working Gemmini RoCC port on the SatGemDualSmall bitstream,
# so adding `v` does not strand the accelerator. `<repo_root>` is
# substituted the same way as GEMMINI's isystem paths.
GEMMINI_Q31_RVV = Backend(
    name="gemmini_q31_rvv",
    description=(
        "Fused Gemmini+Saturn(RVV) on the Q0.31 acc_scale bitstream: "
        "Gemmini kernels for conv2d_s8/add_s8, RVV kernels (kernels/rvv/"
        "*_direct.c, unmodified) for batchnorm2d_s8/maxpool2d_s8/"
        "relu_s8/linear_s8. All of it verified bit-exact (kernel_opt_log "
        "id 1105+); this is the fast headline configuration."
    ),
    kernel_cflags=tuple(
        "-march=rv64imafdcv" if f == "-march=rv64imafdc" else f
        for f in GEMMINI.kernel_cflags
    ) + (
        "-DMODELBLASTER_GEMMINI_Q31_ACC_SCALE=1",
    ),
    # + riscv_vector.h so the RVV fallback kernels (batchnorm2d_s8,
    # maxpool2d_s8, relu_s8, linear_s8 — kernels/gemmini_q31_rvv/)
    # compile alongside gemmini.h's RoCC macros in the same kernels.c TU.
    kernel_includes=GEMMINI.kernel_includes + ("<riscv_vector.h>",),
    # gemmini_q31.conf carries the rvv.conf V stanza (EAGER context
    # switch + CONFIG_RISCV_V_KERNEL_ONLY) on top of GEMMINI's overlay.
    # NOTE (same caveat as GEMMINI.prj_conf_overlay elsewhere in this
    # file): this field is not read by examples/_run_lib.sh's firesim
    # path today — that path selects harness/backends/
    # firesim_chipyard_dual_gemmini_q31.conf directly by GEN_TARGET name
    # (dual_gemmini.conf's SMP/UART stanza + this same V stanza). Kept
    # here for documentation / future wiring.
    prj_conf_overlay="gemmini_q31.conf",
    # `v` added to the isa string so RUNNER=spike / curated_verify's
    # nested spike-harness builds can decode the RVV fallback kernels'
    # intrinsics alongside --extension=gemmini's RoCC decode. Spike
    # itself is single-hart-with-everything (unlike the FPGA's hart-0
    # lacks V / hart-1 has V split), so this combination is expected to
    # "just work", but as of this change it had not been separately
    # regression-tested against a V-containing curated kernel — only
    # against the FPGA hardware run (see kernel_opt_log.jsonl id 800+).
    spike_args=("--extension=gemmini", "--isa=rv64gcv_zicntr"),
    optimization_guide=GEMMINI.optimization_guide,
    verify_method=GEMMINI.verify_method,
    # conv2d_s8 now uses gemmini_im2col_full_C (RVV-vectorized, exact)
    # instead of the drifting gemmini_tiled_conv — see kernels/
    # gemmini_q31_rvv/gemmini_q31_rvv_conv2d_s8_gemmini_im2col_full_C.c.
    # Every op in this target's curated set (kernels/gemmini_q31_rvv/)
    # is bit-exact; tight atol.
    atol_override=1.0,
    rtol_override=0.0,
)


# RVV_HETERO: same kernel codegen as plain RVV (kernels/rvv/*, no
# Saturn OP-V custom instructions) but with a kernel-only-V Kconfig
# overlay so the binary BOOTS on a heterogeneous bitstream where hart 0
# doesn't have V (the GemminiAndOPUShuttleConfig case). Used to test
# Dima's qrb-image kernel set on the existing OPU FireSim bitstream
# without rebuilding the bitstream.
RVV_HETERO = Backend(
    name="rvv_hetero",
    description=(
        "Plain RVV (rv64gcv) kernels with rvv_opu-style kernel-only V "
        "Kconfig overlay. For HETEROGENEOUS bitstreams where hart 0 lacks "
        "V and would otherwise trap on Zephyr's V context init."
    ),
    kernel_cflags=RVV.kernel_cflags,
    kernel_includes=RVV.kernel_includes,
    prj_conf_overlay="rvv_hetero.conf",
    spike_args=RVV.spike_args,
    optimization_guide=RVV.optimization_guide,
    verify_method=RVV.verify_method,
)


BACKENDS: dict[str, Backend] = {
    SCALAR.name: SCALAR,
    RVV.name: RVV,
    RVV_HETERO.name: RVV_HETERO,
    SCALAR_F16.name: SCALAR_F16,
    RVV_F16.name: RVV_F16,
    RVV_OPU.name: RVV_OPU,
    GEMMINI.name: GEMMINI,
    GEMMINI_Q31.name: GEMMINI_Q31,
    GEMMINI_Q31_RVV.name: GEMMINI_Q31_RVV,
}


def get(name: str) -> Backend:
    if name not in BACKENDS:
        raise SystemExit(
            f"unknown target backend: {name}. "
            f"Available: {sorted(BACKENDS)}"
        )
    return BACKENDS[name]
