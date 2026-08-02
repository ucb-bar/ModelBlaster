#!/usr/bin/env bash
# Shared orchestration body for all examples.
#
# Caller responsibilities (set before sourcing/exec'ing this script):
#   MODEL_NAME   the model identifier passed to extract_graph (--model)
#   REPO_ROOT    repo root path; the script cd's into it
#
# Optional env vars (with defaults applied here):
#   BACKEND      reference (default) | llm
#   TARGET       scalar (default) | rvv
#   QUANT        fp32 (default)              # int8 etc. land later
#   OPTIMIZE     0 (default) | 1
#   ALGORITHMS   all (default) | comma list  # forwarded to generate_kernels
#   BEAM/EXPANSIONS/ITERATIONS               # optimize-loop knobs
#
# Layout (per-model, per-quant, per-target):
#   modelblaster/examples/<model>/<quant>/generated/             # IR (target-indep)
#   modelblaster/examples/<model>/<quant>/generated/<target>/    # generated C
#   modelblaster/examples/<model>/<quant>/build/<target>/        # west build
#   modelblaster/examples/<model>/<quant>/cache/<target>/        # kernel cache

set -euo pipefail

: "${MODEL_NAME:?MODEL_NAME must be set by the caller}"
: "${REPO_ROOT:?REPO_ROOT must be set by the caller}"

# Submodule-layout adaptation (LOCAL, uncommitted): ModelBlaster is upstreamed as
# a standalone repo, so this script resolves examples/harness/kernels relative to
# the ModelBlaster root. When embedded as a submodule of zephyr-chipyard-sw, the
# example run.sh scripts set REPO_ROOT to the *parent* (zephyr) repo, so retarget
# it to the nested modelblaster/ dir. No-op in a standalone checkout.
if [[ -d "${REPO_ROOT}/modelblaster/pipeline" ]]; then
    # The parent (zephyr) repo becomes the package root so `python -m
    # modelblaster.pipeline.*` resolves once we cd into the nested dir.
    export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    REPO_ROOT="${REPO_ROOT}/modelblaster"
fi

BACKEND="${BACKEND:-reference}"
TARGET="${TARGET:-scalar}"
QUANT="${QUANT:-fp32}"
OPTIMIZE="${OPTIMIZE:-0}"

# When quant=fp16, promote the target to its fp16-capable backend variant
# (e.g. rvv → rvv_f16) for stages that need zvfh/zfh compiler flags and
# spike ISA extensions. Directory layout still uses TARGET so fp32 and
# fp16 builds share the same paths under the quant-namespaced tree.
#
# For mixed-precision (QUANT=int8 with fp16 ops via get_precision_spec), the
# graph.json contains _f16 ops + cast_{i8_to_f16,f16_to_i8} nodes — we need
# Zfh/Zvfh too. Detected after extract by scanning the IR (deferred to the
# post-extract block below; the initial assignment honors only the static
# QUANT-based override).
GEN_TARGET="${TARGET}"
if [[ "${QUANT}" == "fp16" ]]; then
    GEN_TARGET="${TARGET}_f16"
fi

EXAMPLE_DIR_REL="examples/${MODEL_NAME}"
EXAMPLE_DIR="${REPO_ROOT}/${EXAMPLE_DIR_REL}"

cd "${REPO_ROOT}"
# Avoid the stale Vitis cmake on PATH (it's 3.3.2 and breaks west).
export PATH="/usr/bin:${PATH}"

# Per-stage timing markers. The benchmark harness's arm driver parses
# MODELBLASTER_STAGE_END:<name>:<seconds> lines out of stdout into
# stage_timings.json so the dashboard can show where wall-clock
# actually went (extract / skeleton / kernels / build / run).
_MB_STAGE_T=0
_mb_stage_begin() {
    _MB_STAGE_T="$(date +%s.%N)"
    echo "MODELBLASTER_STAGE_BEGIN:$1"
}
_mb_stage_end() {
    local end="$(date +%s.%N)"
    local delta
    delta=$(awk -v a="${_MB_STAGE_T}" -v b="${end}" \
            'BEGIN{printf "%.3f", b-a}')
    echo "MODELBLASTER_STAGE_END:$1:${delta}"
}

IR_DIR="${EXAMPLE_DIR}/${QUANT}/generated"
GEN_DIR="${IR_DIR}/${TARGET}"
# Two build dirs because the LLM verify path (inside generate_kernels)
# always invokes spike with board=spike_riscv64, while RUNNER=firesim's
# runtime build uses chipyard_riscv64. west refuses to mix board
# targets in the same build dir, so:
#   VERIFY_BUILD_DIR — always spike, used by the BACKEND=llm verify loop
#   BUILD_DIR        — runtime, suffixed _firesim when applicable
# For RUNNER=spike the two are the same (no suffix). cache/ is shared
# (kernel source isn't board-dependent).
VERIFY_BUILD_DIR="${EXAMPLE_DIR}/${QUANT}/build/${TARGET}"
BUILD_SUFFIX=""
if [[ "${RUNNER:-spike}" == "firesim" ]]; then
    BUILD_SUFFIX="_firesim"
fi
BUILD_DIR="${EXAMPLE_DIR}/${QUANT}/build/${TARGET}${BUILD_SUFFIX}"
CACHE_DIR="${EXAMPLE_DIR}/${QUANT}/cache/${TARGET}"
mkdir -p "${GEN_DIR}" "${BUILD_DIR%/*}" "${CACHE_DIR}"

echo "[1/5] extract_graph (quant=${QUANT}) -> ${IR_DIR}"
_mb_stage_begin extract
# Skip the PyTorch extract pass when the IR is already on disk. Useful
# when (a) the active env lacks the model's PyTorch deps (set up the IR
# in a different env first), or (b) iterating on later stages without
# re-running tracing. Set FORCE_EXTRACT=1 to override.
if [[ -f "${IR_DIR}/graph.json" && -f "${IR_DIR}/weights.npz" && -f "${IR_DIR}/io.npz" && "${FORCE_EXTRACT:-0}" != "1" ]]; then
    echo "  (skipped — IR present at ${IR_DIR}; set FORCE_EXTRACT=1 to re-run)"
elif [[ -n "${BENCH_FILE:-}" ]]; then
    # KernelBench mode: caller passes BENCH_FILE (a level1 .py path); route
    # through extract_graph's --bench-file loader instead of --model (the
    # kernelbench MODEL_NAME is a path like kernelbench/kb_<name>, not a
    # registered model).
    # Default sizing: cap baked io to BENCH_TARGET_MB MiB (256), shrinking
    # batch->spatial and protecting channels for good HW utilization. Set
    # BENCH_TARGET_MB=0 for stock dims. BENCH_MAX_ELEMENTS>0 is the LEGACY
    # tiny-element-cap override (kept for reproducing old results).
    # BENCH_TARGET_GFLOPS>0 bounds COMPUTE (forward FLOPs) instead of just io —
    # needed for large-K matmuls / 3D convs that have tiny io but huge FLOPs;
    # BENCH_TARGET_MB stays an io ceiling. Same knob as the ExecuTorch exporter
    # (gen_pte_kb_sized.py --target-gflops), so both flows size identically.
    python -m modelblaster.pipeline.extract_graph \
        --bench-file "${BENCH_FILE}" \
        --out-dir "${IR_DIR}" \
        --quant "${QUANT}" \
        --bench-target-mb "${BENCH_TARGET_MB:-256}" \
        --bench-target-gflops "${BENCH_TARGET_GFLOPS:-0}" \
        --bench-max-elements "${BENCH_MAX_ELEMENTS:-0}"
else
    python -m modelblaster.pipeline.extract_graph \
        --model "${MODEL_NAME}" \
        --out-dir "${IR_DIR}" \
        --quant "${QUANT}" \
        --num-calibration "${NUM_CALIBRATION:-1}"
fi
_mb_stage_end extract

# Post-extract auto-promote: if the IR contains any fp16 ops (mixed
# precision via get_precision_spec) and we haven't already promoted via
# QUANT=fp16, switch GEN_TARGET to the _f16 variant so generate_kernels
# / west build / spike all pull in Zfh + Zvfh.
#
# Only auto-promote from {rvv, scalar} — backends with explicit
# trailing-suffix variants (rvv_opu, rvv_f16, gemmini_q31, ...) need to
# be selected by the user. Auto-promoting rvv_opu→rvv_opu_f16 would
# silently route to a non-existent backend; instead, surface the
# unsupported combination as a hard error at the kernels stage so the
# user knows fp16 ops + that target aren't wired yet.
if [[ ("${TARGET}" == "rvv" || "${TARGET}" == "scalar") \
      && "${GEN_TARGET}" == "${TARGET}" \
      && -f "${IR_DIR}/graph.json" ]]; then
    _HAS_F16=$(python -c "
import json
g = json.load(open('${IR_DIR}/graph.json'))
print('1' if any('f16' in n['op'] for n in g.get('ops', [])) else '0')
" 2>/dev/null || echo "0")
    if [[ "${_HAS_F16}" == "1" ]]; then
        GEN_TARGET="${TARGET}_f16"
        echo "  (auto-promoted GEN_TARGET=${GEN_TARGET} — IR has fp16 ops)"
    fi
fi

echo "[2/5] generate_skeleton (backend=${GEN_TARGET}) -> ${GEN_DIR}"
_mb_stage_begin generate_skeleton
python -m modelblaster.pipeline.generate_skeleton \
    --ir "${IR_DIR}/graph.json" \
    --weights "${IR_DIR}/weights.npz" \
    --io "${IR_DIR}/io.npz" \
    --out-dir "${GEN_DIR}" \
    --backend "${GEN_TARGET}"
_mb_stage_end generate_skeleton

echo "[3/5] generate_kernels (backend=${BACKEND} target=${GEN_TARGET} quant=${QUANT} optimize=${OPTIMIZE}) -> ${GEN_DIR}"
GEN_KERNELS_ARGS=(
    --ir "${IR_DIR}/graph.json"
    --out-dir "${GEN_DIR}"
    --backend "${BACKEND}"
    --target "${GEN_TARGET}"
    --quant "${QUANT}"
    --io "${IR_DIR}/io.npz"
    --repo-root "${REPO_ROOT}"
    --build-dir "${VERIFY_BUILD_DIR}"
    --harness-dir "harness"
    --cache-dir "${CACHE_DIR}"
    --algorithms "${ALGORITHMS:-all}"
)
if [[ -n "${GLOBAL_CURATED_DIR:-}" ]]; then
    GEN_KERNELS_ARGS+=(--global-curated-dir "${GLOBAL_CURATED_DIR}")
fi
# MAX_ACCURACY_CLASS=bit_exact|numeric_drift|approximate restricts kernel
# selection to algorithms that meet at least the given accuracy class. Use
# bit_exact for golden-regression runs; default (unset) keeps the
# atol=8 envelope behavior.
if [[ -n "${MAX_ACCURACY_CLASS:-}" ]]; then
    GEN_KERNELS_ARGS+=(--max-accuracy-class "${MAX_ACCURACY_CLASS}")
fi
if [[ "${OPTIMIZE}" == "1" ]]; then
    GEN_KERNELS_ARGS+=(
        --optimize
        --beam "${BEAM:-2}"
        --expansions "${EXPANSIONS:-3}"
        --iterations "${ITERATIONS:-2}"
    )
    # Memory-aware optimize knobs. Both default off — set FIRESIM_EVAL=1
    # to re-rank top-K spike survivors on the FireSim FPGA and promote
    # the firesim-best to cache. Pair with CACHE_AWARE_PROMPT=1 to also
    # splice the target's memory-hierarchy stanza into the LLM optimize
    # prompt. FIRESIM_OPS is a comma-list to limit re-rank to specific
    # ops (e.g. "conv2d,linear") and skip cheap elementwise ops that
    # don't benefit.
    if [[ "${FIRESIM_EVAL:-0}" == "1" ]]; then
        GEN_KERNELS_ARGS+=(
            --firesim-eval
            --firesim-top-k "${FIRESIM_TOP_K:-3}"
        )
        if [[ -n "${FIRESIM_OPS:-}" ]]; then
            GEN_KERNELS_ARGS+=(--firesim-ops "${FIRESIM_OPS}")
        fi
    fi
    if [[ "${CACHE_AWARE_PROMPT:-0}" == "1" ]]; then
        GEN_KERNELS_ARGS+=(--cache-aware-prompt)
    fi
fi
_mb_stage_begin generate_kernels
python -m modelblaster.pipeline.generate_kernels "${GEN_KERNELS_ARGS[@]}"
_mb_stage_end generate_kernels

# STAGE_ONLY=1 stops after codegen (extract/skeleton/kernels), skipping the
# build+run — used by the multi-model wrapper to stage each bench's
# generated/<target>/ dir before fusing them into one ELF.
if [[ "${STAGE_ONLY:-0}" == "1" ]]; then
    echo "[stage-only] generated -> ${GEN_DIR}"
    return 0 2>/dev/null || exit 0
fi

# RUNNER selects the simulator behind stages 4-5: spike (default; in-process
# spike subprocess) or firesim (build for chipyard_riscv64, copy elf into
# the FireSim sim slot, runworkload, tail uartlog). The build (4/5) and
# run (5/5) split is identical across runners — only the board name and
# the verifier differ.
RUNNER="${RUNNER:-spike}"
case "${RUNNER}" in
    spike)
        BOARD_TARGET="spike_riscv64"
        ;;
    native)
        # Zephyr native_sim: build the harness as a plain x86-64 host binary.
        # Runs at native speed with host-backed memory, so it validates the
        # reference kernels at FULL stock dimensions (spike's 256 MB Zephyr RAM
        # region can't hold them). Requires the host toolchain (not the RISC-V
        # SDK) and a scalar target (native x86 can't use RVV) — the reference
        # kernel math is identical across targets, so this checks the same code.
        BOARD_TARGET="native_sim/native/64"
        export ZEPHYR_TOOLCHAIN_VARIANT=host
        if [[ "${GEN_TARGET}" == rvv* ]]; then
            echo "ERROR: RUNNER=native needs a scalar TARGET (got ${GEN_TARGET}); " \
                 "set TARGET=scalar" >&2
            exit 1
        fi
        ;;
    firesim)
        # Chipyard's quad-rocket-saturn board target. Pulls in the
        # firesim_chipyard.conf overlay (shrunk stack + SMP knobs that
        # the working FireSim Zephyr samples use) so Zephyr boots on
        # the FPGA — the spike-only prj.conf hangs pre-banner there.
        BOARD_TARGET="chipyard_riscv64/rocketchip_virt_riscv64"
        ;;
    *)
        echo "ERROR: unsupported RUNNER=${RUNNER} (expected spike|firesim)" >&2
        exit 1
        ;;
esac

echo "[4/5] west build (board=${BOARD_TARGET}) -> ${BUILD_DIR}"
KERNEL_CFLAGS=$(python -c "
from modelblaster.pipeline.backends import get
b = get('${GEN_TARGET}')
print(';'.join(b.resolved_kernel_cflags('${REPO_ROOT}')))
")
if [[ "${RUNNER}" == "native" ]]; then
    # The scalar backend ships no kernel cflags, so the reference kernels would
    # compile unoptimized — far too slow at stock KernelBench dims (e.g. a
    # 2048^3 matmul). Add -O2 (no -ffast-math, so fp semantics are unchanged).
    KERNEL_CFLAGS="${KERNEL_CFLAGS:+${KERNEL_CFLAGS};}-O2"
fi
WEST_CMAKE_ARGS=(
    -DMODEL_DIR="${GEN_DIR}"
    -DMODELBLASTER_BACKEND="${GEN_TARGET}"
)
if [[ -n "${KERNEL_CFLAGS}" ]]; then
    WEST_CMAKE_ARGS+=(-DMODELBLASTER_KERNEL_CFLAGS="${KERNEL_CFLAGS}")
fi
WEST_BUILD_EXTRA=()
# Auto-size the Zephyr ram0 region from the actual baked-io footprint so the
# default (256 MB target) sizing runs on-target without anyone remembering a
# manual bump — for BOTH spike AND firesim (the stock 256 MB ram0 =
# dts/.../ram0 0x10000000 is a config, not a hw limit). ram0 ~= rodata(io) +
# working buffers + code/stack. SPIKE_RAM_SIZE (hex bytes) overrides the derived
# value; AUTO_RAM0=0 disables. Caps: firesim modeled DRAM (FIRESIM_DRAM_MB,
# default 1024) or spike single-cell 4 GiB.
_RAM0_MB=0
if [[ "${AUTO_RAM0:-1}" == "1" ]]; then
    _io=0
    if [[ -f "${IR_DIR}/io.npz" ]]; then
        _io=$(python -c "import numpy as np,sys; d=np.load(sys.argv[1]); print(int(d['input'].nbytes)+int(d['output'].nbytes))" "${IR_DIR}/io.npz" 2>/dev/null || echo 0)
    fi
    _bytes=$(( _io * 3 + 128*1024*1024 ))                 # io + ~working + margin
    [[ -n "${SPIKE_RAM_SIZE:-}" ]] && _bytes=$(( SPIKE_RAM_SIZE ))
    _sz=$(( ((_bytes + 0x3FFFFFF) / 0x4000000) * 0x4000000 ))   # round up 64 MiB
    if (( _sz > 0x10000000 )); then                        # only if > stock 256 MiB
        if [[ "${RUNNER}" == "firesim" ]]; then _cap=$(( ${FIRESIM_DRAM_MB:-1024} * 1024*1024 )); else _cap=$(( 0xF0000000 )); fi
        if (( _sz > _cap )); then echo "WARN: derived ram0 $((_sz/1048576))MB > ${RUNNER} cap $((_cap/1048576))MB; clamping" >&2; _sz=$_cap; fi
        _RAM_OVL="${IR_DIR}/ram0.overlay"
        printf '/* @generated by _run_lib.sh: auto ram0 for baked io */\n&ram0 { reg = < 0x80000000 0x%x >; };\n' "${_sz}" > "${_RAM_OVL}"
        WEST_BUILD_EXTRA+=(-DEXTRA_DTC_OVERLAY_FILE="${_RAM_OVL}")
        _RAM0_MB=$(( _sz / 1048576 ))
        echo "[ram0] auto-sized to ${_RAM0_MB} MiB (baked io ~$((_io/1048576))MB) for ${RUNNER}" >&2
    fi
fi
if [[ "${RUNNER}" == "firesim" ]]; then
    # Splice the firesim overlay through Zephyr's EXTRA_CONF_FILE knob.
    # `west build -- -DEXTRA_CONF_FILE=...` arrives as a CMake -D, which
    # Zephyr picks up before find_package(Zephyr) processes Kconfig.
    # Pick the overlay matching the active FireSim hwconfig — the
    # quad-rocket and dual-rocket-gemmini bitstreams have different
    # hart counts so MP_MAX_NUM_CPUS must match. Override via
    # FIRESIM_CONF env if running a different config.
    if [[ -n "${FIRESIM_CONF:-}" ]]; then
        FS_CONF="${REPO_ROOT}/harness/backends/${FIRESIM_CONF}"
    elif [[ "${GEN_TARGET}" == "gemmini" || "${GEN_TARGET}" == "gemmini_q31" ]]; then
        # Both float-scale (gemmini) and Q0.31 (gemmini_q31) variants ride
        # the same dual-rocket-saturn-gemmini SoC topology, so the same
        # Zephyr SMP overlay applies. The runtime bitstream is selected
        # via config_runtime.yaml::default_hw_config.
        FS_CONF="${REPO_ROOT}/harness/backends/firesim_chipyard_dual_gemmini.conf"
    else
        FS_CONF="${REPO_ROOT}/harness/backends/firesim_chipyard.conf"
    fi
    WEST_BUILD_EXTRA+=(
        -DEXTRA_CONF_FILE="${FS_CONF}"
    )
elif [[ "${RUNNER}" == "spike" && "${ET_SMP:-0}" != "1" ]]; then
    # Default spike to a 1-core, 1-thread INLINE pthreadpool (+ unbuffered HTIF for
    # live markers). The harness prj.conf is MP_MAX_NUM_CPUS=4, which on spike is
    # >10x slower (4-hart emulation + pthreadpool oversubscription) and looks like
    # a hang. Opt into multicore with ET_SMP=1, or override SPIKE_CONF=.
    WEST_BUILD_EXTRA+=(
        -DEXTRA_CONF_FILE="${REPO_ROOT}/harness/backends/${SPIKE_CONF:-spike_single_core.conf}"
    )
fi
# CMODEL_LARGE=1 selects the RISC-V large code model (auipc+constant-pool /
# R_RISCV_64 indirection) so the program + all static symbols are no longer
# confined to a single 2 GiB window — lifts the R_RISCV_PCREL_HI20 truncation
# that stock-dimension baked io hits under the default medany model. Needs an
# SDK gcc that supports -mcmodel=large for rv64 (Zephyr SDK >= 1.0.0-beta1 /
# gcc 14). Orthogonal to -march; applies on any RISC-V board (spike/firesim).
if [[ "${CMODEL_LARGE:-0}" == "1" ]]; then
    WEST_BUILD_EXTRA+=(-DCONFIG_RISCV_CMODEL_LARGE=y)
fi
# MB_NO_BIGIO_REORDER=1 keeps the baked io in .rodata even under CMODEL_LARGE
# (harness places it above .bss by default). Mainly to A/B the reorder.
if [[ "${MB_NO_BIGIO_REORDER:-0}" == "1" ]]; then
    WEST_BUILD_EXTRA+=(-DMB_NO_BIGIO_REORDER=1)
fi
_mb_stage_begin build
west build -p -b "${BOARD_TARGET}" harness \
    --build-dir "${BUILD_DIR}" \
    -- "${WEST_CMAKE_ARGS[@]}" "${WEST_BUILD_EXTRA[@]}"
_mb_stage_end build

# BUILD_ONLY=1 stops after the west build (link) and skips the run+compare.
# Used to A/B link-time behaviour (e.g. medany R_RISCV_PCREL_HI20 truncation
# vs CMODEL_LARGE) without needing a target that can actually hold/run the io.
if [[ "${BUILD_ONLY:-0}" == "1" ]]; then
    echo "[build-only] link done -> ${BUILD_DIR}/zephyr/zephyr.elf"
    exit 0
fi

echo "[5/5] ${RUNNER} + compare"
_mb_stage_begin run

# Optional IREE-shape per-dispatch profile (PROFILE_OUT_ROOT env).
PROFILE_FLAGS=()
if [[ -n "${PROFILE_OUT_ROOT:-}" ]]; then
    if [[ -z "${PROFILE_BACKEND:-}" ]]; then
        case "${TARGET}" in
            rvv) PROFILE_BACKEND="RVV" ;;
            *)   PROFILE_BACKEND="${TARGET}" ;;
        esac
    fi
    PROFILE_FLAGS+=(
        "--profile-out-root=${PROFILE_OUT_ROOT}"
        "--profile-source=${PROFILE_SOURCE:-${RUNNER}}"
        "--profile-backend=${PROFILE_BACKEND}"
        "--profile-cores=${PROFILE_CORES:-0}"
        "--profile-clock-mhz=${PROFILE_CLOCK_MHZ:-1000.0}"
    )
    if [[ -n "${PROFILE_CPU:-}" ]]; then
        PROFILE_FLAGS+=("--profile-cpu=${PROFILE_CPU}")
    fi
fi

# Per-backend verify tolerance applies to BOTH spike and firesim — gemmini's
# float-scale and Q0.31 requantize paths each drift ~1 int8 LSB per layer
# vs the PyTorch Q0.31 golden, well-covered by atol=8 on shallow nets.
# Backend.atol_override / rtol_override are the authoritative source.
TOL_FLAGS=$(python -c "
from modelblaster.pipeline.backends import get
b = get('${GEN_TARGET}')
parts = []
if b.atol_override is not None:
    parts.append(f'--atol={b.atol_override}')
if b.rtol_override is not None:
    parts.append(f'--rtol={b.rtol_override}')
print(' '.join(parts))
")

if [[ "${RUNNER}" == "spike" ]]; then
    SPIKE_ARGS=$(python -c "
from modelblaster.pipeline.backends import get
b = get('${GEN_TARGET}')
print(' '.join(b.spike_args))
")
    SPIKE_FLAGS=()
    for a in ${SPIKE_ARGS}; do
        SPIKE_FLAGS+=("--spike-arg=${a}")
    done
    # Give spike enough modeled DRAM to cover a bumped ram0 (stock-dimension
    # runs). spike default is 2 GiB at 0x80000000; SPIKE_MEM_MB overrides it.
    # Auto-cover the derived ram0: only override spike's 2 GiB default when the
    # auto-sized ram0 exceeds it (SPIKE_MEM_MB overrides either way).
    _mem="${SPIKE_MEM_MB:-}"
    if [[ -z "${_mem}" && "${_RAM0_MB:-0}" -gt 1984 ]]; then _mem=$(( _RAM0_MB + 64 )); fi
    if [[ -n "${_mem}" ]]; then
        SPIKE_FLAGS+=("--spike-arg=-m${_mem}")
    fi
    # Gemmini backend (incl. Q31 variant) needs the chipyard spike (has
    # --extension=gemmini support + libgemmini.so). Use MODELBLASTER_GEMMINI_SPIKE
    # env if set, else chipyard path. The Q31 acc_scale variant uses the same
    # spike binary; MODELBLASTER_GEMMINI_LIB_DIR is normally per-config under
    # cores/gemmini/include/per_config/<sub>/libgemmini.so — see
    # modelblaster/scripts/validate_q31_matrix.sh.
    SPIKE_BIN_FLAGS=()
    if [[ "${GEN_TARGET}" == "gemmini" || "${GEN_TARGET}" == "gemmini_q31" ]]; then
        _GEMMINI_SPIKE="${MODELBLASTER_GEMMINI_SPIKE:-/scratch2/dima/misc_sw/FreshScheduler/hw/chipyard/.conda-env/riscv-tools/bin/spike}"
        _GEMMINI_LIB_DIR="${MODELBLASTER_GEMMINI_LIB_DIR:-/scratch2/dima/misc_sw/FreshScheduler/hw/chipyard/.conda-env/riscv-tools/lib}"
        if [[ -f "${_GEMMINI_SPIKE}" ]]; then
            SPIKE_BIN_FLAGS+=(--spike "${_GEMMINI_SPIKE}")
            export LD_LIBRARY_PATH="${_GEMMINI_LIB_DIR}:${LD_LIBRARY_PATH:-}"
        fi
    fi
    # rvv_opu backend needs the OPU-extended spike from
    # hw/chipyard/toolchains/riscv-tools/riscv-isa-sim (built via
    # customext/saturn_opu.cc). Use MODELBLASTER_OPU_SPIKE env if set, else
    # the local chipyard-tree path. The customext .so lives next to
    # the binary so LD_LIBRARY_PATH points at the same install lib dir.
    if [[ "${GEN_TARGET}" == "rvv_opu" ]]; then
        _OPU_SPIKE="${MODELBLASTER_OPU_SPIKE:-/scratch2/dima/misc_sw/FreshScheduler/hw/chipyard/.conda-env/riscv-tools/bin/spike}"
        _OPU_LIB_DIR="${MODELBLASTER_OPU_LIB_DIR:-/scratch2/dima/misc_sw/FreshScheduler/hw/chipyard/.conda-env/riscv-tools/lib}"
        if [[ -f "${_OPU_SPIKE}" ]]; then
            SPIKE_BIN_FLAGS+=(--spike "${_OPU_SPIKE}")
            export LD_LIBRARY_PATH="${_OPU_LIB_DIR}:${LD_LIBRARY_PATH:-}"
        fi
    fi
    python -m modelblaster.validation.spike_runner \
        --elf "${BUILD_DIR}/zephyr/zephyr.elf" \
        --io "${IR_DIR}/io.npz" \
        --timeout "${SPIKE_TIMEOUT:-600}" \
        ${TOL_FLAGS} \
        "${SPIKE_BIN_FLAGS[@]}" \
        "${SPIKE_FLAGS[@]}" \
        "${PROFILE_FLAGS[@]}"
elif [[ "${RUNNER}" == "native" ]]; then
    # native_sim: the built artifact is a host executable (zephyr.exe). Run it
    # directly and reuse the shared verify/compare path.
    python -m modelblaster.validation.native_runner \
        --elf "${BUILD_DIR}/zephyr/zephyr.exe" \
        --io "${IR_DIR}/io.npz" \
        --quant "${QUANT}" \
        --timeout "${NATIVE_TIMEOUT:-600}" \
        ${TOL_FLAGS}
else
    # firesim: the runner copies the elf into the sim slot, runs
    # firesim runworkload, tails the uartlog until OUTPUT_END, then
    # firesim kill. FIRESIM_ROOT / FIRESIM_ENV / FIRESIM_SLOT env vars
    # override the install paths.
    FIRESIM_FLAGS=()
    if [[ -n "${FIRESIM_ROOT:-}" ]]; then
        FIRESIM_FLAGS+=("--firesim-root=${FIRESIM_ROOT}")
    fi
    if [[ -n "${FIRESIM_ENV:-}" ]]; then
        FIRESIM_FLAGS+=("--firesim-env=${FIRESIM_ENV}")
    fi
    if [[ -n "${FIRESIM_SLOT:-}" ]]; then
        FIRESIM_FLAGS+=("--firesim-slot=${FIRESIM_SLOT}")
    fi
    if [[ -n "${FIRESIM_TIMEOUT:-}" ]]; then
        FIRESIM_FLAGS+=("--timeout=${FIRESIM_TIMEOUT}")
    fi
    python -m modelblaster.validation.firesim_runner \
        --elf "${BUILD_DIR}/zephyr/zephyr.elf" \
        --io "${IR_DIR}/io.npz" \
        ${TOL_FLAGS} \
        "${FIRESIM_FLAGS[@]}" \
        "${PROFILE_FLAGS[@]}"
fi
_mb_stage_end run
