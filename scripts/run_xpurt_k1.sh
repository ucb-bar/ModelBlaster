#!/usr/bin/env bash
# Execute an XPU-RT schedule as a multi-model periodic workload on the K1,
# entirely through ModelBlaster.
#
# This replaces the earlier IREE-based dispatch scheduler. That runner executed
# IREE VMFBs, which meant every measurement went through the IREE path and that
# ModelBlaster's own generated-C kernels were never what the scheduler placed.
# It also made baseline B3 unreachable: apply_split_hint / apply_fusion_hint
# rewrite ModelBlaster's IR, and a rewritten IR has no VMFB, so a runner that
# resolves each dispatch to a .vmfb on disk cannot execute a granularity change.
# Here the generated C *is* the executable, so a hint change is directly
# runnable.
#
# What this does, in order:
#   1. per (model, backend): extract int8 IR -> generate skeleton -> generate
#      kernels, staged as <base>/<backend>/ which is the layout the harness
#      CMake wants (one weights TU per backend; buffers shared per model).
#   2. ingest the XPU-RT schedule -> dispatch_table.{c,h}
#   3. generate xpurt_main.c (--worker-granularity, default hart)
#   4. cross-build a static executable with harness_xpurt/build_posix.sh
#   5. scp to the board, run pinned, pull back stdout + the trace CSV
#
# Usage:
#   scripts/run_xpurt_k1.sh --schedule <scheduled_*.json> \
#       [--models mlp_control,dronet] [--backends scalar] [--quant int8]
#       [--impl-backends a100:ime=ime_k3_a100,a100:rvv=rvv_k3_a100,x100:rvv=rvv_k3]
#
# --impl-backends serves schedules that choose a per-dispatch implementation
# (XPU-RT `scheduler.enable_impls` writes `impl` on each dispatch): on a core
# of kind K, a dispatch with impl I runs backend B for each K:I=B. Every
# backend named there is built too.
#
# Board selection: MODELBLASTER_BOARD=k1 (default) | k3 changes only the
# defaults (host, remote root, registry, core kinds, backends, output root);
# explicit env vars and flags still win. scripts/run_xpurt_k3.sh sets it.
# On k3 the binary is launched under `taskset -c 0-7` (X100); A100 threads
# register themselves with the kernel inside the runtime, and the registry's
# ai_thread_cpus are handed to build_posix.sh as MODELBLASTER_AI_CPUS.
#
# No credentials are read or written. Board access comes from the ssh config
# entry named by MODELBLASTER_K1_HOST (default "k1"; k3: MODELBLASTER_K3_HOST,
# default "spacemitk3").

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Board defaults (stdlib-only helper; BOARD_* names stay out of the MB_/
# MODELBLASTER_ extraction-config scan below).
BOARD_CFG="$(python3 "${REPO_ROOT}/validation/board_config.py" \
    --board "${MODELBLASTER_BOARD:-k1}" shell)" || exit 2
eval "${BOARD_CFG}"

SCHEDULE=""
MODELS="mlp_control,dronet"
# <network>:<dir>,... -- networks whose IR is ALREADY BUILT and is supplied
# rather than extracted. Two cases need this and neither is exotic:
#
#   * a network whose name is not a `models/` module. `networks_dense2` calls
#     its detector `yolov8_nano_64x96`; that is the `yolov8_nano` module built
#     at a different input size, registered under its own name so specs built
#     against the 160x160 profile still reproduce. `extract_graph --model
#     yolov8_nano_64x96` has nothing to extract.
#   * A REWRITTEN IR. `apply_split_hint` / `apply_fusion_hint` /
#     `apply_unfuse_hint` produce a graph.json that no `--model` can regenerate
#     -- which is the entire point of the granularity loop. Without this the
#     runner cannot execute the thing the loop produces.
#
# The staged IR is COPIED, its sha256 recorded, and its provenance written to
# `.staged_from` -- deliberately NOT `.extract_config`, so a staged graph can
# never be mistaken for an extracted one by the reuse check.
STAGED_IRS=""
# One backend per core kind (parallel to CORE_KINDS below): k1 = rvv_x60 twice,
# k3 = rvv_k3 (x100) + rvv_k3_a100 (a100, zvl1024b).
BACKENDS="${BOARD_BACKEND_P},${BOARD_BACKEND_E}"
QUANT="int8"
# reference = curated/hand-written kernels, no LLM. Codex is only for NEW kernel
# bodies and must be requested explicitly; there is no Bedrock path here.
KERNEL_BACKEND="${BACKEND:-reference}"
# Codex only, no fallback. Consulted only when BACKEND=llm.
export LLM_PROVIDER="${LLM_PROVIDER:-codex}"
# The K1 registry is measured, not assumed -- cluster 0 carries IME, cluster 1
# traps on it. See cores/spacemit_k1.json.
REGISTRY="${REGISTRY:-${BOARD_REGISTRY}}"
# Both K1 clusters are RVV-capable and measured equivalent (0.996 ratio), so a
# single backend kind covers both; the cluster distinction is a placement
# constraint, not a codegen one.
# Two DISTINCT kinds, one per cluster. `kind` is how ingest resolves an
# abstract CPU_P#n / CPU_E#n slot to a hart -- it takes the n-th core of that
# kind -- so a single kind for all eight cores makes CPU_E#n alias CPU_P#n, and
# a 4+4 config silently double-books cluster 0 while never touching cluster 1.
# The two clusters run the same ISA and measure equivalent (0.996), so this is
# a placement-pool distinction, not a codegen one: both compile rvv_x60.
CPU_P_KIND="${CPU_P_KIND:-${BOARD_CPU_P_KIND}}"
CPU_E_KIND="${CPU_E_KIND:-${BOARD_CPU_E_KIND}}"
# core_kind (what the SCHEDULE says, and what the walker matches on) is distinct
# from the backend tag (what the BINARY was built as). rvv_x60 is a K1-specific
# build of the same rvv kind; conflating them makes every worker refuse every
# entry -- strcmp("rvv_x60","rvv") != 0 -- and the run completes having executed
# nothing, with entries_done=0 and an all-zero trace.
CORE_KINDS="${CORE_KINDS:-}"
OUT_ROOT="${BOARD_OUT_ROOT_XPURT}"
TRACE=1
# Per-dispatch-END JSON lines on stdout, for xpu-rt/streaming_feedback.py.
# Off by default: the trace block at exit is enough to EXPLAIN a run, and
# streaming only earns its keep when something is watching live. Pipe the
# run through `tee` when you turn this on -- the lines are interleaved with
# the harness's normal output by design, so one stream carries both.
STREAM="${MB_XPURT_STREAM:-0}"
CPU_IDS=""
JOBS="$(nproc)"

HOST="${BOARD_HOST}"
REMOTE_ROOT="${BOARD_REMOTE_ROOT}"
_rt_var="MODELBLASTER_$(tr '[:lower:]' '[:upper:]' <<<"${BOARD_NAME}")_RT_PRIORITY"
RT_PRIORITY="${!_rt_var:-}"
# No hardcoded chipyard default any more: that was gcc 13.2, which miscompiles
# the RVV vsetvl intrinsics (SIGILL on the board) and lived under another
# user's home. Take CROSS from the environment, else the SpaceMiT toolchain
# (gcc>=14) that XPU-RT's setup script provides.
if [[ -z "${CROSS:-}" ]]; then
    _tc="${SETUP_TOOLCHAIN:-${REPO_ROOT}/../XPU-RT/scripts/setup_spacemit_toolchain.sh}"
    # Never trigger the script's 800 MB download implicitly: only use it when
    # the toolchain is already unpacked; otherwise say how to fetch it.
    _tcroot="${SPACEMIT_TOOLCHAIN_ROOT:-$(dirname "${_tc}")/../tools/riscv-tools-spacemit}"
    if [[ -x "${_tc}" ]] && compgen -G "${_tcroot}/*/bin/riscv64-unknown-linux-gnu-gcc" >/dev/null \
            && CROSS="$("${_tc}" --path)"; then :; else
        echo "CROSS is not set and the SpaceMiT toolchain fallback failed:" >&2
        echo "  tried ${_tc} --path" >&2
        echo "Set CROSS to a gcc>=14 riscv64-unknown-linux-gnu- prefix, e.g." >&2
        echo "  eval \"\$(XPU-RT/scripts/setup_spacemit_toolchain.sh)\"" >&2
        exit 2
    fi
fi
PY="${PY:-python3}"
# Same import root the single-model runner uses.
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Assert we are running THIS checkout's pipeline. There is a second, older
# ModelBlaster clone on this machine, and the venv most people use here carries
# an editable install (_editable_impl_modelblaster.pth) pointing at it. Import
# `modelblaster` without the PYTHONPATH above and you silently get the sibling's
# code -- same module names, different behaviour, no warning.
_mb_file="$(${PY} -c 'from modelblaster.pipeline import backends; print(backends.__file__)' 2>/dev/null || echo '')"
case "${_mb_file}" in
    "${REPO_ROOT}"/*) : ;;
    *) echo "refusing to run: 'modelblaster' resolves to ${_mb_file:-<import failed>}," >&2
       echo "not this checkout (${REPO_ROOT}). Check PYTHONPATH and any editable install." >&2
       exit 2 ;;
esac

while [[ $# -gt 0 ]]; do
    case "$1" in
        --schedule)  SCHEDULE="$2"; shift 2 ;;
        --registry)  REGISTRY="$2"; shift 2 ;;
        --models)    MODELS="$2"; shift 2 ;;
        --staged-ir) STAGED_IRS="${STAGED_IRS:+${STAGED_IRS},}$2"; shift 2 ;;
        --backends)  BACKENDS="$2"; shift 2 ;;
        --impl-backends) IMPL_BACKENDS="$2"; shift 2 ;;
        --quant)     QUANT="$2"; shift 2 ;;
        --out-root)  OUT_ROOT="$2"; shift 2 ;;
        --cpu-ids)   CPU_IDS="$2"; shift 2 ;;
        --no-trace)  TRACE=0; shift ;;
        --jobs)      JOBS="$2"; shift 2 ;;
        -h|--help)   sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done
[[ -n "${SCHEDULE}" ]] || { echo "--schedule is required" >&2; exit 2; }
[[ -f "${SCHEDULE}" ]] || { echo "no such schedule: ${SCHEDULE}" >&2; exit 2; }

IFS=',' read -r -a MODEL_LIST <<< "${MODELS}"
IFS=',' read -r -a BACKEND_LIST <<< "${BACKENDS}"
PRIMARY_BS="${BACKEND_LIST[0]}"
if [[ -z "${CORE_KINDS}" ]]; then
    if [[ "${CPU_P_KIND}" == "${CPU_E_KIND}" ]]; then
        CORE_KINDS="${CPU_P_KIND}"
    else
        CORE_KINDS="${CPU_P_KIND},${CPU_E_KIND}"
    fi
fi
IFS=',' read -r -a CORE_KIND_LIST <<< "${CORE_KINDS}"
# Several core kinds can share one backend -- the K1's two clusters are the
# same ISA, so both pools compile rvv_x60. The GENERATOR needs the per-kind
# list (kind k is executed by backend k); CMake needs the deduped set, or it
# would try to declare the same OBJECT-library target twice.
# Backends that only an --impl-backends triple names are built as well.
IMPL_BACKENDS="${IMPL_BACKENDS:-}"
IMPL_BS=()
if [[ -n "${IMPL_BACKENDS}" ]]; then
    IFS=',' read -r -a _ib <<< "${IMPL_BACKENDS}"
    for t in "${_ib[@]}"; do IMPL_BS+=("${t#*=}"); done
fi
UNIQUE_BACKENDS="$(printf '%s\n' "${BACKEND_LIST[@]}" "${IMPL_BS[@]}" | awk 'NF && !seen[$0]++' | paste -sd, -)"
IFS=',' read -r -a BUILD_BACKEND_LIST <<< "${UNIQUE_BACKENDS}"
if [[ ${#CORE_KIND_LIST[@]} -ne ${#BACKEND_LIST[@]} ]]; then
    echo "core-kinds (${CORE_KINDS}) and backends (${BACKENDS}) must be" >&2
    echo "parallel lists: kind k is executed by backend k." >&2
    exit 2
fi

SCHED_NAME="$(basename "${SCHEDULE}" .json)"
# Absolute: build_posix.sh runs from its own directory.
GEN_DIR="${REPO_ROOT}/${OUT_ROOT}/_gen/${SCHED_NAME}"
BUILD_DIR="${REPO_ROOT}/${OUT_ROOT}/_build/${SCHED_NAME}"
mkdir -p "${GEN_DIR}" "${BUILD_DIR}"

echo "=== 1/5 generate per-(model, backend) sources ==="
MODEL_NAMES=""
MODEL_DIRS_BASE=""

# Stage supplied IRs first, so the loop below finds them in place and treats
# them exactly like an extracted model from that point on.
STAGED_SET=""
if [[ -n "${STAGED_IRS}" ]]; then
    IFS=',' read -r -a _staged <<< "${STAGED_IRS}"
    for spec in "${_staged[@]}"; do
        net="${spec%%:*}"; src="${spec#*:}"
        [[ "${net}" != "${spec}" ]] || { echo "--staged-ir needs <network>:<dir>, got ${spec}" >&2; exit 2; }
        [[ -d "${src}" ]] || { echo "--staged-ir ${net}: no such directory ${src}" >&2; exit 2; }
        for f in graph.json weights.npz io.npz; do
            [[ -f "${src}/${f}" ]] || { echo "--staged-ir ${net}: ${src}/${f} missing" >&2; exit 2; }
        done
        dst="${REPO_ROOT}/${OUT_ROOT}/${net}/${QUANT}"
        _sha="$(sha256sum "${src}/graph.json" | cut -d" " -f1)"
        if [[ -f "${dst}/.staged_from" ]] && grep -q "sha256=${_sha}\b" "${dst}/.staged_from"; then
            echo "  [${net}] staged IR unchanged (sha256 ${_sha:0:12})"
        else
            echo "  [${net}] staging IR from ${src} (sha256 ${_sha:0:12})"
            mkdir -p "${dst}"
            cp "${src}/graph.json" "${src}/weights.npz" "${src}/io.npz" "${dst}/"
            # The IR's own `name` becomes the NETWORK name. Every generated C
            # symbol is mangled by it -- `model_<name>_op_record_t`,
            # `MODEL_<NAME>_OUTPUT_SIZE`, `dispatch_<name>_<id>` -- while
            # xpurt_main.c mangles from the SCHEDULE's network name. Leave them
            # different and the harness fails to compile against a type it just
            # generated under another name. This is a rename, not a rewrite:
            # the graph is otherwise byte-identical, which is why the recorded
            # sha256 is of the SOURCE file.
            ${PY} - "${dst}/graph.json" "${net}" <<'PY_RENAME'
import json, sys
path, net = sys.argv[1], sys.argv[2]
g = json.load(open(path))
was = g.get("name")
if was != net:
    g["name"] = net
    g["_staged_rename"] = {"from": was, "to": net}
    json.dump(g, open(path, "w"), indent=1)
    print(f"      renamed IR {was!r} -> {net!r} (C symbols mangle by this)")
PY_RENAME
            # The generated sources descend from this IR, so they are stale.
            rm -f "${dst}"/*/kernels.c "${dst}"/*/model.c
            printf 'source=%s\nsha256=%s\nstaged_at=%s\n' \
                "$(cd "${src}" && pwd)" "${_sha}" "$(date -Is)" > "${dst}/.staged_from"
        fi
        # A staged graph must never satisfy the extraction reuse check.
        rm -f "${dst}/.extract_config"
        STAGED_SET="${STAGED_SET} ${net} "
    done
fi
for model in "${MODEL_LIST[@]}"; do
    base="${REPO_ROOT}/${OUT_ROOT}/${model}/${QUANT}"
    mkdir -p "${base}"

    # The int8 IR + weights + goldens are per model, not per backend.
    #
    # Reuse is keyed on the EXTRACTION CONFIG, not just on the file existing.
    # Several models read their precision and calibration choices from the
    # environment -- fused_full alone honours MB_FUSED_HYBRID,
    # MB_FUSED_LOWDIM_FLOAT, MB_FUSED_FP16_FC, MB_FUSED_CALIB_PKL and
    # NUM_CALIBRATION -- and none of that is visible in the output path. An
    # existence check therefore silently serves a graph built under different
    # settings.
    #
    # That is not hypothetical. A schedule solved against the fp16 hybrid
    # (17 ops, ids 0-14) was ingested against a cached pure-int8 graph (15 ops,
    # ids 0-12) and died with "dispatch_id=13 not in network 'fused_full'".
    # Dying was the good case: the two graphs differ in PRECISION, so a
    # dispatch-count coincidence would have run the blind int8 model under the
    # hybrid's name and reported its timings.
    if [[ "${STAGED_SET}" == *" ${model} "* ]]; then
        echo "  [${model}] using staged IR ($(sed -n 's/^source=//p' "${base}/.staged_from"))"
        _skip_extract=1
    else
        _skip_extract=0
    fi
    _cfg_file="${base}/.extract_config"
    _cfg="model=${model} quant=${QUANT}"
    for _v in $(compgen -v | grep -E '^(MB_|MODELBLASTER_|NUM_CALIBRATION)' | sort); do
        # Board process policy cannot change IR, weights, or goldens. Keeping
        # it in the extraction key needlessly rebuilt every model when an
        # experiment switched between SCHED_OTHER and SCHED_FIFO.
        [[ "${_v}" == "MODELBLASTER_K1_RT_PRIORITY" ]] && continue
        # Same reasoning for the board selector and the k3 equivalents.
        case "${_v}" in
            MODELBLASTER_BOARD|MODELBLASTER_K3_*|MODELBLASTER_SSH_CONFIG|MB_TOPO_RELATIVE|MB_XPURT_WARMUP) continue ;;
            MB_XPURT_PLACEMENT|MB_XPURT_REPEATS) continue ;;
        esac
        _cfg="${_cfg} ${_v}=${!_v}"
    done
    if [[ "${_skip_extract}" == "1" ]]; then
        :
    elif [[ -f "${base}/graph.json" && -f "${_cfg_file}" ]] \
       && [[ "$(cat "${_cfg_file}")" == "${_cfg}" ]]; then
        echo "  [${model}] reusing ${base}/graph.json"
    else
        if [[ -f "${base}/graph.json" ]]; then
            echo "  [${model}] re-extracting: extraction config changed"
            if [[ -f "${_cfg_file}" ]]; then
                diff <(tr ' ' '\n' < "${_cfg_file}") <(tr ' ' '\n' <<< "${_cfg}") \
                    | grep -E '^[<>]' | sed 's/^/      /' || true
            else
                echo "      (no previous config recorded)"
            fi
        fi
        echo "  [${model}] extract int8 IR"
        ${PY} -m modelblaster.pipeline.extract_graph \
            --model "${model}" --quant "${QUANT}" --out-dir "${base}"
        printf '%s' "${_cfg}" > "${_cfg_file}"
        # The generated sources descend from this IR, so they are stale too.
        rm -f "${base}"/*/kernels.c "${base}"/*/model.c
    fi

    # Materialize the schedule's composite target widths into a codegen-only
    # IR. XPU-RT keeps a shard as one dispatch and records its width in
    # hardware_target (CPU_P#0+CPU_P#1+...); packed RVV convolution weights
    # need that width at skeleton-generation time so each shard receives its
    # own correctly packed OC slice. Keep the extracted graph pristine: the
    # next schedule may choose different widths, and profile/hash checks refer
    # to the unannotated graph.
    CODEGEN_IR="${base}/graph.xpurt_schedule.json"
    ${PY} -m modelblaster.pipeline.schedule_shards \
        --ir "${base}/graph.json" --schedule "${SCHEDULE}" \
        --network "${model}" --out "${CODEGEN_IR}" --backends "${UNIQUE_BACKENDS}"

    for bs in "${BUILD_BACKEND_LIST[@]}"; do
        bdir="${base}/${bs}"
        if [[ -f "${bdir}/kernels.c" && -f "${bdir}/model.c" ]]; then
            # Existence is NOT enough. generate_kernels.py inlines the curated
            # kernel bodies into kernels.c, so once that file exists a later fix
            # to a curated kernel never reaches the board -- the driver prints
            # "reusing" and links the stale copy. That is not hypothetical: the
            # vtype fix in rvv_batchnorm2d_s8_direct.c (explicit vsetvl_e32m8
            # around the float epilogue) was written at 19:01 and the B1 harness
            # built at 17:44 kept SIGILLing on vfmv.v.f under SEW=8, because the
            # generated copy predated the fix. The kernel looked fixed in the
            # source tree and was still broken on the hardware.
            #
            # So reuse only if nothing that FEEDS the generated sources is newer:
            # the curated kernel library and the generators themselves.
            _stale=""
            for _dep in "${REPO_ROOT}/kernels" \
                        "${CODEGEN_IR}" \
                        "${REPO_ROOT}/pipeline/generate_kernels.py" \
                        "${REPO_ROOT}/pipeline/generate_skeleton.py" \
                        "${REPO_ROOT}/pipeline/backends.py"; do
                [[ -e "${_dep}" ]] || continue
                if [[ -n "$(find "${_dep}" -newer "${bdir}/kernels.c" -print -quit 2>/dev/null)" ]]; then
                    _stale="${_dep}"
                    break
                fi
            done
            if [[ -z "${_stale}" ]]; then
                echo "  [${model}/${bs}] reusing generated sources"
                continue
            fi
            echo "  [${model}/${bs}] regenerating -- ${_stale#${REPO_ROOT}/} is newer than the generated kernels.c"
            rm -f "${bdir}/kernels.c" "${bdir}/model.c"
        fi
        echo "  [${model}/${bs}] skeleton + kernels"
        mkdir -p "${bdir}"
        # Default platform: build_posix.sh compiles with MB_PLATFORM_POSIX,
        # which times dispatches with CLOCK_MONOTONIC (rdcycle SIGILLs here).
        ${PY} -m modelblaster.pipeline.generate_skeleton \
            --ir "${CODEGEN_IR}" --weights "${base}/weights.npz" \
            --io "${base}/io.npz" --out-dir "${bdir}" \
            --backend "${bs}"
        ${PY} -m modelblaster.pipeline.generate_kernels \
            --ir "${CODEGEN_IR}" --out-dir "${bdir}" \
            --target "${bs}" --backend "${KERNEL_BACKEND}" --quant "${QUANT}" \
            --global-curated-dir "${REPO_ROOT}/kernels"
    done

    # A vector backend that silently resolved ops to the scalar reference is
    # the single most expensive failure this pipeline has: yolov8_nano measured
    # 0.81x against the scalar build (i.e. SLOWER) purely because the fused
    # conv2d_batchnorm2d_silu_s8 had no curated RVV kernel and nothing said so.
    # Catch it here, before the board time is spent.
    for bs in "${BUILD_BACKEND_LIST[@]}"; do
        [[ "${bs}" == "scalar" ]] && continue
        if ! ${PY} "${REPO_ROOT}/scripts/check_kernel_coverage.py" "${base}/${bs}"; then
            if [[ "${MB_KERNEL_COVERAGE:-strict}" == "warn" ]]; then
                echo "  [${model}/${bs}] coverage gate FAILED -- continuing because MB_KERNEL_COVERAGE=warn"
            else
                echo "  [${model}/${bs}] coverage gate failed; set MB_KERNEL_COVERAGE=warn to profile anyway" >&2
                exit 1
            fi
        fi
    done

    MODEL_NAMES="${MODEL_NAMES:+${MODEL_NAMES};}${model}"
    MODEL_DIRS_BASE="${MODEL_DIRS_BASE:+${MODEL_DIRS_BASE};}${base}"
done

echo "=== 2/5 ingest schedule -> dispatch table ==="
# ingest_xpurt_schedule (not ingest_schedule): the multi-model one, which is
# also where the IR-completion pass lives that synthesizes dispatches the
# scheduler never placed.
SCHED_C="${GEN_DIR}/${SCHED_NAME}.c"
SCHED_H="${GEN_DIR}/${SCHED_NAME}.h"
IR_ARGS=()
for model in "${MODEL_LIST[@]}"; do
    IR_ARGS+=(--ir "${model}:${REPO_ROOT}/${OUT_ROOT}/${model}/${QUANT}/graph.xpurt_schedule.json")
done
${PY} -m modelblaster.pipeline.ingest_xpurt_schedule \
    --schedule "${SCHEDULE}" --registry "${REGISTRY}" \
    "${IR_ARGS[@]}" --out "${SCHED_C}" --name "${SCHED_NAME}" \
    --cpu-p-kind "${CPU_P_KIND}" --cpu-e-kind "${CPU_E_KIND}"
[[ -f "${SCHED_C}" ]] || { echo "ingest produced no dispatch table" >&2; exit 1; }
echo "  ${SCHED_C}"

# Hosted-Linux walker opt-ins (harness_xpurt/LINUX.md):
#   MB_XPURT_WARMUP=1     one untimed pass of the whole schedule before the
#                         timed one (generate_xpurt_main --warmup)
#   MB_XPURT_PLACEMENT=1  record the cpu each entry's kernel ran on; the block is
#                         saved as <sched>_placement.csv (--placement)
#   MB_XPURT_REPEATS=N    run the deployed binary N times (default 1); with N>1
#                         each run's stdout/trace/placement get an _r<k> suffix
WALKER_OPTS=()
[[ "${MB_XPURT_WARMUP:-0}" == "1" ]] && WALKER_OPTS+=(--warmup)
PLACEMENT="${MB_XPURT_PLACEMENT:-0}"
[[ "${PLACEMENT}" == "1" ]] && WALKER_OPTS+=(--placement)
REPEATS="${MB_XPURT_REPEATS:-1}"
[[ "${REPEATS}" =~ ^[1-9][0-9]*$ ]] || { echo "MB_XPURT_REPEATS must be a positive integer" >&2; exit 2; }
# The walker has no live JSON stream yet; say so rather than ignoring the knob.
[[ "${STREAM}" == "1" ]] && \
    echo "note: MB_XPURT_STREAM=1 ignored: the walker has no live stream" >&2
# One pinned worker per (kind, hart) by default: the schedule places
# dispatches on individual harts, and composite targets need hart mode to
# reserve their whole hart set. MB_XPURT_WORKERS=kind gives one worker per
# kind with an intra-op pool over the kind's harts.
WORKERS="${MB_XPURT_WORKERS:-hart}"

echo "=== 3/5 generate the walker (--worker-granularity ${WORKERS}) ==="
MAIN_C="${GEN_DIR}/${SCHED_NAME}_main.c"
${PY} -m modelblaster.pipeline.generate_xpurt_main \
    --schedule "${SCHEDULE}" --out "${MAIN_C}" --name "${SCHED_NAME}" \
    --dispatch-table-header "$(basename "${SCHED_H}")" \
    --core-kinds "${CORE_KINDS}" --backends "${BACKENDS}" \
    ${IMPL_BACKENDS:+--impl-backends "${IMPL_BACKENDS}"} \
    --registry "${REGISTRY}" --worker-granularity "${WORKERS}" \
    ${WALKER_OPTS[@]+"${WALKER_OPTS[@]}"}

echo "=== 4/5 cross-build (harness_xpurt/build_posix.sh) ==="
# A100 harts that must register with the kernel (registry ai_thread_cpus);
# empty on the K1. Kernel flags come from each backend's kernel_cflags.
_ai="$(${PY} "${REPO_ROOT}/pipeline/core_registry.py" "${REGISTRY}" --ai-cpus)" || exit 2
[[ -n "${MODELBLASTER_KERNEL_CC:-}" ]] && \
    echo "note: MODELBLASTER_KERNEL_CC ignored: build_posix.sh builds everything with CROSS" >&2
mkdir -p "${BUILD_DIR}"
BIN="${BUILD_DIR}/xpurt_harness"
MODEL_NAMES="${MODEL_NAMES}" MODEL_DIRS_BASE="${MODEL_DIRS_BASE}" \
MODEL_BACKENDS="${UNIQUE_BACKENDS}" \
XPURT_SCHEDULE_C="${SCHED_C}" XPURT_MAIN_C="${MAIN_C}" XPURT_INCLUDE_DIR="${GEN_DIR}" \
OUT="${BIN}" CC="${CROSS}gcc" LDFLAGS=-static PYTHON="${PY}" \
MODELBLASTER_AI_CPUS="${_ai}" MODELBLASTER_XPURT_TRACE="${TRACE}" \
    bash "${REPO_ROOT}/harness_xpurt/build_posix.sh" >"${BUILD_DIR}/build.log" 2>&1 \
    || { echo "build failed; tail of ${BUILD_DIR}/build.log:" >&2; tail -40 "${BUILD_DIR}/build.log" >&2; exit 1; }
[[ -x "${BIN}" ]] || { echo "no binary at ${BIN}" >&2; exit 1; }
echo "  $(file -b "${BIN}" | cut -c1-70)"

echo "=== 5/5 deploy and run on ${HOST} ==="
# k3's ssh is flaky: generous handshake timeout, and retry the idempotent
# setup/copy/fetch steps (never the run itself). k1 keeps plain ssh/scp.
SSH_OPTS=()
RETRIES=1
if [[ "${BOARD_NAME}" != "k1" ]]; then SSH_OPTS=(-o ConnectTimeout=60); RETRIES=4; fi
# Optional ssh config file (e.g. a tunnel); same knob as validation/k1_runner.py.
if [[ -n "${MODELBLASTER_SSH_CONFIG:-}" ]]; then SSH_OPTS=(-F "${MODELBLASTER_SSH_CONFIG}" "${SSH_OPTS[@]}"); fi
_retry() { local i; for ((i = 1; i <= RETRIES; i++)); do "$@" && return 0; [[ $i -lt ${RETRIES} ]] && sleep 5; done; return 1; }
_retry ssh "${SSH_OPTS[@]}" "${HOST}" "mkdir -p ${REMOTE_ROOT}/xpurt"
_retry scp "${SSH_OPTS[@]}" -q "${BIN}" "${HOST}:${REMOTE_ROOT}/xpurt/${SCHED_NAME}"
RUN="cd ${REMOTE_ROOT}/xpurt && ulimit -n 8192 &&"
if [[ -n "${BOARD_LAUNCH_CPUS}" ]]; then
    # k3: always START on the X100. Widening to the A100 is the runtime's job
    # (threads register + set their own affinity); a wider mask here would let
    # the main thread land on an unregistered A100 core.
    [[ -n "${CPU_IDS}" && "${CPU_IDS}" != "${BOARD_LAUNCH_CPUS}" ]] && \
        echo "  note: --cpu-ids ${CPU_IDS} ignored on ${BOARD_NAME}; launching under taskset -c ${BOARD_LAUNCH_CPUS}" >&2
    RUN="${RUN} taskset -c ${BOARD_LAUNCH_CPUS}"
elif [[ -n "${CPU_IDS}" ]]; then
    RUN="${RUN} taskset -c ${CPU_IDS}"
fi
RUNNER_POLICY="SCHED_OTHER"
RUNNER_PRIORITY=0
if [[ -n "${RT_PRIORITY}" ]]; then
    [[ "${RT_PRIORITY}" =~ ^[0-9]+$ ]] && (( RT_PRIORITY >= 1 && RT_PRIORITY <= 99 )) \
        || { echo "MODELBLASTER_K1_RT_PRIORITY must be an integer in 1..99" >&2; exit 2; }
    RUN="${RUN} chrt -f ${RT_PRIORITY}"
    RUNNER_POLICY="SCHED_FIFO"
    RUNNER_PRIORITY="${RT_PRIORITY}"
fi
RUN="${RUN} ./${SCHED_NAME}"
# Redirect ON THE BOARD and scp the file back, rather than streaming stdout
# through ssh. A three-model schedule emits ~1600 trace rows, and streaming that
# volume reproducibly killed the board's ssh daemon:
#
#   sshd-session[...]: unhandled signal 7 code 0x1 in libcrypto.so.3
#   status: ... badaddr: 0000002ab877e7ea cause: 0000000000000006
#
# cause 6 is a misaligned store, signal 7 is SIGBUS, and Comm is sshd-session --
# so the harness was fine and the transport died under it. The visible symptom
# was a truncated trace (575 of 1617 rows) plus a nonzero exit, which reads
# exactly like a crash in the run being measured. Writing to a file first makes
# the run's completion independent of the link, and the exit status is the
# program's own.
final_rc=0
for ((rep = 1; rep <= REPEATS; rep++)); do
sfx=""
(( REPEATS > 1 )) && sfx="_r${rep}"
OUT="${GEN_DIR}/${SCHED_NAME}_stdout${sfx}.txt"
REMOTE_LOG="${REMOTE_ROOT}/xpurt/${SCHED_NAME}_stdout${sfx}.txt"
(( REPEATS > 1 )) && echo "  --- run ${rep}/${REPEATS} ---"
set +e
ssh "${SSH_OPTS[@]}" "${HOST}" \
    "printf '%s\\n' 'xpurt_runner: sched_policy=${RUNNER_POLICY} priority=${RUNNER_PRIORITY}' >${REMOTE_LOG}; ${RUN} >>${REMOTE_LOG} 2>&1; echo \$? >${REMOTE_LOG}.rc"
_retry scp "${SSH_OPTS[@]}" -q "${HOST}:${REMOTE_LOG}" "${OUT}"
rc=$(_retry ssh "${SSH_OPTS[@]}" "${HOST}" "cat ${REMOTE_LOG}.rc" 2>/dev/null || echo 255)
set -e
echo "  exit=${rc}  stdout -> ${OUT}"
grep -E 'MODELBLASTER_VERIFY|max_abs_err|FAIL|PASS' "${OUT}" | head -20 || true

if [[ "${TRACE}" == "1" ]] && grep -q 'MODELBLASTER_XPURT_TRACE_BEGIN' "${OUT}"; then
    TRACE_CSV="${GEN_DIR}/${SCHED_NAME}_trace${sfx}.csv"
    awk '/MODELBLASTER_XPURT_TRACE_BEGIN/{f=1;next} /MODELBLASTER_XPURT_TRACE_END/{f=0} f' \
        "${OUT}" > "${TRACE_CSV}"
    echo "  trace -> ${TRACE_CSV} ($(($(wc -l <"${TRACE_CSV}") - 1)) rows)"
fi
if [[ "${PLACEMENT}" == "1" ]] && grep -q 'MODELBLASTER_XPURT_PLACEMENT_BEGIN' "${OUT}"; then
    PLACEMENT_CSV="${GEN_DIR}/${SCHED_NAME}_placement${sfx}.csv"
    awk '/MODELBLASTER_XPURT_PLACEMENT_BEGIN/{f=1;next} /MODELBLASTER_XPURT_PLACEMENT_END/{f=0} f' \
        "${OUT}" > "${PLACEMENT_CSV}"
    echo "  placement -> ${PLACEMENT_CSV} ($(($(wc -l <"${PLACEMENT_CSV}") - 1)) rows)"
fi
[[ "${rc}" == "0" ]] || final_rc="${rc}"
done
exit "${final_rc}"
