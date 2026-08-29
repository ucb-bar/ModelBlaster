#!/usr/bin/env bash
# Execute an XPU-RT schedule as a multi-model periodic workload on the K1,
# entirely through ModelBlaster.
#
# This replaces merlin's merlin-dispatch-scheduler. That runner executes IREE
# VMFBs, which meant every measurement went through the IREE path and that
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
#      CMake wants (one weights/buffers TU per model, shared across backends).
#   2. ingest the XPU-RT schedule -> dispatch_table.{c,h}
#   3. generate xpurt_main.c with --platform linux (rdtime + POSIX semaphores)
#   4. cross-build harness_xpurt_linux
#   5. scp to the board, run pinned, pull back stdout + the trace CSV
#
# Usage:
#   scripts/run_xpurt_k1.sh --schedule <scheduled_*.json> \
#       [--models mlp_control,dronet] [--backends scalar] [--quant int8]
#
# No credentials are read or written. Board access comes from the ssh config
# entry named by MODELBLASTER_K1_HOST (default "k1").

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

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
BACKENDS="rvv_x60,rvv_x60"
QUANT="int8"
# reference = curated/hand-written kernels, no LLM. Codex is only for NEW kernel
# bodies and must be requested explicitly; there is no Bedrock path here.
KERNEL_BACKEND="${BACKEND:-reference}"
# Codex only, no fallback. Consulted only when BACKEND=llm.
export LLM_PROVIDER="${LLM_PROVIDER:-codex}"
# The K1 registry is measured, not assumed -- cluster 0 carries IME, cluster 1
# traps on it. See cores/spacemit_k1.json.
REGISTRY="${REGISTRY:-cores/spacemit_k1.json}"
# Both K1 clusters are RVV-capable and measured equivalent (0.996 ratio), so a
# single backend kind covers both; the cluster distinction is a placement
# constraint, not a codegen one.
# Two DISTINCT kinds, one per cluster. `kind` is how ingest resolves an
# abstract CPU_P#n / CPU_E#n slot to a hart -- it takes the n-th core of that
# kind -- so a single kind for all eight cores makes CPU_E#n alias CPU_P#n, and
# a 4+4 config silently double-books cluster 0 while never touching cluster 1.
# The two clusters run the same ISA and measure equivalent (0.996), so this is
# a placement-pool distinction, not a codegen one: both compile rvv_x60.
CPU_P_KIND="${CPU_P_KIND:-rvv}"
CPU_E_KIND="${CPU_E_KIND:-rvv_c1}"
# core_kind (what the SCHEDULE says, and what the walker matches on) is distinct
# from the backend tag (what the BINARY was built as). rvv_x60 is a K1-specific
# build of the same rvv kind; conflating them makes every worker refuse every
# entry -- strcmp("rvv_x60","rvv") != 0 -- and the run completes having executed
# nothing, with entries_done=0 and an all-zero trace.
CORE_KINDS="${CORE_KINDS:-}"
OUT_ROOT="build/k1_xpurt"
TRACE=1
# Per-dispatch-END JSON lines on stdout, for xpu-rt/streaming_feedback.py.
# Off by default: the trace block at exit is enough to EXPLAIN a run, and
# streaming only earns its keep when something is watching live. Pipe the
# run through `tee` when you turn this on -- the lines are interleaved with
# the harness's normal output by design, so one stream carries both.
STREAM="${MB_XPURT_STREAM:-0}"
CPU_IDS=""
JOBS="$(nproc)"

HOST="${MODELBLASTER_K1_HOST:-k1}"
REMOTE_ROOT="${MODELBLASTER_K1_REMOTE_ROOT:-/root/mb_k1}"
CROSS="${CROSS:-/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-linux-gnu-}"
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
UNIQUE_BACKENDS="$(printf '%s\n' "${BACKEND_LIST[@]}" | awk '!seen[$0]++' | paste -sd, -)"
if [[ ${#CORE_KIND_LIST[@]} -ne ${#BACKEND_LIST[@]} ]]; then
    echo "core-kinds (${CORE_KINDS}) and backends (${BACKENDS}) must be" >&2
    echo "parallel lists: kind k is executed by backend k." >&2
    exit 2
fi

SCHED_NAME="$(basename "${SCHEDULE}" .json)"
# Absolute: CMake resolves relative source paths against its own source dir,
# not the invoking cwd, so a relative XPURT_MAIN_C is looked up under
# harness_xpurt_linux/ and not found.
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

    for bs in "${BACKEND_LIST[@]}"; do
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
        # --platform linux swaps rdcycle (SIGILLs here) for rdtime.
        ${PY} -m modelblaster.pipeline.generate_skeleton \
            --ir "${base}/graph.json" --weights "${base}/weights.npz" \
            --io "${base}/io.npz" --out-dir "${bdir}" \
            --backend "${bs}" --platform linux
        ${PY} -m modelblaster.pipeline.generate_kernels \
            --ir "${base}/graph.json" --out-dir "${bdir}" \
            --target "${bs}" --backend "${KERNEL_BACKEND}" --quant "${QUANT}" \
            --global-curated-dir "${REPO_ROOT}/kernels"
    done

    # A vector backend that silently resolved ops to the scalar reference is
    # the single most expensive failure this pipeline has: yolov8_nano measured
    # 0.81x against the scalar build (i.e. SLOWER) purely because the fused
    # conv2d_batchnorm2d_silu_s8 had no curated RVV kernel and nothing said so.
    # Catch it here, before the board time is spent.
    for bs in "${BACKEND_LIST[@]}"; do
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
    IR_ARGS+=(--ir "${model}:${REPO_ROOT}/${OUT_ROOT}/${model}/${QUANT}/graph.json")
done
${PY} -m modelblaster.pipeline.ingest_xpurt_schedule \
    --schedule "${SCHEDULE}" --registry "${REGISTRY}" \
    "${IR_ARGS[@]}" --out "${SCHED_C}" --name "${SCHED_NAME}" \
    --cpu-p-kind "${CPU_P_KIND}" --cpu-e-kind "${CPU_E_KIND}"
[[ -f "${SCHED_C}" ]] || { echo "ingest produced no dispatch table" >&2; exit 1; }
echo "  ${SCHED_C}"

echo "=== 3/5 generate the walker (platform linux) ==="
MAIN_C="${GEN_DIR}/${SCHED_NAME}_main.c"
${PY} -m modelblaster.pipeline.generate_xpurt_main \
    --schedule "${SCHEDULE}" --out "${MAIN_C}" --name "${SCHED_NAME}" \
    --dispatch-table-header "$(basename "${SCHED_H}")" \
    --platform linux --core-kinds "${CORE_KINDS}" --backends "${BACKENDS}" \
    --networks "${MODELS}"

echo "=== 4/5 cross-build ==="
CMAKE_ARGS=(
    "-DCMAKE_C_COMPILER=${CROSS}gcc"
    "-DCMAKE_C_FLAGS=-O2 -static"
    "-DCMAKE_SYSTEM_NAME=Linux"
    "-DCMAKE_SYSTEM_PROCESSOR=riscv64"
    "-DPYTHON_EXECUTABLE=${PY}"
    "-DMODEL_BACKENDS=${UNIQUE_BACKENDS}"
    "-DMODEL_NAMES=${MODEL_NAMES}"
    "-DMODEL_DIRS_BASE=${MODEL_DIRS_BASE}"
    "-DXPURT_SCHEDULE_C=${SCHED_C}"
    "-DXPURT_MAIN_C=${MAIN_C}"
    "-DXPURT_INCLUDE_DIR=${GEN_DIR}"
)
for bs in "${BACKEND_LIST[@]}"; do
    BS_UPPER=$(echo "${bs}" | tr '[:lower:]' '[:upper:]')
    KF=$(${PY} -c "
from modelblaster.pipeline.backends import get
print(' '.join(get('${bs}').resolved_kernel_cflags('${REPO_ROOT}')))
" 2>/dev/null || echo "")
    [[ -n "${KF}" ]] && CMAKE_ARGS+=("-DMODELBLASTER_KERNEL_CFLAGS_${BS_UPPER}=${KF}")
done
[[ "${TRACE}" == "1" ]] && CMAKE_ARGS+=("-DMODELBLASTER_XPURT_TRACE=ON")
[[ "${STREAM}" == "1" ]] && CMAKE_ARGS+=("-DMODELBLASTER_XPURT_STREAM=ON")
# Compile kernels.c with a different compiler than the rest. Needed for the
# Zvfh fp16 kernels, whose intrinsics do not exist in GCC 13.2 -- the only
# riscv64-unknown-linux-gnu compiler here. Same variable and same meaning as
# scripts/run_model_k1.sh, so the standalone and scheduled paths agree.
# Unset by default; see harness_xpurt_linux/CMakeLists.txt for why it is an
# explicit opt-in rather than a search.
[[ -n "${MODELBLASTER_KERNEL_CC:-}" ]] && \
    CMAKE_ARGS+=("-DMODELBLASTER_KERNEL_CC=${MODELBLASTER_KERNEL_CC}")

cmake -S harness_xpurt_linux -B "${BUILD_DIR}" "${CMAKE_ARGS[@]}" >"${BUILD_DIR}/cmake.log" 2>&1 \
    || { echo "cmake configure failed; tail of ${BUILD_DIR}/cmake.log:" >&2; tail -30 "${BUILD_DIR}/cmake.log" >&2; exit 1; }
cmake --build "${BUILD_DIR}" -j "${JOBS}" >"${BUILD_DIR}/build.log" 2>&1 \
    || { echo "build failed; tail of ${BUILD_DIR}/build.log:" >&2; tail -40 "${BUILD_DIR}/build.log" >&2; exit 1; }
BIN="${BUILD_DIR}/xpurt_harness"
[[ -x "${BIN}" ]] || { echo "no binary at ${BIN}" >&2; exit 1; }
echo "  $(file -b "${BIN}" | cut -c1-70)"

echo "=== 5/5 deploy and run on ${HOST} ==="
ssh "${HOST}" "mkdir -p ${REMOTE_ROOT}/xpurt"
scp -q "${BIN}" "${HOST}:${REMOTE_ROOT}/xpurt/${SCHED_NAME}"
RUN="cd ${REMOTE_ROOT}/xpurt && ulimit -n 8192 &&"
[[ -n "${CPU_IDS}" ]] && RUN="${RUN} taskset -c ${CPU_IDS}"
RUN="${RUN} ./${SCHED_NAME}"
OUT="${GEN_DIR}/${SCHED_NAME}_stdout.txt"
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
REMOTE_LOG="${REMOTE_ROOT}/xpurt/${SCHED_NAME}_stdout.txt"
set +e
ssh "${HOST}" "${RUN} >${REMOTE_LOG} 2>&1; echo \$? >${REMOTE_LOG}.rc"
scp -q "${HOST}:${REMOTE_LOG}" "${OUT}"
rc=$(ssh "${HOST}" "cat ${REMOTE_LOG}.rc" 2>/dev/null || echo 255)
set -e
echo "  exit=${rc}  stdout -> ${OUT}"
grep -E 'MODELBLASTER_VERIFY|max_abs_err|FAIL|PASS' "${OUT}" | head -20 || true

if [[ "${TRACE}" == "1" ]] && grep -q 'MODELBLASTER_XPURT_TRACE_BEGIN' "${OUT}"; then
    TRACE_CSV="${GEN_DIR}/${SCHED_NAME}_trace.csv"
    awk '/MODELBLASTER_XPURT_TRACE_BEGIN/{f=1;next} /MODELBLASTER_XPURT_TRACE_END/{f=0} f' \
        "${OUT}" > "${TRACE_CSV}"
    echo "  trace -> ${TRACE_CSV} ($(($(wc -l <"${TRACE_CSV}") - 1)) rows)"
fi
exit "${rc}"
