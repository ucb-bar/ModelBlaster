#!/usr/bin/env bash
# Copyright (c) 2026 Dima Nikiforov <vnikiforov@berkeley.edu>
# SPDX-License-Identifier: Apache-2.0
#
# build_posix.sh -- build the XPU-RT schedule walker (harness_xpurt) as a
# hosted Linux executable instead of a Zephyr image.
#
# Mirrors harness_xpurt/CMakeLists.txt one-for-one, minus Zephyr:
#   * each (model, backend) pair compiles model.c + kernels.c with the
#     backend_rename -D map so several backends of one model link together;
#   * weights.c is per backend; buffers.c / test_io.S come from the primary dir;
#   * <name>_model.h / <name>_test_io.h are staged for the walker main;
#   * modelblaster_pool and every TU build with -DMB_PLATFORM_POSIX, which
#     routes their Zephyr kernel calls through runtime/mb_posix.h.
#
# Usage (all paths absolute; lists are ';'-separated, as for the CMake build):
#   MODEL_NAMES=mlp_control MODEL_DIRS_BASE=/.../generated \
#   MODEL_BACKENDS=scalar XPURT_SCHEDULE_C=/.../sched.c XPURT_MAIN_C=/.../sched_main.c \
#   XPURT_INCLUDE_DIR=/.../ OUT=/.../walker \
#   [CC=gcc] [CFLAGS_COMMON=-O2] [LDFLAGS=-static] [KERNEL_CFLAGS_<BACKEND>="-march=..."] \
#   [MODELBLASTER_AI_CPUS=8-15] \
#   [MODELBLASTER_POOL_THREADS=N] [MODELBLASTER_XPURT_TRACE=1] \
#       bash harness_xpurt/build_posix.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MB_ROOT="$(cd "${HERE}/.." && pwd)"
PKG_ROOT="$(cd "${MB_ROOT}/.." && pwd)"
# src/ holds the `modelblaster` shim, so `python -m modelblaster.*` resolves
# whatever this checkout is called; PKG_ROOT keeps the submodule layout working.
MB_PYTHONPATH="${MB_ROOT}/src:${PKG_ROOT}"

for v in MODEL_NAMES MODEL_DIRS_BASE MODEL_BACKENDS XPURT_SCHEDULE_C XPURT_MAIN_C XPURT_INCLUDE_DIR OUT; do
    [[ -n "${!v:-}" ]] || { echo "build_posix: $v is required" >&2; exit 2; }
done
CC="${CC:-gcc}"
CFLAGS_COMMON="${CFLAGS_COMMON:--O2}"
PYTHON="${PYTHON:-python3}"

IFS=';' read -ra NAMES <<< "${MODEL_NAMES}"
IFS=';' read -ra BASES <<< "${MODEL_DIRS_BASE}"
IFS=';,' read -ra BACKENDS <<< "${MODEL_BACKENDS}"
[[ ${#NAMES[@]} -eq ${#BASES[@]} ]] || {
    echo "build_posix: MODEL_NAMES and MODEL_DIRS_BASE differ in length" >&2; exit 2; }
PRIMARY="${BACKENDS[0]}"

WORK="${OUT}.d"
STAGE="${WORK}/stage"
rm -rf "${WORK}"; mkdir -p "${STAGE}"

DEFS=(-D_GNU_SOURCE -DMB_PLATFORM_POSIX -DMODELBLASTER_DISABLE_UNMANGLED)
# CPUs whose threads must register before they may run there (the K3 A100
# cluster, "8-15"); see runtime/mb_placement.h. Empty on the K1 and the host.
[[ -n "${MODELBLASTER_AI_CPUS:-}" ]] && DEFS+=(-DMODELBLASTER_AI_CPUS="\"${MODELBLASTER_AI_CPUS}\"")
# Kernel flags default to the backend's kernel_cflags (pipeline/backends.py)
# when the compiler targets RISC-V: they set -march (zvl1024b for the K3 A100
# backends, whose code is illegal on the X100) and the weight-layout define that
# must match weights.c. KERNEL_CFLAGS_<BACKEND> overrides.
RISCV_TARGET=0
[[ "$("${CC}" -dumpmachine 2>/dev/null)" == riscv* ]] && RISCV_TARGET=1
backend_cflags() {
    PYTHONPATH="${MB_PYTHONPATH}" "${PYTHON}" -c "
import sys
from modelblaster.pipeline.backends import BACKENDS
b = BACKENDS.get(sys.argv[1])
print(' '.join(f.replace('<repo_root>', sys.argv[2]) for f in (b.kernel_cflags if b else ())))
" "$1" "${MB_ROOT}"
}
INC=(-I"${MB_ROOT}/runtime" -I"${MB_ROOT}/runtime/modelblaster_pool")
OBJS=()

cc() {  # cc <out.o> <src> [extra flags...]
    local o="$1" s="$2"; shift 2
    # shellcheck disable=SC2086  # CFLAGS_COMMON is a deliberate word list
    "${CC}" ${CFLAGS_COMMON} -std=gnu11 "${DEFS[@]}" "${INC[@]}" "$@" -c "$s" -o "$o"
    OBJS+=("$o")
}

cc "${WORK}/modelblaster_pool.o" "${MB_ROOT}/runtime/modelblaster_pool/modelblaster_pool.c"
cc "${WORK}/mb_placement.o" "${MB_ROOT}/runtime/mb_placement.c"

for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"; base="${BASES[$i]}"; pdir="${base}/${PRIMARY}"
    # One copy of the intermediate buffers per instance of this network in the
    # schedule (ingest writes XPURT_N_INSTANCES_<NAME> into the table header),
    # so overlapping instances never share scratch. 1 if the header says nothing.
    n_inst="$(grep -hs "^#define XPURT_N_INSTANCES_${name^^} " "${XPURT_INCLUDE_DIR}"/*.h | awk '{print $3}' | head -1)"
    BUFSETS=(-DMB_N_BUF_SETS="${n_inst:-1}")
    echo "build_posix: ${name} buffer sets: ${n_inst:-1}"
    for f in model.h test_io.h weights.c buffers.c; do
        [[ -f "${pdir}/$f" ]] || { echo "build_posix: ${pdir} missing $f" >&2; exit 1; }
    done
    cp "${pdir}/model.h"   "${STAGE}/${name}_model.h"
    # test_io.h pulls in a bare "model.h"; point it at the staged copy so
    # the walker main sees one definition of the model types.
    sed "s|#include \"model.h\"|#include \"${name}_model.h\"|" \
        "${pdir}/test_io.h" > "${STAGE}/${name}_test_io.h"

    for bs in "${BACKENDS[@]}"; do
        bdir="${base}/${bs}"
        for f in model.c kernels.c weights.c; do
            [[ -f "${bdir}/$f" ]] || { echo "build_posix: ${bdir} missing $f" >&2; exit 1; }
        done
        # A failed helper would otherwise leave RENAME or KFLAGS empty and the
        # build would go on with wrong symbols or flags, so stop instead.
        rename_out="$(PYTHONPATH="${MB_PYTHONPATH}" "${PYTHON}" -m modelblaster.pipeline.backend_rename \
            --ir "${base}/graph.json" --backend "${bs}" --separator ' ')" \
            || { echo "build_posix: backend_rename failed for ${name}/${bs}" >&2; exit 1; }
        read -ra RENAME <<< "${rename_out}"
        kvar="KERNEL_CFLAGS_${bs^^}"
        if [[ -n "${!kvar+x}" ]]; then
            read -ra KFLAGS <<< "${!kvar}"
        elif [[ "${RISCV_TARGET}" == 1 ]]; then
            kflags_out="$(backend_cflags "${bs}")" \
                || { echo "build_posix: cannot read kernel_cflags of ${bs}" >&2; exit 1; }
            read -ra KFLAGS <<< "${kflags_out}"
        else
            KFLAGS=()
        fi
        echo "build_posix: ${name}/${bs} kernel flags: ${KFLAGS[*]:-(none)}"
        cc "${WORK}/${name}_${bs}_model.o"   "${bdir}/model.c"   -I"${bdir}" \
            -DMODELBLASTER_USE_POOL=1 "${BUFSETS[@]}" "${RENAME[@]}"
        cc "${WORK}/${name}_${bs}_kernels.o" "${bdir}/kernels.c" -I"${bdir}" \
            -DMODELBLASTER_USE_POOL=1 "${RENAME[@]}" "${KFLAGS[@]}"
        # Weights are backend-suffixed at codegen (each backend may pack
        # them differently), so every backend links its own weights.c.
        cc "${WORK}/${name}_${bs}_weights.o" "${bdir}/weights.c" -I"${bdir}"
    done
    cc "${WORK}/${name}_buffers.o" "${pdir}/buffers.c" -I"${pdir}" "${BUFSETS[@]}"
    if [[ -f "${pdir}/test_io.S" ]]; then
        # test_io.S .incbin's the input/golden blobs by the absolute path they
        # were generated at; strip it so the model dir can be built anywhere.
        sed -E 's|(\.incbin[[:space:]]+")[^"]*/([^"/]+")|\1\2|' "${pdir}/test_io.S" \
            > "${WORK}/${name}_test_io.S"
        ( cd "${pdir}" && "${CC}" -c "${WORK}/${name}_test_io.S" -o "${WORK}/${name}_test_io.o" )
        OBJS+=("${WORK}/${name}_test_io.o")
    fi
done

MAIN_DEFS=()
[[ -n "${MODELBLASTER_POOL_THREADS:-}" ]] && MAIN_DEFS+=(-DMODELBLASTER_POOL_THREADS="${MODELBLASTER_POOL_THREADS}")
[[ "${MODELBLASTER_XPURT_TRACE:-0}" == 1 ]] && MAIN_DEFS+=(-DMODELBLASTER_XPURT_TRACE=1)
# -iquote keeps these dirs from shadowing system headers such as <sched.h>.
cc "${WORK}/xpurt_sched.o" "${XPURT_SCHEDULE_C}" -iquote "${STAGE}" -iquote "${XPURT_INCLUDE_DIR}"
cc "${WORK}/xpurt_main.o"  "${XPURT_MAIN_C}"     -iquote "${STAGE}" -iquote "${XPURT_INCLUDE_DIR}" "${MAIN_DEFS[@]}"

# shellcheck disable=SC2086  # LDFLAGS is a deliberate word list
"${CC}" "${OBJS[@]}" ${LDFLAGS:-} -pthread -lm -o "${OUT}"
echo "build_posix: wrote ${OUT}"
