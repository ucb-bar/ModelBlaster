# SPDX-License-Identifier: Apache-2.0
#
# Multi-model modelblaster harness. Builds N models into one Zephyr binary,
# each a self-contained compilation unit with its own kernels.h /
# weights.h scope (via per-source include directories on per-model
# OBJECT libraries). The generated multi_main.c invokes them all
# sequentially.
#
# Pass:
#   -DMODELBLASTER_BACKEND=<scalar|rvv>          common HW backend for ALL models
#   -DMODEL_NAMES=name1;name2;...          parallel list to MODEL_DIRS
#   -DMODEL_DIRS=<absdir1>;<absdir2>;...   each is a per-target generated/
#                                          dir (holds model.c, kernels.c,
#                                          weights.c, model.h, test_io.h)
#   -DMODELBLASTER_KERNEL_CFLAGS=...             optional per-backend cflags
#                                          applied ONLY to kernels.c

cmake_minimum_required(VERSION 3.20.0)

if(NOT DEFINED MODELBLASTER_BACKEND)
    set(MODELBLASTER_BACKEND "scalar")
endif()
if(NOT DEFINED MODEL_NAMES OR NOT DEFINED MODEL_DIRS)
    message(FATAL_ERROR
        "Multi-harness requires -DMODEL_NAMES and -DMODEL_DIRS (parallel "
        "semicolon-separated lists).")
endif()
list(LENGTH MODEL_NAMES _n_names)
list(LENGTH MODEL_DIRS _n_dirs)
if(NOT _n_names EQUAL _n_dirs)
    message(FATAL_ERROR
        "MODEL_NAMES and MODEL_DIRS must have the same length "
        "(${_n_names} names vs ${_n_dirs} dirs).")
endif()

# Apply the vendored POSIX-affinity patch (Phase A —
# pthread_attr_setaffinity_np) idempotently before find_package(Zephyr).
# The patch lives in zephyr_patches/ and is applied with `git apply` at
# configure time. Re-applying is a no-op: we test with --check first and
# skip if it's already in place. See modelblaster/notes/posix_affinity_investigation.md.
set(_AFFINITY_PATCH "${CMAKE_CURRENT_LIST_DIR}/zephyr_patches/posix-affinity.patch")
set(_ZEPHYR_TREE "$ENV{ZEPHYR_BASE}")
if(EXISTS "${_AFFINITY_PATCH}" AND IS_DIRECTORY "${_ZEPHYR_TREE}")
    # Already applied?
    execute_process(
        COMMAND git -C "${_ZEPHYR_TREE}" apply --reverse --check "${_AFFINITY_PATCH}"
        RESULT_VARIABLE _patch_already_applied
        OUTPUT_QUIET ERROR_QUIET
    )
    if(_patch_already_applied EQUAL 0)
        message(STATUS "modelblaster: POSIX-affinity patch already applied to Zephyr tree")
    else()
        execute_process(
            COMMAND git -C "${_ZEPHYR_TREE}" apply --check "${_AFFINITY_PATCH}"
            RESULT_VARIABLE _patch_clean
            OUTPUT_VARIABLE _patch_check_out
            ERROR_VARIABLE _patch_check_err
        )
        if(NOT _patch_clean EQUAL 0)
            message(FATAL_ERROR
                "modelblaster: POSIX-affinity patch does not apply cleanly to "
                "${_ZEPHYR_TREE}. Patch output:\n${_patch_check_err}")
        endif()
        execute_process(
            COMMAND git -C "${_ZEPHYR_TREE}" apply "${_AFFINITY_PATCH}"
            RESULT_VARIABLE _patch_apply
        )
        if(NOT _patch_apply EQUAL 0)
            message(FATAL_ERROR "modelblaster: POSIX-affinity patch failed to apply.")
        endif()
        message(STATUS "modelblaster: applied POSIX-affinity patch to ${_ZEPHYR_TREE}")
    endif()
endif()

# Splice the backend prj.conf overlay BEFORE find_package(Zephyr).
set(CONF_FILE "prj.conf")
set(_BACKEND_CONF "${CMAKE_CURRENT_LIST_DIR}/backends/${MODELBLASTER_BACKEND}.conf")
if(EXISTS "${_BACKEND_CONF}")
    list(APPEND CONF_FILE "${_BACKEND_CONF}")
else()
    message(FATAL_ERROR
        "MODELBLASTER_BACKEND='${MODELBLASTER_BACKEND}' but no overlay at ${_BACKEND_CONF}")
endif()

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(modelblaster_harness_multi)

# Parallel-for pool: modelblaster/runtime/modelblaster_pool (raw pthreads +
# k_sem). Object library pulled in via add_subdirectory; we link it
# into every per-model objs lib + the app itself.
#
# History: the harness used to link pthreadpool (xnnpack-vendored). It
# was retired in Phase 4 of the pthreadpool->modelblaster_pool migration
# after the bench under modelblaster/microbench/threadpool/ measured a ~650x
# per-call dispatch overhead reduction (27 M cycles -> 20 k cycles at
# N=4) on the FireSim quad-rocket-saturn hwconfig and the multi_demo
# sweep confirmed up to 13x faster end-to-end at p=4. See
# modelblaster/runtime/modelblaster_pool/results/firesim_overhead_after.md.
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../runtime/modelblaster_pool"
                 modelblaster_pool_build)

# Stage per-model headers under build/modelblaster_multi/<name>_model.h and
# <name>_test_io.h so multi_main.c can include all of them with
# disambiguated names.
set(_STAGE_DIR "${CMAKE_CURRENT_BINARY_DIR}/modelblaster_multi")
file(MAKE_DIRECTORY "${_STAGE_DIR}")

set(_MODEL_SPECS "")
math(EXPR _last "${_n_dirs} - 1")
foreach(_i RANGE ${_last})
    list(GET MODEL_NAMES ${_i} _name)
    list(GET MODEL_DIRS  ${_i} _dir)

    foreach(_required model.c model.h kernels.c kernels.h weights.c weights.h buffers.c test_io.h test_io.S test_input.bin test_golden.bin)
        if(NOT EXISTS "${_dir}/${_required}")
            message(FATAL_ERROR
                "MODEL_DIRS[${_i}]='${_dir}' is missing ${_required}. "
                "Re-run modelblaster/pipeline/generate_skeleton.py — older "
                "checkouts emitted decimal-formatted const arrays in "
                "test_io.h; the current pipeline emits test_io.S "
                "(.incbin pulls test_input.bin / test_golden.bin into "
                "rodata, sidesteps cc1 OOM on kernelbench-scale inputs).")
        endif()
    endforeach()

    # Per-model OBJECT library: each compiled with its OWN include path
    # so its model.c sees the right kernels.h / weights.h / model.h. This
    # is the key to letting two models with same-named headers (different
    # mangled symbols inside) link into one binary. buffers.c lives here
    # too — model.c references its symbols via extern, so the per-model
    # OBJECT lib must define them. test_io.S uses .incbin to pull
    # test_input.bin / test_golden.bin into rodata at link time —
    # avoids cc1 choking on multi-GB decimal const arrays for
    # kernelbench-shaped tensors.
    add_library(modelblaster_${_name}_objs OBJECT
        "${_dir}/model.c"
        "${_dir}/kernels.c"
        "${_dir}/weights.c"
        "${_dir}/buffers.c"
        "${_dir}/test_io.S"
    )
    set_source_files_properties("${_dir}/test_io.S" PROPERTIES LANGUAGE ASM)
    target_include_directories(modelblaster_${_name}_objs PRIVATE "${_dir}")
    target_link_libraries(modelblaster_${_name}_objs PRIVATE zephyr_interface)
    # MODELBLASTER_USE_POOL switches the static parallel_<op> wrappers from a
    # sequential no-op fallback to the modelblaster_pool dispatch path. Linking
    # against modelblaster_pool propagates its include path (modelblaster_pool.h) to
    # model.c so the wrappers can reach the API.
    target_compile_definitions(modelblaster_${_name}_objs PRIVATE MODELBLASTER_USE_POOL=1)
    target_link_libraries(modelblaster_${_name}_objs PRIVATE modelblaster_pool)

    # Apply per-backend kernel cflags (e.g. -march=rv64gcv) ONLY to
    # kernels.c, not to model.c / weights.c. Same convention as the
    # single-model harness.
    if(DEFINED MODELBLASTER_KERNEL_CFLAGS AND NOT "${MODELBLASTER_KERNEL_CFLAGS}" STREQUAL "")
        set_source_files_properties(
            "${_dir}/kernels.c"
            PROPERTIES COMPILE_OPTIONS "${MODELBLASTER_KERNEL_CFLAGS}"
        )
    endif()

    # Stage a renamed copy of model.h + test_io.h under the build dir so
    # multi_main.c can include all models without name collisions.
    configure_file("${_dir}/model.h"   "${_STAGE_DIR}/${_name}_model.h"   COPYONLY)
    configure_file("${_dir}/test_io.h" "${_STAGE_DIR}/${_name}_test_io.h" COPYONLY)

    list(APPEND _MODEL_SPECS "${_name}:${_dir}")
endforeach()

# Generate multi_main.c via Python. Input deps: all model.h + test_io.h
# files (so editing any of them re-runs the generator).
set(_MULTI_MAIN "${_STAGE_DIR}/multi_main.c")
set(_GEN_PY "${CMAKE_CURRENT_LIST_DIR}/../pipeline/generate_multi_main.py")
set(_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")

set(_GEN_DEPS "")
foreach(_dir ${MODEL_DIRS})
    list(APPEND _GEN_DEPS "${_dir}/model.h" "${_dir}/test_io.h")
endforeach()

# Optional: pool-sweep mode. When MODELBLASTER_POOL_SIZES is set (comma-list),
# the generator emits a binary that runs every model once per pool size
# and tags blocks `[<model>@p<N>]`. Lets us collect topo_0 / topo_0_1 /
# topo_0_1_2_3 results in a single FireSim run, amortizing infrasetup.
set(_POOL_SIZES_ARG "")
if(DEFINED MODELBLASTER_POOL_SIZES AND NOT "${MODELBLASTER_POOL_SIZES}" STREQUAL "")
    set(_POOL_SIZES_ARG --pool-sizes "${MODELBLASTER_POOL_SIZES}")
endif()

add_custom_command(
    OUTPUT "${_MULTI_MAIN}"
    COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${_REPO_ROOT}"
            python3 -m modelblaster.pipeline.generate_multi_main
            --models ${_MODEL_SPECS}
            --out "${_MULTI_MAIN}"
            ${_POOL_SIZES_ARG}
    DEPENDS ${_GEN_PY} ${_GEN_DEPS}
    WORKING_DIRECTORY "${_REPO_ROOT}"
    COMMENT "Generating multi_main.c for: ${_MODEL_SPECS} pool_sizes=${MODELBLASTER_POOL_SIZES}"
    VERBATIM
)
add_custom_target(modelblaster_multi_main_gen ALL DEPENDS "${_MULTI_MAIN}")

target_sources(app PRIVATE "${_MULTI_MAIN}")
target_include_directories(app PRIVATE "${_STAGE_DIR}")
add_dependencies(app modelblaster_multi_main_gen)

# Optional: fixed modelblaster_pool worker count baked into multi_main.c.
# When unset the generator's `#define MODELBLASTER_POOL_THREADS 0` falls back
# to a NULL pool (parallel_<op> runs sequentially on the master).
if(DEFINED MODELBLASTER_POOL_THREADS)
    target_compile_definitions(app PRIVATE MODELBLASTER_POOL_THREADS=${MODELBLASTER_POOL_THREADS})
endif()

# Pull in every model's compiled objects.
foreach(_name ${MODEL_NAMES})
    target_link_libraries(app PRIVATE modelblaster_${_name}_objs)
endforeach()

target_link_libraries(app PRIVATE modelblaster_pool)

# Mirror the single-model harness: disable RISC-V gp-relative
# relaxation so big-data models (mlp_control, dronet) don't blow the
# 2KB R_RISCV_GPREL_I window. medany handles the rest.
zephyr_link_libraries(-Wl,--no-relax)

message(STATUS "modelblaster multi-harness: backend=${MODELBLASTER_BACKEND} models=${MODEL_NAMES}")
