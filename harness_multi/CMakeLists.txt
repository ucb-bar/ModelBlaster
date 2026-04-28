# SPDX-License-Identifier: Apache-2.0
#
# Multi-model agents harness. Builds N models into one Zephyr binary,
# each a self-contained compilation unit with its own kernels.h /
# weights.h scope (via per-source include directories on per-model
# OBJECT libraries). The generated multi_main.c invokes them all
# sequentially.
#
# Pass:
#   -DAGENTS_BACKEND=<scalar|rvv>          common HW backend for ALL models
#   -DMODEL_NAMES=name1;name2;...          parallel list to MODEL_DIRS
#   -DMODEL_DIRS=<absdir1>;<absdir2>;...   each is a per-target generated/
#                                          dir (holds model.c, kernels.c,
#                                          weights.c, model.h, test_io.h)
#   -DAGENTS_KERNEL_CFLAGS=...             optional per-backend cflags
#                                          applied ONLY to kernels.c

cmake_minimum_required(VERSION 3.20.0)

if(NOT DEFINED AGENTS_BACKEND)
    set(AGENTS_BACKEND "scalar")
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
# skip if it's already in place. See agents/notes/posix_affinity_investigation.md.
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
        message(STATUS "agents: POSIX-affinity patch already applied to Zephyr tree")
    else()
        execute_process(
            COMMAND git -C "${_ZEPHYR_TREE}" apply --check "${_AFFINITY_PATCH}"
            RESULT_VARIABLE _patch_clean
            OUTPUT_VARIABLE _patch_check_out
            ERROR_VARIABLE _patch_check_err
        )
        if(NOT _patch_clean EQUAL 0)
            message(FATAL_ERROR
                "agents: POSIX-affinity patch does not apply cleanly to "
                "${_ZEPHYR_TREE}. Patch output:\n${_patch_check_err}")
        endif()
        execute_process(
            COMMAND git -C "${_ZEPHYR_TREE}" apply "${_AFFINITY_PATCH}"
            RESULT_VARIABLE _patch_apply
        )
        if(NOT _patch_apply EQUAL 0)
            message(FATAL_ERROR "agents: POSIX-affinity patch failed to apply.")
        endif()
        message(STATUS "agents: applied POSIX-affinity patch to ${_ZEPHYR_TREE}")
    endif()
endif()

# Splice the backend prj.conf overlay BEFORE find_package(Zephyr).
set(CONF_FILE "prj.conf")
set(_BACKEND_CONF "${CMAKE_CURRENT_LIST_DIR}/backends/${AGENTS_BACKEND}.conf")
if(EXISTS "${_BACKEND_CONF}")
    list(APPEND CONF_FILE "${_BACKEND_CONF}")
else()
    message(FATAL_ERROR
        "AGENTS_BACKEND='${AGENTS_BACKEND}' but no overlay at ${_BACKEND_CONF}")
endif()

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(agents_harness_multi)

# pthreadpool — same source the samples/xnnpack project pulls in. Build it
# directly (not through XNNPACK) so we get the threadpool target with
# minimal extra surface. PTHREADPOOL_SOURCE_DIR / FXDIV_SOURCE_DIR point
# at the executorch-vendored submodules so CMake doesn't try to download
# them at configure time.
set(_REPO_ROOT_ABS "${CMAKE_CURRENT_LIST_DIR}/../..")
set(PTHREADPOOL_SOURCE_DIR
    "${_REPO_ROOT_ABS}/third-party/executorch/backends/xnnpack/third-party/pthreadpool"
    CACHE STRING "pthreadpool source directory")
set(FXDIV_SOURCE_DIR
    "${_REPO_ROOT_ABS}/third-party/executorch/backends/xnnpack/third-party/FXdiv"
    CACHE STRING "FXdiv source directory")
set(PTHREADPOOL_BUILD_TESTS      OFF CACHE BOOL "")
set(PTHREADPOOL_BUILD_BENCHMARKS OFF CACHE BOOL "")
set(PTHREADPOOL_ALLOW_DEPRECATED_API OFF CACHE BOOL "")
add_subdirectory("${PTHREADPOOL_SOURCE_DIR}" pthreadpool_build)
# pthreadpool's CMake builds it without Zephyr's interface flags by
# default; pull them in so it sees the same -march/-mabi and includes as
# the rest of the app.
target_link_libraries(pthreadpool PRIVATE zephyr_interface)

# Stage per-model headers under build/agents_multi/<name>_model.h and
# <name>_test_io.h so multi_main.c can include all of them with
# disambiguated names.
set(_STAGE_DIR "${CMAKE_CURRENT_BINARY_DIR}/agents_multi")
file(MAKE_DIRECTORY "${_STAGE_DIR}")

set(_MODEL_SPECS "")
math(EXPR _last "${_n_dirs} - 1")
foreach(_i RANGE ${_last})
    list(GET MODEL_NAMES ${_i} _name)
    list(GET MODEL_DIRS  ${_i} _dir)

    foreach(_required model.c model.h kernels.c kernels.h weights.c weights.h test_io.h)
        if(NOT EXISTS "${_dir}/${_required}")
            message(FATAL_ERROR
                "MODEL_DIRS[${_i}]='${_dir}' is missing ${_required}")
        endif()
    endforeach()

    # Per-model OBJECT library: each compiled with its OWN include path
    # so its model.c sees the right kernels.h / weights.h / model.h. This
    # is the key to letting two models with same-named headers (different
    # mangled symbols inside) link into one binary.
    add_library(agents_${_name}_objs OBJECT
        "${_dir}/model.c"
        "${_dir}/kernels.c"
        "${_dir}/weights.c"
    )
    target_include_directories(agents_${_name}_objs PRIVATE "${_dir}")
    target_link_libraries(agents_${_name}_objs PRIVATE zephyr_interface)
    # AGENTS_USE_POOL switches the static parallel_<op> wrappers from a
    # sequential no-op fallback to the pthreadpool dispatch path. Linking
    # against the pthreadpool target propagates pthreadpool.h's include
    # dir to model.c.
    target_compile_definitions(agents_${_name}_objs PRIVATE AGENTS_USE_POOL=1)
    target_link_libraries(agents_${_name}_objs PRIVATE pthreadpool)

    # Apply per-backend kernel cflags (e.g. -march=rv64gcv) ONLY to
    # kernels.c, not to model.c / weights.c. Same convention as the
    # single-model harness.
    if(DEFINED AGENTS_KERNEL_CFLAGS AND NOT "${AGENTS_KERNEL_CFLAGS}" STREQUAL "")
        set_source_files_properties(
            "${_dir}/kernels.c"
            PROPERTIES COMPILE_OPTIONS "${AGENTS_KERNEL_CFLAGS}"
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

add_custom_command(
    OUTPUT "${_MULTI_MAIN}"
    COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${_REPO_ROOT}"
            python3 -m agents.pipeline.generate_multi_main
            --models ${_MODEL_SPECS}
            --out "${_MULTI_MAIN}"
    DEPENDS ${_GEN_PY} ${_GEN_DEPS}
    WORKING_DIRECTORY "${_REPO_ROOT}"
    COMMENT "Generating multi_main.c for: ${_MODEL_SPECS}"
    VERBATIM
)
add_custom_target(agents_multi_main_gen ALL DEPENDS "${_MULTI_MAIN}")

target_sources(app PRIVATE "${_MULTI_MAIN}")
target_include_directories(app PRIVATE "${_STAGE_DIR}")
add_dependencies(app agents_multi_main_gen)

# Optional: fixed pthreadpool worker count baked into multi_main.c.
# When unset the generator's `#define AGENTS_POOL_THREADS 0` falls back
# to pthreadpool's auto-detect (CONFIG_MP_MAX_NUM_CPUS).
if(DEFINED AGENTS_POOL_THREADS)
    target_compile_definitions(app PRIVATE AGENTS_POOL_THREADS=${AGENTS_POOL_THREADS})
endif()

# Pull in every model's compiled objects.
foreach(_name ${MODEL_NAMES})
    target_link_libraries(app PRIVATE agents_${_name}_objs)
endforeach()

target_link_libraries(app PRIVATE pthreadpool)

# Mirror the single-model harness: disable RISC-V gp-relative
# relaxation so big-data models (mlp_control, dronet) don't blow the
# 2KB R_RISCV_GPREL_I window. medany handles the rest.
zephyr_link_libraries(-Wl,--no-relax)

message(STATUS "agents multi-harness: backend=${AGENTS_BACKEND} models=${MODEL_NAMES}")
