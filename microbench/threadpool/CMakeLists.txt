# SPDX-License-Identifier: Apache-2.0
#
# Threadpool microbench harness. Builds one of three benchmark binaries
# (selected via -DBENCH_TARGET={pthreadpool,zephyr_threads,pthreads_raw})
# with the same SMP / POSIX / pthreadpool footprint the multi-model
# harness uses, so the measurements transfer directly.
#
# Pass:
#   -DBENCH_TARGET=<pthreadpool|zephyr_threads|pthreads_raw>
#   -DBENCH_PTHREADPOOL_VARIANT=<default|spin>   (only meaningful when
#       BENCH_TARGET=pthreadpool; "spin" disables the cond_wait fallback
#       by linking a pthreadpool that's been compiled with
#       PTHREADPOOL_SYNC_PRIMITIVE=condvar plus a large
#       PTHREADPOOL_SPIN_WAIT_ITERATIONS, effectively keeping the
#       master in its spin loop the whole wait.) Two variants get
#       distinct CMake target names so they can co-exist.
#
# Includes the same vendored POSIX-affinity patch hook as
# agents/harness_multi/CMakeLists.txt and a copy of the patch file in
# zephyr_patches/.

cmake_minimum_required(VERSION 3.20.0)

if(NOT DEFINED BENCH_TARGET)
    set(BENCH_TARGET "pthreadpool")
endif()
set(_VALID_TARGETS "pthreadpool" "zephyr_threads" "pthreads_raw")
list(FIND _VALID_TARGETS "${BENCH_TARGET}" _idx)
if(_idx EQUAL -1)
    message(FATAL_ERROR
        "BENCH_TARGET='${BENCH_TARGET}' must be one of: ${_VALID_TARGETS}")
endif()

if(NOT DEFINED BENCH_PTHREADPOOL_VARIANT)
    set(BENCH_PTHREADPOOL_VARIANT "default")
endif()

# Apply the vendored POSIX-affinity patch idempotently (same logic as
# harness_multi). Workers in bench_pthreads_raw need
# pthread_attr_setaffinity_np to land on the right hart.
set(_AFFINITY_PATCH
    "${CMAKE_CURRENT_LIST_DIR}/../../harness_multi/zephyr_patches/posix-affinity.patch")
set(_ZEPHYR_TREE "$ENV{ZEPHYR_BASE}")
if(EXISTS "${_AFFINITY_PATCH}" AND IS_DIRECTORY "${_ZEPHYR_TREE}")
    execute_process(
        COMMAND git -C "${_ZEPHYR_TREE}" apply --reverse --check "${_AFFINITY_PATCH}"
        RESULT_VARIABLE _patch_already_applied
        OUTPUT_QUIET ERROR_QUIET
    )
    if(_patch_already_applied EQUAL 0)
        message(STATUS "microbench: POSIX-affinity patch already applied")
    else()
        execute_process(
            COMMAND git -C "${_ZEPHYR_TREE}" apply --check "${_AFFINITY_PATCH}"
            RESULT_VARIABLE _patch_clean
            ERROR_VARIABLE _patch_check_err
        )
        if(NOT _patch_clean EQUAL 0)
            message(FATAL_ERROR
                "microbench: POSIX-affinity patch does not apply cleanly:\n"
                "${_patch_check_err}")
        endif()
        execute_process(
            COMMAND git -C "${_ZEPHYR_TREE}" apply "${_AFFINITY_PATCH}"
            RESULT_VARIABLE _patch_apply
        )
        if(NOT _patch_apply EQUAL 0)
            message(FATAL_ERROR "microbench: POSIX-affinity patch failed.")
        endif()
        message(STATUS "microbench: applied POSIX-affinity patch")
    endif()
endif()

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(agents_microbench_threadpool)

# Wire pthreadpool only if the target needs it. We keep the same
# vendored copy as harness_multi.
if(BENCH_TARGET STREQUAL "pthreadpool")
    set(_REPO_ROOT_ABS "${CMAKE_CURRENT_LIST_DIR}/../../..")
    set(PTHREADPOOL_SOURCE_DIR
        "${_REPO_ROOT_ABS}/third-party/executorch/backends/xnnpack/third-party/pthreadpool"
        CACHE STRING "pthreadpool source directory")
    set(FXDIV_SOURCE_DIR
        "${_REPO_ROOT_ABS}/third-party/executorch/backends/xnnpack/third-party/FXdiv"
        CACHE STRING "FXdiv source directory")
    set(PTHREADPOOL_BUILD_TESTS      OFF CACHE BOOL "")
    set(PTHREADPOOL_BUILD_BENCHMARKS OFF CACHE BOOL "")
    set(PTHREADPOOL_ALLOW_DEPRECATED_API OFF CACHE BOOL "")

    if(BENCH_PTHREADPOOL_VARIANT STREQUAL "spin")
        # "spin" variant: ratchet PTHREADPOOL_SPIN_WAIT_ITERATIONS to a
        # value larger than any reasonable cross-hart wake time so the
        # master never falls back to pthread_cond_wait. The default in
        # threadpool-common.h is 1_000_000; on FireSim a single spin
        # iteration is one yield()-equivalent which is ~kcycles, so
        # 1M may already be enough — but bumping it removes any doubt.
        # We override via a target-level compile definition. To make
        # the override take effect on the THIRD-PARTY pthreadpool target
        # (which is added by add_subdirectory), we set it as a CMake
        # CACHE variable that the pthreadpool target picks up via its
        # own definitions list.
        # NOTE: pthreadpool's CMakeLists doesn't actually expose this
        # constant, so we patch it via a compile-definition that
        # overrides the macro for the pthreadpool TU. The PRIVATE
        # define on the pthreadpool target post-add_subdirectory works
        # because the macro is referenced in src/pthreads.c which is
        # compiled into the pthreadpool target.
    endif()

    add_subdirectory("${PTHREADPOOL_SOURCE_DIR}" pthreadpool_build)
    target_link_libraries(pthreadpool PRIVATE zephyr_interface)

    if(BENCH_PTHREADPOOL_VARIANT STREQUAL "spin")
        # 100 million spin iters — enough for any cross-hart wake on
        # this RTL. We can't override it via -D (pthreadpool's
        # threadpool-common.h unconditionally `#define`s the macro
        # later in the TU) and the brief forbids editing the vendored
        # source. The trick: -include forcibly inserts our header at
        # the very top of every pthreadpool .c, so by the time
        # threadpool-common.h is `#include`d its include guard is
        # already satisfied (our copy uses the same `#pragma once`).
        # spin_overrides/threadpool-common.h is a byte-identical copy
        # of upstream's header except for the bumped spin constant.
        target_compile_options(pthreadpool PRIVATE
            -include "${CMAKE_CURRENT_LIST_DIR}/spin_overrides/spin_inject.h")
        message(STATUS "microbench: pthreadpool variant=spin "
                       "(100M spin iters via -include override)")
    else()
        message(STATUS "microbench: pthreadpool variant=default "
                       "(condvar fallback after 1M spins)")
    endif()
endif()

# Pick the source file. All three share bench_common.h — pulled in
# via target_include_directories.
set(_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/src")
if(BENCH_TARGET STREQUAL "pthreadpool")
    target_sources(app PRIVATE "${_SRC_DIR}/bench_pthreadpool.c")
elseif(BENCH_TARGET STREQUAL "zephyr_threads")
    target_sources(app PRIVATE "${_SRC_DIR}/bench_zephyr_threads.c")
elseif(BENCH_TARGET STREQUAL "pthreads_raw")
    target_sources(app PRIVATE "${_SRC_DIR}/bench_pthreads_raw.c")
endif()
target_include_directories(app PRIVATE "${_SRC_DIR}")

if(BENCH_TARGET STREQUAL "pthreadpool")
    target_link_libraries(app PRIVATE pthreadpool)
    target_compile_definitions(app PRIVATE
        BENCH_PTHREADPOOL_VARIANT_NAME="${BENCH_PTHREADPOOL_VARIANT}")
endif()

# Same RISC-V relax workaround the model harnesses apply.
zephyr_link_libraries(-Wl,--no-relax)

message(STATUS "microbench: BENCH_TARGET=${BENCH_TARGET} "
               "variant=${BENCH_PTHREADPOOL_VARIANT}")
