# SPDX-License-Identifier: Apache-2.0
#
# Standalone unit-test for agents_pool. Builds a Zephyr binary that:
#   1. Creates an agents_pool of size 4.
#   2. Runs agents_pool_parallelize_1d with a function that sums into
#      per-worker buckets, then re-checks the bucket totals.
#   3. Prints PASS/FAIL between begin/end markers + a tiny per-call
#      cycles datapoint so we can sanity-check it's in the same
#      ballpark as the bench's pthreads_raw row.
#
# Build:
#   west build -p -b spike_riscv64 \
#       agents/runtime/agents_pool/test_app
#   spike -p4 --isa=rv64gcv_zicntr build/zephyr/zephyr.elf
#
# For FireSim, layer in the existing chipyard overlay:
#   west build -p -b chipyard_riscv64/rocketchip_virt_riscv64 \
#       agents/runtime/agents_pool/test_app -- \
#       -DEXTRA_CONF_FILE=$(realpath agents/harness/backends/firesim_chipyard.conf)

cmake_minimum_required(VERSION 3.20.0)

# Apply the vendored POSIX-affinity patch idempotently — same logic as
# harness_multi/. Workers in agents_pool need
# pthread_attr_setaffinity_np to land on the right hart.
set(_AFFINITY_PATCH
    "${CMAKE_CURRENT_LIST_DIR}/../../../harness_multi/zephyr_patches/posix-affinity.patch")
set(_ZEPHYR_TREE "$ENV{ZEPHYR_BASE}")
if(EXISTS "${_AFFINITY_PATCH}" AND IS_DIRECTORY "${_ZEPHYR_TREE}")
    execute_process(
        COMMAND git -C "${_ZEPHYR_TREE}" apply --reverse --check "${_AFFINITY_PATCH}"
        RESULT_VARIABLE _patch_already_applied
        OUTPUT_QUIET ERROR_QUIET
    )
    if(NOT _patch_already_applied EQUAL 0)
        execute_process(
            COMMAND git -C "${_ZEPHYR_TREE}" apply --check "${_AFFINITY_PATCH}"
            RESULT_VARIABLE _patch_clean
            ERROR_VARIABLE _patch_check_err
        )
        if(NOT _patch_clean EQUAL 0)
            message(FATAL_ERROR
                "agents_pool/test_app: POSIX-affinity patch does not apply cleanly:\n"
                "${_patch_check_err}")
        endif()
        execute_process(
            COMMAND git -C "${_ZEPHYR_TREE}" apply "${_AFFINITY_PATCH}"
            RESULT_VARIABLE _patch_apply
        )
        if(NOT _patch_apply EQUAL 0)
            message(FATAL_ERROR
                "agents_pool/test_app: POSIX-affinity patch failed to apply.")
        endif()
    endif()
endif()

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(agents_pool_test_app)

# Pull in agents_pool as an OBJECT lib via add_subdirectory.
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/.. agents_pool_build)

target_sources(app PRIVATE src/main.c)
target_link_libraries(app PRIVATE agents_pool)

# Same RISC-V relax workaround the model harnesses apply, in case a
# future test variant grows large data tables.
zephyr_link_libraries(-Wl,--no-relax)
