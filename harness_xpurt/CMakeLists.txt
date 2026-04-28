# SPDX-License-Identifier: Apache-2.0
#
# Schedule-driven multi-network harness. Round-trip companion to
# agents/harness_multi/, which runs networks straight-line in IR order.
# This harness instead walks an XPU-RT-emitted dispatch table generated
# by agents/pipeline/ingest_xpurt_schedule + generate_xpurt_main.
#
# Pass:
#   -DAGENTS_BACKEND=<scalar|rvv>          common HW backend for ALL models
#   -DMODEL_NAMES=name1;name2;...          parallel list to MODEL_DIRS
#   -DMODEL_DIRS=<absdir1>;<absdir2>;...   per-target generated/ dirs
#   -DAGENTS_KERNEL_CFLAGS=...             optional per-backend cflags
#   -DXPURT_SCHEDULE_C=<path>              generated dispatch_table.c
#   -DXPURT_MAIN_C=<path>                  generated xpurt_main.c
#   -DXPURT_INCLUDE_DIR=<path>             dir holding the .h files

cmake_minimum_required(VERSION 3.20.0)

if(NOT DEFINED AGENTS_BACKEND)
    set(AGENTS_BACKEND "scalar")
endif()
foreach(v MODEL_NAMES MODEL_DIRS XPURT_SCHEDULE_C XPURT_MAIN_C XPURT_INCLUDE_DIR)
    if(NOT DEFINED ${v})
        message(FATAL_ERROR "harness_xpurt requires -D${v}")
    endif()
endforeach()
list(LENGTH MODEL_NAMES _n_names)
list(LENGTH MODEL_DIRS _n_dirs)
if(NOT _n_names EQUAL _n_dirs)
    message(FATAL_ERROR "MODEL_NAMES and MODEL_DIRS must have equal length")
endif()

# Vendored POSIX-affinity patch (same as harness_multi).
set(_AFFINITY_PATCH "${CMAKE_CURRENT_LIST_DIR}/../harness_multi/zephyr_patches/posix-affinity.patch")
set(_ZEPHYR_TREE "$ENV{ZEPHYR_BASE}")
if(EXISTS "${_AFFINITY_PATCH}" AND IS_DIRECTORY "${_ZEPHYR_TREE}")
    execute_process(
        COMMAND git -C "${_ZEPHYR_TREE}" apply --reverse --check "${_AFFINITY_PATCH}"
        RESULT_VARIABLE _patch_already_applied
        OUTPUT_QUIET ERROR_QUIET)
    if(NOT _patch_already_applied EQUAL 0)
        execute_process(COMMAND git -C "${_ZEPHYR_TREE}" apply "${_AFFINITY_PATCH}"
            RESULT_VARIABLE _ok)
        if(NOT _ok EQUAL 0)
            message(FATAL_ERROR "harness_xpurt: failed to apply affinity patch")
        endif()
    endif()
endif()

# Backend overlay (same setup as harness_multi).
set(CONF_FILE "prj.conf")
set(_BACKEND_CONF "${CMAKE_CURRENT_LIST_DIR}/backends/${AGENTS_BACKEND}.conf")
if(EXISTS "${_BACKEND_CONF}")
    list(APPEND CONF_FILE "${_BACKEND_CONF}")
endif()

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(agents_harness_xpurt)

# pthreadpool — same vendored source path.
set(_REPO_ROOT_ABS "${CMAKE_CURRENT_LIST_DIR}/../..")
set(PTHREADPOOL_SOURCE_DIR
    "${_REPO_ROOT_ABS}/third-party/executorch/backends/xnnpack/third-party/pthreadpool"
    CACHE STRING "")
set(FXDIV_SOURCE_DIR
    "${_REPO_ROOT_ABS}/third-party/executorch/backends/xnnpack/third-party/FXdiv"
    CACHE STRING "")
set(PTHREADPOOL_BUILD_TESTS      OFF CACHE BOOL "")
set(PTHREADPOOL_BUILD_BENCHMARKS OFF CACHE BOOL "")
set(PTHREADPOOL_ALLOW_DEPRECATED_API OFF CACHE BOOL "")
add_subdirectory("${PTHREADPOOL_SOURCE_DIR}" pthreadpool_build)
target_link_libraries(pthreadpool PRIVATE zephyr_interface)

# Stage per-model headers under <build>/agents_xpurt/<name>_model.h
# matching harness_multi's convention so xpurt_main.c #include lines line up.
set(_STAGE_DIR "${CMAKE_CURRENT_BINARY_DIR}/agents_xpurt")
file(MAKE_DIRECTORY "${_STAGE_DIR}")

math(EXPR _last "${_n_dirs} - 1")
foreach(_i RANGE ${_last})
    list(GET MODEL_NAMES ${_i} _name)
    list(GET MODEL_DIRS  ${_i} _dir)
    foreach(_required model.c model.h kernels.c kernels.h weights.c weights.h test_io.h)
        if(NOT EXISTS "${_dir}/${_required}")
            message(FATAL_ERROR "MODEL_DIRS[${_i}]='${_dir}' missing ${_required}")
        endif()
    endforeach()
    add_library(agents_${_name}_objs OBJECT
        "${_dir}/model.c" "${_dir}/kernels.c" "${_dir}/weights.c")
    target_include_directories(agents_${_name}_objs PRIVATE "${_dir}")
    target_link_libraries(agents_${_name}_objs PRIVATE zephyr_interface)
    target_compile_definitions(agents_${_name}_objs PRIVATE AGENTS_USE_POOL=1)
    target_link_libraries(agents_${_name}_objs PRIVATE pthreadpool)
    if(DEFINED AGENTS_KERNEL_CFLAGS AND NOT "${AGENTS_KERNEL_CFLAGS}" STREQUAL "")
        set_source_files_properties("${_dir}/kernels.c"
            PROPERTIES COMPILE_OPTIONS "${AGENTS_KERNEL_CFLAGS}")
    endif()
    configure_file("${_dir}/model.h"   "${_STAGE_DIR}/${_name}_model.h"   COPYONLY)
    configure_file("${_dir}/test_io.h" "${_STAGE_DIR}/${_name}_test_io.h" COPYONLY)
endforeach()

# Schedule-driven app: xpurt_main.c + the dispatch table .c.
target_sources(app PRIVATE "${XPURT_MAIN_C}" "${XPURT_SCHEDULE_C}")
target_include_directories(app PRIVATE "${_STAGE_DIR}" "${XPURT_INCLUDE_DIR}")

if(DEFINED AGENTS_POOL_THREADS)
    target_compile_definitions(app PRIVATE AGENTS_POOL_THREADS=${AGENTS_POOL_THREADS})
endif()

foreach(_name ${MODEL_NAMES})
    target_link_libraries(app PRIVATE agents_${_name}_objs)
endforeach()
target_link_libraries(app PRIVATE pthreadpool)

# Same RISC-V gp-relative relax fix as harness_multi.
zephyr_link_libraries(-Wl,--no-relax)

message(STATUS "harness_xpurt: backend=${AGENTS_BACKEND} models=${MODEL_NAMES}")
message(STATUS "harness_xpurt: xpurt_main=${XPURT_MAIN_C}")
message(STATUS "harness_xpurt: dispatch_table=${XPURT_SCHEDULE_C}")
