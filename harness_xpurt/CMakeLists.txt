# SPDX-License-Identifier: Apache-2.0
#
# Schedule-driven multi-network harness with HETEROGENEOUS backends.
#
# Each model is compiled once per HW backend (e.g. scalar + rvv) into a
# separate OBJECT library. The externally-visible symbols of model.c +
# kernels.c get suffixed via -D renames (see agents/pipeline/backend_rename.py)
# so both copies can link into one binary without collision. The walker
# (xpurt_main.c) dispatches per entry by matching the entry's core_kind to
# one of the per-backend dispatch tables.
#
# Pass:
#   -DMODEL_BACKENDS=scalar,rvv             comma list of HW backends to build
#                                          for. Must cover every core_kind
#                                          appearing in the schedule.
#   -DMODEL_NAMES=name1;name2;...           cmake-list of constituent networks
#   -DMODEL_DIRS_BASE=<base1>;<base2>;...   parallel cmake-list of per-quant
#                                          base dirs. Each must contain a
#                                          <backend>/ subdir (with model.c,
#                                          kernels.c, weights.c) for every
#                                          backend in MODEL_BACKENDS, plus a
#                                          graph.json for the renamer.
#   -DAGENTS_KERNEL_CFLAGS_<BS>=...         optional cflags for that backend's
#                                          kernels.c (e.g. -march=rv64gcv).
#                                          Applied to each (model, backend)
#                                          OBJECT lib.
#   -DXPURT_SCHEDULE_C=<path>               generated dispatch_table.c
#   -DXPURT_MAIN_C=<path>                   generated xpurt_main.c
#   -DXPURT_INCLUDE_DIR=<path>              dir holding the .h files
#   -DAGENTS_XPURT_TRACE=ON                 (optional) enable trace capture

cmake_minimum_required(VERSION 3.20.0)

if(NOT DEFINED MODEL_BACKENDS)
    set(MODEL_BACKENDS "scalar")
endif()
# MODEL_BACKENDS arrives as a comma list (matches the CLI other tools take);
# convert to a CMake list for foreach() iteration.
string(REPLACE "," ";" MODEL_BACKENDS "${MODEL_BACKENDS}")

foreach(v MODEL_NAMES MODEL_DIRS_BASE XPURT_SCHEDULE_C XPURT_MAIN_C XPURT_INCLUDE_DIR)
    if(NOT DEFINED ${v})
        message(FATAL_ERROR "harness_xpurt requires -D${v}")
    endif()
endforeach()
list(LENGTH MODEL_NAMES _n_names)
list(LENGTH MODEL_DIRS_BASE _n_bases)
if(NOT _n_names EQUAL _n_bases)
    message(FATAL_ERROR
        "MODEL_NAMES and MODEL_DIRS_BASE must have equal length "
        "(${_n_names} vs ${_n_bases})")
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

# Stack every requested backend's prj.conf overlay. Order doesn't matter
# (each overlay enables a separate Kconfig). RVV is a strict superset of
# scalar at the ISA level so a binary built with rvv.conf still runs
# scalar-only kernels correctly.
set(CONF_FILE "prj.conf")
foreach(_bs IN LISTS MODEL_BACKENDS)
    set(_BACKEND_CONF "${CMAKE_CURRENT_LIST_DIR}/backends/${_bs}.conf")
    if(EXISTS "${_BACKEND_CONF}")
        list(APPEND CONF_FILE "${_BACKEND_CONF}")
    endif()
endforeach()

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(agents_harness_xpurt)

# AGENTS_POOL_BACKEND={pthreadpool,raw} — see agents/harness_multi for
# rationale. `pthreadpool` is the historical default; `raw` swaps in
# agents/runtime/agents_pool.
if(NOT DEFINED AGENTS_POOL_BACKEND)
    # Phase 3 default — see agents/harness_multi/CMakeLists.txt rationale.
    set(AGENTS_POOL_BACKEND "raw")
endif()
set(_VALID_POOL_BACKENDS "pthreadpool" "raw")
list(FIND _VALID_POOL_BACKENDS "${AGENTS_POOL_BACKEND}" _pb_idx)
if(_pb_idx EQUAL -1)
    message(FATAL_ERROR
        "AGENTS_POOL_BACKEND='${AGENTS_POOL_BACKEND}' must be one of: "
        "${_VALID_POOL_BACKENDS}")
endif()
message(STATUS "agents harness_xpurt: AGENTS_POOL_BACKEND=${AGENTS_POOL_BACKEND}")

set(_REPO_ROOT_ABS "${CMAKE_CURRENT_LIST_DIR}/../..")
if(AGENTS_POOL_BACKEND STREQUAL "pthreadpool")
    # pthreadpool — same vendored source path.
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
else()
    add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../runtime/agents_pool"
                     agents_pool_build)
endif()

# Stage per-model headers under <build>/agents_xpurt/<name>_model.h
# matching harness_multi's convention so xpurt_main.c #include lines line up.
# The header is backend-agnostic (struct types & macros are identical across
# backends) — pick the first backend's copy.
set(_STAGE_DIR "${CMAKE_CURRENT_BINARY_DIR}/agents_xpurt")
file(MAKE_DIRECTORY "${_STAGE_DIR}")

list(GET MODEL_BACKENDS 0 _PRIMARY_BS)

math(EXPR _last "${_n_bases} - 1")
foreach(_i RANGE ${_last})
    list(GET MODEL_NAMES ${_i} _name)
    list(GET MODEL_DIRS_BASE ${_i} _base)

    set(_primary_dir "${_base}/${_PRIMARY_BS}")
    foreach(_required model.h test_io.h)
        if(NOT EXISTS "${_primary_dir}/${_required}")
            message(FATAL_ERROR "MODEL_DIRS_BASE[${_i}]/${_PRIMARY_BS}='${_primary_dir}' missing ${_required}")
        endif()
    endforeach()
    configure_file("${_primary_dir}/model.h"   "${_STAGE_DIR}/${_name}_model.h"   COPYONLY)
    configure_file("${_primary_dir}/test_io.h" "${_STAGE_DIR}/${_name}_test_io.h" COPYONLY)

    # Per-(model, backend) OBJECT lib over model.c + kernels.c, with -D
    # renames so the externally-visible symbols don't collide across
    # backends. Weights are NOT renamed (link once per model below).
    foreach(_bs IN LISTS MODEL_BACKENDS)
        set(_backend_dir "${_base}/${_bs}")
        foreach(_required model.c kernels.c weights.c)
            if(NOT EXISTS "${_backend_dir}/${_required}")
                message(FATAL_ERROR "MODEL_DIRS_BASE[${_i}]/${_bs}='${_backend_dir}' missing ${_required}")
            endif()
        endforeach()

        # Drive backend_rename.py to compute the -D flags. We emit them as
        # a CMake-list (semicolon-separated) so we can splice straight into
        # target_compile_definitions.
        set(_graph_json "${_base}/graph.json")
        if(NOT EXISTS "${_graph_json}")
            message(FATAL_ERROR "MODEL_DIRS_BASE[${_i}]='${_base}' missing graph.json (needed by backend_rename)")
        endif()
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${_REPO_ROOT_ABS}"
                python -m agents.pipeline.backend_rename
                    --ir "${_graph_json}"
                    --backend "${_bs}"
                    --separator ";"
            OUTPUT_VARIABLE _rename_flags
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _rename_rc)
        if(NOT _rename_rc EQUAL 0)
            message(FATAL_ERROR "backend_rename.py failed for ${_name}/${_bs}")
        endif()
        # backend_rename emits "-Dold=new;-Dold2=new2;..." — strip the
        # leading -D from each before handing to target_compile_definitions
        # (which prepends -D itself).
        set(_rename_defs "")
        foreach(_flag IN LISTS _rename_flags)
            string(REGEX REPLACE "^-D" "" _kv "${_flag}")
            list(APPEND _rename_defs "${_kv}")
        endforeach()

        set(_obj_tgt "agents_${_name}_${_bs}_objs")
        add_library(${_obj_tgt} OBJECT
            "${_backend_dir}/model.c" "${_backend_dir}/kernels.c")
        target_include_directories(${_obj_tgt} PRIVATE "${_backend_dir}")
        target_link_libraries(${_obj_tgt} PRIVATE zephyr_interface)
        if(AGENTS_POOL_BACKEND STREQUAL "raw")
            target_link_libraries(${_obj_tgt} PRIVATE agents_pool)
        else()
            target_link_libraries(${_obj_tgt} PRIVATE pthreadpool)
        endif()
        # AGENTS_DISABLE_UNMANGLED keeps the unmangled aliases out of these
        # TUs (multi-model linking would alias-collide otherwise).
        target_compile_definitions(${_obj_tgt} PRIVATE
            AGENTS_USE_POOL=1
            AGENTS_DISABLE_UNMANGLED=1)
        if(AGENTS_POOL_BACKEND STREQUAL "raw")
            target_compile_definitions(${_obj_tgt} PRIVATE
                AGENTS_POOL_BACKEND_RAW=1)
        endif()
        if(_rename_defs)
            target_compile_definitions(${_obj_tgt} PRIVATE ${_rename_defs})
        endif()

        # Per-backend kernel cflags. Applied only to kernels.c (model.c
        # doesn't use intrinsics; keeping the flag scoped avoids surprises).
        set(_cflags_var "AGENTS_KERNEL_CFLAGS_${_bs}")
        string(TOUPPER "${_cflags_var}" _cflags_var)
        if(DEFINED ${_cflags_var} AND NOT "${${_cflags_var}}" STREQUAL "")
            set_source_files_properties("${_backend_dir}/kernels.c"
                TARGET_DIRECTORY ${_obj_tgt}
                PROPERTIES COMPILE_OPTIONS "${${_cflags_var}}")
        endif()

        message(STATUS "harness_xpurt: ${_name}/${_bs} -> ${_obj_tgt}")
    endforeach()

    # Backend-agnostic weights — single OBJECT lib, picked from the
    # primary backend's dir (all copies are identical).
    set(_weights_tgt "agents_${_name}_weights_obj")
    add_library(${_weights_tgt} OBJECT "${_primary_dir}/weights.c")
    target_include_directories(${_weights_tgt} PRIVATE "${_primary_dir}")
    target_link_libraries(${_weights_tgt} PRIVATE zephyr_interface)
    target_compile_definitions(${_weights_tgt} PRIVATE AGENTS_DISABLE_UNMANGLED=1)

    # Intermediate-buffer storage — also single OBJECT lib per model, NOT
    # per backend. Without this, each per-backend model.c TU would carry
    # its own private file-static `buf_*` and cross-backend dispatches
    # within one network would read zeroed scratch (caused dronet output
    # corruption on FireSim where rvv ran maxpool1 but scalar's
    # conv_modules.3 read its own backend's `buf_maxpool1`). model.c's
    # extern declarations resolve against this single TU's definitions.
    if(NOT EXISTS "${_primary_dir}/buffers.c")
        message(FATAL_ERROR
            "MODEL_DIRS_BASE[${_i}]/${_PRIMARY_BS}='${_primary_dir}' missing buffers.c — "
            "regenerate the skeleton with the latest agents/pipeline/generate_skeleton.py")
    endif()
    set(_buffers_tgt "agents_${_name}_buffers_obj")
    add_library(${_buffers_tgt} OBJECT "${_primary_dir}/buffers.c")
    target_include_directories(${_buffers_tgt} PRIVATE "${_primary_dir}")
    target_link_libraries(${_buffers_tgt} PRIVATE zephyr_interface)
endforeach()

# Schedule-driven app: xpurt_main.c + the dispatch table .c.
target_sources(app PRIVATE "${XPURT_MAIN_C}" "${XPURT_SCHEDULE_C}")
target_include_directories(app PRIVATE "${_STAGE_DIR}" "${XPURT_INCLUDE_DIR}")
target_compile_definitions(app PRIVATE AGENTS_DISABLE_UNMANGLED=1)

if(DEFINED AGENTS_POOL_THREADS)
    target_compile_definitions(app PRIVATE AGENTS_POOL_THREADS=${AGENTS_POOL_THREADS})
endif()

# Optional trace collection. When ON, the schedule-driven main records
# each entry's actual start/end mtime-cycle window and prints a
# parseable CSV block at the end. Host-side `agents/scripts/plot_xpurt_trace.py`
# reads it and renders a Gantt timeline overlaying XPU-RT's prediction.
if(AGENTS_XPURT_TRACE)
    target_compile_definitions(app PRIVATE AGENTS_XPURT_TRACE=1)
endif()

# Link every (model, backend) OBJECT lib + the per-model weights and
# buffers OBJECT libs into the final binary. weights and buffers are
# shared across all backends of one model — see comments where they're
# declared.
foreach(_name ${MODEL_NAMES})
    foreach(_bs IN LISTS MODEL_BACKENDS)
        target_link_libraries(app PRIVATE agents_${_name}_${_bs}_objs)
    endforeach()
    target_link_libraries(app PRIVATE agents_${_name}_weights_obj)
    target_link_libraries(app PRIVATE agents_${_name}_buffers_obj)
endforeach()
if(AGENTS_POOL_BACKEND STREQUAL "raw")
    target_compile_definitions(app PRIVATE AGENTS_POOL_BACKEND_RAW=1)
    target_link_libraries(app PRIVATE agents_pool)
else()
    target_link_libraries(app PRIVATE pthreadpool)
endif()

# Same RISC-V gp-relative relax fix as harness_multi.
zephyr_link_libraries(-Wl,--no-relax)

message(STATUS "harness_xpurt: backends=${MODEL_BACKENDS} models=${MODEL_NAMES}")
message(STATUS "harness_xpurt: xpurt_main=${XPURT_MAIN_C}")
message(STATUS "harness_xpurt: dispatch_table=${XPURT_SCHEDULE_C}")
