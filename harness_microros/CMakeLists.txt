# SPDX-License-Identifier: Apache-2.0
#
# micro-ROS baseline harness — fixed-HW per-network ROS nodes.
#
# Each network = one rclc executor pinned to one hart with one rmw
# session over the in-target loopback broker (lib/microros_local). Each
# thread runs its full dispatch graph SEQUENTIALLY through the backend
# the network is assigned to (no cross-kind routing) — this is the
# fixed-HW counterpart to harness_xpurt, used as a baseline to compare
# against the xpurt scheduler.
#
# Reuses harness_xpurt's per-(model, backend) OBJECT lib infrastructure
# verbatim (compile each model.c + kernels.c per backend with backend-
# rename -D flags so symbols don't collide). Drops xpurt_main.c +
# dispatch_table.c; replaces them with a hand-rolled src/main.c that
# spins one rclc executor per network.
#
# Pass:
#   -DMODEL_BACKENDS=<bs1>,<bs2>,...        comma list of HW backends to
#                                          BUILD into the binary. Must
#                                          cover every PIN_BACKEND below.
#   -DMODEL_NAMES=name1;name2;...           cmake-list of constituent networks
#   -DMODEL_DIRS_BASE=<base1>;<base2>;...   parallel cmake-list of per-quant
#                                          base dirs. Each must contain a
#                                          <backend>/ subdir for every
#                                          backend in MODEL_BACKENDS.
#   -DMODEL_PIN_BACKENDS=<bs1>;<bs2>;...    parallel cmake-list — which
#                                          backend each network's executor
#                                          should run kernels through.
#                                          Surfaced to main.c via
#                                          per-network -D flags.
#   -DMODEL_PIN_HARTS=<n1>;<n2>;...         hart IDs to pin each network's
#                                          executor to. Broker pins to the
#                                          highest available hart in
#                                          MICROROS_BROKER_HART (default
#                                          MP_MAX_NUM_CPUS-1).
#   -DMODEL_PERIODS_MS=<p1>;<p2>;...        timer period (0 = one-shot).
#                                          Periodic networks run the full
#                                          model graph every period_ms;
#                                          zero means run once and publish
#                                          /done.
#   -DAGENTS_KERNEL_CFLAGS_<BS>=...         per-backend kernel cflags
#                                          (same as harness_xpurt).

cmake_minimum_required(VERSION 3.20.0)

if(NOT DEFINED MODEL_BACKENDS)
    set(MODEL_BACKENDS "scalar")
endif()
string(REPLACE "," ";" MODEL_BACKENDS "${MODEL_BACKENDS}")

foreach(v MODEL_NAMES MODEL_DIRS_BASE MODEL_PIN_BACKENDS MODEL_PIN_HARTS MODEL_PERIODS_MS)
    if(NOT DEFINED ${v})
        message(FATAL_ERROR "harness_microros requires -D${v}")
    endif()
endforeach()
list(LENGTH MODEL_NAMES _n_names)
list(LENGTH MODEL_DIRS_BASE _n_bases)
list(LENGTH MODEL_PIN_BACKENDS _n_pinbs)
list(LENGTH MODEL_PIN_HARTS _n_pinharts)
list(LENGTH MODEL_PERIODS_MS _n_periods)
if(_n_names LESS 2 OR _n_names GREATER 3)
    message(FATAL_ERROR
        "harness_microros supports 2 or 3 networks; got ${_n_names}")
endif()
foreach(_pair _n_bases _n_pinbs _n_pinharts _n_periods)
    if(NOT ${_pair} EQUAL ${_n_names})
        message(FATAL_ERROR
            "MODEL_NAMES (${_n_names}) and ${_pair} (${${_pair}}) "
            "must have equal length")
    endif()
endforeach()

# Reuse harness_multi's vendored POSIX-affinity patch (same Zephyr issue
# as harness_xpurt + harness_multi). Idempotent — applies once per tree.
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
            message(FATAL_ERROR "harness_microros: failed to apply affinity patch")
        endif()
    endif()
endif()

# Stack each requested backend's prj.conf overlay (identical pattern to
# harness_xpurt). Plus a microros.conf for the executor + broker bits.
set(CONF_FILE "prj.conf")
foreach(_bs IN LISTS MODEL_BACKENDS)
    set(_BACKEND_CONF "${CMAKE_CURRENT_LIST_DIR}/backends/${_bs}.conf")
    if(EXISTS "${_BACKEND_CONF}")
        list(APPEND CONF_FILE "${_BACKEND_CONF}")
    endif()
endforeach()

# micro-ROS module + local broker module must be on ZEPHYR_EXTRA_MODULES
# BEFORE find_package(Zephyr) so the find_package step picks up the
# zephyr_library_named()'d targets and links them automatically.
set(_REPO_ROOT_ABS "${CMAKE_CURRENT_LIST_DIR}/../..")
list(APPEND ZEPHYR_EXTRA_MODULES
    "${_REPO_ROOT_ABS}/third-party/micro_ros_zephyr_module/modules/libmicroros"
    "${_REPO_ROOT_ABS}/lib/microros_local")

# rcl/rcutils still poke at __STDC_WANT_LIB_EXT1__ helpers; setting it
# to 0 mirrors samples/micro_ros_local's main.
add_compile_definitions(__STDC_WANT_LIB_EXT1__=0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(agents_harness_microros)

# agents_pool: same parallel-for backend the model code expects.
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../runtime/agents_pool"
                 agents_pool_build)

# Stage per-model headers under <build>/agents_microros/<name>_model.h
# so main.c #include lines line up regardless of MODEL_DIRS_BASE.
set(_STAGE_DIR "${CMAKE_CURRENT_BINARY_DIR}/agents_microros")
file(MAKE_DIRECTORY "${_STAGE_DIR}")

list(GET MODEL_BACKENDS 0 _PRIMARY_BS)

# Build per-(model, backend) OBJECT libs the same way harness_xpurt does
# — see harness_xpurt/CMakeLists.txt for the rationale (-D rename flags
# so externally-visible symbols don't alias across backends; weights +
# buffers shared across backends as a single TU per model).
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

    foreach(_bs IN LISTS MODEL_BACKENDS)
        set(_backend_dir "${_base}/${_bs}")
        foreach(_required model.c kernels.c weights.c)
            if(NOT EXISTS "${_backend_dir}/${_required}")
                message(FATAL_ERROR "MODEL_DIRS_BASE[${_i}]/${_bs}='${_backend_dir}' missing ${_required}")
            endif()
        endforeach()

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
        set(_rename_defs "")
        foreach(_flag IN LISTS _rename_flags)
            string(REGEX REPLACE "^-D" "" _kv "${_flag}")
            list(APPEND _rename_defs "${_kv}")
        endforeach()

        set(_obj_tgt "agents_microros_${_name}_${_bs}_objs")
        add_library(${_obj_tgt} OBJECT
            "${_backend_dir}/model.c" "${_backend_dir}/kernels.c")
        target_include_directories(${_obj_tgt} PRIVATE "${_backend_dir}")
        target_link_libraries(${_obj_tgt} PRIVATE zephyr_interface)
        target_link_libraries(${_obj_tgt} PRIVATE agents_pool)
        target_compile_definitions(${_obj_tgt} PRIVATE
            AGENTS_USE_POOL=1
            AGENTS_DISABLE_UNMANGLED=1)
        if(_rename_defs)
            target_compile_definitions(${_obj_tgt} PRIVATE ${_rename_defs})
        endif()

        set(_cflags_var "AGENTS_KERNEL_CFLAGS_${_bs}")
        string(TOUPPER "${_cflags_var}" _cflags_var)
        if(DEFINED ${_cflags_var} AND NOT "${${_cflags_var}}" STREQUAL "")
            set_source_files_properties("${_backend_dir}/kernels.c"
                TARGET_DIRECTORY ${_obj_tgt}
                PROPERTIES COMPILE_OPTIONS "${${_cflags_var}}")
        endif()

        message(STATUS "harness_microros: ${_name}/${_bs} -> ${_obj_tgt}")
    endforeach()

    # Backend-agnostic weights + buffers + test_io — single TU per model
    # (matches harness_xpurt's reasoning: cross-backend dispatches in one
    # network must read each other's writes through one buffer storage).
    set(_weights_tgt "agents_microros_${_name}_weights_obj")
    add_library(${_weights_tgt} OBJECT "${_primary_dir}/weights.c")
    target_include_directories(${_weights_tgt} PRIVATE "${_primary_dir}")
    target_link_libraries(${_weights_tgt} PRIVATE zephyr_interface)
    target_compile_definitions(${_weights_tgt} PRIVATE AGENTS_DISABLE_UNMANGLED=1)

    if(NOT EXISTS "${_primary_dir}/buffers.c")
        message(FATAL_ERROR
            "MODEL_DIRS_BASE[${_i}]/${_PRIMARY_BS}='${_primary_dir}' missing buffers.c")
    endif()
    set(_buffers_tgt "agents_microros_${_name}_buffers_obj")
    add_library(${_buffers_tgt} OBJECT "${_primary_dir}/buffers.c")
    target_include_directories(${_buffers_tgt} PRIVATE "${_primary_dir}")
    target_link_libraries(${_buffers_tgt} PRIVATE zephyr_interface)

    if(EXISTS "${_primary_dir}/test_io.S")
        set(_test_io_tgt "agents_microros_${_name}_test_io_obj")
        add_library(${_test_io_tgt} OBJECT "${_primary_dir}/test_io.S")
        set_source_files_properties("${_primary_dir}/test_io.S"
            PROPERTIES LANGUAGE ASM)
        target_include_directories(${_test_io_tgt} PRIVATE "${_primary_dir}")
        target_link_libraries(${_test_io_tgt} PRIVATE zephyr_interface)
    endif()
endforeach()

# Hand-rolled main.c — supports 2 or 3 networks (NET_C gated by
# MICROROS_3NET). The per-network -D flags below let main.c reference
# each network's symbols + pinning without cmake-list parsing in C.
target_sources(app PRIVATE "${CMAKE_CURRENT_LIST_DIR}/src/main.c")
target_include_directories(app PRIVATE "${_STAGE_DIR}")
target_compile_definitions(app PRIVATE AGENTS_DISABLE_UNMANGLED=1)

# Surface MODEL_NAMES + per-model pinning to main.c via -D flags.
# NET_A + NET_B are always present; NET_C is conditional (3 models).
list(GET MODEL_NAMES         0 _n0)
list(GET MODEL_NAMES         1 _n1)
list(GET MODEL_PIN_BACKENDS  0 _pb0)
list(GET MODEL_PIN_BACKENDS  1 _pb1)
list(GET MODEL_PIN_HARTS     0 _ph0)
list(GET MODEL_PIN_HARTS     1 _ph1)
list(GET MODEL_PERIODS_MS    0 _pm0)
list(GET MODEL_PERIODS_MS    1 _pm1)
string(TOUPPER "${_n0}" _N0_UP)
string(TOUPPER "${_n1}" _N1_UP)
string(TOUPPER "${_pb0}" _PB0_UP)
string(TOUPPER "${_pb1}" _PB1_UP)
target_compile_definitions(app PRIVATE
    NET_A_NAME=\"${_n0}\"
    NET_A_NAME_C=${_n0}
    NET_A_NAME_UP=${_N0_UP}
    NET_A_BACKEND=${_pb0}
    NET_A_BACKEND_UP=${_PB0_UP}
    NET_A_HART=${_ph0}
    NET_A_PERIOD_MS=${_pm0}
    NET_B_NAME=\"${_n1}\"
    NET_B_NAME_C=${_n1}
    NET_B_NAME_UP=${_N1_UP}
    NET_B_BACKEND=${_pb1}
    NET_B_BACKEND_UP=${_PB1_UP}
    NET_B_HART=${_ph1}
    NET_B_PERIOD_MS=${_pm1})

if(_n_names GREATER_EQUAL 3)
    list(GET MODEL_NAMES         2 _n2)
    list(GET MODEL_PIN_BACKENDS  2 _pb2)
    list(GET MODEL_PIN_HARTS     2 _ph2)
    list(GET MODEL_PERIODS_MS    2 _pm2)
    string(TOUPPER "${_n2}" _N2_UP)
    string(TOUPPER "${_pb2}" _PB2_UP)
    target_compile_definitions(app PRIVATE
        MICROROS_3NET=1
        NET_C_NAME=\"${_n2}\"
        NET_C_NAME_C=${_n2}
        NET_C_NAME_UP=${_N2_UP}
        NET_C_BACKEND=${_pb2}
        NET_C_BACKEND_UP=${_PB2_UP}
        NET_C_HART=${_ph2}
        NET_C_PERIOD_MS=${_pm2})
endif()

if(DEFINED MICROROS_BROKER_HART)
    target_compile_definitions(app PRIVATE
        MICROROS_BROKER_HART=${MICROROS_BROKER_HART})
endif()

# Pass-through for the run_graph_b interrupt-mask debug knob. One of:
#   MICROROS_MASK_ALL     irq_lock — no interrupts in run_graph_b
#   MICROROS_MASK_TIMER   only mie.MTIE cleared (default)
#   MICROROS_MASK_IPI     only mie.MSIE cleared
#   MICROROS_MASK_EXT     only mie.MEIE cleared
foreach(knob MICROROS_MASK_ALL MICROROS_MASK_TIMER MICROROS_MASK_IPI MICROROS_MASK_EXT
             MICROROS_SINGLE_EXECUTOR MICROROS_NO_BROKER MICROROS_NO_MICROROS
             MICROROS_NO_PUBLISH
             MICROROS_NO_LOCK_A MICROROS_NO_FPREGS_C MICROROS_SKIP_TRACE
             MICROROS_2EXEC_BC MICROROS_2EXEC_FIRE_FAST
             MICROROS_2EXEC_NORCLC MICROROS_2EXEC_FUSE_BC
             MICROROS_FUSE_BC_NO_C)
    if(${knob})
        target_compile_definitions(app PRIVATE ${knob}=1)
    endif()
endforeach()

# Generate two tiny header shims (net_a_includes.h / net_b_includes.h)
# that pull in the per-network model.h + test_io.h. We can't do
# `#include STR(NET_A_NAME_C##_model.h)` directly because token-paste
# doesn't run inside #include's argument expansion.
file(WRITE  "${_STAGE_DIR}/net_a_includes.h"
     "#include \"${_n0}_model.h\"\n#include \"${_n0}_test_io.h\"\n")
file(WRITE  "${_STAGE_DIR}/net_b_includes.h"
     "#include \"${_n1}_model.h\"\n#include \"${_n1}_test_io.h\"\n")
if(_n_names GREATER_EQUAL 3)
    file(WRITE "${_STAGE_DIR}/net_c_includes.h"
         "#include \"${_n2}_model.h\"\n#include \"${_n2}_test_io.h\"\n")
endif()

if(DEFINED AGENTS_POOL_THREADS)
    target_compile_definitions(app PRIVATE
        AGENTS_POOL_THREADS=${AGENTS_POOL_THREADS})
endif()

# Link every (model, backend) + the per-model weights/buffers/test_io.
foreach(_name ${MODEL_NAMES})
    foreach(_bs IN LISTS MODEL_BACKENDS)
        target_link_libraries(app PRIVATE agents_microros_${_name}_${_bs}_objs)
    endforeach()
    target_link_libraries(app PRIVATE agents_microros_${_name}_weights_obj)
    target_link_libraries(app PRIVATE agents_microros_${_name}_buffers_obj)
    if(TARGET agents_microros_${_name}_test_io_obj)
        target_link_libraries(app PRIVATE agents_microros_${_name}_test_io_obj)
    endif()
endforeach()
target_link_libraries(app PRIVATE agents_pool)

# Same RISC-V gp-relative relax fix as harness_xpurt.
zephyr_link_libraries(-Wl,--no-relax)

message(STATUS "harness_microros: backends=${MODEL_BACKENDS}")
message(STATUS "harness_microros: ${_n0} -> ${_pb0} on hart ${_ph0} (period ${_pm0}ms)")
message(STATUS "harness_microros: ${_n1} -> ${_pb1} on hart ${_ph1} (period ${_pm1}ms)")
if(_n_names GREATER_EQUAL 3)
    message(STATUS "harness_microros: ${_n2} -> ${_pb2} on hart ${_ph2} (period ${_pm2}ms)")
endif()
