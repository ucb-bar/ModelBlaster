# Schedule-driven multi-network harness for a hosted riscv64 board (SpaceMiT K1).
#
# The Linux counterpart of ../harness_xpurt, which is Zephyr-only (it does
# find_package(Zephyr) and stacks prj.conf overlays). Everything structural is
# kept identical -- per-(model, backend) OBJECT libraries with backend_rename.py
# symbol suffixes, one weights/buffers/test_io TU per model, and the generated
# walker plus dispatch table as the app -- so a schedule that runs under Zephyr
# builds here without regenerating anything except xpurt_main.c
# (--platform linux).
#
# Why the per-model/per-backend split is not simplified: model.c's `buf_*`
# intermediates must have exactly ONE definition per model, shared across
# backends. Giving each backend its own copy made a cross-backend dispatch
# within one network read zeroed scratch -- that is what corrupted DroNet's
# output on FireSim when rvv ran maxpool1 and scalar's conv_modules.3 then read
# its own backend's buf_maxpool1. Same reason weights are linked once.
#
# Pass:
#   -DMODEL_BACKENDS=scalar,rvv           comma list; must cover every core_kind
#                                         in the schedule
#   -DMODEL_NAMES=a;b;...                 cmake-list of networks
#   -DMODEL_DIRS_BASE=<base_a>;<base_b>   parallel list; each holds
#                                         <backend>/{model.c,kernels.c,
#                                         weights.c,buffers.c,model.h,test_io.h}
#                                         plus graph.json
#   -DXPURT_SCHEDULE_C=<path>             generated dispatch_table.c
#   -DXPURT_MAIN_C=<path>                 generated xpurt_main.c (--platform linux)
#   -DXPURT_INCLUDE_DIR=<path>            dir holding the generated .h files
#   -DMODELBLASTER_KERNEL_CFLAGS_<BACKEND>=...   e.g. "-march=rv64gcv_zvl256b"
#   -DMODELBLASTER_XPURT_TRACE=ON         emit the trace CSV block
#   -DMODELBLASTER_POOL_THREADS=N         parallel-for pool size

cmake_minimum_required(VERSION 3.16)
project(modelblaster_harness_xpurt_linux C ASM)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT DEFINED MODEL_BACKENDS)
    set(MODEL_BACKENDS "scalar")
endif()
string(REPLACE "," ";" MODEL_BACKENDS "${MODEL_BACKENDS}")

foreach(v MODEL_NAMES MODEL_DIRS_BASE XPURT_SCHEDULE_C XPURT_MAIN_C XPURT_INCLUDE_DIR)
    if(NOT DEFINED ${v})
        message(FATAL_ERROR "harness_xpurt_linux requires -D${v}")
    endif()
endforeach()

list(LENGTH MODEL_NAMES _n_names)
list(LENGTH MODEL_DIRS_BASE _n_bases)
if(NOT _n_names EQUAL _n_bases)
    message(FATAL_ERROR
        "MODEL_NAMES and MODEL_DIRS_BASE must have equal length "
        "(${_n_names} vs ${_n_bases})")
endif()

set(_REPO_ROOT_ABS "${CMAKE_CURRENT_LIST_DIR}/..")
set(_RUNTIME_DIR "${_REPO_ROOT_ABS}/runtime")

find_package(Threads REQUIRED)

# The parallel-for pool. Built here directly rather than via its own
# CMakeLists, which links zephyr_interface.
add_library(modelblaster_pool_linux STATIC
    "${_RUNTIME_DIR}/modelblaster_pool/modelblaster_pool.c")
target_include_directories(modelblaster_pool_linux PUBLIC
    "${_RUNTIME_DIR}/modelblaster_pool" "${_RUNTIME_DIR}")
target_compile_definitions(modelblaster_pool_linux PUBLIC
    MODELBLASTER_PLATFORM_LINUX=1)
target_link_libraries(modelblaster_pool_linux PUBLIC Threads::Threads)

# Stage per-model headers as <build>/modelblaster_xpurt/<name>_model.h, the
# layout the generated walker's #include lines assume.
set(_STAGE_DIR "${CMAKE_CURRENT_BINARY_DIR}/modelblaster_xpurt")
file(MAKE_DIRECTORY "${_STAGE_DIR}")
list(GET MODEL_BACKENDS 0 _PRIMARY_BS)

add_executable(xpurt_harness "${XPURT_MAIN_C}" "${XPURT_SCHEDULE_C}")
target_include_directories(xpurt_harness PRIVATE
    "${_STAGE_DIR}" "${XPURT_INCLUDE_DIR}"
    "${_RUNTIME_DIR}" "${_RUNTIME_DIR}/modelblaster_pool")
target_compile_definitions(xpurt_harness PRIVATE
    MODELBLASTER_DISABLE_UNMANGLED=1
    MODELBLASTER_PLATFORM_LINUX=1)
target_link_libraries(xpurt_harness PRIVATE
    modelblaster_pool_linux Threads::Threads m)

if(DEFINED MODELBLASTER_POOL_THREADS)
    target_compile_definitions(xpurt_harness PRIVATE
        MODELBLASTER_POOL_THREADS=${MODELBLASTER_POOL_THREADS})
endif()
if(MODELBLASTER_XPURT_TRACE)
    target_compile_definitions(xpurt_harness PRIVATE MODELBLASTER_XPURT_TRACE=1)
endif()

math(EXPR _last "${_n_bases} - 1")
foreach(_i RANGE ${_last})
    list(GET MODEL_NAMES ${_i} _name)
    list(GET MODEL_DIRS_BASE ${_i} _base)
    set(_primary_dir "${_base}/${_PRIMARY_BS}")

    foreach(_required model.h test_io.h)
        if(NOT EXISTS "${_primary_dir}/${_required}")
            message(FATAL_ERROR
                "MODEL_DIRS_BASE[${_i}]/${_PRIMARY_BS}='${_primary_dir}' "
                "missing ${_required}")
        endif()
    endforeach()
    configure_file("${_primary_dir}/model.h"   "${_STAGE_DIR}/${_name}_model.h"   COPYONLY)
    configure_file("${_primary_dir}/test_io.h" "${_STAGE_DIR}/${_name}_test_io.h" COPYONLY)

    set(_graph_json "${_base}/graph.json")
    if(NOT EXISTS "${_graph_json}")
        message(FATAL_ERROR
            "MODEL_DIRS_BASE[${_i}]='${_base}' missing graph.json "
            "(needed by backend_rename)")
    endif()

    foreach(_bs IN LISTS MODEL_BACKENDS)
        set(_backend_dir "${_base}/${_bs}")
        foreach(_required model.c kernels.c weights.c)
            if(NOT EXISTS "${_backend_dir}/${_required}")
                message(FATAL_ERROR
                    "MODEL_DIRS_BASE[${_i}]/${_bs}='${_backend_dir}' "
                    "missing ${_required}")
            endif()
        endforeach()

        # Symbol suffixes, so two backends of one model link without
        # colliding. Same invocation the Zephyr harness uses.
        # PYTHONPATH mirrors scripts/run_xpurt_k1.sh: `src` first (the shim
        # package whose __path__ points back at this checkout's repo root),
        # then the repo root itself. It used to be `${_REPO_ROOT_ABS}/..`,
        # which only works when the checkout's own directory is *named*
        # `modelblaster` -- so a checkout named ModelBlaster/ (or vendored
        # under another repo) failed configure with "No module named
        # 'modelblaster'", and a machine with a sibling clone next door
        # silently ran THAT checkout's backend_rename instead. The symbol
        # suffixes it prints decide which kernel each dispatch links against,
        # so resolving to the wrong checkout is not a cosmetic error.
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E env
                "PYTHONPATH=${_REPO_ROOT_ABS}/src:${_REPO_ROOT_ABS}"
                ${PYTHON_EXECUTABLE} -m modelblaster.pipeline.backend_rename
                    --ir "${_graph_json}" --backend "${_bs}" --separator ";"
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

        set(_obj_tgt "mb_${_name}_${_bs}_objs")
        add_library(${_obj_tgt} OBJECT
            "${_backend_dir}/model.c" "${_backend_dir}/kernels.c")
        target_include_directories(${_obj_tgt} PRIVATE
            "${_backend_dir}" "${_RUNTIME_DIR}" "${_RUNTIME_DIR}/modelblaster_pool")
        target_compile_definitions(${_obj_tgt} PRIVATE
            MODELBLASTER_USE_POOL=1
            MODELBLASTER_DISABLE_UNMANGLED=1
            MODELBLASTER_PLATFORM_LINUX=1)
        if(_rename_defs)
            target_compile_definitions(${_obj_tgt} PRIVATE ${_rename_defs})
        endif()

        # Kernel cflags are scoped to kernels.c only: model.c uses no
        # intrinsics, and keeping -march off it avoids surprises.
        string(TOUPPER "MODELBLASTER_KERNEL_CFLAGS_${_bs}" _cflags_var)
        if(DEFINED ${_cflags_var} AND NOT "${${_cflags_var}}" STREQUAL "")
            separate_arguments(_kflags UNIX_COMMAND "${${_cflags_var}}")
            set_source_files_properties("${_backend_dir}/kernels.c"
                TARGET_DIRECTORY ${_obj_tgt}
                PROPERTIES COMPILE_OPTIONS "${_kflags}")
        endif()

        target_link_libraries(xpurt_harness PRIVATE ${_obj_tgt})
        message(STATUS "harness_xpurt_linux: ${_name}/${_bs} -> ${_obj_tgt}")
    endforeach()

    # One TU per model, NOT per backend -- see the header comment.
    foreach(_what weights buffers)
        if(NOT EXISTS "${_primary_dir}/${_what}.c")
            message(FATAL_ERROR
                "${_primary_dir} missing ${_what}.c -- regenerate the skeleton")
        endif()
        set(_tgt "mb_${_name}_${_what}_obj")
        add_library(${_tgt} OBJECT "${_primary_dir}/${_what}.c")
        target_include_directories(${_tgt} PRIVATE "${_primary_dir}")
        target_compile_definitions(${_tgt} PRIVATE MODELBLASTER_PLATFORM_LINUX=1)
        if(_what STREQUAL "weights")
            target_compile_definitions(${_tgt} PRIVATE
                MODELBLASTER_DISABLE_UNMANGLED=1)
        endif()
        target_link_libraries(xpurt_harness PRIVATE ${_tgt})
    endforeach()

    # .incbin pulls test_input.bin / test_golden.bin into rodata, which is what
    # keeps cc1 from OOMing on kernelbench-scale decimal const arrays.
    if(EXISTS "${_primary_dir}/test_io.S")
        set(_tio "mb_${_name}_test_io_obj")
        add_library(${_tio} OBJECT "${_primary_dir}/test_io.S")
        set_source_files_properties("${_primary_dir}/test_io.S"
            PROPERTIES LANGUAGE ASM)
        target_include_directories(${_tio} PRIVATE "${_primary_dir}")
        target_link_libraries(xpurt_harness PRIVATE ${_tio})
    endif()
endforeach()

message(STATUS "harness_xpurt_linux: backends=${MODEL_BACKENDS} models=${MODEL_NAMES}")
message(STATUS "harness_xpurt_linux: xpurt_main=${XPURT_MAIN_C}")
