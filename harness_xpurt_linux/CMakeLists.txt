# Schedule-driven multi-network harness for a hosted riscv64 board (SpaceMiT K1).
#
# The Linux counterpart of ../harness_xpurt, which is Zephyr-only (it does
# find_package(Zephyr) and stacks prj.conf overlays). Everything structural is
# kept identical -- per-(model, backend) OBJECT libraries with backend_rename.py
# symbol suffixes, one weights TU per (model, backend), one buffers/test_io TU
# per model, and the generated walker plus dispatch table as the app -- so a
# schedule that runs under Zephyr builds here without regenerating anything
# except xpurt_main.c (--platform linux).
#
# Why the per-model/per-backend split is not simplified: model.c's `buf_*`
# intermediates must have exactly ONE definition per model, shared across
# backends. Giving each backend its own copy made a cross-backend dispatch
# within one network read zeroed scratch -- that is what corrupted DroNet's
# output on FireSim when rvv ran maxpool1 and scalar's conv_modules.3 then read
# its own backend's buf_maxpool1. Weights are different: codegen may pack them
# differently per backend and gives every backend distinct weight symbols.
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
#   -DMODELBLASTER_KERNEL_CC=<abs path>   compile every kernels.c with THIS
#                                         compiler instead of CMAKE_C_COMPILER
#   -DMODELBLASTER_XPURT_TRACE=ON         emit the trace CSV block at exit
#   -DMODELBLASTER_XPURT_STREAM=ON        emit one JSON line per dispatch END,
#                                         live, for streaming_feedback.py
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
if(MODELBLASTER_XPURT_STREAM)
    # One JSON line per dispatch END on stdout, for the host-side feedback
    # tailer. Independent of the TRACE block, which only prints at exit.
    target_compile_definitions(xpurt_harness PRIVATE MODELBLASTER_XPURT_STREAM=1)
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
    # test_io.h's own `#include "model.h"` -- needed by its arity-agnostic
    # model_{mid}_run_test() wrapper, which references model_{mid}_output_t --
    # must be rewritten to the staged, per-model-prefixed name. COPYONLY does
    # not rewrite contents, so the staged header kept including a bare
    # "model.h" that does not exist in the staging dir, and several models'
    # unqualified model.h would collide there anyway.
    #
    # The Zephyr harness has done this since 84a98cc; this is the same fix,
    # ported. Without it the xpurt build fails at
    #   fatal error: model.h: No such file or directory
    # which reads like a missing generated file rather than a staging bug.
    file(READ "${_primary_dir}/test_io.h" _test_io_contents)
    string(REPLACE "#include \"model.h\"" "#include \"${_name}_model.h\""
           _test_io_contents "${_test_io_contents}")
    file(WRITE "${_STAGE_DIR}/${_name}_test_io.h" "${_test_io_contents}")

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
        set(_kflags "")
        if(DEFINED ${_cflags_var} AND NOT "${${_cflags_var}}" STREQUAL "")
            separate_arguments(_kflags UNIX_COMMAND "${${_cflags_var}}")
            set_source_files_properties("${_backend_dir}/kernels.c"
                TARGET_DIRECTORY ${_obj_tgt}
                PROPERTIES COMPILE_OPTIONS "${_kflags}")
        endif()

        # MODELBLASTER_KERNEL_CC compiles kernels.c with a DIFFERENT compiler,
        # mirroring harness_linux/Makefile's KERNEL_CC. The Zvfh intrinsics the
        # curated fp16 kernels use (vfloat16m4_t, vfmacc.vv f16, vfredusum.vs
        # f16) do not exist in GCC 13.2, the only riscv64-unknown-linux-gnu
        # compiler installed here; GCC >= 14 has them. kernels.c is a leaf TU
        # whose only external references are libm, so an object built by a
        # bare-metal riscv64-unknown-elf gcc of the same lp64d ABI links into
        # this glibc binary and runs -- verified on the board via harness_linux.
        #
        # Without this the fp16 build can only run standalone, never in a
        # schedule. That is not an academic gap: the pure-int8 fused_full the
        # scheduler could otherwise build is the configuration whose fuse layer
        # requantizes the camera and ToF branches to all-zero (scale ratios
        # 0.0011 and 0.0026 against the output scale), so the only fused_full
        # that can be SCHEDULED is the one that cannot see.
        #
        # Done with an explicit custom command, not C_COMPILER_LAUNCHER: a
        # launcher PREFIXES the compiler rather than replacing it, so it would
        # have quietly kept using 13.2 -- the silent-fallback shape this
        # pipeline has been bitten by twice. CMake takes a .o as a source, so
        # the object joins the link the ordinary way.
        #
        # Explicit opt-in, never a search: a build must not silently change
        # compilers. The fp16 kernels carry an #error naming this variable, so a
        # 13.2 build fails by name rather than falling back to scalar.
        # Apply it ONLY to the files that need it. This harness compiles many
        # models into one binary, unlike harness_linux which builds one at a
        # time -- so a global override silently recompiled EVERY model's
        # kernels.c with the bare-metal toolchain. yolov8_nano's int8 kernels
        # built that way produced a binary that ran all 767 dispatches and then
        # died in printf with a float bit pattern where a char* belonged
        # (SIGSEGV in __strlen from __printf_buffer, badaddr 0xffffffffc8ffc9f8
        # -- the same value in two different binaries, which is what gave it
        # away as a computed value rather than stray memory). fused_full ALONE
        # was fine, because then nothing else was miscompiled.
        #
        # The rule is "use the alternate compiler exactly where the default one
        # cannot compile the file", and the file itself says so: the Zvfh
        # kernels are the ones declaring _Float16. Detecting that beats a
        # per-model flag, which would be one more thing to keep in sync with
        # which models happen to have an fp16 tail.
        set(_needs_alt_cc FALSE)
        if(DEFINED MODELBLASTER_KERNEL_CC AND NOT "${MODELBLASTER_KERNEL_CC}" STREQUAL "")
            file(READ "${_backend_dir}/kernels.c" _kernels_src)
            if(_kernels_src MATCHES "_Float16")
                set(_needs_alt_cc TRUE)
            else()
                message(STATUS
                    "harness_xpurt_linux: ${_name}/${_bs} kernels.c has no "
                    "_Float16; keeping ${CMAKE_C_COMPILER}")
            endif()
            unset(_kernels_src)
        endif()

        if(_needs_alt_cc)
            if(NOT EXISTS "${MODELBLASTER_KERNEL_CC}")
                message(FATAL_ERROR
                    "MODELBLASTER_KERNEL_CC='${MODELBLASTER_KERNEL_CC}' does not "
                    "exist. Pass an absolute path to a riscv64 GCC >= 14, or "
                    "leave it unset to use ${CMAKE_C_COMPILER}.")
            endif()
            # Drop kernels.c from the CMake-compiled object library; the custom
            # command below owns it. model.c stays on the default compiler --
            # it is the TU that references the runtime and the pool.
            get_target_property(_srcs ${_obj_tgt} SOURCES)
            list(REMOVE_ITEM _srcs "${_backend_dir}/kernels.c")
            set_property(TARGET ${_obj_tgt} PROPERTY SOURCES ${_srcs})

            set(_kobj "${CMAKE_CURRENT_BINARY_DIR}/mb_${_name}_${_bs}_kernels.o")
            set(_kdefs -DMODELBLASTER_USE_POOL=1
                       -DMODELBLASTER_DISABLE_UNMANGLED=1
                       -DMODELBLASTER_PLATFORM_LINUX=1)
            foreach(_kv IN LISTS _rename_defs)
                list(APPEND _kdefs "-D${_kv}")
            endforeach()
            add_custom_command(
                OUTPUT "${_kobj}"
                COMMAND "${MODELBLASTER_KERNEL_CC}"
                        -O2 ${_kflags} ${_kdefs}
                        -I "${_backend_dir}" -I "${_RUNTIME_DIR}"
                        -I "${_RUNTIME_DIR}/modelblaster_pool"
                        -c "${_backend_dir}/kernels.c" -o "${_kobj}"
                DEPENDS "${_backend_dir}/kernels.c"
                COMMENT "kernels.c [${_name}/${_bs}] with ${MODELBLASTER_KERNEL_CC}"
                VERBATIM)
            target_sources(xpurt_harness PRIVATE "${_kobj}")
            set_source_files_properties("${_kobj}" PROPERTIES
                EXTERNAL_OBJECT TRUE GENERATED TRUE)
            message(STATUS
                "harness_xpurt_linux: ${_name}/${_bs} kernels.c -> "
                "${MODELBLASTER_KERNEL_CC}")
        endif()

        target_link_libraries(xpurt_harness PRIVATE ${_obj_tgt})
        message(STATUS "harness_xpurt_linux: ${_name}/${_bs} -> ${_obj_tgt}")
    endforeach()

    # One weights TU per (model, backend), matching the Zephyr harness and
    # generate_skeleton._weight_name(). Backends can pack the same logical
    # tensor in different physical layouts, and generated model.c references
    # backend-suffixed symbols such as ..._weight_q_rvv_x60. Linking only the
    # primary backend's weights therefore either fails (distinct symbols) or,
    # with older unsuffixed codegen, silently feeds a kernel the wrong layout.
    foreach(_bs IN LISTS MODEL_BACKENDS)
        set(_backend_dir "${_base}/${_bs}")
        set(_weights_tgt "mb_${_name}_${_bs}_weights_obj")
        add_library(${_weights_tgt} OBJECT "${_backend_dir}/weights.c")
        target_include_directories(${_weights_tgt} PRIVATE "${_backend_dir}")
        target_compile_definitions(${_weights_tgt} PRIVATE
            MODELBLASTER_DISABLE_UNMANGLED=1
            MODELBLASTER_PLATFORM_LINUX=1)
        target_link_libraries(xpurt_harness PRIVATE ${_weights_tgt})
    endforeach()

    # Intermediate buffers are deliberately shared across every backend for a
    # model. Cross-backend dispatches must see the producer's scratch storage.
    if(NOT EXISTS "${_primary_dir}/buffers.c")
        message(FATAL_ERROR
            "${_primary_dir} missing buffers.c -- regenerate the skeleton")
    endif()
    set(_buffers_tgt "mb_${_name}_buffers_obj")
    add_library(${_buffers_tgt} OBJECT "${_primary_dir}/buffers.c")
    target_include_directories(${_buffers_tgt} PRIVATE "${_primary_dir}")
    target_compile_definitions(${_buffers_tgt} PRIVATE
        MODELBLASTER_PLATFORM_LINUX=1)
    target_link_libraries(xpurt_harness PRIVATE ${_buffers_tgt})

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
