# SPDX-License-Identifier: Apache-2.0
#
# Shared-input, per-variant-output modelblaster harness. Runs N kernel
# implementations of the SAME KernelBench problem in one Zephyr boot,
# sharing the input tensor in rodata (baked once via test_io.S) and
# writing each variant's output to its own buffer in ram0.
#
# Motivation: for RL sweeps that generate N candidate kernels of the
# same problem, running all N in one FireSim boot amortizes the multi-
# minute per-boot cost across the batch. Additionally, sharing the
# 100+ MiB baked input means only ONE copy in the ELF's rodata instead
# of N copies — the memory-limit saving Dima flagged.
#
# Pass:
#   -DMODELBLASTER_BACKEND=<scalar|rvv>   common HW backend for the shared model.
#   -DMODEL_DIR=<abs>                     ONE generated/<target>/ dir
#                                         (holds model.c, model.h, kernels.h,
#                                         weights.c, weights.h, buffers.c,
#                                         test_io.h, test_io.S,
#                                         test_input.bin, test_golden.bin).
#   -DVARIANT_SOURCES=<c1>;<c2>;...       semicolon list of variant kernels.c
#                                         paths (each has the LLM-emitted
#                                         kernel body; same declared function
#                                         name as MODEL_DIR/kernels.h).
#   -DVARIANT_TAGS=<t1>;<t2>;...          OPTIONAL semicolon list of tag
#                                         strings for MODELBLASTER_WALL_CYCLES
#                                         markers. Defaults to v0,v1,... —
#                                         must be parallel to VARIANT_SOURCES.
#   -DMODELBLASTER_KERNEL_CFLAGS=...      optional per-backend cflags
#                                         applied ONLY to kernels.c.

cmake_minimum_required(VERSION 3.20.0)

if(NOT DEFINED MODELBLASTER_BACKEND)
    set(MODELBLASTER_BACKEND "scalar")
endif()
if(NOT DEFINED MODEL_DIR OR NOT DEFINED VARIANT_SOURCES)
    message(FATAL_ERROR
        "Shared-input harness requires -DMODEL_DIR and -DVARIANT_SOURCES "
        "(semicolon list of variant kernels.c paths).")
endif()

list(LENGTH VARIANT_SOURCES _n_variants)
if(_n_variants LESS 1)
    message(FATAL_ERROR "VARIANT_SOURCES is empty.")
endif()

# ``_last_variant`` is used by every foreach(_i RANGE ...) loop below.
# Compute it ONCE up-front, unconditionally — a prior version only set
# it inside the "TAGS not defined" branch, so caller-provided TAGS
# left it undefined and the RANGE fell through to 0..0. Symptoms
# looked like "only the first variant links; second one is missing"
# — spent an hour chasing an imagined dependency race.
math(EXPR _last_variant "${_n_variants} - 1")

# Default tags = v0, v1, ..., vN-1.
if(NOT DEFINED VARIANT_TAGS OR "${VARIANT_TAGS}" STREQUAL "")
    set(VARIANT_TAGS "")
    foreach(_i RANGE ${_last_variant})
        list(APPEND VARIANT_TAGS "v${_i}")
    endforeach()
endif()
list(LENGTH VARIANT_TAGS _n_tags)
if(NOT _n_tags EQUAL _n_variants)
    message(FATAL_ERROR
        "VARIANT_TAGS length (${_n_tags}) != VARIANT_SOURCES length "
        "(${_n_variants}).")
endif()

# Splice the backend prj.conf overlay BEFORE find_package(Zephyr).
# Reuse harness_multi's backend configs — the compilation-unit shape is
# identical (one OBJECT lib per model+variant, same shared Zephyr link).
set(CONF_FILE "prj.conf")
set(_BACKEND_CONF "${CMAKE_CURRENT_LIST_DIR}/../harness_multi/backends/${MODELBLASTER_BACKEND}.conf")
if(EXISTS "${_BACKEND_CONF}")
    list(APPEND CONF_FILE "${_BACKEND_CONF}")
else()
    message(FATAL_ERROR
        "MODELBLASTER_BACKEND='${MODELBLASTER_BACKEND}' but no overlay at ${_BACKEND_CONF}")
endif()

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(modelblaster_harness_shared_input)

# Pull in the modelblaster_pool runtime (raw-pthread parallel-for) —
# same integration as harness_multi. All variants share the same pool
# instance; each variant runs to completion before the next starts, so
# there is no cross-variant contention on worker threads.
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../runtime/modelblaster_pool"
                 modelblaster_pool_build)

# Validate MODEL_DIR has the expected shape (matches harness_multi's check).
foreach(_required model.c model.h kernels.c kernels.h weights.c weights.h buffers.c test_io.h test_io.S test_input.bin test_golden.bin)
    if(NOT EXISTS "${MODEL_DIR}/${_required}")
        message(FATAL_ERROR
            "MODEL_DIR='${MODEL_DIR}' is missing ${_required}. "
            "Re-run modelblaster/pipeline/generate_skeleton.py.")
    endif()
endforeach()

# Stage per-variant generated headers under build/modelblaster_shared/<tag>/
# so shared_input_main.c can include them with disambiguated names, and
# each variant's OBJECT lib compiles against its own mangled-symbol
# rename header (auto-generated below).
set(_STAGE_DIR "${CMAKE_CURRENT_BINARY_DIR}/modelblaster_shared")
file(MAKE_DIRECTORY "${_STAGE_DIR}")

# ---------------------------------------------------------------------
# Auto-generate per-variant symbol rename headers.
# gen_variant_renames.py parses MODEL_DIR/{model.h,model.c,kernels.h} to
# enumerate every mangled runtime symbol (run_model_<mid>,
# MODEL_<MID>_DISPATCH_FNS, dispatch_<mid>_<N>, kernel_<op>_<mid>,
# model_<mid>_profile_records / _wall_cycles / _set_wall_cycles /
# _reset_profile) and emits <tag>_renames.h with the tag suffix
# applied. Weights.c / buffers.c / test_io.S symbols are NOT renamed —
# their data is shared across all variants (same weights, same input).
# ---------------------------------------------------------------------
set(_GEN_PY "${CMAKE_CURRENT_LIST_DIR}/gen_variant_renames.py")
set(_MAIN_GEN_PY "${CMAKE_CURRENT_LIST_DIR}/gen_shared_input_main.py")

set(_VARIANT_RENAME_HEADERS "")
foreach(_i RANGE ${_last_variant})
    list(GET VARIANT_TAGS ${_i} _tag)
    set(_rename_h "${_STAGE_DIR}/${_tag}_renames.h")
    add_custom_command(
        OUTPUT "${_rename_h}"
        COMMAND ${CMAKE_COMMAND} -E env
                "PYTHONPATH=${CMAKE_CURRENT_LIST_DIR}/.."
                "${Python3_EXECUTABLE}" "${_GEN_PY}"
                --model-dir "${MODEL_DIR}"
                --tag "${_tag}"
                --out "${_rename_h}"
        DEPENDS "${_GEN_PY}"
                "${MODEL_DIR}/model.h"
                "${MODEL_DIR}/model.c"
                "${MODEL_DIR}/kernels.h"
        COMMENT "harness_shared_input: renames for variant ${_tag}"
        VERBATIM
    )
    list(APPEND _VARIANT_RENAME_HEADERS "${_rename_h}")
endforeach()

# ---------------------------------------------------------------------
# Per-variant OBJECT library. Each recompiles model.c + weights.c +
# buffers.c + test_io.S from MODEL_DIR + its own VARIANT_SOURCES[_i]
# (LLM-emitted kernels.c), with the auto-generated rename header
# -include'd first so every mangled symbol picks up the tag suffix.
#
# Note: weights.c + test_io.S produce identical binary blobs for every
# variant. The linker will merge identically-named .rodata section
# contents into one — no bloat despite the per-variant compilation.
# ---------------------------------------------------------------------
set(_ALL_VARIANT_LIBS "")
foreach(_i RANGE ${_last_variant})
    list(GET VARIANT_SOURCES ${_i} _kern_c)
    list(GET VARIANT_TAGS    ${_i} _tag)
    list(GET _VARIANT_RENAME_HEADERS ${_i} _rename_h)
    if(NOT EXISTS "${_kern_c}")
        message(FATAL_ERROR
            "VARIANT_SOURCES[${_i}]='${_kern_c}' does not exist.")
    endif()

    # Per-variant OBJECT lib carries ONLY code that gets a per-variant
    # symbol namespace: model.c (renamed dispatch table + run_model +
    # introspection helpers) and the variant's kernels.c. weights.c,
    # buffers.c, and test_io.S are SHARED — same content in every
    # variant, so we compile them exactly once (into the app itself,
    # below) to avoid multiply-defined rodata / extern-linkage symbols
    # at the final link. model.c's references to weights_*/buf_*/
    # test_input etc. stay unrenamed and resolve to the single shared
    # copy — gen_variant_renames.py explicitly excludes those symbols.
    add_library(modelblaster_${_tag}_objs OBJECT
        "${MODEL_DIR}/model.c"
        "${_kern_c}"
    )

    # Rename header -include'd first so every subsequent header sees the
    # macro-mangled symbol names. This is what actually makes N copies
    # of run_model_<mid>() link into one binary without collision.
    target_compile_options(modelblaster_${_tag}_objs PRIVATE
        "-include" "${_rename_h}")

    # Make the OBJECT lib depend on the auto-gen header so its own build
    # blocks until gen_variant_renames.py has emitted it.
    add_custom_target(gen_${_tag}_renames DEPENDS "${_rename_h}")
    add_dependencies(modelblaster_${_tag}_objs gen_${_tag}_renames)

    target_include_directories(modelblaster_${_tag}_objs PRIVATE "${MODEL_DIR}")
    target_link_libraries(modelblaster_${_tag}_objs PRIVATE zephyr_interface)
    # MODELBLASTER_DISABLE_UNMANGLED strips the model.h convenience alias
    # block ("#define model_wall_cycles model_<mid>_wall_cycles" etc.).
    # For shared-input, those aliases would collide across N variants
    # (they resolve to per-variant mangled names, and the alias itself
    # is single-model). Same reason harness_multi defines it.
    target_compile_definitions(modelblaster_${_tag}_objs PRIVATE
        MODELBLASTER_USE_POOL=1
        MODELBLASTER_DISABLE_UNMANGLED=1)
    target_link_libraries(modelblaster_${_tag}_objs PRIVATE modelblaster_pool)

    # Backend cflags apply ONLY to the LLM-emitted kernels.c — same
    # convention as harness_multi.
    if(DEFINED MODELBLASTER_KERNEL_CFLAGS AND NOT "${MODELBLASTER_KERNEL_CFLAGS}" STREQUAL "")
        set_source_files_properties(
            "${_kern_c}"
            PROPERTIES COMPILE_OPTIONS "${MODELBLASTER_KERNEL_CFLAGS}"
        )
    endif()

    list(APPEND _ALL_VARIANT_LIBS "modelblaster_${_tag}_objs")
endforeach()

# ---------------------------------------------------------------------
# Generate shared_input_main.c — the top-level dispatcher that loads
# the shared input once, calls each run_model_<mid>_v<tag>() with the
# per-variant output buffer, times the call, verifies against the
# shared golden, and emits MODELBLASTER_* markers keyed on <mid>@<tag>.
# ---------------------------------------------------------------------
set(_MAIN_C "${_STAGE_DIR}/shared_input_main.c")
string(REPLACE ";" "," _TAGS_CSV "${VARIANT_TAGS}")
add_custom_command(
    OUTPUT "${_MAIN_C}"
    COMMAND ${CMAKE_COMMAND} -E env
            "PYTHONPATH=${CMAKE_CURRENT_LIST_DIR}/.."
            "${Python3_EXECUTABLE}" "${_MAIN_GEN_PY}"
            --model-dir "${MODEL_DIR}"
            --tags "${_TAGS_CSV}"
            --out "${_MAIN_C}"
    DEPENDS "${_MAIN_GEN_PY}"
            "${MODEL_DIR}/model.h"
            "${MODEL_DIR}/test_io.h"
    COMMENT "harness_shared_input: shared_input_main.c (${_n_variants} variants)"
    VERBATIM
)

# App carries the shared-data TUs: weights.c + buffers.c + test_io.S.
# Compiled ONCE into libapp.a; the per-variant OBJECT libs' extern refs
# to these symbols resolve here at final link. Same convention as the
# single-model harness (single copy of weights + baked I/O), just with
# an explicit target instead of implicit via the model dir.
target_sources(app PRIVATE
    "${_MAIN_C}"
    "${MODEL_DIR}/weights.c"
    "${MODEL_DIR}/buffers.c"
    "${MODEL_DIR}/test_io.S"
)
set_source_files_properties("${MODEL_DIR}/test_io.S" PROPERTIES LANGUAGE ASM)
target_include_directories(app PRIVATE "${MODEL_DIR}" "${_STAGE_DIR}")
# App defines MODELBLASTER_DISABLE_UNMANGLED so its own inclusion of
# model.h (via shared_input_main.c) does not emit the single-model
# alias block — which is defined in every variant OBJECT lib and
# would collide with the app's copy.
target_compile_definitions(app PRIVATE MODELBLASTER_DISABLE_UNMANGLED=1)
target_link_libraries(app PRIVATE modelblaster_pool)
foreach(_lib IN LISTS _ALL_VARIANT_LIBS)
    target_link_libraries(app PRIVATE ${_lib})
endforeach()

message(STATUS "harness_shared_input: ${_n_variants} variants of ${MODEL_DIR}, tags=${VARIANT_TAGS}")
