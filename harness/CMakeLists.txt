# SPDX-License-Identifier: Apache-2.0
#
# Generic modelblaster harness. Build for a specific model+target by passing
# MODEL_DIR (where the generated C lives) and MODELBLASTER_BACKEND (the HW target):
#
#   west build -p -b spike_riscv64 modelblaster/harness -- \
#       -DMODEL_DIR=$PWD/modelblaster/examples/mlp/generated/scalar \
#       -DMODELBLASTER_BACKEND=scalar
#
# MODELBLASTER_BACKEND selects:
#   - the prj.conf overlay under modelblaster/harness/backends/<name>.conf
#   - the per-source compile flags applied to kernels.c
#   (both come from modelblaster/pipeline/backends.py — passed in via -D)
#
# Defaults to "scalar" if unset.

cmake_minimum_required(VERSION 3.20.0)

if(NOT DEFINED MODEL_DIR)
    message(FATAL_ERROR
        "MODEL_DIR not set. Pass -DMODEL_DIR=<absolute path to a generated/<target>/ dir>.")
endif()
get_filename_component(MODEL_DIR "${MODEL_DIR}" ABSOLUTE)
if(NOT IS_DIRECTORY "${MODEL_DIR}")
    message(FATAL_ERROR "MODEL_DIR='${MODEL_DIR}' is not a directory.")
endif()

if(NOT DEFINED MODELBLASTER_BACKEND)
    set(MODELBLASTER_BACKEND "scalar")
endif()

# Splice the backend prj.conf overlay BEFORE find_package(Zephyr) so its
# CONFIG_* lines reach Kconfig.
set(CONF_FILE "prj.conf")
set(_BACKEND_CONF "${CMAKE_CURRENT_LIST_DIR}/backends/${MODELBLASTER_BACKEND}.conf")
if(EXISTS "${_BACKEND_CONF}")
    list(APPEND CONF_FILE "${_BACKEND_CONF}")
else()
    message(FATAL_ERROR
        "MODELBLASTER_BACKEND='${MODELBLASTER_BACKEND}' but no overlay at ${_BACKEND_CONF}")
endif()

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(modelblaster_harness)

foreach(_required model.c model.h kernels.c kernels.h weights.c weights.h buffers.c test_io.h test_io.S test_input.bin test_golden.bin)
    if(NOT EXISTS "${MODEL_DIR}/${_required}")
        message(FATAL_ERROR
            "MODEL_DIR='${MODEL_DIR}' is missing required file: ${_required}. "
            "Run modelblaster/pipeline/generate_skeleton.py and generate_kernels.py first.")
    endif()
endforeach()
message(STATUS "modelblaster harness MODEL_DIR=${MODEL_DIR}  backend=${MODELBLASTER_BACKEND}")

target_sources(app PRIVATE
    src/main.c
    ${MODEL_DIR}/model.c
    ${MODEL_DIR}/kernels.c
    ${MODEL_DIR}/weights.c
    ${MODEL_DIR}/buffers.c
    ${MODEL_DIR}/test_io.S
)
set_source_files_properties(${MODEL_DIR}/test_io.S PROPERTIES LANGUAGE ASM)

# Big models (mlp_control's hidden layers, dronet's conv weights) push
# .sdata past the ±2KB window that the linker tries to fold into
# `gp`-relative loads via R_RISCV_GPREL_I, producing
# "relocation truncated to fit" against e.g. z_malloc_heap. Disable
# linker relaxation here — medany still gives ±2GB addressing, we just
# lose the gp-relative shortcut. No measurable code-size impact.
zephyr_link_libraries(-Wl,--no-relax)
target_include_directories(app PRIVATE
    src
    ${MODEL_DIR}
)

# Apply backend-specific compile flags ONLY to kernels.c so we don't perturb
# Zephyr's own build. Flags are passed in via -DMODELBLASTER_KERNEL_CFLAGS=...
# as a CMake list (semicolon-separated).
if(DEFINED MODELBLASTER_KERNEL_CFLAGS AND NOT "${MODELBLASTER_KERNEL_CFLAGS}" STREQUAL "")
    set_source_files_properties(
        ${MODEL_DIR}/kernels.c
        PROPERTIES COMPILE_OPTIONS "${MODELBLASTER_KERNEL_CFLAGS}"
    )
    message(STATUS "modelblaster harness kernel cflags=${MODELBLASTER_KERNEL_CFLAGS}")
endif()
