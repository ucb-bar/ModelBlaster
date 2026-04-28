# SPDX-License-Identifier: Apache-2.0
#
# Generic agents harness. Build for a specific model+target by passing
# MODEL_DIR (where the generated C lives) and AGENTS_BACKEND (the HW target):
#
#   west build -p -b spike_riscv64 agents/harness -- \
#       -DMODEL_DIR=$PWD/agents/examples/mlp/generated/scalar \
#       -DAGENTS_BACKEND=scalar
#
# AGENTS_BACKEND selects:
#   - the prj.conf overlay under agents/harness/backends/<name>.conf
#   - the per-source compile flags applied to kernels.c
#   (both come from agents/pipeline/backends.py — passed in via -D)
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

if(NOT DEFINED AGENTS_BACKEND)
    set(AGENTS_BACKEND "scalar")
endif()

# Splice the backend prj.conf overlay BEFORE find_package(Zephyr) so its
# CONFIG_* lines reach Kconfig.
set(CONF_FILE "prj.conf")
set(_BACKEND_CONF "${CMAKE_CURRENT_LIST_DIR}/backends/${AGENTS_BACKEND}.conf")
if(EXISTS "${_BACKEND_CONF}")
    list(APPEND CONF_FILE "${_BACKEND_CONF}")
else()
    message(FATAL_ERROR
        "AGENTS_BACKEND='${AGENTS_BACKEND}' but no overlay at ${_BACKEND_CONF}")
endif()

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(agents_harness)

foreach(_required model.c model.h kernels.c kernels.h weights.c weights.h test_io.h)
    if(NOT EXISTS "${MODEL_DIR}/${_required}")
        message(FATAL_ERROR
            "MODEL_DIR='${MODEL_DIR}' is missing required file: ${_required}. "
            "Run agents/pipeline/generate_skeleton.py and generate_kernels.py first.")
    endif()
endforeach()
message(STATUS "agents harness MODEL_DIR=${MODEL_DIR}  backend=${AGENTS_BACKEND}")

target_sources(app PRIVATE
    src/main.c
    ${MODEL_DIR}/model.c
    ${MODEL_DIR}/kernels.c
    ${MODEL_DIR}/weights.c
)

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
# Zephyr's own build. Flags are passed in via -DAGENTS_KERNEL_CFLAGS=...
# as a CMake list (semicolon-separated).
if(DEFINED AGENTS_KERNEL_CFLAGS AND NOT "${AGENTS_KERNEL_CFLAGS}" STREQUAL "")
    set_source_files_properties(
        ${MODEL_DIR}/kernels.c
        PROPERTIES COMPILE_OPTIONS "${AGENTS_KERNEL_CFLAGS}"
    )
    message(STATUS "agents harness kernel cflags=${AGENTS_KERNEL_CFLAGS}")
endif()
