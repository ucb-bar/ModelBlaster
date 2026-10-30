# SPDX-License-Identifier: Apache-2.0
#
# modelblaster_pool — raw-pthread parallel-for pool. Builds an OBJECT library
# that any modelblaster harness (harness, harness_multi, harness_xpurt) can
# pull in via:
#
#     add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../runtime/modelblaster_pool
#                      modelblaster_pool_build)
#     target_link_libraries(<your target> PRIVATE modelblaster_pool)
#
# Standalone (test_app/) builds with the same target name; the parent
# harness CMakeLists is the one that decides whether to link
# pthreadpool or modelblaster_pool.
#
# This lib does NOT apply the POSIX-affinity Zephyr patch on its own —
# the harness CMakeLists that pull it in already do (idempotent). For
# the test_app/ standalone build we apply it locally before
# find_package(Zephyr).

if(NOT TARGET modelblaster_pool)
    add_library(modelblaster_pool OBJECT
        ${CMAKE_CURRENT_LIST_DIR}/modelblaster_pool.c
    )
    target_include_directories(modelblaster_pool PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}
    )
    # zephyr_interface gives us Zephyr's compile flags (incl. -march/-mabi
    # matching the rest of the app) plus the kernel headers needed by
    # k_sem and k_thread.
    if(TARGET zephyr_interface)
        target_link_libraries(modelblaster_pool PRIVATE zephyr_interface)
    endif()
endif()
