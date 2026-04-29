# SPDX-License-Identifier: Apache-2.0
#
# agents_pool — raw-pthread parallel-for pool. Builds an OBJECT library
# that any agents harness (harness, harness_multi, harness_xpurt) can
# pull in via:
#
#     add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../runtime/agents_pool
#                      agents_pool_build)
#     target_link_libraries(<your target> PRIVATE agents_pool)
#
# Standalone (test_app/) builds with the same target name; the parent
# harness CMakeLists is the one that decides whether to link
# pthreadpool or agents_pool.
#
# This lib does NOT apply the POSIX-affinity Zephyr patch on its own —
# the harness CMakeLists that pull it in already do (idempotent). For
# the test_app/ standalone build we apply it locally before
# find_package(Zephyr).

if(NOT TARGET agents_pool)
    add_library(agents_pool OBJECT
        ${CMAKE_CURRENT_LIST_DIR}/agents_pool.c
    )
    target_include_directories(agents_pool PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}
    )
    # zephyr_interface gives us Zephyr's compile flags (incl. -march/-mabi
    # matching the rest of the app) plus the kernel headers needed by
    # k_sem and k_thread.
    if(TARGET zephyr_interface)
        target_link_libraries(agents_pool PRIVATE zephyr_interface)
    endif()
endif()
