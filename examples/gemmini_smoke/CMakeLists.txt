cmake_minimum_required(VERSION 3.20)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(gemmini_smoke)

target_sources(app PRIVATE src/main.c)
target_include_directories(app SYSTEM PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../cores/gemmini/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../cores/gemmini"
)
target_compile_definitions(app PRIVATE GEMMINI_ROCC BAREMETAL)
