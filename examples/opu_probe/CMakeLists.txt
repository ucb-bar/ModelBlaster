cmake_minimum_required(VERSION 3.20)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(opu_probe)

target_sources(app PRIVATE src/main.c)
# Compile with full RVV — the probe deliberately exercises opcodes we
# expect the bitstream might trap on.
target_compile_options(app PRIVATE -march=rv64gcv -mabi=lp64d)
target_link_options(app PRIVATE -march=rv64gcv -mabi=lp64d)
