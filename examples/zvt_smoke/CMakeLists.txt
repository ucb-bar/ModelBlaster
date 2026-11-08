# Zvt linear_s8 smoke test: Zvt tile kernel vs the scalar reference (bit-exact)
# and the curated RVV kernel, on spike with Zvt (riscv-isa-sim >= 72958551).
cmake_minimum_required(VERSION 3.20)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(zvt_smoke)

target_sources(app PRIVATE src/main.c src/ref_linear_s8.c src/zvt_linear_s8.c src/rvv_linear_s8.c
    src/ref_conv2d_s8.c src/zvt_conv2d_s8.c src/ref_linear_f32.c src/zvt_linear_f32.c src/rvv_linear_f32.c)
target_include_directories(app SYSTEM PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../cores/zvt/include")
# CONFIG_RISCV_V_KERNEL_ONLY drops 'v' from the global -march; the kernels
# opt back in per file (as the harness does via the backend kernel_cflags).
set_source_files_properties(src/zvt_linear_s8.c src/rvv_linear_s8.c src/zvt_conv2d_s8.c
    src/zvt_linear_f32.c src/rvv_linear_f32.c
    PROPERTIES COMPILE_OPTIONS "-march=rv64gcv;-mabi=lp64d")
# fp32 reference without FMA contraction (the semantics vtfmm implements).
set_source_files_properties(src/ref_linear_f32.c PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
