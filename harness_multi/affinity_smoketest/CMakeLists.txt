# Smoke test for the vendored Phase-A POSIX affinity patch
# (zephyr_patches/posix-affinity.patch). Spawns N pthreads, each pinned
# to a distinct hart via pthread_attr_setaffinity_np, and prints the
# hart id each thread observes via arch_curr_cpu()->id.
#
# Build (from repo root, after sourcing scripts/set_envvars_sdk.sh):
#   west build -p -b spike_riscv64 modelblaster/harness_multi/affinity_smoketest \
#       --build-dir build/affinity_smoketest
#   spike -p4 build/affinity_smoketest/zephyr/zephyr.elf

cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(affinity_smoketest)

target_sources(app PRIVATE src/main.c)
