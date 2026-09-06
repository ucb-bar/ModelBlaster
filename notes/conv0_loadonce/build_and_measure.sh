#!/usr/bin/env bash
# Reproducible build + on-bench measurement of the load-once conv0 kernel via an
# isolated copy of gemmini_ubench (exact DroNet c0g shape + fused 3x3-s2 maxpool,
# Gemmini EX/LD/ST perf counters, bit-exact vs tiled_conv_auto+pool AND a scalar
# golden). Off-bench build; guarded on-bench JTAG-load (SRAM only, no reflash).
#
# Prereqs (one-time): the isolated ubench app dir lives in the parent chipyard tree
#   ZCS=/home/cobble/Tools/zephyr-chipyard-sw
#   cp -r $ZCS/gemmini_ubench $ZCS/gemmini_ubench_lo
#   cp <this repo>/notes/conv0_loadonce/ubench_bench.c $ZCS/gemmini_ubench_lo/src/bench.c
# (ubench_bench.c #includes this repo's kernels/gemmini_q31_rvv/conv2d_pool_loadonce.h
#  by absolute path -- edit that path if the worktree moves.)
set -uo pipefail
ZCS=/home/cobble/Tools/zephyr-chipyard-sw
APP=gemmini_ubench_lo
BUILD=$ZCS/build_ubench_lo
SC="$(mktemp -d)"
OCD_CFG="${OCD_CFG:-/home/cobble/Tools/riskybird/scripts/openocd/arty200t_rocket.cfg}"   # 6010 build; use the 6011 cfg if the binary FTDI enumerates 6011
FTDI_LOC="${FTDI_LOCATION:-3-6}"
CON="${CON:-/dev/ttyUSB1}"

echo "=== [off-bench] build ==="
cd "$ZCS"
source scripts/activate_conda.sh   >/dev/null 2>&1
source scripts/set_envvars_sdk.sh  >/dev/null 2>&1
export PATH="/usr/bin:${PATH}"
west build -b chipyard_riscv64 "$APP" --build-dir "$BUILD" 2>&1 | tail -3
ELF="$BUILD/zephyr/zephyr.elf"
[ -f "$ELF" ] || { echo "build failed"; exit 1; }

echo "=== [bench guard] ==="
if pgrep -x openocd >/dev/null || pgrep -x openFPGALoader >/dev/null || pgrep -x vivado >/dev/null; then
  echo "!! BENCH BUSY (openocd/openFPGALoader/vivado) -- aborting"; exit 3; fi

CONLOG="$SC/con.log"
stty -F "$CON" 115200 raw -echo -echoe -echok -ixon -crtscts 2>/dev/null || true
: > "$CONLOG"; timeout 30 cat "$CON" > "$CONLOG" 2>/dev/null & CATPID=$!
sleep 1
echo "=== [on-bench] JTAG load + resume (FTDI $FTDI_LOC) ==="
FTDI_LOCATION="$FTDI_LOC" FTDI_SPEED=1000 timeout 90 openocd -f "$OCD_CFG" \
  -c "init; halt; reg mstatus 0x0; reg mie 0x0; load_image $ELF; resume 0x80000000; shutdown" 2>&1 \
  | grep -iE "misa|Error|Failed|shutdown" | head -6
wait $CATPID 2>/dev/null
echo "=== RESULTS ==="
grep -aE "^cfg,|^UBP,|^UBV," "$CONLOG"
