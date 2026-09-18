#!/usr/bin/env bash
# Build the CANONICAL riskybird KU040 dual-core (2-hart) DroNet deploy/profile ELF,
# NHWC variant: the Gemmini conv island runs native NHWC (no per-op layout conversion).
#
#   hart0 = Rocket + Saturn RVV  (batchnorm/add/relu/linear/sigmoid)
#   hart1 = Rocket + 32x32 Gemmini (conv0+pool + conv1..conv9 NHWC + the boundary relayouts)
#
# On-board vs the NCHW baseline: conv1-9 1587K->174K (9x), conv0 934K->602K,
# gemmini total 2521K->1429K (-43%, incl. 8 relayouts), ~24-27 fps, VERIFY max_abs_err=3.
# See notes/DRONET_KU040_DUAL_REPRODUCTION.md for the full recipe + measured table.
#
# Key difference vs build_ku040_dual.sh (NCHW): assign_layouts --policy islands, and the
# gemmini side compiles the gemmini_q31_rvv backend (only place the NHWC fused conv2d_pool
# + gemmini_blocked_tb32 relayout kernels live), mapped to the gemmini_q31 worker kind via
# --backends rvv,gemmini_q31_rvv so the on-wire impl stays a valid core-kind (no runtime FATAL).
#
# Same env gotcha as the NCHW script: do NOT source set_envvars_sdk.sh under set -e/pipefail.
set -u
: "${MB:=$(cd "$(dirname "$0")/../../.." && pwd)}"
: "${ZCS:=/scratch2/dima/misc_sw/XPU-RT/zephyr-chipyard-sw}"
: "${ZEPHYR_SDK:=$ZCS/tools-manual/zephyr-sdk-1.0.0-beta1}"
: "${CONDA_SH:=/scratch2/dima/miniforge3/etc/profile.d/conda.sh}"
: "${CONDA_ENV:=zephyr}"
: "${GEMMINI_SPIKE:=/scratch2/dima/chipyard-fsim/.conda-env/riscv-tools/bin/spike}"
: "${GEMMINI_LIB:=/scratch2/dima/chipyard-fsim/.conda-env/riscv-tools/lib/libgemmini.so}"
: "${CONSOLE_OVERLAY:=/scratch2/dima/misc_sw/ku040_dronedual_console.overlay}"
: "${CONSOLE_CONF:=/scratch2/dima/misc_sw/ku040_dronedual_console.conf}"
: "${GEMMINI_CONFIG:=q31ws_32x32_acc}"
: "${REGEN_MODEL:=1}"     # 1 = re-extract + assign_layouts + regen both backends
: "${REGEN_SCHED:=0}"     # 1 = regenerate schedule/main; 0 = use committed ku040_dual_nhwc/ artifacts
: "${BUILD_DIR:=$MB/examples/dronet/int8/build/ku040_dronet_nhwc}"

GEN=$MB/examples/dronet/int8/generated
DUAL=$MB/examples/dronet/int8/ku040_dual_nhwc   # committed NHWC schedule + dispatch table + main
SO=$MB/examples/xpurt_demo/int8/generated       # scratch (gitignored)

source "$CONDA_SH"; conda activate "$CONDA_ENV"
export ZEPHYR_BASE="$ZCS/zephyr_ws/zephyr"
export ZEPHYR_SDK_INSTALL_DIR="$ZEPHYR_SDK"
export ZEPHYR_TOOLCHAIN_VARIANT=zephyr
export PATH="/usr/bin:$ZEPHYR_SDK/gnu/riscv64-zephyr-elf/bin:$PATH"
export PYTHONPATH="$(dirname "$MB")"
export MODELBLASTER_GEMMINI_CONFIG="$GEMMINI_CONFIG"
export MODELBLASTER_CURATED_VERIFY=0
export MODELBLASTER_DRONET_CHANNELS=1
export MB_ENABLE_FUSION=1
export MODELBLASTER_GEMMINI_SPIKE="$GEMMINI_SPIKE"
export MODELBLASTER_GEMMINI_LIB="$GEMMINI_LIB"
cd "$MB"

if [ "$REGEN_MODEL" = "1" ]; then
  echo "== extract (grayscale, fused) =="
  python -m modelblaster.pipeline.extract_graph --model dronet --out-dir "$GEN" \
      --quant int8 --num-calibration 1 --fusion-target gemmini_q31_rvv
  echo "== assign_layouts --policy islands (NHWC island around the conv chain) =="
  python pipeline/assign_layouts.py "$GEN/graph.json" "$GEN/graph.json" --policy islands \
      --hint examples/dronet/int8/dronet_layout_hint.json --model dronet --report
  # gemmini_q31_rvv = NHWC fused conv2d_pool + conv2d_nhwc + gemmini_blocked_tb32 relayouts
  # (+ rvv bn/add, which are unused on hart1); rvv = hart0 bn/add/relu/linear/sigmoid.
  for BK in gemmini_q31_rvv rvv; do
    echo "== generate_skeleton [$BK] =="
    python -m modelblaster.pipeline.generate_skeleton --ir "$GEN/graph.json" \
        --weights "$GEN/weights.npz" --io "$GEN/io.npz" --out-dir "$GEN/$BK" --backend "$BK"
    echo "== generate_kernels [$BK] =="
    python -m modelblaster.pipeline.generate_kernels --ir "$GEN/graph.json" \
        --out-dir "$GEN/$BK" --backend reference --target "$BK" --quant int8 \
        --io "$GEN/io.npz" --repo-root "$MB" --build-dir "$MB/examples/dronet/int8/build/${BK}_nhwc" \
        --harness-dir "$MB/harness" --cache-dir "$MB/examples/dronet/int8/cache/${BK}_nhwc" \
        --algorithms all --global-curated-dir "$MB/kernels"
  done
fi
# NOTE: no bake_bn_int.py -- BN runs on the rvv hart (direct), not fused into a gemmini conv.

if [ "$REGEN_SCHED" = "1" ]; then
  echo "== regenerate NHWC schedule (conv/pool + relayouts -> gemmini/hart1; rest -> rvv/hart0) =="
  mkdir -p "$SO" "$DUAL"
  python scripts/gen_hetero_schedule.py --ir "$GEN/graph.json" \
      --out "$SO/dronet_ku040_nhwc_sched.json" --job-name dronet --policy gemmini_main_opu_skip
  python - "$GEN/graph.json" "$SO/dronet_ku040_nhwc_sched.json" <<'PY'
import json,sys
g=json.load(open(sys.argv[1])); opk={o.get("dispatch_id"):o.get("op") for o in g["ops"] if o.get("dispatch_id") is not None}
S=json.load(open(sys.argv[2]))
# relayouts use gemmini_blocked_tb32 (a Gemmini RoCC transpose) -> hart1, adjacent to the conv island.
GEM={"conv2d_s8","conv2d_pool_s8","nchw_to_nhwc_s8","nhwc_to_nchw_s8"}
for k,v in S["dispatches"].items():
    v.pop("impl",None)                                    # impl defaults to core_kind (rvv/gemmini_q31)
    v["hardware_target"]="CPU_P#0" if opk.get(v.get("id")) in GEM else "CPU_E#0"
json.dump(S,open(sys.argv[2],"w"),indent=2)
PY
  python -m modelblaster.pipeline.ingest_xpurt_schedule --schedule "$SO/dronet_ku040_nhwc_sched.json" \
      --registry cores/chipyard_ku040_dronedual_q31.json --ir dronet:"$GEN/graph.json" \
      --cpu-p-kind gemmini_q31 --cpu-e-kind rvv --name ku040_dronet_nhwc_dev --out "$SO/ku040_dronet_nhwc_dev.c"
  python -m modelblaster.pipeline.generate_xpurt_main --schedule "$SO/dronet_ku040_nhwc_sched.json" \
      --out "$SO/ku040_dronet_nhwc_dev_main.c" --name ku040_dronet_nhwc_dev \
      --dispatch-table-header ku040_dronet_nhwc_dev.h --platform zephyr \
      --core-kinds rvv,gemmini_q31 --backends rvv,gemmini_q31_rvv \
      --model-gen-dir dronet="$GEN/gemmini_q31_rvv" --networks dronet --registry cores/chipyard_ku040_dronedual_q31.json
  cp "$SO"/dronet_ku040_nhwc_sched.json "$SO"/ku040_dronet_nhwc_dev.{c,h} "$SO"/ku040_dronet_nhwc_dev_main.c "$DUAL/"
fi

# ---- west build: kind gemmini_q31 -> backend gemmini_q31_rvv (NHWC kernels); impl stays gemmini_q31
SOFTFLOAT_CONF=$(mktemp)
cat > "$SOFTFLOAT_CONF" <<CONF
CONFIG_FPU=n
CONFIG_RISCV_ISA_EXT_V=y
CONFIG_RISCV_ISA_EXT_V_LAZY=n
CONFIG_RISCV_V_KERNEL_ONLY=y
CONFIG_MP_MAX_NUM_CPUS=2
CONFIG_RV_BOOT_HART=0
CONF
PC=$MB/cores/gemmini/include/per_config/$GEMMINI_CONFIG
# gemmini_q31_rvv kernels.c carries rvv bn/add (need zve64x) + the NHWC conv2d_pool that
# #includes conv2d_pool_loadonce.h (need -isystem kernels/gemmini_q31_rvv).
GCF="-march=rv64imac_zve64x;-mabi=lp64;-isystem$PC;-isystem$MB/cores/gemmini/include;-isystem$MB/cores/gemmini;-isystem$MB/kernels/gemmini_q31_rvv;-DGEMMINI_ROCC;-DBAREMETAL;-DMODELBLASTER_GEMMINI_HWIO_WEIGHTS=1;-DMODELBLASTER_GEMMINI_Q31_ACC_SCALE=1"
RCF="-march=rv64imac_zve64x;-mabi=lp64;-DMODELBLASTER_RVV_IHWOC_WEIGHTS=1"
echo "== west build -> $BUILD_DIR =="
west build -p always -b chipyard_riscv64 harness_xpurt --build-dir "$BUILD_DIR" -- \
  -DMODEL_NAMES=dronet -DMODEL_DIRS_BASE="$GEN" -DMODEL_BACKENDS=gemmini_q31_rvv,rvv \
  -DXPURT_SCHEDULE_C="$DUAL/ku040_dronet_nhwc_dev.c" -DXPURT_MAIN_C="$DUAL/ku040_dronet_nhwc_dev_main.c" \
  -DXPURT_INCLUDE_DIR="$DUAL" \
  -DMODELBLASTER_KERNEL_CFLAGS_GEMMINI_Q31_RVV="$GCF" -DMODELBLASTER_KERNEL_CFLAGS_RVV="$RCF" \
  -DEXTRA_CONF_FILE="$MB/harness_xpurt/backends/rvv.conf;$MB/harness/backends/firesim_chipyard_dual_gemmini.conf;$CONSOLE_CONF;$SOFTFLOAT_CONF" \
  -DEXTRA_DTC_OVERLAY_FILE="$CONSOLE_OVERLAY"
rc=$?
rm -f "$SOFTFLOAT_CONF"
echo "build rc=$rc  elf=$BUILD_DIR/zephyr/zephyr.elf"
