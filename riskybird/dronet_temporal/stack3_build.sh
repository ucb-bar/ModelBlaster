#!/usr/bin/env bash
echo "ENV_SENTINEL: stack3_build $(date)"
M=/scratch2/dima/misc_sw/xpurt_repro_wt/modelblaster
ZCS=/scratch2/dima/misc_sw/XPU-RT/zephyr-chipyard-sw; SDK=$ZCS/tools-manual/zephyr-sdk-1.0.0-beta1
source /scratch2/dima/miniforge3/etc/profile.d/conda.sh; conda activate zephyr
export ZEPHYR_BASE=$ZCS/zephyr_ws/zephyr ZEPHYR_SDK_INSTALL_DIR=$SDK ZEPHYR_TOOLCHAIN_VARIANT=zephyr
export PATH=/usr/bin:$SDK/gnu/riscv64-zephyr-elf/bin:$PATH ; export PYTHONPATH="$(dirname "$M")"
export MODELBLASTER_GEMMINI_CONFIG=q31ws_32x32_acc MODELBLASTER_CURATED_VERIFY=0
export MODELBLASTER_DRONET_CHANNELS=3 MB_ENABLE_FUSION=1
export MODELBLASTER_DRONET_CKPT=/scratch2/dima/misc_sw/XPU-RT/logs/dronet_himax_stack3/best.pt
export MODELBLASTER_GEMMINI_SPIKE=/scratch2/dima/chipyard-fsim/.conda-env/riscv-tools/bin/spike
export MODELBLASTER_GEMMINI_LIB=/scratch2/dima/chipyard-fsim/.conda-env/riscv-tools/lib/libgemmini.so
cd "$M"; GEN=examples/dronet/int8/generated_stack3; SO=examples/xpurt_demo/int8/generated
echo "== extract grayscale+fused =="
python -m modelblaster.pipeline.extract_graph --model dronet --out-dir $GEN --quant int8 --num-calibration 1 --fusion-target gemmini_q31_rvv 2>&1 | tail -2
echo "== assign_layouts --policy islands (NHWC) =="
python pipeline/assign_layouts.py $GEN/graph.json $GEN/graph.json --policy islands \
  --hint examples/dronet/int8/dronet_layout_hint.json --model dronet --report 2>&1 | tail -3
python - <<PY
import json;g=json.load(open("examples/dronet/int8/generated_stack3/graph.json"))
ops=[o for o in g["ops"] if o.get("dispatch_id") is not None]
from collections import Counter
print("dispatch ops:",len(ops),"kinds:",dict(Counter(o["op"] for o in ops)))
PY
for BK in gemmini_q31_rvv rvv; do
  echo "== generate_skeleton [$BK] =="
  python -m modelblaster.pipeline.generate_skeleton --ir $GEN/graph.json --weights $GEN/weights.npz --io $GEN/io.npz --out-dir $GEN/$BK --backend $BK 2>&1 | tail -1
  echo "== generate_kernels [$BK] =="
  python -m modelblaster.pipeline.generate_kernels --ir $GEN/graph.json --out-dir $GEN/$BK \
    --backend reference --target $BK --quant int8 --io $GEN/io.npz --repo-root "$M" \
    --build-dir examples/dronet/int8/build/${BK}_stack3 --harness-dir "$M/harness" \
    --cache-dir examples/dronet/int8/cache/${BK}_stack3 --algorithms all --global-curated-dir "$M/kernels" 2>&1 | tail -2
  python -c "import json;p=json.load(open('$GEN/$BK/kernel_picks.json'));print('  picks[$BK]:',{k:(v.get('algorithm') if isinstance(v,dict) else v) for k,v in (p.get('picks',p).items() if isinstance(p,dict) else [])})" 2>/dev/null
done
echo "== schedule (relayouts+conv -> gemmini/hart1; rest -> rvv/hart0) =="
python scripts/gen_hetero_schedule.py --ir $GEN/graph.json --out $SO/dronet_ku040_stack3_sched.json --job-name dronet --policy gemmini_main_opu_skip 2>&1 | tail -1
python - <<PY
import json
g=json.load(open("examples/dronet/int8/generated_stack3/graph.json"))
opk={o.get("dispatch_id"):o.get("op") for o in g["ops"] if o.get("dispatch_id") is not None}
S=json.load(open("examples/xpurt_demo/int8/generated/dronet_ku040_stack3_sched.json"))
GEM={"conv2d_s8","conv2d_pool_s8","nchw_to_stack3_s8","nhwc_to_nchw_s8"}
nP=nE=0
for k,v in S["dispatches"].items():
    v.pop("impl",None)
    if opk.get(v.get("id")) in GEM: v["hardware_target"]="CPU_P#0"; nP+=1
    else: v["hardware_target"]="CPU_E#0"; nE+=1
json.dump(S,open("examples/xpurt_demo/int8/generated/dronet_ku040_stack3_sched.json","w"),indent=2)
print(f"  CPU_P(gemmini/hart1)={nP} CPU_E(rvv/hart0)={nE}")
PY
mkdir -p examples/dronet/int8/ku040_dual_stack3
python -m modelblaster.pipeline.ingest_xpurt_schedule --schedule $SO/dronet_ku040_stack3_sched.json \
  --registry cores/chipyard_ku040_dronedual_q31.json --ir dronet:$GEN/graph.json \
  --cpu-p-kind gemmini_q31 --cpu-e-kind rvv --name ku040_dronet_stack3_dev --out $SO/ku040_dronet_stack3_dev.c 2>&1 | tail -2
grep -oE "\.impl = \"[a-z0-9_]+\"" $SO/ku040_dronet_stack3_dev.c | sort | uniq -c
python -m modelblaster.pipeline.generate_xpurt_main --schedule $SO/dronet_ku040_stack3_sched.json \
  --out $SO/ku040_dronet_stack3_dev_main.c --name ku040_dronet_stack3_dev --dispatch-table-header ku040_dronet_stack3_dev.h \
  --platform zephyr --core-kinds rvv,gemmini_q31 --backends rvv,gemmini_q31_rvv \
  --model-gen-dir dronet=$GEN/gemmini_q31_rvv --networks dronet --registry cores/chipyard_ku040_dronedual_q31.json 2>&1 | tail -1
cp $SO/dronet_ku040_stack3_sched.json $SO/ku040_dronet_stack3_dev.{c,h} $SO/ku040_dronet_stack3_dev_main.c examples/dronet/int8/ku040_dual_stack3/
echo "== west build (NHWC, backends rvv+gemmini_q31_rvv) =="
cat > /tmp/sf.conf <<CONF
CONFIG_FPU=n
CONFIG_RISCV_ISA_EXT_V=y
CONFIG_RISCV_ISA_EXT_V_LAZY=n
CONFIG_RISCV_V_KERNEL_ONLY=y
CONFIG_MP_MAX_NUM_CPUS=2
CONFIG_RV_BOOT_HART=0
CONF
PC=$M/cores/gemmini/include/per_config/q31ws_32x32_acc
GCF="-march=rv64imac_zve64x;-mabi=lp64;-isystem$PC;-isystem$M/cores/gemmini/include;-isystem$M/cores/gemmini;-isystem$M/kernels/gemmini_q31_rvv;-DGEMMINI_ROCC;-DBAREMETAL;-DMODELBLASTER_GEMMINI_HWIO_WEIGHTS=1;-DMODELBLASTER_GEMMINI_Q31_ACC_SCALE=1"
RCF="-march=rv64imac_zve64x;-mabi=lp64;-DMODELBLASTER_RVV_IHWOC_WEIGHTS=1"
D=examples/dronet/int8/ku040_dual_stack3
west build -p always -b chipyard_riscv64 harness_xpurt --build-dir examples/xpurt_demo/int8/build/ku040_dronet_stack3 -- \
  -DMODEL_NAMES=dronet -DMODEL_DIRS_BASE=$M/$GEN -DMODEL_BACKENDS=gemmini_q31_rvv,rvv \
  -DXPURT_SCHEDULE_C=$M/$D/ku040_dronet_stack3_dev.c -DXPURT_MAIN_C=$M/$D/ku040_dronet_stack3_dev_main.c -DXPURT_INCLUDE_DIR=$M/$D \
  -DMODELBLASTER_KERNEL_CFLAGS_GEMMINI_Q31_RVV="$GCF" -DMODELBLASTER_KERNEL_CFLAGS_RVV="$RCF" \
  -DEXTRA_CONF_FILE="$M/harness_xpurt/backends/rvv.conf;$M/harness/backends/firesim_chipyard_dual_gemmini.conf;/scratch2/dima/misc_sw/ku040_dronedual_console.conf;/tmp/sf.conf" \
  -DEXTRA_DTC_OVERLAY_FILE=/scratch2/dima/misc_sw/ku040_dronedual_console.overlay > /tmp/ninja_stack3.log 2>&1
echo "BUILD_RC=$?"
grep -nE "error:|FAILED:|undefined reference" /tmp/ninja_stack3.log | head -12
E=examples/xpurt_demo/int8/build/ku040_dronet_stack3/zephyr/zephyr.elf
[ -f "$E" ] && echo "ELF_OK $(stat -c %s "$E")"
echo STACK3_BUILD_DONE
