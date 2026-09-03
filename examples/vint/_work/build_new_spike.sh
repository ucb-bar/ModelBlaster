#!/usr/bin/env bash
source /scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw/modelblaster/examples/vint/_work/vint_env.sh
cd $MB
export MODEL_NAME=vint REPO_ROOT=$MB QUANT=int8 TARGET=rvv BACKEND=reference OPTIMIZE=0
export RUNNER=spike STOP_AFTER=build
export GLOBAL_CURATED_DIR=$W/curated_new
export MODELBLASTER_CURATED_VERIFY=1
bash examples/_run_lib.sh
