ZCS=/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw
MB=$ZCS/modelblaster
W=$MB/examples/vint/_work
set +u
source $ZCS/scripts/activate_conda.sh  >/dev/null 2>&1
source $ZCS/scripts/set_envvars_sdk.sh >/dev/null 2>&1
set -u
export PYTHONPATH="${ZCS}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="/usr/bin:${PATH}"
