. /tmp/krisd/hw/env.sh
cd /tmp/krisd/hw/code
export MB_TEST_TMP=/tmp/krisd/hw/tmp2; mkdir -p $MB_TEST_TMP
python /tmp/krisd/hw/flow_variance.py 2>&1 | grep -E "^(off|on)[AB] "
