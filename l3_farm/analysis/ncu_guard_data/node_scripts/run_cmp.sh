. /tmp/krisd/ng/env.sh
cd /tmp/krisd/ng; mkdir -p out
python scripts/ncu_cmp.py /tmp/krisd/ng/out/ncu_cmp.json 3
echo exit $?
