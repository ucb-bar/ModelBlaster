. /tmp/krisd/pa/env.sh
cd /tmp/krisd/pa/code/ModelBlaster
echo "== MB pipeline suite $(date)"
python -m pytest -q -p no:cacheprovider pipeline/tests 2>&1 | tail -40
echo "== MB other tests $(date)"
python -m pytest -q -p no:cacheprovider tests 2>&1 | tail -15
echo "== KB suite $(date)"
cd third_party/KernelBlaster && python -m pytest -q -p no:cacheprovider tests 2>&1 | tail -30
echo SUITEDONE $(date)
