. /tmp/krisd/bar/env.sh
cd /tmp/krisd/bar/code
mkdir -p /tmp/krisd/bar/out; export MB_TEST_TMP=/tmp/krisd/bar/tmp; mkdir -p $MB_TEST_TMP
timeout 3000 python -m pytest ModelBlaster/pipeline/tests/test_kb_barrier.py -q -s -p no:cacheprovider -rfEs 2>&1 | tail -40
