. /tmp/krisd/hw/env.sh
cd /tmp/krisd/hw/code
mkdir -p /tmp/krisd/hw/out
export MB_TEST_TMP=/tmp/krisd/hw/tmp
echo "== GPU flow: halfway (rounds + interleaved)"; date
timeout 3000 python -m pytest ModelBlaster/pipeline/tests/test_kb_halfway_flow.py -q -s -p no:cacheprovider > /tmp/krisd/hw/out/flow_halfway.txt 2>&1; tail -5 /tmp/krisd/hw/out/flow_halfway.txt
echo "== A4 / flow tests (halfway off): graph_search_flow, kb_graph_opt_flow"; date
timeout 3000 python -m pytest ModelBlaster/pipeline/tests/test_graph_search_flow.py ModelBlaster/pipeline/tests/test_kb_graph_opt_flow.py -q -p no:cacheprovider > /tmp/krisd/hw/out/flow_a4.txt 2>&1; tail -3 /tmp/krisd/hw/out/flow_a4.txt
echo "== MB pipeline suite"; date
timeout 3000 python -m pytest ModelBlaster/pipeline/tests -q -p no:cacheprovider -rfE > /tmp/krisd/hw/out/mb_suite.txt 2>&1; tail -25 /tmp/krisd/hw/out/mb_suite.txt
echo "== KB suite"; date
cd ModelBlaster/third_party/KernelBlaster
PYTHONPATH=src:tests timeout 3000 python -m pytest tests -q -p no:cacheprovider -rfE > /tmp/krisd/hw/out/kb_suite.txt 2>&1; tail -15 /tmp/krisd/hw/out/kb_suite.txt
date; echo SUITE-DONE
