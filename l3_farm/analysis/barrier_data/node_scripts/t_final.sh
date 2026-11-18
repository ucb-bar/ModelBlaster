. /tmp/krisd/bar/env.sh
cd /tmp/krisd/bar/code
export MB_TEST_TMP=/tmp/krisd/bar/tmp; mkdir -p $MB_TEST_TMP
timeout 2400 python -m pytest ModelBlaster/pipeline/tests/test_kb_barrier.py ModelBlaster/pipeline/tests/test_kb_halfway.py ModelBlaster/pipeline/tests/test_kb_global.py ModelBlaster/pipeline/tests/test_graph_search.py ModelBlaster/pipeline/tests/test_graph_search_flow.py ModelBlaster/pipeline/tests/test_kb_resume.py -q -p no:cacheprovider -rfE 2>&1 | tail -15
