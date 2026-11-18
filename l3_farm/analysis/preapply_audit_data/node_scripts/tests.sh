. /tmp/krisd/pa/env.sh
cd /tmp/krisd/pa/code/ModelBlaster
python -m modelblaster.pipeline.kb_graph_opt --help 2>&1 | grep -E "graph-preapply" | head
python -m pytest -q -p no:cacheprovider pipeline/tests/test_preapply_audit.py pipeline/tests/test_graph_search.py pipeline/tests/test_graph_search_flow.py pipeline/tests/test_graph_priors_standard.py pipeline/tests/test_kb_efficiency.py pipeline/tests/test_graph_hints.py 2>&1 | tail -15
