. /tmp/krisd/venv/bin/activate; CUDA_N=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04; export PATH=$CUDA_N/bin:$PATH CUDA_HOME=$CUDA_N; cd /tmp/krisd/mc/mb3/ModelBlaster; export PYTHONPATH=/tmp/krisd/mc/mb3:/tmp/krisd/mc/ip/pylib
for grp in "test_graph_prune.py" "test_cuda_learnings.py" "test_kb_efficiency.py" "test_experiments_flag.py"; do
 echo "== $grp"; python -m pytest pipeline/tests/$grp pipeline/tests/test_kb_trace.py -q -p no:cacheprovider -k "not test_graph_search_flow" 2>&1 | grep -E "^FAILED|passed|failed"; done; echo DONE
