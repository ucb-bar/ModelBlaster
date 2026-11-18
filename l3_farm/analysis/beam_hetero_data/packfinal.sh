# Node side: stream the pieces needed to re-time an arm's final model elsewhere. Args: <tag>
cd /tmp/krisd/btest_$1/43_MinGPTCausalAttention && tar czf - work/cache work/graph_hints.json irs problem.py
