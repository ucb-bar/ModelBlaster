pgrep -af "modelblaster.pipeline.kb_graph_opt" | grep "pilot/B " | cut -c1-120
for p in $(pgrep -f "modelblaster.pipeline.kb_graph_opt.*--work /tmp/krisd/family/pilot/B "); do kill -TERM $p; done
for i in $(seq 1 60); do pgrep -f "work /tmp/krisd/family/pilot/B " >/dev/null || break; sleep 5; done
pgrep -af "pilot/B" | cut -c1-100; cat /tmp/krisd/family/pilot/B/status; tail -3 /tmp/krisd/family/pilot/B/run.log | cut -c1-200
