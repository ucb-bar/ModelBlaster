# held by a detached srun step (a step teardown kills setsid children of a non-detached step)
exec -a "kb_graph_opt_hw_guard /tmp/krisd/hw_on/ /tmp/krisd/hw_off/ /tmp/krisd/hw_pooled/ /tmp/krisd/hw_x/" sleep 30000
