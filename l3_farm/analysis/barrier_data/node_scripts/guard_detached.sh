# held by a detached srun step: keeps the production dispatcher (which claims any idle L40S job of mine) from starting a real
# problem here: a process with "kb_graph_opt" in its command line outside /tmp/krisd/farm/ is "foreign" for it. The test
# instances' own dirs are listed so their probes (grep -v "$F/") do not count it.
exec -a "kb_graph_opt_bar_guard /tmp/krisd/bar_ind/ /tmp/krisd/bar_pool/ /tmp/krisd/bar_x/" sleep 36000
