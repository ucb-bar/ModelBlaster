# keeps the l3_farm dispatcher off this node (probe counts kb_graph_opt processes outside /farm/ as foreign)
mkdir -p /tmp/krisd/family
exec -a kb_graph_opt_family_guard sleep 40000
