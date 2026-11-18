echo "setup.status: $(cat /tmp/krisd/farm/setup.status 2>/dev/null)"; echo "tree: $(cat /tmp/krisd/farm/tree.stamp 2>/dev/null)"
pgrep -af "llm_server|relay|kb_graph_opt|setup.sh" | cut -c1-150 | head
cat /tmp/krisd/farm/relay.env 2>/dev/null
df -h /tmp | tail -1; nproc
/tmp/krisd/venv/bin/python -c "import torch;print(torch.__version__, torch.cuda.is_available())"
