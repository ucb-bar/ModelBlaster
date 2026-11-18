# E0: status of this node's replicates (no secrets).
E=/tmp/krisd/e0
echo "host=$(hostname) tree=$(cat $E/tree.stamp 2>/dev/null)"
for r in $E/runs/*/; do n=$(basename $r); s=$(cat $r/status 2>/dev/null)
  [ "$s" = running -o "$s" = remeasuring ] && ! pgrep -f "e0_worker.sh" >/dev/null && s="$s:died"
  nr=$(python3 -c "import json;print(len(json.load(open('$r/work/state.json'))['rounds']))" 2>/dev/null)
  echo "run $n $s rounds=${nr:-0} t0=$(cat $r/t0 2>/dev/null) now=$(date +%s)"; done
echo "gpu_apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ';')"
L=/tmp/krisd/farm/llm_ledger.jsonl; [ -f $L ] && echo "ledger_lines=$(wc -l < $L)"
exit 0
