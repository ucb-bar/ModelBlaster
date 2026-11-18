# merge checks: status of the runs on this node (no secrets). Args: [label]
E=/tmp/krisd/mc
echo "host=$(hostname) job=${SLURM_JOB_ID:-?}"
for r in $E/runs/*/*/*/; do [ -f $r/status ] || continue; n=${r#$E/runs/}; s=$(cat $r/status)
  [ "$s" = running -o "$s" = remeasuring ] && ! pgrep -f "mc_worker.sh ${r%/}" >/dev/null && s="$s:died"
  nr=$(python3 -c "import json;print(len(json.load(open('$r/work/state.json'))['rounds']))" 2>/dev/null)
  echo "run $n $s rounds=${nr:-0} t0=$(cat $r/t0 2>/dev/null) now=$(date +%s)"; done
echo "gpu_apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ';')"
echo "clocks=$(nvidia-smi --query-gpu=clocks.sm,clocks.mem,clocks.max.mem --format=csv,noheader)"
exit 0
