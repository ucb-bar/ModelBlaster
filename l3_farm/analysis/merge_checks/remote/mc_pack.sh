# merge checks: stream one run's result files (tar.gz on stdout). Args: <run dir>. Keeps the final cache and hints (needed to re-measure
# the final model later) and the per-round work files; drops traces, binaries, objects and files > 5 MB.
cd $1 || exit 1
L=/tmp/mc_pack_$$.list
{ ls status flow_status t0 t1 t2 t_extract cmdline.txt run.env extract.log run.log remeasure.json remeasure.log smi_start.txt smi_end.txt smi_samples.txt ledger.jsonl plugin.py 2>/dev/null
  find irs -maxdepth 3 -type f -name 'graph.json' 2>/dev/null
  find work remeasure -type f -size -5M -not -path '*/trace/*' -not -path '*/run/work/*' \
    \( -name '*.json' -o -name '*.jsonl' -o -name '*.log' -o -name '*.cu' -o -name '*.cuh' -o -name '*.md' -o -name '*.txt' -o -name '*.csv' -o -name '*.cpp' -o -name '*.h' \) 2>/dev/null
} > $L
tar czf - -T $L 2>/dev/null
rm -f $L
exit 0
