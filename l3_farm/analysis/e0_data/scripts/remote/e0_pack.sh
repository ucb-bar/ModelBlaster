# E0: stream one replicate's small result files (+ trace blobs when < 200 MB) as tar.gz on stdout. Args: <run name>
cd /tmp/krisd/e0/runs/$1 || exit 1
L=/tmp/e0_pack_$1.list
{ ls status flow_status t0 t_extract t1 t2 extract.log run.log remeasure.json remeasure.log posthoc.json posthoc.log posthoc2.json posthoc2.log gpu_apps_end.txt ledger.jsonl 2>/dev/null
  find work remeasure -type f -size -5M \( -name '*.json' -o -name '*.jsonl' -o -name '*.log' -o -name '*.cu' -o -name '*.md' -o -name '*.txt' \) \
    -not -path '*/cache/*' -not -path '*/run/work/*' -not -path '*/problems/*' -not -path '*/trace/blobs/*' 2>/dev/null
  B=$(find work -type d -path '*/trace/blobs' 2>/dev/null)
  if [ -n "$B" ] && [ $(du -scm $B | tail -1 | cut -f1) -lt 200 ]; then find $B -type f; fi
} > $L
tar czf - -T $L 2>/dev/null
rm -f $L
exit 0
