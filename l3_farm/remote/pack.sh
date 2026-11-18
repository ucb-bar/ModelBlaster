# Args: <name>. Streams the small, resumable/result files of a problem (state, summaries, logs, knowledge db,
# winning sources); build trees, caches and binaries (several GB) stay on the node.
cd /tmp/krisd/farm/$1 || exit 1
{ ls status job.env extract.log run.log worker.out 2>/dev/null
  find work -type f -size -3M \( -name '*.json' -o -name '*.jsonl' -o -name '*.log' -o -name '*.cu' -o -name '*.md' -o -name '*.db' -o -name '*.txt' \) \
    -not -path '*/cache/*' -not -path '*/run/work/*' -not -path '*/problems/*' 2>/dev/null
  # the winners cache (work/cache: small index + winner sources) is what makes a resumed run keep its earlier rounds' kernels
  find work/cache -type f -size -20M 2>/dev/null
} | tar czf - -T - 2>/dev/null
exit 0
