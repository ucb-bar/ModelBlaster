# Read-only: lists every surviving knowledge.db under /tmp/krisd/farm/*/work/knowledge (except the problem
# named in $1, e.g. one still running) as "name status bytes". Nothing is written.
SKIP=$1
for d in /tmp/krisd/farm/*/; do n=$(basename $d); [ "$n" = "$SKIP" ] && continue
  f=$d/work/knowledge/knowledge.db; [ -f $f ] || continue
  echo "db $n $(cat $d/status 2>/dev/null || echo none) $(stat -c %s $f) wal=$(stat -c %s $f-wal 2>/dev/null || echo 0)"
done
exit 0
