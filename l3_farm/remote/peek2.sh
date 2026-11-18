# peek.sh for a given farm dir. Args: <farm dir>
F=${1:-/tmp/krisd/farm}
for d in $F/*/; do n=$(basename $d); [ -f $d/status ] || continue
  echo "== $n $(cat $d/status) | $(ls $d/work 2>/dev/null | tr '\n' ' ')"
  tail -2 $d/extract.log 2>/dev/null | cut -c1-200
  tail -3 $d/run.log 2>/dev/null | cut -c1-300
  ls -d $d/work/r0* 2>/dev/null | tr '\n' ' '; ls $d/gk_refresh $d/gk_refresh/done 2>/dev/null | tr '\n' ' '; echo
done
