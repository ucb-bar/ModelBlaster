for d in /tmp/krisd/farm/*/; do n=$(basename $d); [ -f $d/status ] || continue
  echo "== $n $(cat $d/status) | $(ls $d/work 2>/dev/null | tr '\n' ' ')"
  tail -2 $d/extract.log 2>/dev/null | cut -c1-200
  tail -2 $d/run.log 2>/dev/null | cut -c1-220
  ls -d $d/work/r0* 2>/dev/null | tr '\n' ' '; echo
done
