cd /tmp/krisd/a2
for d in ab5/S_gemm ab5/S_rec ab5/M_rec reg/run runs/view31/run; do
  [ -d $d ] || continue
  f=$(ls $d/*/progress.json $d/*/*/progress.json 2>/dev/null | head -1)
  echo "== $d $(date +%H:%M) $(ls $d/*/trace 2>/dev/null | wc -l)"
  grep -h "step .* done\|new best\|status\|baseline" $d/../$(basename $d).log 2>/dev/null | tail -3 | cut -c1-250
  tail -2 $d.log 2>/dev/null | cut -c1-250
done
tail -3 logs/ab5.log logs/reg_llm.log 2>/dev/null | cut -c1-300
