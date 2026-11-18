cd /tmp/krisd/a2
for d in ab5/S_gemm ab5/S_rec ab5/M_rec reg/run runs/view31/run; do
  [ -d $d ] || continue
  echo "== $d $(date +%H:%M)"
  grep -h "accepted\|improv\|regress\|failed\|step" $d.log 2>/dev/null | grep -v "efficiency" | tail -4 | cut -c1-260
done
grep "===" logs/ab5.log logs/reg_llm.log 2>/dev/null
