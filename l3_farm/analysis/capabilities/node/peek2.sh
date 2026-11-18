cd /tmp/krisd/a2; for f in logs/ab5.log logs/reg_llm.log; do [ -f $f ] && { echo "== $f"; tail -5 $f | cut -c1-600; }; done
for d in ab5/S_gemm ab5/S_rec ab5/M_rec reg/run; do [ -d $d ] && { echo "== $d"; grep -h "step\|best\|accepted\|improved" $d/*.log $d.log 2>/dev/null | tail -3 | cut -c1-300; ls $d | head -20 | tr '\n' ' '; echo; }; done
