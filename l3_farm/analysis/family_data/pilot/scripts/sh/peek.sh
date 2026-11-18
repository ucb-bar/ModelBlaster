# Args: <name>
WK=/tmp/krisd/family/pilot/$1; cat $WK/status; tail -c 1500 $WK/run.log; echo; ls $WK/r00 2>/dev/null; ls $WK/r00/problems 2>/dev/null
f=$(ls -d $WK/r00/run/*/ 2>/dev/null | head -1); [ -n "$f" ] && ls $f && tail -5 $WK/r00/run/run.log 2>/dev/null
tail -2 /tmp/krisd/farm/llm_ledger.jsonl 2>/dev/null | cut -c1-300
