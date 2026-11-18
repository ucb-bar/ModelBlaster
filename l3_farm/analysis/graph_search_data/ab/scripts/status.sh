# Args: <arm>. Status, last log lines, rounds so far, ledger lines since start.
D=/tmp/krisd/a4/runs/$1
echo "status $(cat $D/status 2>/dev/null)"; echo "start $(cat $D/t_start 2>/dev/null) now $(date +%s)"
echo "ledger $(cat $D/ledger_start 2>/dev/null) -> $(wc -l < /tmp/krisd/farm/llm_ledger.jsonl)"
tail -n ${2:-15} $D/run.log 2>/dev/null | cut -c1-400
ls $D/work 2>/dev/null | tr '\n' ' '; echo
tail -n 5 $D/work/graph_decisions.jsonl 2>/dev/null | cut -c1-300
