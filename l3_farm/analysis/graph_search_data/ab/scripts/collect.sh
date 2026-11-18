# Args: <arm>. Pack the arm's results (no KB problem dirs / traces) -> /tmp/krisd/a4/runs/<arm>.tgz; prints its size.
D=/tmp/krisd/a4/runs/$1; cd $D
L0=$(cat ledger_start 2>/dev/null || echo 0); L1=$(cat ledger_end 2>/dev/null || wc -l < /tmp/krisd/farm/llm_ledger.jsonl)
sed -n "$((L0+1)),${L1}p" /tmp/krisd/farm/llm_ledger.jsonl > ledger_slice.jsonl 2>/dev/null
F="status t_start t_end ledger_start ledger_end ledger_slice.jsonl run.log extract.log"
W="work/state.json work/summary.json work/graph_hints.json work/graph_decisions.jsonl work/model_runtime.json work/telemetry.jsonl work/graph_search_state.json work/cache"
R=$(ls -d work/r[0-9][0-9] 2>/dev/null)
X=""; for r in $R; do X="$X $(ls -d $r/round.json $r/plan.json $r/progress.json $r/char/headroom.json $r/run/*/result.json $r/variants/*/run/*/result.json $r/run.log 2>/dev/null)"; done
tar czf /tmp/krisd/a4/runs/$1.tgz $(ls -d $F $W irs 2>/dev/null) $X 2>/dev/null
ls -la /tmp/krisd/a4/runs/$1.tgz
