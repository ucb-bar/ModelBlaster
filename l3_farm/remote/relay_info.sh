# Read-only: prints the running llm_server relay's settings from its command line, e.g. "conc=96 queue=1024 budget=10000 port=23456".
# Fields absent from the command line are omitted; prints nothing when no relay runs. No secrets (the key is in the environment, not argv).
L=$(pgrep -af "[l]lm_server --port" | head -1)
[ -n "$L" ] || exit 0
o=""
for k in "conc:--max-concurrency" "queue:--max-queue" "budget:--budget-usd" "port:--port"; do
  v=$(echo "$L" | sed -n "s/.*${k#*:} \([0-9.]*\).*/\1/p")
  [ -n "$v" ] && o="$o ${k%%:*}=$v"
done
echo "$o"
exit 0
