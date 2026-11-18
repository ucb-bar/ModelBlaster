# Long-lived step (see farm.detach): key arrives on stdin (first line, read -r KEY); the relay runs in the foreground
# of this step so it lives as long as the step does. Args: <concurrency> <budget_usd> [max_queue]   (max_queue omitted = server default, as before)
CONC=$1; BUD=$2; QUEUE=$3
F=/tmp/krisd/farm; mkdir -p $F; umask 077
. /tmp/krisd/venv/bin/activate
KB=/tmp/krisd/ModelBlaster/third_party/KernelBlaster
[ -d $KB ] || KB=$(ls -dt /tmp/krisd/*/ModelBlaster/third_party/KernelBlaster 2>/dev/null | head -1)   # an instance's own code root (fresh node: no /tmp/krisd/ModelBlaster)
PORT=$((20000 + RANDOM % 30000))
PYTHONPATH=$KB/src python -m kernelblaster.servers.llm_server --write-token $F/token >/dev/null
read -r KEY
rm -f $F/relay.env
( for i in $(seq 1 30); do curl -sf -H "Authorization: Bearer $(cat $F/token)" http://127.0.0.1:$PORT/v1/health >/dev/null && break; sleep 2; done
  printf 'export KB_LLM_SERVER_URL=http://127.0.0.1:%s\nexport KB_LLM_SERVER_TOKEN_FILE=%s/token\n' $PORT $F > $F/relay.env ) &
cd $KB
export OPENAI_API_KEY="$KEY" OPENAI_BASE_URL=https://inference-api.nvidia.com/v1 OPENAI_MAX_TOKENS=32000 PYTHONPATH=$KB/src
unset KEY
exec python -m kernelblaster.servers.llm_server --port $PORT --token-file $F/token \
  --max-concurrency $CONC ${QUEUE:+--max-queue $QUEUE} --allow-models 'azure/.*' --ledger $F/llm_ledger.jsonl --budget-usd $BUD \
  --log-file $F/llm_server.log > $F/llm_server.out 2>&1 < /dev/null
