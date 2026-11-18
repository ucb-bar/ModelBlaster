# Restart the driver's LLM relay on the SAME port/token with a new concurrency (controllers keep their KB_LLM_SERVER_URL).
# Key arrives on stdin (farm.detach key=True). Args: <concurrency> <max_queue> <budget_usd> <port>
CONC=$1; QUEUE=$2; BUD=$3; PORT=$4
F=/tmp/krisd/farm; umask 077
. /tmp/krisd/venv/bin/activate
KB=/tmp/krisd/ModelBlaster/third_party/KernelBlaster
[ -d $KB ] || KB=$(ls -dt /tmp/krisd/*/ModelBlaster/third_party/KernelBlaster 2>/dev/null | head -1)   # an instance's own code root (fresh node: no /tmp/krisd/ModelBlaster)
read -r KEY
pkill -f "[l]lm_server --port $PORT"
for i in $(seq 1 30); do curl -s -o /dev/null http://127.0.0.1:$PORT/v1/health || break; sleep 0.5; done
cd $KB
export OPENAI_API_KEY="$KEY" OPENAI_BASE_URL=https://inference-api.nvidia.com/v1 OPENAI_MAX_TOKENS=32000 PYTHONPATH=$KB/src
unset KEY
exec python -m kernelblaster.servers.llm_server --port $PORT --token-file $F/token \
  --max-concurrency $CONC --max-queue $QUEUE --allow-models 'azure/.*' --ledger $F/llm_ledger.jsonl --budget-usd $BUD \
  --log-file $F/llm_server.log > $F/llm_server.out 2>&1 < /dev/null
