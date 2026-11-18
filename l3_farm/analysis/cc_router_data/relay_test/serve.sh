#!/bin/bash
# Local relay server for the claude-code: tests (loopback, token file, no output of the token).
# usage: serve.sh <workdir> <port> <max_conc> <cli_parallel> <budget_usd> <ledger_who> [cli_timeout_s]
W=$1; PORT=$2; MC=$3; CP=$4; BUD=$5; WHO=$6; TO=${7:-600}
S=/private/tmp/claude-501/-Users-krisd-Documents/bf39cca1-7e22-482f-8c91-bf7b1d5240f4/scratchpad
K=/Users/krisd/Documents/ModelBlaster-clone/ModelBlaster/third_party/KernelBlaster/src
mkdir -p $W; umask 077
[ -f $W/token ] || PYTHONPATH=$K $S/venv/bin/python -m kernelblaster.servers.llm_server --write-token $W/token >/dev/null
export KB_CLAUDE_CLI_PARALLEL=$CP KB_CLAUDE_CLI_TIMEOUT_S=$TO PYTHONPATH=$K
exec $S/venv/bin/python -m kernelblaster.servers.llm_server --host 127.0.0.1 --port $PORT --token-file $W/token \
  --max-concurrency $MC --allow-models 'claude-code:.*' --ledger $W/ledger.jsonl --ledger-who $WHO --budget-usd $BUD \
  --log-file $W/server.log >$W/server.out 2>&1 </dev/null
