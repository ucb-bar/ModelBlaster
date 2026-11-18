#!/bin/bash
# usage: client.sh <port> <workdir-with-token> <drive.py args...>   (token path only; the token is never printed)
PORT=$1; W=$2; shift 2
S=/private/tmp/claude-501/-Users-krisd-Documents/bf39cca1-7e22-482f-8c91-bf7b1d5240f4/scratchpad
export PYTHONPATH=/Users/krisd/Documents/ModelBlaster-clone/ModelBlaster/third_party/KernelBlaster/src
export KB_LLM_SERVER_URL=http://127.0.0.1:$PORT KB_LLM_SERVER_TOKEN_FILE=$W/token
exec $S/venv/bin/python -I -X utf8 -c "import sys,runpy; sys.path.insert(0,'$PYTHONPATH'); sys.argv=['drive.py']+sys.argv[1:]; runpy.run_path('/Users/krisd/Documents/ModelBlaster-clone/l3_farm/analysis/cc_router_data/relay_test/drive.py', run_name='__main__')" "$@"
