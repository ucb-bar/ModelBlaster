#!/bin/bash
# usage: scale_series.sh <port> <workdir> <model> <tag> <offset0> <conc:n> ...   (distinct prompts per run: offset advances)
PORT=$1; W=$2; MODEL=$3; TAG=$4; OFF=$5; shift 5
D=$(dirname "$0"); S=/private/tmp/claude-501/-Users-krisd-Documents/bf39cca1-7e22-482f-8c91-bf7b1d5240f4/scratchpad
for cn in "$@"; do
  C=${cn%%:*}; N=${cn##*:}
  RT_OUT=$S/scale_${TAG}_$C.json KB_LLM_SERVER_RETRIES=2 $D/client.sh $PORT $W scale $C $N $MODEL $OFF 2>&1 | tail -1
  OFF=$((OFF+N))
done
