#!/bin/bash
# Pack the sweep pieces a3_node.py needs for the given problems -> $1 (tar.gz); data/<NAME>/...
set -e
OUT=$1; shift
L3=~/Documents/ModelBlaster-clone/l3_farm
T=$(mktemp -d)
for P in "$@"; do
  D=$T/data/$P; mkdir -p $D/work
  cp $L3/problems/$P.py $D/problem.py
  cp $L3/results/$P/work/graph_hints.json $D/work/ 2>/dev/null || true
  for R in $L3/results/$P/work/r[0-9][0-9]; do
    r=$(basename $R)
    [ -d $R/winners ] && mkdir -p $D/work/$r && cp -r $R/winners $D/work/$r/
    for pf in $R/profile/*/profile.json; do [ -f $pf ] && { q=$(basename $(dirname $pf)); mkdir -p $D/work/$r/profile/$q; cp $pf $D/work/$r/profile/$q/; }; done
  done
  F=$L3/results/$P/work/final/$P/problem
  mkdir -p $D/work/final/$P/problem $D/work/final/$P/ir && cp $F/manifest.json $D/work/final/$P/problem/ && cp -r $F/slots $D/work/final/$P/problem/ && cp $L3/results/$P/work/final/$P/ir/graph.json $D/work/final/$P/ir/
done
COPYFILE_DISABLE=1 tar --no-xattrs -czf $OUT -C $T data
rm -rf $T
ls -la $OUT
