# Args: <old stamp> <new stamp>. New tree = old frozen tree + the uploaded graph_search.py (absorb fix only).
set -eu; A=/tmp/krisd/a4; rm -rf $A/$2; cp -a $A/$1 $A/$2; ln -sfn $A/$2/ModelBlaster $A/$2/modelblaster
cp $A/graph_search.py $A/$2/ModelBlaster/pipeline/graph_search.py; sha256sum $A/$2/ModelBlaster/pipeline/graph_search.py
