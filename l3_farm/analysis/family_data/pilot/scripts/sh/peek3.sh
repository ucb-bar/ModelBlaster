WK=/tmp/krisd/family/pilot/$1
cat $WK/status; grep -v "^\s*$" $WK/r00/run.log | tail -${2:-25} | cut -c1-400
