# Args: <runs dir> <name>. Status / progress of a smoke-test run (read-only).
D=$1/$2
echo "status=$(cat $D/status 2>/dev/null || echo none)"
pgrep -f "gkb/worker4.sh $2" >/dev/null && echo alive=1 || echo alive=0
tail -3 $D/run.log 2>/dev/null | cut -c1-300
tail -2 $D/extract.log 2>/dev/null | cut -c1-200
exit 0
