# Args: <runs dir> <name>. Status / progress of an eff_smoke.py run (read-only; gkb_status.sh for worker5.sh).
D=$1/$2
echo "status=$(cat $D/status 2>/dev/null || echo none)"
pgrep -f "effsmoke/worker5.sh $2" >/dev/null && echo alive=1 || echo alive=0
grep -h "efficiency" $D/run.log 2>/dev/null | tail -2 | cut -c1-300
tail -2 $D/run.log 2>/dev/null | cut -c1-300
exit 0
