for d in /tmp/krisd/farm/*/; do n=$(basename $d); [ "$(cat $d/status 2>/dev/null)" = running ] || continue
  p=$d/work/r00/plan.json; [ -f $p ] || { echo "$n no-plan-yet"; continue; }
  echo "$n items=$(python3 -c "import json;print(len(json.load(open('$p'))['items']))") done=$(find $d/work/r00/run -name result.json 2>/dev/null | wc -l) rounds=$(ls -d $d/work/r0* | wc -l)"
done
