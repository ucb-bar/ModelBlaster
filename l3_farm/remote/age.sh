now=$(date +%s)
for d in /tmp/krisd/farm/*/; do n=$(basename $d); [ "$(cat $d/status 2>/dev/null)" = running ] || continue
  f=$(find $d/work -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1); a=$(( now - ${f%.*} ))
  echo "$n idle_s=$a rounds=$(ls -d $d/work/r[0-9][0-9] 2>/dev/null | wc -l) done=$(python3 -c "import json;print(len(json.load(open('$d/work/state.json'))['rounds']))" 2>/dev/null)"
done
