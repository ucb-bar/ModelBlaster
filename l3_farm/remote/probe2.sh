# probe.sh for the scheduling features / a second farm instance (analysis/scheduling.md). Args: [farm dir, default
# /tmp/krisd/farm]. Prints key=value facts about this node, "job <name> <status>" per problem dir and
# "rounds <name> <n>" (completed rounds in its work/state.json) for running ones, and "halfway <name> <round> <hb age s>"
# for a run parked at its halfway point (knowledge sync, analysis/halfway_sync.md). No secrets.
F=${1:-/tmp/krisd/farm}; S=/tmp/krisd/farm          # S: the node's setup / relay (shared with the production farm)
echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
ST=$(cat $S/setup.status 2>/dev/null || echo none)
[ "$ST" = running ] && ! pgrep -f "farm/setup.sh" >/dev/null && ST=failed
echo "setup=$ST"
echo "tree=$(cat $F/tree.stamp 2>/dev/null || echo none)"
if [ -f $S/relay.env ]; then
  . $S/relay.env
  if curl -sf -m 5 -H "Authorization: Bearer $(cat $KB_LLM_SERVER_TOKEN_FILE)" $KB_LLM_SERVER_URL/v1/health >/dev/null; then echo relay=ok; else echo relay=down; fi
else echo relay=none; fi
echo "foreign=$(pgrep -af kb_graph_opt | grep -v "$F/" | grep -vc 'pgrep')"
for d in $F/*/; do n=$(basename $d); [ -f $d/status ] || continue; s=$(cat $d/status)
  [ "$s" = running ] && ! pgrep -f "worker[0-9]*.sh $n" >/dev/null && s=failed:died
  echo "job $n $s"
  if [ "$s" = running ] && [ -f $d/work/state.json ]; then
    echo "rounds $n $(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1])).get("rounds", [])))' $d/work/state.json 2>/dev/null || echo 0)"
  fi
  # halfway sync: the run parked at its halfway point (round dir's halfway.json stage "paused") waits for the controller
  if [ "$s" = running ] && ls $d/work/r[0-9][0-9]/halfway.json >/dev/null 2>&1; then
    python3 - $d $n <<"PY" 2>/dev/null
import glob, json, os, sys, time
d, n = sys.argv[1], sys.argv[2]
for f in sorted(glob.glob(d + "/work/r[0-9][0-9]/halfway.json"))[-1:]:
    try:
        h = json.load(open(f))
    except (OSError, ValueError):
        continue
    if h.get("stage") == "paused":
        try:
            age = int(time.time() - os.path.getmtime(os.path.join(os.path.dirname(f), "halfway.hb")))
        except OSError:
            age = -1
        print("halfway %s %s %d" % (n, h.get("round", 0), age))      # <name> <round> <heartbeat age s>
PY
  fi
  # characterization sync (X-151): waiting after A4's characterization pass (round dir's char_sync.json stage "paused")
  if [ "$s" = running ] && ls $d/work/r[0-9][0-9]/char_sync.json >/dev/null 2>&1; then
    python3 - $d $n <<"PY" 2>/dev/null
import glob, json, os, sys, time
d, n = sys.argv[1], sys.argv[2]
for f in sorted(glob.glob(d + "/work/r[0-9][0-9]/char_sync.json"))[-1:]:
    try:
        h = json.load(open(f))
    except (OSError, ValueError):
        continue
    if h.get("stage") == "paused":
        try:
            age = int(time.time() - os.path.getmtime(os.path.join(os.path.dirname(f), "char_sync.hb")))
        except OSError:
            age = -1
        print("charsync %s %s %d" % (n, h.get("round", 0), age))      # <name> <round> <heartbeat age s>
PY
  fi
done
exit 0
