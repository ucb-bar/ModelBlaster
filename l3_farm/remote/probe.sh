# Prints key=value facts about this node. No secrets.
F=/tmp/krisd/farm
echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
S=$(cat $F/setup.status 2>/dev/null || echo none)
[ "$S" = running ] && ! pgrep -f "farm/setup.sh" >/dev/null && S=failed
echo "setup=$S"
echo "tree=$(cat $F/tree.stamp 2>/dev/null || echo none)"
if [ -f $F/relay.env ]; then
  . $F/relay.env
  if curl -sf -m 5 -H "Authorization: Bearer $(cat $KB_LLM_SERVER_TOKEN_FILE)" $KB_LLM_SERVER_URL/v1/health >/dev/null; then echo relay=ok; else echo relay=down; fi
else echo relay=none; fi
echo "foreign=$(pgrep -af kb_graph_opt | grep -vc '/farm/' )"
for d in $F/*/; do n=$(basename $d); [ -f $d/status ] || continue; s=$(cat $d/status)
  [ "$s" = running ] && ! pgrep -f "farm/worker[0-9]*.sh $n" >/dev/null && s=failed:died
  echo "job $n $s"; done
exit 0
