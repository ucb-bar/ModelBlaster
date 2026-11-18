# Pooled mode (analysis/scheduling.md): can this node (a problem controller) reach the pool hosts over ssh and
# run on their GPU? Args: <host>... Prints one line per host. No secrets.
for h in "$@"; do
  out=$(ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "$h" \
        'echo gpu=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader | head -1); command -v flock >/dev/null && echo flock=yes; command -v rsync >/dev/null && echo rsync=yes; mkdir -p /tmp/krisd/pool && echo dir=ok' 2>&1 | tr '\n' ' ')
  echo "host $h rc=$? $out"
done
echo "self=$(hostname) cpus=$(nproc)"
exit 0
