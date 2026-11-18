# Pooled-mode diagnosis of host-key checking (no secrets: prints only ssh's verdict lines). Args: <host>
h=$1
echo "HOME=$HOME user=$(id -un) host=$(hostname)"
ls -la ~/.ssh/known_hosts 2>&1 | cut -c1-120
ssh-keygen -F $h 2>&1 | grep -c "$h" | sed 's/^/known entries for host: /'
ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new $h true; echo "accept-new rc=$?"
ssh -o BatchMode=yes -o ConnectTimeout=10 $h true; echo "batch rc=$?"
ssh -v -o BatchMode=yes -o ConnectTimeout=10 $h true 2>&1 | grep -iE "known_hosts|host key|verification|Server host key" | cut -c1-160 | head -12
