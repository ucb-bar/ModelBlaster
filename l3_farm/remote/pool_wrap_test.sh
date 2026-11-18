# Pooled-mode check of the worker5 ssh wrapper. Args: <tmp dir> <host>
D=$1; mkdir -p $D/.sshbin; command -v ssh
printf '#!/bin/sh\nexec /usr/bin/ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=%s/.known_hosts "$@"\n' $D > $D/.sshbin/ssh
chmod +x $D/.sshbin/ssh; export PATH=$D/.sshbin:$PATH
ssh -o BatchMode=yes -o ConnectTimeout=15 $2 true </dev/null; echo "first rc=$?"
ssh -o BatchMode=yes -o ConnectTimeout=15 $2 nvidia-smi -L </dev/null; echo "second rc=$?"
