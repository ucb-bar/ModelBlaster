mkdir -p /tmp/krisd/cpool/mux /tmp/krisd/cpool/sshbin
printf '#!/bin/sh\nexec /usr/bin/ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/tmp/krisd/cpool/.known_hosts "$@"\n' > /tmp/krisd/cpool/sshbin/ssh
chmod +x /tmp/krisd/cpool/sshbin/ssh
