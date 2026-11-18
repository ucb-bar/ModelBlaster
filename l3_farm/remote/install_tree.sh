# Args: <stamp>. Unpacks /tmp/krisd/farm/mb.tar.gz into /tmp/krisd.
set -eu
F=/tmp/krisd/farm; cd /tmp/krisd
tar xzf $F/mb.tar.gz && find ModelBlaster -name '._*' -delete
ln -sfn /tmp/krisd/ModelBlaster /tmp/krisd/modelblaster
echo "$1" > $F/tree.stamp
