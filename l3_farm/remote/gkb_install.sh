# Args: <dir>. Unpacks <dir>/mb.tar.gz into <dir> (a private ModelBlaster tree, e.g. /tmp/krisd/gkb for the global
# knowledge smoke test) and links <dir>/modelblaster -> <dir>/ModelBlaster. Touches nothing else.
set -eu
G=$1; cd $G
rm -rf ModelBlaster && tar xzf mb.tar.gz && find ModelBlaster -name '._*' -delete
ln -sfn $G/ModelBlaster $G/modelblaster
echo installed $(date +%s)
