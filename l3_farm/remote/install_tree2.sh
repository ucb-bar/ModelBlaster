# Args: <stamp> <farm dir> <code root>. Unpacks <farm dir>/mb.tar.gz into <code root> (a second farm instance's own
# tree; the production tree /tmp/krisd/ModelBlaster is not touched).
set -eu
STAMP=$1; F=$2; CR=$3
mkdir -p $CR $F; cd $CR
rm -rf $CR/ModelBlaster.new && mkdir $CR/ModelBlaster.new
tar xzf $F/mb.tar.gz -C $CR/ModelBlaster.new && find $CR/ModelBlaster.new -name '._*' -delete
rm -rf $CR/ModelBlaster && mv $CR/ModelBlaster.new/ModelBlaster $CR/ModelBlaster && rm -rf $CR/ModelBlaster.new
ln -sfn $CR/ModelBlaster $CR/modelblaster
echo "$STAMP" > $F/tree.stamp
