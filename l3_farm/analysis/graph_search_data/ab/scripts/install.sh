# Args: <stamp> <sha256>. Unpack the frozen tree to /tmp/krisd/a4/<stamp> (ModelBlaster/ + modelblaster symlink).
set -eu
A=/tmp/krisd/a4; cd $A
echo "$2  $1.tar.gz" | sha256sum -c -
mkdir -p $1 && tar xzf $1.tar.gz -C $1 && find $1/ModelBlaster -name '._*' -delete
ln -sfn $A/$1/ModelBlaster $A/$1/modelblaster
echo "$1 $2" > $1/STAMP; ls $1; du -sh $1
