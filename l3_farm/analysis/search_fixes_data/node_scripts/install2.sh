set -e
mkdir -p /tmp/krisd/a3/code && cd /tmp/krisd/a3/code
rm -rf ModelBlaster && tar xzf /tmp/krisd/a3/mb_a3.tar.gz 2>/dev/null && find ModelBlaster -name '._*' -delete
ln -sfn /tmp/krisd/a3/code/ModelBlaster /tmp/krisd/a3/code/modelblaster
sha256sum /tmp/krisd/a3/mb_a3.tar.gz; echo installed
