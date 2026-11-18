set -e
mkdir -p /tmp/krisd/cl/code && cd /tmp/krisd/cl/code
rm -rf ModelBlaster && tar xzf /tmp/krisd/cl/mb_cl.tar.gz 2>/dev/null && find ModelBlaster -name '._*' -delete
ln -sfn /tmp/krisd/cl/code/ModelBlaster /tmp/krisd/cl/code/modelblaster
cd /tmp/krisd/cl && rm -rf cdata && tar xzf cdata.tar.gz 2>/dev/null && find cdata -name '._*' -delete
echo installed
