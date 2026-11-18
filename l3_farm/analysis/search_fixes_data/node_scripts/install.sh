set -e
mkdir -p /tmp/krisd/a3/code && cd /tmp/krisd/a3/code
rm -rf ModelBlaster && tar xzf /tmp/krisd/a3/mb_a3.tar.gz 2>/dev/null && find ModelBlaster -name '._*' -delete
ln -sfn /tmp/krisd/a3/code/ModelBlaster /tmp/krisd/a3/code/modelblaster
cd /tmp/krisd/a3 && rm -rf data && tar xzf data.tar.gz 2>/dev/null && find data -name '._*' -delete
sha256sum /tmp/krisd/a3/mb_a3.tar.gz
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
echo installed
