set -e
mkdir -p /tmp/krisd/cl/code && cd /tmp/krisd/cl/code
rm -rf ModelBlaster && tar xzf /tmp/krisd/cl/mb_cl.tar.gz 2>/dev/null && find ModelBlaster -name '._*' -delete
ln -sfn /tmp/krisd/cl/code/ModelBlaster /tmp/krisd/cl/code/modelblaster
cd /tmp/krisd/cl && rm -rf data && tar xzf data.tar.gz 2>/dev/null && find data -name '._*' -delete
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo installed
