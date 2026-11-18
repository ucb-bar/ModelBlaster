set -e
mkdir -p /tmp/krisd/cpool/code && cd /tmp/krisd/cpool/code
rm -rf ModelBlaster && tar xzf /tmp/krisd/cpool/mb_cpool.tar.gz 2>/dev/null && find ModelBlaster -name '._*' -delete
ln -sfn /tmp/krisd/cpool/code/ModelBlaster /tmp/krisd/cpool/code/modelblaster
sha256sum /tmp/krisd/cpool/mb_cpool.tar.gz
. /tmp/krisd/cpool/env.sh
python -c "import pytest" 2>/dev/null || pip -q install pytest
python -c "import pytest,torch;print('pytest',pytest.__version__,'torch',torch.__version__)"
echo installed
