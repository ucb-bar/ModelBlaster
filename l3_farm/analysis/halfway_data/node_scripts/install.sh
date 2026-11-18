set -e
mkdir -p /tmp/krisd/hw/code && cd /tmp/krisd/hw/code
rm -rf ModelBlaster && tar xzf /tmp/krisd/hw/mb_pa.tar.gz 2>/dev/null && find ModelBlaster -name '._*' -delete
ln -sfn /tmp/krisd/hw/code/ModelBlaster /tmp/krisd/hw/code/modelblaster
sha256sum /tmp/krisd/hw/mb_pa.tar.gz
. /tmp/krisd/hw/env.sh
python -c "import pytest" 2>/dev/null || pip -q install pytest
python -c "import pytest,torch;print('pytest',pytest.__version__,'torch',torch.__version__)"
echo installed
