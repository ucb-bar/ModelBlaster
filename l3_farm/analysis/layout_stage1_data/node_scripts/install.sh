set -e
mkdir -p /tmp/krisd/x142/code && cd /tmp/krisd/x142/code
rm -rf ModelBlaster && tar xzf /tmp/krisd/x142/mb_x142.tar.gz 2>/dev/null && find ModelBlaster -name '._*' -delete
ln -sfn /tmp/krisd/x142/code/ModelBlaster /tmp/krisd/x142/code/modelblaster
sha256sum /tmp/krisd/x142/mb_x142.tar.gz
. /tmp/krisd/x142/env.sh
python -c "import pytest" 2>/dev/null || pip -q install pytest
python -c "import pytest,torch;print('pytest',pytest.__version__,'torch',torch.__version__)"
echo installed
