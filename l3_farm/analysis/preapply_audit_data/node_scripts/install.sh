set -e
mkdir -p /tmp/krisd/pa/code && cd /tmp/krisd/pa/code
rm -rf ModelBlaster && tar xzf /tmp/krisd/pa/mb_pa.tar.gz 2>/dev/null && find ModelBlaster -name '._*' -delete
ln -sfn /tmp/krisd/pa/code/ModelBlaster /tmp/krisd/pa/code/modelblaster
sha256sum /tmp/krisd/pa/mb_pa.tar.gz
. /tmp/krisd/pa/env.sh
python -c "import pytest" 2>/dev/null || pip -q install pytest
python -c "import pytest,torch;print('pytest',pytest.__version__,'torch',torch.__version__)"
echo installed
