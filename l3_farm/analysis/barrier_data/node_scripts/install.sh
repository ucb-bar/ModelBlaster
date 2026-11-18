set -e
mkdir -p /tmp/krisd/bar/code && cd /tmp/krisd/bar/code
rm -rf ModelBlaster && tar xzf /tmp/krisd/bar/mb_bar.tar.gz 2>/dev/null && find ModelBlaster -name '._*' -delete
ln -sfn /tmp/krisd/bar/code/ModelBlaster /tmp/krisd/bar/code/modelblaster
sha256sum /tmp/krisd/bar/mb_bar.tar.gz
. /tmp/krisd/bar/env.sh
python -c "import pytest" 2>/dev/null || pip -q install pytest
python -c "import pytest,torch;print('pytest',pytest.__version__,'torch',torch.__version__)"
echo installed
