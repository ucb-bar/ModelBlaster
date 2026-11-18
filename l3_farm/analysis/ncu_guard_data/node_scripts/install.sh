set -e
mkdir -p /tmp/krisd/ng/code && cd /tmp/krisd/ng/code
rm -rf ModelBlaster && tar xzf /tmp/krisd/ng/mb_ng.tar.gz 2>/dev/null && find ModelBlaster -name '._*' -delete
ln -sfn /tmp/krisd/ng/code/ModelBlaster /tmp/krisd/ng/code/modelblaster
sha256sum /tmp/krisd/ng/mb_ng.tar.gz
. /tmp/krisd/ng/env.sh
python -c "import pytest" 2>/dev/null || pip -q install pytest
python -c "import pytest,torch;print('pytest',pytest.__version__,'torch',torch.__version__)"
echo installed
