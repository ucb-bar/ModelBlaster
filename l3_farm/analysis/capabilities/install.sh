# usage (on node via a2run.py): STAMP=... bash install.sh ; tarball pre-put at /tmp/krisd/a2/mb_$STAMP.tar.gz
set -e
mkdir -p /tmp/krisd/a2/code/$STAMP && cd /tmp/krisd/a2/code/$STAMP
rm -rf ModelBlaster && tar xzf /tmp/krisd/a2/mb_$STAMP.tar.gz 2>/dev/null && find ModelBlaster -name '._*' -delete
ln -sfn /tmp/krisd/a2/code/$STAMP/ModelBlaster /tmp/krisd/a2/code/$STAMP/modelblaster
ln -sfn /tmp/krisd/a2/code/$STAMP /tmp/krisd/a2/code/cur
ln -sfn /tmp/krisd/a2/code/$STAMP /tmp/krisd/a2/code/cur
. /tmp/krisd/a2/env.sh; python -c "import torch, einops; print(torch.__version__)"; which nvcc
echo installed $STAMP
