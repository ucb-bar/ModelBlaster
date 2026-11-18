# args: ARM STAMP   (tarball at /tmp/krisd/mt_ARM/mb.tar.gz, b41.tgz, force_tech_order.patch, mt_prep.py, mkprob.py already put)
set -e
ARM=$1; STAMP=$2; R=/tmp/krisd/mt_$ARM; cd $R
mkdir -p code/$STAMP && cd code/$STAMP && rm -rf ModelBlaster && tar xzf $R/mb.tar.gz 2>/dev/null && find ModelBlaster -name '._*' -delete
ln -sfn $R/code/$STAMP/ModelBlaster $R/code/$STAMP/modelblaster; ln -sfn $R/code/$STAMP $R/code/cur
if [ "$ARM" = S8_first ]; then patch -p1 < $R/force_tech_order.patch; fi
cd $R
cat > env.sh <<EOF
export PATH=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04/bin:\$PATH
export CUDA_HOME=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04
export MB_CUPTI_LIB=\$CUDA_HOME/extras/CUPTI/lib64/libcupti.so.12
. /tmp/krisd/venv/bin/activate
export PYTHONPATH=$R/code/cur
EOF
. $R/env.sh
rm -rf b41 && tar xzf b41.tgz 2>/dev/null; mv b41 b41dir 2>/dev/null || true
python mt_prep.py $R/b41dir $R/prune
export MB_RECURRENT_CLASS=1
rm -rf prob; python mkprob.py rev $R/b41dir $R/prune/ir_final $R/prob
sha256sum mb.tar.gz | cut -c1-16; echo prepared
