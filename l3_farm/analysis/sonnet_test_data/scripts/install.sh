# Node side. Args: <sha256> <code root>. Unpacks the frozen tarball uploaded to /tmp/krisd/sonnet_in/mb.tar.gz into
# <code root>, verifies sha, applies fix_follow.patch (model_router.py + one line in opt_multislot.py) in that copy only.
set -eu
SHA=$1; CR=$2; I=/tmp/krisd/sonnet_in
echo "$SHA  $I/mb.tar.gz" | sha256sum -c - >/dev/null
rm -rf $CR && mkdir -p $CR && cd $CR
tar xzf $I/mb.tar.gz && find ModelBlaster -name '._*' -delete
ln -sfn $CR/ModelBlaster $CR/modelblaster
if command -v patch >/dev/null; then patch -p1 < $I/fix_follow.patch; else echo "no patch(1)"; exit 3; fi
sha256sum $I/fix_follow.patch | cut -c1-16
grep -c "fix-follow" $CR/ModelBlaster/third_party/KernelBlaster/src/kernelblaster/agents/model_router.py
grep -c "writer=model" $CR/ModelBlaster/third_party/KernelBlaster/src/kernelblaster/agents/opt_multislot.py
/tmp/krisd/venv/bin/python -c "import torch, einops; print('venv_ok', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "host=$(hostname)"
