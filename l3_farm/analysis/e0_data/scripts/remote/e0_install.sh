# E0: install the FROZEN tree (tarball uploaded to /tmp/krisd/e0/mb.tar.gz) under /tmp/krisd/e0. Args: <sha256> <stamp>
set -eu
E=/tmp/krisd/e0; mkdir -p $E/runs; cd $E
echo "$1  mb.tar.gz" | sha256sum -c - >/dev/null
rm -rf ModelBlaster && tar xzf mb.tar.gz && find ModelBlaster -name '._*' -delete
ln -sfn $E/ModelBlaster $E/modelblaster
echo "$2 $1" > $E/tree.stamp
/tmp/krisd/venv/bin/python -c "import torch, einops; print('venv_ok', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "host=$(hostname) gpus=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader | tr '\n' ';')"
echo "installed $(cat $E/tree.stamp)"
