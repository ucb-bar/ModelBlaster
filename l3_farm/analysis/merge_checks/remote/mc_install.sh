# merge checks: install a FROZEN tree (uploaded to /tmp/krisd/mc/<label>.tar.gz) under /tmp/krisd/mc/<label>.
# Args: <label> <sha256> <stamp>. Only /tmp/krisd/mc is touched.
set -eu
L=$1; SHA=$2; ST=$3; E=/tmp/krisd/mc; mkdir -p $E/runs; cd $E
echo "$SHA  $L.tar.gz" | sha256sum -c - >/dev/null
rm -rf $E/$L && mkdir -p $E/$L && tar xzf $L.tar.gz -C $E/$L && find $E/$L -name '._*' -delete
ln -sfn $E/$L/ModelBlaster $E/$L/modelblaster
echo "$ST $SHA" > $E/$L/tree.stamp
/tmp/krisd/venv/bin/python -c "import torch, einops; print('venv_ok', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "host=$(hostname) job=${SLURM_JOB_ID:-?}"
echo "installed $L $(cat $E/$L/tree.stamp)"
