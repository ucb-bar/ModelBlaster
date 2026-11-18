# Pooled-mode diagnosis: run KernelBlaster's GPU inventory command over ssh and show what comes back. Args: <code root> <host>
. /tmp/krisd/venv/bin/activate
cd $1/ModelBlaster/third_party/KernelBlaster
CMD=$(PYTHONPATH=src python -c 'from kernelblaster.servers.utils import resources as r; print(r.inventory_command())')
echo "cmd: $CMD"
ssh -o BatchMode=yes -o ConnectTimeout=15 $2 "$CMD"; echo "rc=$?"
echo "--- local"; bash -c "$CMD" | head -5
