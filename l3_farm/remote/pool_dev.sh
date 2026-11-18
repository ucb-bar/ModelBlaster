# Pooled-mode diagnosis: KernelBlaster's remote GPU detection from this node. Args: <code root> <host>...
CR=$1; shift
. /tmp/krisd/venv/bin/activate
cd $CR/ModelBlaster/third_party/KernelBlaster
for h in "$@"; do
PYTHONPATH=src python - "$h" <<'PY'
import asyncio, sys, logging
logging.basicConfig(level=logging.INFO)
from kernelblaster.servers.utils import resources as r
h = sys.argv[1]
print("inventory:", r.gpu_inventory(h))
for spec in ("all", "auto", None):
    try:
        print(spec, asyncio.run(r.remote_devices(h, spec)))
    except Exception as e:
        print(spec, "ERR", repr(e)[:300])
PY
done
