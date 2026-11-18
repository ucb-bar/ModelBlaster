import sys, os
sys.path.insert(0, os.path.expanduser("~/Documents/ModelBlaster-clone/l3_farm"))
import farm
Q = 'gpu.product_name="*L40S*"'
cmds = [
 f"crun -q '{Q}' --no-shell --gpus 1 -t 10:00:00 --dry-run 2>&1 | head -40",
 f"crun -q '{Q}' --no-shell --gpus 1 -t 24:00:00 --dry-run 2>&1 | head -40",
 f"crun --help 2>&1 | head -120",
 f"cdb find '{Q}' -s idle,mixed -c hostname,slurm.state,slurm.partitions 2>&1 | head -15",
 "cat /computelab-cluster/tools/bin/crun | head -c 600; ls /computelab-cluster/tools/bin | head -50",
]
for c in cmds:
    rc, out, err = farm.ssh(c, b"", 120)
    print("=== $", c, "rc", rc); print(out.decode()[:7000]); print(err.decode()[:500])
