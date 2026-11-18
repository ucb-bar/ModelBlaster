import sys, os
sys.path.insert(0, os.path.expanduser("~/Documents/ModelBlaster-clone/l3_farm"))
import farm
Q = 'gpu.product_name="*L40S*"'
P = "l40s@cr+mp/romed8nl/1gpu-32cpu-128gb"
cmds = [
 f"crun -q '{Q}' --no-shell --gpus 1 -t 10:00:00 --dry-run --debug 2>&1 | grep -v -E 'Node options|^$' | cut -c1-900 | head -60",
 f"scontrol show job 5321587 2>&1 | head -40",
]
for t in ["10:00:00","1-00:00:00","1-12:00:00","3-00:00:00"]:
    cmds.append(f"sbatch --test-only -J mb-batchproto -p '{P}' --gpus=1 -A cuopt_fallback -t {t} --wrap='sleep 1' 2>&1 | tail -5")
cmds.append(f"sbatch --test-only -J mb-batchproto -p '{P}' --gpus=1 -A cuopt_fallback --qos=batch -t 1-00:00:00 --wrap='sleep 1' 2>&1 | tail -5")
cmds.append(f"sbatch --test-only -J mb-batchproto -p '{P}' --gpus=1 -A cuopt_fallback --qos=preemptable -t 3-00:00:00 --wrap='sleep 1' 2>&1 | tail -5")
cmds.append(f"sbatch --test-only -J mb-batchproto -p '{P}' --gpus=1 -A cuopt_preemptable -t 3-00:00:00 --wrap='sleep 1' 2>&1 | tail -5")
for c in cmds:
    rc, out, err = farm.ssh(c, b"", 180)
    print("=== $", c, "rc", rc); print(out.decode()[:6000]); print(err.decode()[:500])
