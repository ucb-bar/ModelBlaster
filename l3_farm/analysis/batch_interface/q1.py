import sys, os
sys.path.insert(0, os.path.expanduser("~/Documents/ModelBlaster-clone/l3_farm"))
import farm
cmds = [
 "hostname; date; squeue --me -h -o '%i|%j|%T|%N|%L|%P|%q|%a' | head -40",
 "sacctmgr -n -P show qos format=Name,Priority,MaxWall,Preempt,PreemptMode,Flags,MaxTRESPU,MaxJobsPU,MaxSubmitPU,GraceTime 2>&1",
 "sacctmgr -n -P show assoc user=$USER format=Account,Partition,QOS,DefaultQOS,MaxJobs,MaxSubmit,GrpTRES 2>&1 | head -30",
 "scontrol show partition 'l40s@cr+mp/romed8nl/1gpu-32cpu-128gb' 2>&1",
 "sinfo -h -o '%P|%l|%a|%D|%T' | grep -i l40s | head -20",
 "which crun; file $(which crun) | head -2",
]
for c in cmds:
    rc, out, err = farm.ssh(c, b"", 90)
    print("=== $", c, "rc", rc); print(out.decode()[:6000]); print(err.decode()[:500])
