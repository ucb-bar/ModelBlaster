import sys, os
sys.path.insert(0, os.path.expanduser("~/Documents/ModelBlaster-clone/l3_farm"))
import farm, batch_nodes as B
P = ",".join(B.L40S_PARTITIONS)
cmds = [
 "scontrol show config | grep -i -E 'KillWait|MaxJobCount|PreemptType|PreemptMode|PreemptExemptTime|UnkillableStepTimeout|MinJobAge|SchedulerType|DefaultTime|MaxTime|OverTimeLimit|SlurmVersion|SLURM_VERSION|^Prolog|Epilog|SelectType'",
 f"sbatch --test-only -J mb-batchproto -p '{P}' --gpus=1 -A cuopt_fallback -t 1-00:00:00 --signal=B:USR1@600 --no-requeue --wrap='sleep 1' 2>&1 | tail -3",
 f"sbatch --test-only -J mb-batchproto -p '{P}' --gpus=1 -A cuopt_fallback --qos=reservation -t 2-00:00:00 --wrap='sleep 1' 2>&1 | tail -3",
 f"sbatch --test-only -J mb-batchproto -p '{P}' --gpus=1 -A cuopt_fallback --qos=interactive-isolated -t 12:00:00 --wrap='sleep 1' 2>&1 | tail -3",
 "sinfo -T 2>&1 | head; sacctmgr -n -P show qos format=Name,GrpTRES,MaxTRESPerUser,MaxJobsPerUser 2>&1 | head",
 f"scontrol show config | grep -i -E 'Version|ResumeTimeout'",
 "sacct -u krisd -S 2026-10-20 -X -n -P -o JobID,JobName,State,Elapsed,Timelimit,QOS | awk -F'|' '$2 ~ /crun/ {c[$3]++} END{for(k in c) print k, c[k]}'",
]
for c in cmds:
    rc, out, err = farm.ssh(c, b"", 120)
    print("=== $", c[:150], "rc", rc); print(out.decode()[:3000]); print(err.decode()[:300])
