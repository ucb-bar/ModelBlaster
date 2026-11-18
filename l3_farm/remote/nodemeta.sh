# Per-node hardware facts for run_meta (X-128). No secrets. key=value lines.
echo "job_id=${SLURM_JOB_ID:-}"
echo "host=$(hostname)"
echo "ts=$(date +%s)"
nvidia-smi --query-gpu=name,uuid,driver_version,vbios_version,clocks.max.memory,clocks.mem,clocks.max.sm,clocks.sm,pstate,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | sed 's/^/smi=/'
echo "kernel=$(uname -r)"
exit 0
