ps -eo pid,ppid,etime,stat,pcpu,cmd --sort=start_time | grep -vE "ps -eo|grep|sshd|bash -s|srun" | tail -25 | cut -c1-200
nvidia-smi --query-gpu=clocks.sm,utilization.gpu,pstate --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
