tail -c 3000 /tmp/krisd/cl/logs/rc_A.log 2>/dev/null; tail -c 3000 /tmp/krisd/cl/logs/rc_B.log 2>/dev/null
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
