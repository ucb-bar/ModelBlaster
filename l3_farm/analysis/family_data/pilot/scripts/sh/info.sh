hostname; nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader; nvidia-smi --query-compute-apps=pid,name --format=csv,noheader
df -h /tmp | tail -1; nproc; free -g | head -2
ls /tmp/krisd; ls /tmp/krisd/farm; cat /tmp/krisd/farm/tree.stamp
which ncu; ls /home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04/bin | grep -i ncu
pgrep -af kb_graph_opt
