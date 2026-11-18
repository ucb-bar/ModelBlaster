hostname; nvidia-smi --query-gpu=name,clocks.max.mem,clocks.mem,clocks.sm,clocks.max.sm,driver_version --format=csv,noheader
sudo -l 2>&1 | head -40
which ncu nvcc; ncu --version | tail -2
ls /tmp/krisd 2>&1 | head
rusu get system.is_driver_locked 2>&1
