# documented allowlist only (computelab:rusu skill, "GPU reset"): sudo rusu reset-gpu -i <id>
Q="nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm,clocks.mem,clocks_throttle_reasons.active,utilization.gpu,pstate --format=csv,noheader"
echo "before: $($Q)"
sudo rusu reset-gpu -i 0; echo "rc=$?"
sleep 8; echo "after: $($Q)"
