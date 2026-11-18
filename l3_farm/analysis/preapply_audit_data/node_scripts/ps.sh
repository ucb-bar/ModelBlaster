ps aux | grep -E "gpu_audit_test|nvcc|cicc|ptxas|problem" | grep -v grep | cut -c1-150 | head; ls -t /tmp/krisd/pa/gpuwork_8_ResNetBasicBlock/* 2>/dev/null | head -5
