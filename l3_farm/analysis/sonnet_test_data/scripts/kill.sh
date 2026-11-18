# Node side: stop one arm (cost / time cap exceeded). Args: <arm>. Only processes whose command line names this arm's dir.
pkill -f "sonnet_$1/"
echo failed:killed-cap > /tmp/krisd/sonnet_$1/43_MinGPTCausalAttention/status
