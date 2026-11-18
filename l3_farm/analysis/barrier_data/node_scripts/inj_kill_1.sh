pkill -f "/tmp/krisd/bar_x/1_MLP/" ; pkill -f "worker5.sh 1_MLP"; sleep 1; pgrep -af "bar_x/1_MLP" | head -2; echo killed
