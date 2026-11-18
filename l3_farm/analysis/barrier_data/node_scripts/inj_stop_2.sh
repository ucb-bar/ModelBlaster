pkill -STOP -f "/tmp/krisd/bar_x/2_ShallowWideMLP/"; echo stopped; ps -eo pid,stat,cmd | grep "bar_x/2_Shallow" | grep -v grep | cut -c1-100 | head -3
