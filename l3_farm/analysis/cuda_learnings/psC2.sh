ps -u krisd -o etime,args | grep conc_check | grep -v grep | cut -c1-150; tail -c 600 /tmp/krisd/cl/logs/rc_C.log
