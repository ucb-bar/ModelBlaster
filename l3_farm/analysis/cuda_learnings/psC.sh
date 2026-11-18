ps -u krisd -o pid,etime,pcpu,rss,args | grep -E "conc_check|extract_graph|problem" | grep -v grep | cut -c1-200; ls -la /tmp/krisd/cl/cruns/6_GoogleNetInceptionModule/ 2>/dev/null; nproc; uptime
