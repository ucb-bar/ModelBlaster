import sys
sys.path.insert(0, '/Users/krisd/Documents/ModelBlaster-clone/l3_farm/analysis/e1_tools')
import e1_run
J = sys.argv[1]; P = sys.argv[2]
sh = f"""cd /tmp/krisd/e1; date; ps -eo pid,etime,cmd | grep -E 'e1_node|torch_bench|worker' | grep -v grep | head; ls -la runs/{P}; tail -15 runs/{P}/node.log; grep -c '"t1"' runs/{P}/e1.json; tail -3 logs/worker_1792988937.log"""
rc, out, err = e1_run.run(J, sh, 120)
print(out, err[-300:])
