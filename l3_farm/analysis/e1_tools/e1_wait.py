import sys, time
sys.path.insert(0, '/Users/krisd/Documents/ModelBlaster-clone/l3_farm/analysis/e1_tools')
import e1_run
J = sys.argv[1]
while True:
    rc, out, err = e1_run.run(J, "cd /tmp/krisd/e1; tail -2 logs/worker_1792988937.log; ls runs/31_VisionAttention | tr '\\n' ' '", 120)
    print(time.strftime('%T'), out.strip()[-300:], flush=True)
    if 'WORKER_DONE' in out: break
    time.sleep(240)
