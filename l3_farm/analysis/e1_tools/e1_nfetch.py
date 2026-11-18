"""fetch every finished runs_nearest problem from my live nodes: e1_nfetch.py"""
import os, sys, json
os.environ["E1_MODE"] = "nearest"
sys.path.insert(0, '/Users/krisd/Documents/ModelBlaster-clone/l3_farm/analysis/e1_tools')
import e1_run, nodes
asg = json.load(open(e1_run.DATA / "nearest_assign.json"))
asg.setdefault("5304009", []).append("39_GRU")
alive = nodes.squeue()
for j, ps in asg.items():
    if int(j) not in alive:
        print("node gone", j); continue
    rc, out, err = e1_run.run(j, "cd /tmp/krisd/e1/runs_nearest 2>/dev/null && for d in *; do [ -f $d/torch_3.txt ] && grep -q '\"t1\"' $d/e1.json && echo $d; done", 120)
    done = out.split()
    print(j, len(done), "done of", len(ps))
    if done:
        e1_run.fetch(j, done)
