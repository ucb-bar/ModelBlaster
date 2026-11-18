import sys
sys.path.insert(0, '/Users/krisd/Documents/ModelBlaster-clone/l3_farm/analysis/e1_tools')
import e1_run
J, cmd = sys.argv[1], sys.argv[2]
print(e1_run.run(J, "cd /tmp/krisd/e1; " + cmd, 120)[1][-1500:])
