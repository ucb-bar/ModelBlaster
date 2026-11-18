"""Re-run 31_VisionAttention without --audit (the wide global pass hit the 5400 s timeout on 16384x16384 bmm)."""
import sys
sys.path.insert(0, '/Users/krisd/Documents/ModelBlaster-clone/l3_farm/analysis/e1_tools')
import e1_run
J = sys.argv[1]
e1_run.AUDIT.discard("31_VisionAttention")
e1_run.pushbin(J)
e1_run.start(J, ["31_VisionAttention"])
