import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "merge_checks"))
import mc_common as C
C.MINE = HERE.parent / "my_jobs.json"
job=int(sys.argv[1]); print(C.put(job, sys.argv[2], sys.argv[3])[0])
