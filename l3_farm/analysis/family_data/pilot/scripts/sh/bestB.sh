J=$(ls /tmp/krisd/family/pilot/B/r00/run/kb_*/journal/state.json | head -1)
python3 - $J <<'PY'
import json,sys
s=json.load(open(sys.argv[1])); b=s["best"]["conv2d_f16"]; src=s["code"][b["source"]]
open("/tmp/krisd/family/pilot/B_best_now.cu","w").write(src)
import re
print(b["slot_us"], b["techniques"], len(src.splitlines()), "lines")
print([l for l in src.splitlines() if "MB_NUM_CONFIGS" in l or "mb_choose_config" in l or "MB_FORCE" in l or "MB_TUNED" in l][:8])
PY
