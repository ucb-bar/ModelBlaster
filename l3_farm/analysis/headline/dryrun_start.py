"""Dry run of farm.start() with kb.headline_config set: nothing is launched (put/ssh/detach/meta are stubbed), prints job.env KB_EXTRA.
Does not touch state/farm.json or config.json (CFG is modified in memory only)."""
import copy, sys, json
from pathlib import Path
L3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(L3))
import farm
cap = {}
farm.put = lambda job, src, dst: cap.__setitem__(str(dst), src)
farm.ssh = lambda *a, **k: (0, b"", b"")
farm.detach = lambda *a, **k: True
farm.meta_event = lambda *a, **k: None
farm.save = lambda *a, **k: None
farm._S = None
farm.gk_enabled = lambda p: False
farm.NEW = True
for label, hc in (("defaults (no headline_config)", None), ("headline", "analysis/headline/headline.json")):
    farm.CFG = copy.deepcopy(farm.CFG)
    farm.CFG["kb"].pop("headline_config", None)
    if hc: farm.CFG["kb"]["headline_config"] = hc
    cap.clear()
    ok = farm.start(1, "1_MLP", {})
    env = next((v for k, v in cap.items() if k.endswith("job.env")), b"").decode()
    print("==", label, "launched=", ok)
    print("\n".join(l for l in env.splitlines() if not any(s in l.upper() for s in ("KEY", "TOKEN", "SECRET"))) or "(empty env)")
    print("scheduling:", farm.SC["mode"], (farm.SC.get("pool") or {}).get("order"), "| gk:", farm.GK.get("path"), farm.GK.get("merge_on"), farm.GK.get("refresh"), farm.GK.get("env"), "| relay:", farm.CFG["kb"]["relay_concurrency"], farm.CFG["kb"]["relay_budget_usd"], "| instance:", farm.INST.get("name"), farm.STATE, farm.RES)
