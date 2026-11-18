"""Dry run of farm.start() for one L1 and one capped L2 problem with analysis/l12_sweep/config_l12.json: nothing is launched (put / ssh / detach stubbed); prints the job.env (secrets filtered)."""
import copy, os, sys
from pathlib import Path
L3 = Path(__file__).resolve().parents[2]
os.environ["FARM_CONFIG"] = str(L3 / "analysis/l12_sweep/config_l12.json")
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
names = [n for n in (p.stem for p in (L3 / "problems").glob("6[12]*.py"))]
pick = [next(n for n in sorted(names) if n.startswith("61001")), next(n for n in sorted(names) if n.startswith("61045")), next(n for n in sorted(names) if n.startswith("62055"))]
for name in pick:
    cap.clear()
    ok = farm.start(1, name, {})
    env = next((v for k, v in cap.items() if k.endswith("job.env")), b"").decode()
    print("==", name, "launched=", ok)
    print("\n".join(l[:230] for l in env.splitlines() if not any(s in l.upper() for s in ("KEY", "TOKEN", "SECRET")) and any(t in l for t in ("KB_EXTRA", "BENCH_TARGET", "MODEL", "ROUNDS", "ROLLOUTS", "STEPS", "TIMEOUT"))))
print("scheduling:", farm.SC["mode"], "| pool:", farm.SC.get("pool"), "| WORKER:", farm.WORKER, "| max_nodes:", farm.CFG["max_nodes"])
