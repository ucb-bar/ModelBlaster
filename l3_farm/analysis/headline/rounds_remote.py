import json, glob, os
EXP = {0: ["profile", "gs_char", "gs_char_profile", "gs_preapply", "plan", "export", "cands", "gs_inter", "ingest", "reuse"],
       "n": ["profile", "gs_preapply", "plan", "export", "cands", "gs_inter", "ingest", "reuse"]}
out = []
for d in sorted(glob.glob("/tmp/krisd/farm_headline/*/")):
    n = os.path.basename(d.rstrip("/")); w = d + "work/"
    try: st = json.load(open(w + "state.json"))
    except Exception: out.append([n, None, None, None, None]); continue
    done = len(st.get("rounds", [])); rs = sorted(int(os.path.basename(x)[1:]) for x in glob.glob(w + "r0[0-9]"))
    cur = max(rs) if rs else 0
    ph = []
    try: ph = [k for k, v in json.load(open(w + f"r0{cur}/progress.json"))["phases"].items()]
    except Exception: pass
    exp = EXP[0] if cur == 0 else EXP["n"]
    frac = min(1.0, len([p for p in ph if p in exp]) / len(exp)) if cur >= done else 0
    out.append([n, done, cur, round(frac, 2), ph])
print(json.dumps(out))
