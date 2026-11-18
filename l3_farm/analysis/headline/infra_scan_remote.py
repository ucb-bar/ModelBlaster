import os, glob, json, re, sys
roots = sys.argv[1:] or ["/tmp/krisd/farm_headline"]
PAT = {"scp_closed": r"scp: Connection closed", "infra_tag": r"\[infra\]", "nodevice": r"NoDevice|no usable GPU", "alloc_x": r"alloc x",
       "so_load": r"cannot open shared object|dlopen|\.so load", "conn_reset": r"Connection reset|Broken pipe|Connection timed out|ssh: connect",
       "final_infra": r"final_infra", "illegal_mem": r"illegal memory access", "relay_503": r"queue is full"}
CP = {k: re.compile(v) for k, v in PAT.items()}
out = {}
for root in roots:
    for d in sorted(glob.glob(root + "/*/")):
        n = os.path.basename(d.rstrip("/")); r = {k: 0 for k in PAT}; per_round = {}
        for f in glob.glob(d + "run.log") + glob.glob(d + "work/r0*/run.log") + glob.glob(d + "work/r0*/char/run.log"):
            m = re.search(r"/(r0\d)/", f); rd = m.group(1) if m else "top"
            try: txt = open(f, errors="ignore").read()
            except Exception: continue
            for k, c in CP.items():
                v = len(c.findall(txt)); r[k] += v
                if v and k in ("scp_closed", "infra_tag", "nodevice", "so_load", "conn_reset", "final_infra"): per_round.setdefault(rd, {})[k] = per_round.setdefault(rd, {}).get(k, 0) + v
        rej = []
        try:
            for l in open(d + "work/graph_decisions.jsonl"):
                j = json.loads(l)
                if j.get("event") == "decision" and j.get("status") == "rejected" and j.get("measured") is False and j.get("phase") != "preapply" and not str(j.get("hint", "")).startswith("region"):
                    rej.append((j.get("round"), j.get("hint")))
        except Exception: pass
        sent = 0
        try:
            st = json.load(open(d + "work/state.json")); rounds = len(st.get("rounds", []))
        except Exception: rounds = None
        out[n] = {"counts": r, "per_round": per_round, "rejected_unmeasured": rej, "rounds": rounds}
print(json.dumps(out))
