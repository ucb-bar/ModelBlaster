"""Headline sweep spend from the relay ledger on the driver node (relay est_usd is 0 for these models, so the relay's $ cap does
NOT bind): tokens priced with the assumed prices ($/M: Opus 5 in / 25 out / 0.5 cache read / 6.25 cache write; Haiku 0.10 / 0.50 /
0.01 / 0.125; Sonnet 2 / 10 / 0.2 / 2.5). Usage (from l3_farm): FARM_CONFIG=analysis/headline/config_headline.json python3 analysis/headline/spend.py [JOB]"""
import sys, json, collections
sys.path.insert(0, ".")
import farm
# main instance: the pooled driver's ledger (all controllers share one relay); heavy instance: one relay per node (extra jobs as args)
s = farm.load()
drv = (s.get("sched") or {}).get("driver") or sorted(s.get("nodes", {}))[-1]
jobs = [int(drv)] + [int(a) for a in sys.argv[1:]]            # (heavy nodes 5312377/5312379 were released: their ~$292 is added below)
BACKUPS = []   # (first-driver ledger already inside FROZEN_BEFORE_1711 below)
# /tmp/krisd persists across jobs on a reused host: count only ledger lines written after this sweep's own start per node
START = {int(drv): "2026-10-26T03:49:00" if int(drv) == 5310063 else ("2026-10-26T17:10:00" if int(drv) == 5321587 else "2026-10-26T13:41:00")}
out = b""
for j in jobs:
    rc, o, err = farm.ssh(f"srun --jobid={j} --overlap bash -c 'cat /tmp/krisd/farm/llm_ledger.jsonl'")
    t0 = START.get(j, "2026-10-26T03:49:00")
    for ln in o.decode().splitlines():
        try:
            if json.loads(ln).get("ts", "") >= t0: out += (ln + "\n").encode()
        except Exception: pass
for bf in BACKUPS:
    try:
        for ln in open(bf):
            try:
                if json.loads(ln).get("ts", "") >= "2026-10-26T03:49:00": out += ln.encode()
            except Exception: pass
    except OSError: pass
P = {"opus": (5, 25, 0.5, 6.25), "haiku": (0.10, 0.50, 0.01, 0.125), "sonnet": (2, 10, 0.2, 2.5)}
tot = collections.Counter(); calls = collections.Counter(); roles = collections.Counter()
for ln in out.decode().splitlines():
    try: r = json.loads(ln)
    except Exception: continue
    m = "haiku" if "haiku" in r.get("model", "") else "sonnet" if "sonnet" in r.get("model", "") else "opus"
    i, o, cr, cw = P[m]
    pt, ct = r.get("prompt_tokens", 0), r.get("completion_tokens", 0)
    c_r, c_w = r.get("cache_read_input_tokens", 0), r.get("cache_write_input_tokens", 0)
    usd = ((max(pt - c_r - c_w, 0)) * i + c_r * cr + c_w * cw + ct * o) / 1e6
    tot[m] += usd; calls[m] += r.get("calls", 1)
    role = (r.get("what", "").split("role=")[1].split()[0] if "role=" in r.get("what", "") else "?")
    roles[(m, role)] += usd
print("calls", dict(calls), "| spend by model $", {k: round(v, 2) for k, v in tot.items()}, "| TOTAL $", round(sum(tot.values()), 2))
print("by role:", {f"{m}/{r}": round(v, 2) for (m, r), v in sorted(roles.items(), key=lambda kv: -kv[1])[:8]})
FROZEN_BEFORE_1711 = 3083.0 - 292.0   # main spend up to 17:11 (est.): the second driver 5313059 expired before its ledger was saved; last measured GRAND total 3,013 at 16:36 (incl. heavy 292) + ~35 min at ~$2/min
BACKUPS = []                          # the first-driver backup is already inside FROZEN_BEFORE_1711
HEAVY_USD = 292.0   # heavy instance (44, 18; nodes 5312377/5312379, released) priced from their ledgers at 08:35-11:00
print("main spend up to 17:11 (estimated, driver 5313059 ledger lost) $", FROZEN_BEFORE_1711, "+ new driver since 17:10 above")
print("heavy instance (released) ~$", HEAVY_USD, "| GRAND TOTAL ~$", round(sum(tot.values()) + HEAVY_USD + FROZEN_BEFORE_1711, 2))
