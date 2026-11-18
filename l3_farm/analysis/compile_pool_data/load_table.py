import json
from pathlib import Path
rows = [json.loads(l) for l in (Path(__file__).parent / "results/load_summaries.jsonl").read_text().splitlines() if l.strip()]
rows.sort(key=lambda r: (r["N"], r["think"], ["local", "shared", "remote", "remote2", "remote3"].index(r["label"].split("_")[0])))
print("| N | think s | arm | wall s | sets | units | compiles (nvcc -c run) | builds/min | set latency p50 / p90 / max s | driver avg / peak cores | driver core-s | clean pool hosts core-s (0578+0572) | local nvcc queue wait s | remote fallbacks / FULL polls | failed sets |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    c = r["cpu"]; d = c["driver"] or {}
    hs = sum((c[h] or {}).get("busy_core_s", 0) for h in r["cpu"] if h not in ("driver", "a4u8g-mil-0024"))
    print(f"| {r['N']} | {r['think']:g} | {r['label'].split('_')[0]} | {r['wall']} | {r['sets']} | {r['units']} | {r['compiles_total']} | {r['builds_per_min']} | "
          f"{r['set_lat_p50']} / {r['set_lat_p90']} / {r['set_lat_max']} | {d.get('avg_cores')} / {d.get('peak_cores')} | {d.get('busy_core_s')} | {hs:.0f} | "
          f"{r['sem_wait_s']} | {r['remote_fallback_local']} / {r['remote_full']} | {r['failed_sets']} |")
