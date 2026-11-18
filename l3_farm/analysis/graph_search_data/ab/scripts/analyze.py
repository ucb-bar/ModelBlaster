#!/usr/bin/env python3
"""A/B analysis (X-133): per arm, the time-to-quality curve (model partition time at each round start, after the
characterization pass, and the final verified / model-run time) against cumulative wall time, LLM calls and $
(relay ledger: tokens x assumed Opus prices $5/$25/$6.25/$0.50 per M), plus the graph decisions with their times.
Usage: analyze.py DIR_R DIR_I > ab.json   (DIR_* = unpacked <arm>.tgz)"""
import json, sys
from pathlib import Path

P = {"in": 5.0, "out": 25.0, "cw": 6.25, "cr": 0.50}


def usd_row(d):
    u = d.get("usage") or d
    pt = u.get("prompt_tokens") or u.get("input_tokens") or 0
    ct = u.get("completion_tokens") or u.get("output_tokens") or 0
    cr = u.get("cache_read_tokens") or u.get("cache_read_input_tokens") or (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    cw = u.get("cache_write_input_tokens") or u.get("cache_creation_input_tokens") or u.get("cache_write_tokens") or 0
    unc = max(0, pt - cr - cw)
    return (unc * P["in"] + ct * P["out"] + cw * P["cw"] + cr * P["cr"]) / 1e6, pt, ct, cr, cw


def arm(d):
    d = Path(d)
    st = json.loads((d / "work" / "state.json").read_text())
    summ = json.loads((d / "work" / "summary.json").read_text()) if (d / "work" / "summary.json").exists() else {}
    t0 = int((d / "t_start").read_text())
    t1 = int((d / "t_end").read_text()) if (d / "t_end").exists() else None
    led = [json.loads(x) for x in (d / "ledger_slice.jsonl").read_text().splitlines() if x.strip()] \
        if (d / "ledger_slice.jsonl").exists() else []
    rows = []
    for x in led:
        c, pt, ct, cr, cw = usd_row(x)
        ts = x.get("ts") or 0
        if isinstance(ts, str):
            import time as _t
            ts = _t.mktime(_t.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S"))
        rows.append({"ts": ts, "usd": c, "pt": pt, "ct": ct, "cr": cr, "cw": cw})
    rows.sort(key=lambda r: r["ts"])

    def spend_until(t):
        s = [r for r in rows if r["ts"] and r["ts"] <= t]
        return round(sum(r["usd"] for r in s), 3), len(s)
    curve = []
    tcur = t0
    for r in st["rounds"]:
        pu = list(r["partition_us"].values())[0]
        curve.append({"point": f"r{r['round']} start", "partition_us": pu})
        tcur += r.get("seconds") or 0
        gsr = r.get("graph_search") or {}
        curve[-1]["round_seconds"] = r.get("seconds")
        curve[-1]["decisions"] = [(x["hint"], x["status"], round(x["gain"], 3) if x.get("gain") is not None else None)
                                  for x in r.get("decisions", [])]
        curve[-1]["t_end_round"] = tcur
        curve[-1]["usd_end_round"], curve[-1]["calls_end_round"] = spend_until(tcur)
    ev = []
    f = d / "work" / "graph_decisions.jsonl"
    if f.exists():
        for x in f.read_text().splitlines():
            e = json.loads(x)
            if e["event"] in ("decision", "characterize", "stop", "survival"):
                ev.append({k: e.get(k) for k in ("event", "t_run_s", "round", "phase", "hint", "status", "steps", "llm_calls",
                                                 "usd", "steps_saved", "problems", "partition_us_after")})
    ir = next(iter(summ.get("irs", {}).values()), {})
    return {"t_start": t0, "t_end": t1, "wall_s": (t1 - t0) if t1 else None, "ledger_calls": len(rows),
            "usd": round(sum(r["usd"] for r in rows), 3),
            "tokens": {k: sum(r[k] for r in rows) for k in ("pt", "ct", "cr", "cw")},
            "kb_llm_calls": summ.get("llm_calls"), "curve": curve, "events": ev,
            "final": {k: ir.get(k) for k in ("first_us", "final_us", "model_e2e_us", "seed_e2e_us", "model_speedup",
                                              "passed", "fused", "torch_us", "speedup_vs_torch")},
            "accepted": summ.get("accepted"), "rejected": summ.get("rejected"), "rounds": summ.get("rounds")}


if __name__ == "__main__":
    print(json.dumps({Path(p).name: arm(p) for p in sys.argv[1:]}, indent=1, default=str))
