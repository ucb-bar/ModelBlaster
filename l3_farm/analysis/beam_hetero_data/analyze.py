#!/usr/bin/env python3
"""Analyze collected arms: python3 analyze.py DIR [DIR ...]  (DIR = res/<tag> holding work/, ledger.jsonl, markers.txt,
or a sweep results dir). Writes DIR/analysis.json and prints markdown tables."""
import json, re, sys
from collections import defaultdict, Counter
from pathlib import Path

# $/M tokens: (input, output, cache write, cache read). Opus at the user's ASSUMED prices; Haiku/Sonnet 5.5 at list
# prices (cache write 1.25x input, read 0.1x input); Haiku 5.5 long-context (>100K prompt) $0.50/$2.50.
PRICE = {"opus": (5.0, 25.0, 6.25, 0.50), "sonnet": (2.0, 10.0, 2.5, 0.20), "haiku": (0.10, 0.50, 0.125, 0.01)}
HAIKU_LONG = (0.50, 2.50, 0.625, 0.05)
HARD = ("shared_memory_tiling", "tensor_core", "wmma", "mma", "kernel_fusion", "fused_operations", "register_tiling",
        "register_blocking", "algorithmic", "composite", "hybrid", "implicit_gemm", "double_buffering", "flash",
        "online_softmax", "split_k", "persistent", "cp_async", "single_pass_fusion", "local_memory_tiling", "winograd",
        "im2col", "epilogue_fusion")


def fam(model):
    m = (model or "").lower()
    return next((k for k in PRICE if k in m), "opus")


def usd(model, prompt, out, cw, cr):
    p = PRICE[fam(model)]
    if fam(model) == "haiku" and prompt > 100_000:
        p = HAIKU_LONG
    unc = max(0, prompt - cw - cr)
    return (unc * p[0] + out * p[1] + cw * p[2] + cr * p[3]) / 1e6


def tsf(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        import time as _t
        return _t.mktime(_t.strptime(str(v)[:19], "%Y-%m-%dT%H:%M:%S"))


def tier(tech):
    return "hard" if any(h in (tech or "").lower() for h in HARD) else "simple"


def analyze(d):
    d = Path(d)
    w = d / "work"
    A = {"dir": str(d)}
    summ = json.loads((w / "summary.json").read_text()) if (w / "summary.json").exists() else {}
    ir = next(iter((summ.get("irs") or {}).values()), {})
    A["final"] = {k: ir.get(k) for k in ("model_e2e_us", "seed_e2e_us", "model_speedup", "model_passed", "passed",
                                          "final_us", "first_us")}
    A["accepted"] = summ.get("accepted")
    st = json.loads((w / "state.json").read_text()) if (w / "state.json").exists() else {}
    A["rounds"] = [{"round": r["round"], "start_partition_us": r["partition_us"].get("kb_problem"),
                    "llm_calls_total": r.get("llm_calls_total"), "seconds": r.get("seconds"),
                    "decisions": [(x.get("hint"), x.get("status"), round(x.get("gain") or 0, 3)) for x in r.get("decisions", [])]}
                   for r in st.get("rounds", [])]
    # ---- ledger (ground truth of spend; rows of this arm's time window)
    mk = (d / "markers.txt").read_text().split() if (d / "markers.txt").exists() else []
    t0 = float(mk[1]) if len(mk) > 1 else 0
    t1 = float(mk[5]) if len(mk) > 5 else 9e18
    led = []
    if (d / "ledger.jsonl").exists():
        for l in (d / "ledger.jsonl").read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                if t0 - 5 <= tsf(r.get("ts", 0)) <= t1 + 5:
                    led.append(r)
    A["wall_s"] = (t1 - t0) if len(mk) > 5 else None
    lm = defaultdict(lambda: Counter())
    for r in led:
        c = lm[r.get("model")]
        c["calls"] += int(r.get("calls") or 1)
        for k in ("prompt_tokens", "completion_tokens", "cache_write_input_tokens", "cache_read_input_tokens"):
            c[k] += int(r.get(k) or 0)
        c["usd"] += usd(r.get("model"), int(r.get("prompt_tokens") or 0), int(r.get("completion_tokens") or 0),
                        int(r.get("cache_write_input_tokens") or 0), int(r.get("cache_read_input_tokens") or 0))
    A["ledger"] = {m: dict(c) for m, c in lm.items()}
    A["ledger_total"] = {"calls": sum(c["calls"] for c in lm.values()), "usd": sum(c["usd"] for c in lm.values()),
                         "out": sum(c["completion_tokens"] for c in lm.values())}
    lr = defaultdict(lambda: Counter())
    for r in led:
        m = re.search(r"role=(\S+)", r.get("what") or "")
        c = lr[f"{m.group(1) if m else '?'}|{fam(r.get('model'))}"]
        c["calls"] += 1
        c["in"] += int(r.get("prompt_tokens") or 0); c["out"] += int(r.get("completion_tokens") or 0)
        c["cw"] += int(r.get("cache_write_input_tokens") or 0); c["cr"] += int(r.get("cache_read_input_tokens") or 0)
        c["usd"] += usd(r.get("model"), int(r.get("prompt_tokens") or 0), int(r.get("completion_tokens") or 0),
                        int(r.get("cache_write_input_tokens") or 0), int(r.get("cache_read_input_tokens") or 0))
    A["ledger_what"] = {k: dict(v) for k, v in lr.items()}
    led_ts = sorted(tsf(r["ts"]) for r in led)
    # ---- journals: per role/model, truncation
    jr = defaultdict(lambda: Counter())
    trunc = Counter()
    calls_ts = []
    for jf in sorted(w.glob("r[0-9][0-9]/run/*/journal/llm.jsonl")):
        for l in jf.read_text().splitlines():
            if not l.strip():
                continue
            x = json.loads(l)
            u = x.get("usage") or {}
            key = (x.get("role"), x.get("model"))
            c = jr[key]
            c["calls"] += 1
            p, o = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
            cw, cr = int(u.get("cache_creation_input_tokens") or 0), int(u.get("cache_read_input_tokens") or 0)
            c["prompt"] += p; c["out"] += o; c["cw"] += cw; c["cr"] += cr
            c["usd"] += usd(x.get("model"), p, o, cw, cr)
            stops = ((u.get("telemetry") or {}).get("stop_reasons")) or []
            if any(s in ("length", "max_tokens") for s in stops if s):
                trunc[key] += 1
            calls_ts.append(float(x.get("ts") or 0))
    A["journal_by_role"] = {f"{k[0]}|{k[1]}": dict(v) for k, v in jr.items()}
    A["truncations"] = {f"{k[0]}|{k[1]}": v for k, v in trunc.items()}
    # ---- steps
    steps = []
    beams = []
    for rf in sorted(w.glob("r[0-9][0-9]/run/*/result.json")):
        r = json.loads(rf.read_text())
        prob = rf.parent.name
        rnd = rf.parent.parent.parent.name
        jts = defaultdict(float)
        jf = rf.parent / "journal" / "llm.jsonl"
        if jf.exists():
            for l in jf.read_text().splitlines():
                if l.strip():
                    x = json.loads(l)
                    m = re.match(r"r(\d+)s(\d+)", x["key"])
                    if m:
                        jts[(int(m.group(1)), int(m.group(2)))] = max(jts[(int(m.group(1)), int(m.group(2)))], float(x.get("ts") or 0))
        for s in r.get("steps", []):
            ctx = s.get("context") or {}
            steps.append({"round": rnd, "problem": prob, "rollout": s.get("rollout"), "step": s.get("step"),
                          "slot": s.get("slot"), "technique": s.get("technique"), "status": s.get("status"),
                          "tier": tier(s.get("technique")), "model": ctx.get("model"), "role": ctx.get("model_role"),
                          "why": ctx.get("model_why"), "fix_attempts": s.get("fix_attempts") or 0,
                          "fix_models": ctx.get("fix_models"), "e2e_before": s.get("e2e_before"),
                          "e2e_after": s.get("e2e_after"), "reward": s.get("reward"),
                          "beam": ctx.get("beam"), "ts": jts.get((s.get("rollout"), s.get("step")), 0.0),
                          "base_e2e": r.get("baseline_e2e_us"), "final_e2e": r.get("final_e2e_us")})
        if r.get("search"):
            beams.append({"problem": prob, "round": rnd, **r["search"]})
        # kb-level runs
    A["n_steps"] = len(steps)
    A["steps"] = steps
    A["beams"] = beams
    A["kb_runs"] = []
    for rf in sorted(w.glob("r[0-9][0-9]/run/*/result.json")):
        r = json.loads(rf.read_text())
        A["kb_runs"].append({"problem": rf.parent.name, "base_e2e": r.get("baseline_e2e_us"), "final_e2e": r.get("final_e2e_us"),
                             "e2e_speedup": r.get("e2e_speedup"), "llm_calls": r.get("llm_calls"), "steps": len(r.get("steps", []))})
    # status by role/model/tier
    tab = defaultdict(Counter)
    for s in steps:
        k = f"{s['role']}|{fam(s['model'])}|{s['tier']}"
        tab[k][s["status"]] += 1
        tab[k]["n"] += 1
        tab[k]["with_fix"] += s["fix_attempts"] > 0
        tab[k]["fix_calls"] += s["fix_attempts"]
        fm = s.get("fix_models") or []
        if isinstance(fm, list):
            tab[k]["fix_escalated"] += sum(1 for f in fm if "escal" in json.dumps(f))
    A["status_table"] = {k: dict(v) for k, v in tab.items()}
    A["why"] = dict(Counter(f"{s['role']}: {re.sub(r' [a-z0-9_]+$', ' <tech>', s['why'] or '')}" for s in steps))
    (d / "analysis.json").write_text(json.dumps(A, indent=1, default=str))
    return A


if __name__ == "__main__":
    for x in sys.argv[1:]:
        A = analyze(x)
        print("##", x)
        print("final", A["final"], "accepted", A["accepted"], "wall_s", A["wall_s"])
        print("rounds", json.dumps(A["rounds"]))
        print("ledger_total", A["ledger_total"])
        for m, c in A["ledger"].items():
            print("  ledger", m, {k: round(v, 2) if isinstance(v, float) else v for k, v in c.items()})
        print("  ledger what", A["ledger_what"])
        for k, v in A["journal_by_role"].items():
            print("  journal", k, {kk: round(vv, 2) if isinstance(vv, float) else vv for kk, vv in v.items()})
        print("  truncations", A["truncations"])
        print("  steps", A["n_steps"])
        for k, v in sorted(A["status_table"].items()):
            print("  status", k, v)
        print("  why", A["why"])
        for b in A["beams"]:
            print("  beam", json.dumps(b)[:600])
