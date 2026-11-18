"""Client driver for the claude-code: relay tests (uses the real client path llm_relay.generate_via_relay).

  drive.py e2e <n> <model>              n prompts: relay vs direct CLI (text sim, usage fields, latency)
  drive.py scale <conc> <n> <model>     n requests with <conc> in flight; latency distribution, errors
  drive.py one <model> [rid-file]       a single request (failure tests); prints outcome only
Env: KB_LLM_SERVER_URL, KB_LLM_SERVER_TOKEN_FILE (never printed). Results: JSON on stdout last line + file.
"""
import asyncio, difflib, json, os, statistics, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import prompts
from kernelblaster.agents.utils import llm_relay, agentic_llm

OUT = os.environ.get("RT_OUT", "/tmp/rt_out.json")


def pct(v, p):
    v = sorted(v)
    return round(v[min(len(v) - 1, int(p * len(v)))], 2) if v else None


async def timed(fn, *a):
    t0 = time.time()
    try:
        r = await fn(*a)
        return r, None, time.time() - t0
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:160]}", time.time() - t0


def row(r, err, dt, p):
    d = {"role": p["role"], "chars": p["chars"], "wall_s": round(dt, 2), "err": err}
    if r is not None:
        u = r.usage or {}
        t = u.get("telemetry") or {}
        d.update(in_tok=u.get("prompt_tokens"), out_tok=u.get("completion_tokens"), cost=u.get("cost_usd"),
                 srv_ms=t.get("server_latency_ms"), q=t.get("relay_queue_s"), net=t.get("relay_network_s"),
                 att=t.get("relay_attempts"), out_chars=len(r.generations[0]) if r.generations else 0)
    return d


async def e2e(n, model):
    P = prompts.load()[:n]
    rows = []
    for p in P:
        # sequential: direct CLI then relay, so the wall times are comparable
        rd, ed, td = await timed(agentic_llm.generate_code_agentic, p["messages"], 1, model)
        rr, er, tr = await timed(llm_relay.generate_via_relay, p["messages"], 1, model)
        sim = None
        if rd is not None and rr is not None:
            sim = round(difflib.SequenceMatcher(None, rd.generations[0].splitlines(), rr.generations[0].splitlines(),
                                                autojunk=False).ratio(), 3)
        rows.append({"direct": row(rd, ed, td, p), "relay": row(rr, er, tr, p), "sim": sim,
                     "relay_usage_keys": sorted((rr.usage or {}).keys()) if rr else None,
                     "relay_tel_keys": sorted(((rr.usage or {}).get("telemetry") or {}).keys()) if rr else None})
        print(json.dumps(rows[-1]), flush=True)
    return rows


async def scale(conc, n, model, seed_off=0):
    P = prompts.load(tuple(os.environ.get('RT_ROLES','discovery,reflect,hypothesis').split(',')), int(os.environ.get('RT_LO','3000')), int(os.environ.get('RT_HI','16000')))
    P = [P[(i + seed_off) % len(P)] for i in range(n)]
    sem = asyncio.Semaphore(conc)
    rows = []

    async def one(p):
        async with sem:
            r, e, dt = await timed(llm_relay.generate_via_relay, p["messages"], 1, model)
            rows.append(row(r, e, dt, p))

    t0 = time.time()
    await asyncio.gather(*[one(p) for p in P])
    wall = time.time() - t0
    ok = [r for r in rows if not r["err"]]
    lat = [r["wall_s"] for r in ok]
    outt = sum(r.get("out_tok") or 0 for r in ok)
    s = {"conc": conc, "n": n, "model": model, "wall_s": round(wall, 1), "ok": len(ok), "err": len(rows) - len(ok),
         "throughput_req_per_min": round(60 * len(ok) / wall, 1), "agg_out_tok_per_s": round(outt / wall, 1),
         "lat_p50": pct(lat, .5), "lat_p90": pct(lat, .9), "lat_max": max(lat) if lat else None,
         "lat_mean": round(statistics.mean(lat), 2) if lat else None,
         "queue_p50": pct([r.get("q") or 0 for r in ok], .5), "queue_max": max([r.get("q") or 0 for r in ok] or [0]),
         "cost_usd": round(sum(r.get("cost") or 0 for r in ok), 4),
         "per_req_tok_s": round(statistics.median([(r["out_tok"] or 0) / max(r["srv_ms"] or 1, 1) * 1000 for r in ok]), 1) if ok else None,
         "errors": sorted({r["err"] for r in rows if r["err"]})}
    return s, rows


async def main():
    mode = sys.argv[1]
    if mode == "e2e":
        res = await e2e(int(sys.argv[2]), sys.argv[3])
    elif mode == "scale":
        res = await scale(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 0)
        print(json.dumps(res[0]))
    elif mode == "one":
        p = prompts.load(tuple(os.environ.get("RT_ROLES", "discovery,reflect,hypothesis").split(",")), int(os.environ.get("RT_LO", "3000")), int(os.environ.get("RT_HI", "16000")))[int(os.environ.get("RT_IDX", "0"))]
        r, e, dt = await timed(llm_relay.generate_via_relay, p["messages"], 1, sys.argv[2])
        res = row(r, e, dt, p)
        print(json.dumps(res))
    elif mode == "tiny":
        # server-path overhead: alternate direct CLI / relay on a trivial prompt
        n = int(sys.argv[3]); m = [{"role": "user", "content": "Reply with exactly: OK"}]
        d, r = [], []
        for i in range(n):
            for lst, fn in ((d, agentic_llm.generate_code_agentic), (r, llm_relay.generate_via_relay)):
                x, e, dt = await timed(fn, m, 1, sys.argv[2])
                if e: print("ERR", e)
                else: lst.append(dt)
        res = {"n": n, "direct_s": [round(x, 3) for x in d], "relay_s": [round(x, 3) for x in r],
               "direct_med": round(statistics.median(d), 3), "relay_med": round(statistics.median(r), 3)}
        print(json.dumps(res))
    elif mode == "health":
        res = await llm_relay.check_health()
        res = {k: res[k] for k in ("concurrency", "budget", "providers", "allow_models")}
        print(json.dumps(res))
    json.dump(res, open(OUT, "w"), indent=1)


asyncio.run(main())
