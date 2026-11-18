#!/usr/bin/env python3
"""X-144 analysis: per-step tables, best-so-far curves, equal-steps / equal-$ comparisons, technique reach, multi-turn
records. Reads arms/<ARM>/out/<prob>/{result.json,multiturn.jsonl,journal/llm.jsonl,experiments/}, remeasure.json.
Writes summary.json, tables.md, experiments.jsonl, plots (png). Run with a python that has matplotlib.
Cost: $5 / $25 per M input / output tokens on the arms' llm_usage (A2's convention; cache discounts ignored)."""
import json, re, sys
from pathlib import Path
H = Path(__file__).resolve().parent
A2 = H.parent / "capabilities" / "results" / "5299224" / "ab5"
PIN, POUT = 5.0, 25.0
PROB = "a2_41_gru_rev_l1_5"
ARMS = ["M8_s0", "M8_s1", "S12_s0", "S12_s1", "S8_first_r1", "S8_first_r2"]
REFS = {"A2_S_rec": A2 / "S_rec" / PROB, "A2_M_rec": A2 / "M_rec" / PROB}
RECSPEC = {"persistent_recurrence_state_on_chip", "recurrence_input_projection_hoisting", "step_sync_minimization",
           "layer_wavefront_pipeline", "recurrent_gate_fast_math", "per_step_latency_probe"}


def usd(i, o):
    return (i * PIN + o * POUT) / 1e6


def load(name, d):
    r = json.loads((d / "result.json").read_text())
    base = r["baseline_e2e_us"]
    rows, cum, ccalls, best = [], 0.0, 0, base
    for s in r["steps"]:
        c = s["context"]; e = c.get("e2e", {}); t = c.get("telemetry", {}); l = t.get("llm", {})
        i, o, n = l.get("in", 0), l.get("out", 0), l.get("calls", 0)
        cost = usd(i, o); cum += cost; ccalls += n
        after_abs = e.get("confirm_us") or e.get("after_us")
        if s["status"] == "improved" and after_abs:
            best = min(best, after_abs)
        rows.append(dict(step=s["step"], technique=s["technique"], kind=c.get("kind", "technique"), status=s["status"],
                         e2e_before=s.get("e2e_before"), e2e_after=s.get("e2e_after"), cand_abs_us=after_abs,
                         paired_gain=e.get("gain"), reward=s["reward"], seconds=s["seconds"], calls=n, tok_in=i,
                         tok_out=o, cost=cost, cum_cost=cum, cum_calls=ccalls, best_abs=best,
                         truncated=l.get("truncated", 0), fixes=s.get("fix_attempts")))
    mt = r.get("multiturn")
    u = r["llm_usage"]
    return dict(name=name, baseline=base, final=r["final_e2e_us"], calls=r["llm_calls"], tok_in=u["prompt_tokens"],
                tok_out=u["completion_tokens"], cost=usd(u["prompt_tokens"], u["completion_tokens"]), steps=rows,
                multiturn=mt, wall_s=sum(x["seconds"] for x in rows), result=r)


def at_cost(a, budget):
    b = a["baseline"]
    for s in a["steps"]:
        if s["cum_cost"] <= budget:
            b = s["best_abs"]
    return b


def at_step(a, k):
    return a["steps"][k - 1]["best_abs"] if len(a["steps"]) >= k else (a["steps"][-1]["best_abs"] if a["steps"] else a["baseline"])


def experiments(name, d):
    """One record per experiment run: what the existing logs hold, joined from multiturn.jsonl + journal + experiments/."""
    mt = [json.loads(l) for l in (d / "multiturn.jsonl").read_text().splitlines()] if (d / "multiturn.jsonl").exists() else []
    J = {}
    for l in (d / "journal" / "llm.jsonl").read_text().splitlines():
        j = json.loads(l); J[j["key"]] = j
    res = json.loads((d / "result.json").read_text())
    steps = {f"r0s{s['step']}": s for s in res["steps"]}
    out, k = [], 0
    for m in mt:
        s = steps.get(m["key"], {})
        for it in m["items"]:
            k += 1
            tk = f"{m['key']}c0" + (f"t{it['turn']}" if it["turn"] else "")
            nxt = f"{m['key']}c0t{it['turn'] + 1}"
            txt = J.get(tk, {}).get("text", "")
            prose = re.sub(r"```.*?```", "", txt, flags=re.S).strip()
            src_p = d / "experiments" / f"exp{k:03d}" / "exp.cu"
            nums = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*us", it["out_head"])]
            ntxt = re.sub(r"```.*?```", "", J.get(nxt, {}).get("text", ""), flags=re.S).strip()
            out.append(dict(arm=name, exp_no=k, step_key=m["key"], technique=m["technique"], turn=it["turn"],
                            purpose=prose[:700], source_file=str(src_p.relative_to(H)) if src_p.exists() else None,
                            source_bytes=src_p.stat().st_size if src_p.exists() else None, exit=it["exit"],
                            wall_s=it["wall_s"], stdout_head=it["out_head"], parsed_us=nums,
                            next_turn_text=ntxt[:900], step_status=s.get("status"),
                            step_e2e_before=s.get("e2e_before"), step_e2e_after=s.get("e2e_after")))
    return out


def main():
    arms = {}
    for a in ARMS:
        d = H / "arms" / a / "out" / PROB
        if (d / "result.json").exists():
            arms[a] = load(a, d)
    for a, d in REFS.items():
        if (d / "result.json").exists():
            arms[a] = load(a, d)
    rm = json.loads((H / "remeasure.json").read_text()) if (H / "remeasure.json").exists() else {}
    summ = {}
    md = []
    for n, a in arms.items():
        md.append(f"\n#### {n}  (baseline {a['baseline']:.0f} us, final {a['final']:.0f} us, {len(a['steps'])} steps, "
                  f"{a['calls']} LLM calls, tokens in/out {a['tok_in']/1e3:.0f}k/{a['tok_out']/1e3:.0f}k, "
                  f"${a['cost']:.2f}, step time {a['wall_s']/60:.0f} min)\n")
        md.append("| step | technique | status | e2e before | e2e after (paired chain) | candidate abs us | reward | best abs so far | calls | $ step | $ cum | s |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for s in a["steps"]:
            f = lambda x: "-" if x is None else f"{x:,.0f}"
            md.append(f"| {s['step']} | {s['technique']}{' (hyp)' if s['kind']=='hypothesis' else ''} | {s['status']}{' (trunc)' if s['truncated'] else ''} | "
                      f"{f(s['e2e_before'])} | {f(s['e2e_after']) if s['status'] in ('improved','neutral') else '-'} | "
                      f"{f(s['cand_abs_us'])} | {s['reward']:+.3f} | {f(s['best_abs'])} | {s['calls']} | {s['cost']:.2f} | {s['cum_cost']:.2f} | {s['seconds']:.0f} |")
        tech = {}
        for s in a["steps"]:
            tech.setdefault(s["technique"], []).append(s["step"])
        summ[n] = dict(baseline=a["baseline"], final_in_run=a["final"], calls=a["calls"], tok_in=a["tok_in"], tok_out=a["tok_out"],
                       cost=round(a["cost"], 2), steps=len(a["steps"]), step_min=round(a["wall_s"] / 60, 1),
                       cost_per_step=round(a["cost"] / max(1, len(a["steps"])), 2), multiturn=a["multiturn"],
                       techniques={k: v for k, v in tech.items()},
                       reached_step_sync=min(tech.get("step_sync_minimization", [None]) or [None], key=lambda x: 99 if x is None else x),
                       best_at_step={k: at_step(a, k) for k in (1, 2, 3, 4, 5, 6, 7, 8, 12)},
                       best_at_cost={f"${b}": at_cost(a, b) for b in (2, 4, 6, 8, 10)},
                       truncations=sum(s["truncated"] for s in a["steps"]),
                       compile_errors=sum(s["status"] == "compile_error" for s in a["steps"]),
                       n_improved=sum(s["status"] == "improved" for s in a["steps"]))
        if n in rm:
            v = sorted(x["deployed_us"] for x in rm[n] if x.get("ok") and x.get("deployed_us"))
            summ[n]["remeasure_us"] = v
            summ[n]["remeasure_median"] = v[len(v) // 2] if v else None
    (H / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    (H / "tables.md").write_text("\n".join(md))
    ex = []
    for a in ARMS:
        d = H / "arms" / a / "out" / PROB
        if (d / "multiturn.jsonl").exists():
            ex += experiments(a, d)
    (H / "experiments.jsonl").write_text("\n".join(json.dumps(x) for x in ex))
    plots(arms)
    print(json.dumps({n: {k: v for k, v in s.items() if k in ("final_in_run", "cost", "steps", "calls", "remeasure_median", "best_at_step")} for n, s in summ.items()}, indent=1, default=str))


def plots(arms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"M8_s0": "#1f77b4", "M8_s1": "#6baed6", "S12_s0": "#d62728", "S12_s1": "#ff9896", "S8_first_r1": "#2ca02c", "S8_first_r2": "#98df8a",
           "A2_S_rec": "#7f7f7f", "A2_M_rec": "#bcbd22"}
    ls = {"A2_S_rec": ":", "A2_M_rec": ":"}
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for n, a in arms.items():
        xs = [0] + [s["step"] + 1 for s in a["steps"]]
        ys = [a["baseline"]] + [s["best_abs"] for s in a["steps"]]
        ax[0].step(xs, ys, where="post", label=n, color=col[n], ls=ls.get(n, "-"), lw=2)
        c = [0] + [s["cum_cost"] for s in a["steps"]]
        ax[1].step(c, ys, where="post", label=n, color=col[n], ls=ls.get(n, "-"), lw=2)
        k = [0] + [s["cum_calls"] for s in a["steps"]]
        ax[2].step(k, ys, where="post", label=n, color=col[n], ls=ls.get(n, "-"), lw=2)
    for x, t in zip(ax, ("steps", "cumulative LLM $ (assumed $5/$25 per M)", "cumulative LLM calls")):
        x.set_xlabel(t); x.set_ylabel("best e2e so far (us, abs deployed timing)"); x.grid(alpha=.3)
        x.axhline(2785, color="k", lw=.8, ls="--"); x.set_ylim(2400, 5100)
    ax[0].legend(fontsize=8)
    ax[0].set_title("best-so-far vs steps (dashed: forward winner 2,785)")
    fig.tight_layout(); fig.savefig(H.parent / "multiturn_test_curves.png", dpi=130)
    fig, ax = plt.subplots(figsize=(9, 4))
    names = list(arms)
    ax.bar(range(len(names)), [arms[n]["cost"] / max(1, len(arms[n]["steps"])) for n in names], color=[col[n] for n in names])
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, fontsize=8); ax.set_ylabel("$ per step")
    fig.tight_layout(); fig.savefig(H.parent / "multiturn_test_cost_per_step.png", dpi=130)


if __name__ == "__main__":
    main()
