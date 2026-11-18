#!/usr/bin/env python3
"""E7 (X-101): build the step-level dataset of the Level 3 sweep from the logged trajectories (read-only, zero LLM spend).

Outputs (next to this script):
  steps.csv     one row per KernelBlaster step (one technique attempt incl. its fix calls) of every slot problem,
                plus 'orphan' rows for LLM calls that never committed a step (killed / in-flight)
  kbruns.csv    one row per slot problem (kb run): baseline/final, share of model time, spend, fused-away flag
  rounds.csv    one row per (problem, round): measured model time (partition_us) at round start, telemetry spend by role
  problems.csv  one row per problem: seed/final model time, speedup, family, sample flag, spend
  decisions.csv one row per fusion-variant decision (accepted / rejected, gain, measured us)
  join_check.json  reconciliation of journal calls vs telemetry calls

Usage: python3 e7_build.py   (paths are relative to l3_farm/)
"""
import json, glob, os, re, csv, collections, math

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(ROOT, "results")
OUT = os.path.dirname(os.path.abspath(__file__))
PRICE = dict(uncached=5.0, cache_read=0.5, cache_write=6.25, output=25.0)  # $/M tokens (assumed Opus list price, as cost_breakdown.py)

FAM = {
    "CNN": {4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27},
    "Attention/transformer": {28,29,30,31,32,43,44,45,50},
    "Recurrent": {33,34,35,36,37,38,39,40,41,42},
    "MLP": {1,2,3},
    "Other": {46,47,48,49},
}
def family(name):
    k = int(name.split("_")[0]); return next(f for f, s in FAM.items() if k in s)

def usd(uncached, cw, cr, out):
    return (uncached * PRICE["uncached"] + cw * PRICE["cache_write"] + cr * PRICE["cache_read"] + out * PRICE["output"]) / 1e6

CONV = ("conv2d", "conv_transpose")
GEMM = ("linear", "matmul")
ATTN = ("bmm",)
RNN = ("lstm_seq", "gru_seq")
RED = ("softmax", "layer_norm", "reduce", "sum_dim", "maxpool", "avgpool", "adaptive_avg_pool")
COPY = ("strided_copy", "cat", "gather")
def op_kind(slot):
    return re.sub(r"__v\d+$", "", slot or "")
def op_class(kind):
    base = kind[len("fused__"):] if kind.startswith("fused__") else kind
    first = base.split("__")[0]
    for pref, c in ((CONV, "conv"), (ATTN, "attention_bmm"), (GEMM, "gemm"), (RNN, "recurrent"), (RED, "reduction"), (COPY, "copy")):
        if first.startswith(pref): return c
    return "elementwise"   # relu/relu6/gelu/tanh/sigmoid/add/mul/ewise*/batchnorm (inference affine)

def load(p):
    with open(p) as f: return json.load(f)

def final_alive(op, fused_groups, fused_parts):
    """Is plan op `op` (possibly a fused name 'a+b') still a kernel of the final model?"""
    if op in fused_groups: return True
    parts = op.split("+")
    return not any(p in fused_parts for p in parts)

steps_rows, kb_rows, round_rows, prob_rows, dec_rows = [], [], [], [], []
join = dict(per_problem={})

def problem_dirs():
    out = []
    for d in sorted(glob.glob(os.path.join(RES, "[0-9]*"))):
        out.append((os.path.basename(d), d, "main"))
    for d in sorted(glob.glob(os.path.join(RES, "_v1_pre_plan_fix", "[0-9]*"))):
        out.append((os.path.basename(d), d, "v1_pre_plan_fix"))
    return out

for name, pdir, sample in problem_dirs():
    w = os.path.join(pdir, "work")
    if not os.path.exists(os.path.join(w, "state.json")):
        prob_rows.append(dict(problem=name, family=family(name), sample="failed_no_state", first_us="", final_us=""))
        continue
    if sample == "main" and name.startswith("16_"): sample = "degraded"
    pid = f"{name}" if sample != "v1_pre_plan_fix" else f"{name}@v1"
    S = load(os.path.join(w, "state.json"))
    summ = load(os.path.join(w, "summary.json")) if os.path.exists(os.path.join(w, "summary.json")) else {}
    ir = next(iter(summ.get("irs", {}).values()), {})
    fused_groups = set(ir.get("fused", []) or [])
    fused_parts = set(p for g in fused_groups for p in g.split("+"))
    rounds = {r["round"]: r for r in S["rounds"]}
    # ---- per round: telemetry spend, measured model time ----
    tel_calls_cg = 0; tel_calls_all = 0
    part = {r: sum(rr["partition_us"].values()) for r, rr in rounds.items()}
    for r, rr in sorted(rounds.items()):
        llm = (((rr.get("telemetry") or {}).get("kb") or {}).get("llm") or {})
        row = dict(problem=pid, family=family(name), sample=sample, round=r, partition_us=part[r],
                   n_kb_runs=len(rr["kb_runs"]), n_decisions=len(rr["decisions"]), seconds=rr.get("seconds"),
                   budget=json.dumps((load(os.path.join(w, "r%02d" % r, "plan.json")) if os.path.exists(os.path.join(w, "r%02d" % r, "plan.json")) else {}).get("budget")))
        tot = collections.Counter()
        for br in llm.get("by_role_model", []):
            role = br["role"]
            c = dict(uncached=br.get("uncached_input_tokens", 0), cw=br.get("cache_write_input_tokens", 0),
                     cr=br.get("cache_read_input_tokens", 0), out=br.get("output_tokens", 0))
            row[f"calls_{role}"] = br.get("calls", 0); row[f"usd_{role}"] = usd(c["uncached"], c["cw"], c["cr"], c["out"])
            row[f"trunc_{role}"] = br.get("truncated", 0)
            tot["calls"] += br.get("calls", 0); tot["usd"] += row[f"usd_{role}"]
            tel_calls_all += br.get("calls", 0)
            if role.startswith(("codegen", "fix")): tel_calls_cg += br.get("calls", 0)
        row["calls"] = tot["calls"]; row["usd"] = tot["usd"]
        round_rows.append(row)
        for d in rr["decisions"]:
            dec_rows.append(dict(problem=pid, sample=sample, round=r, hint=d.get("hint"), status=d.get("status"), gain=d.get("gain"),
                                 original_us=d.get("original_us"), variant_us=d.get("variant_us"), n_ops=len(d.get("ops") or [])))
    last_round = max(rounds)
    first_us = ir.get("first_us", part.get(0)); final_us = ir.get("final_us")
    prob_rows.append(dict(problem=pid, family=family(name), sample=sample, first_us=first_us, final_us=final_us,
                          speedup=(first_us / final_us) if final_us else "", model_e2e_us=ir.get("model_e2e_us"),
                          seed_e2e_us=ir.get("seed_e2e_us"), passed=ir.get("passed"), model_passed=ir.get("model_passed"),
                          n_rounds=len(rounds), tel_calls=tel_calls_all, tel_calls_codegen_fix=tel_calls_cg,
                          **{f"part_r{r}": part[r] for r in sorted(part)}))
    # ---- kb runs ----
    jl_calls = 0; orphan_calls = 0
    rdirs = sorted(glob.glob(os.path.join(w, "r0*")))
    for rd in rdirs:
        rbase = os.path.basename(rd)
        stale = ".stale" in rbase
        r = int(rbase[1:3])
        plan_p = os.path.join(rd, "plan.json")
        plan = load(plan_p) if os.path.exists(plan_p) else {"items": []}
        decs = {d["hint"]: d for d in rounds.get(r, {}).get("decisions", [])} if not stale else {}
        kbdirs = [(d, "opt") for d in sorted(glob.glob(os.path.join(rd, "run", "kb_*")))] + \
                 [(d, "variant") for d in sorted(glob.glob(os.path.join(rd, "variants", "wave*", "kb_*")) + glob.glob(os.path.join(rd, "variants", "*", "run", "kb_*")))]
        for kd, phase in kbdirs:
            kb = os.path.basename(kd)
            jp = os.path.join(kd, "journal", "state.json"); rp = os.path.join(kd, "result.json")
            J = load(jp) if os.path.exists(jp) else {}
            R = load(rp) if os.path.exists(rp) else {}
            steps = R.get("steps") or J.get("steps_log") or []
            if J.get("steps_log") and len(J["steps_log"]) >= len(steps): steps = J["steps_log"]   # commit order, complete
            calls = []
            lp = os.path.join(kd, "journal", "llm.jsonl")
            if os.path.exists(lp):
                for ln in open(lp):
                    try: calls.append(json.loads(ln))
                    except Exception: pass
            jl_calls += len(calls)
            bykey = collections.defaultdict(list)
            for c in calls:
                m = re.match(r"r(\d+)s(\d+)c(\d+)", c.get("key", ""))
                if m: bykey[(int(m.group(1)), int(m.group(2)))].append(c)
            # slot share
            item = None; idx = None; side = None; hint = None; dec = None
            if phase == "opt":
                m = re.match(r".*_(\d+)__r\d+$", kb)
                if m:
                    idx = int(m.group(1))
                    if idx < len(plan.get("items", [])): item = plan["items"][idx]
            else:
                m = re.match(r"kb_problem__(fuse|split)_(.+)__(orig|variant)$", kb)
                if m:
                    hint = m.group(1) + ":" + m.group(2); side = m.group(3); dec = decs.get(hint)
            base_part = R.get("baseline_partition_us") or (J.get("baseline") or {}).get("partition_us")
            fin_part = R.get("final_partition_us") if R else None
            base_part = R.get("baseline_partition_us") or (J.get("baseline") or {}).get("partition_us")
            if phase == "opt" and base_part:
                # kb index == plan item index only in round 0; otherwise (or if inconsistent) match by op kind + nearest time
                kind0 = op_kind((steps[0]["slot"]) if steps else re.sub(r"_\d+__r\d+$", "", kb.replace("kb_problem__", "")))
                def ok(it): return it and it.get("op") == kind0 and 0.6 < it["total_us"] / base_part < 1.6
                if not ok(item):
                    cands = [it for it in plan.get("items", []) if it.get("op") == kind0 and it.get("total_us")]
                    item = min(cands, key=lambda it: abs(math.log(it["total_us"] / base_part))) if cands else None
                    if item and not ok(item): item = None
            # model-us of the slot problem: its own measured baseline partition (the item's ops in isolation; ~= plan item us)
            item_us = base_part if phase == "opt" else (item["total_us"] if item else None)
            model_us = part.get(r) if not stale else None
            share = (item_us / model_us) if (item_us and model_us) else None
            if phase == "variant" and dec and model_us:
                share = (dec.get("original_us") or 0) / model_us
            item_ops = [o for occ in (item or {}).get("occurrences", []) for o in occ.get("ops", [])]
            alive = [final_alive(o, fused_groups, fused_parts) for o in item_ops]
            fused_away_frac = (1 - sum(alive) / len(alive)) if alive else None
            # slot baseline at attribution level
            slots_in = sorted(set(s["slot"] for s in steps))
            base_attr = {}
            for s in steps:
                base_attr.setdefault(s["slot"], None)
            # first step's slot_us_before on rollout state = seed attr
            for s in steps:
                if s.get("step") == 0 and base_attr.get(s["slot"]) is None: base_attr[s["slot"]] = s.get("slot_us_before")
            for s in steps:
                if base_attr.get(s["slot"]) is None: base_attr[s["slot"]] = s.get("slot_us_before")
            scale = None
            if phase == "opt" and item_us and base_part:
                scale = 1.0       # partition-us == model-us of the slot's ops (multi-occurrence items included)
            best = dict(base_attr)
            seen = set(); kb_usd = 0.0; kb_calls = 0
            rows_kb = []
            for order, s in enumerate(steps):
                ro, st = s.get("rollout"), s.get("step")
                cs = bykey.get((ro, st), []); seen.add((ro, st))
                ctx = s.get("context") or {}
                tl = (ctx.get("telemetry") or {}).get("llm") or {}
                pt = sum((c.get("usage") or {}).get("prompt_tokens", 0) or 0 for c in cs)
                out = sum((c.get("usage") or {}).get("completion_tokens", 0) or 0 for c in cs)
                cr = sum((c.get("usage") or {}).get("cache_read_input_tokens", 0) or 0 for c in cs)
                cw = sum((c.get("usage") or {}).get("cache_creation_input_tokens", 0) or 0 for c in cs)
                src = "journal"
                if not cs and tl.get("calls"):
                    # fall back on the step's own telemetry (no cache-write split: assume prompt - cache_read was written)
                    pt, out, cr = tl.get("in", 0), tl.get("out", 0), tl.get("cache_read", 0); cw = max(pt - cr, 0); src = "step_telemetry"
                unc = max(pt - cr - cw, 0)
                ncalls = len(cs) if cs else (tl.get("calls") or 0)
                nfix = sum(1 for c in cs if str(c.get("role", "")).startswith("fix"))
                def _cu(c):
                    u = c.get("usage") or {}; p_, o_ = u.get("prompt_tokens", 0) or 0, u.get("completion_tokens", 0) or 0
                    r_, w_ = u.get("cache_read_input_tokens", 0) or 0, u.get("cache_creation_input_tokens", 0) or 0
                    return usd(max(p_ - r_ - w_, 0), w_, r_, o_)
                fix_usd = sum(_cu(c) for c in cs if str(c.get("role", "")).startswith("fix"))
                cost = usd(unc, cw, cr, out)
                kb_usd += cost; kb_calls += ncalls
                slot = s.get("slot"); kind = op_kind(slot)
                status = s.get("status")
                b0 = best.get(slot)
                after = s.get("slot_us_after")
                new_best = bool(status == "improved" and after is not None and b0 is not None and after < b0)
                if new_best: best[slot] = after
                delta = (b0 - best[slot]) if new_best else 0.0
                tech = s.get("technique", "")
                rows_kb.append(dict(
                    problem=pid, family=family(name), sample=("stale" if stale else sample), round=r, phase=phase, kb_run=kb,
                    kb_index=idx, variant_side=side, hint=hint, slot=slot, op_kind=kind, op_class=op_class(kind),
                    share=share, item_us=item_us, order=order, rollout=ro, step=st, kind=ctx.get("kind"),
                    technique=tech, technique_group=("hypothesis" if tech.startswith("h_") else tech),
                    role=ctx.get("model_role"), model=ctx.get("model"), explore=ctx.get("explore"),
                    status=status, outcome=("failed" if status in ("compile_error", "wrong_output", "runtime_error") else status),
                    fix_attempts=s.get("fix_attempts", 0), needed_fix=int((s.get("fix_attempts") or 0) > 0),
                    n_calls=ncalls, n_fix_calls=nfix, prompt_tokens=pt, output_tokens=out, cache_read=cr, cache_write=cw,
                    uncached=unc, usd=cost, fix_usd=fix_usd, truncated=tl.get("truncated", 0), tok_src=src, seconds=s.get("seconds"),
                    llm_latency_s=tl.get("latency_s"),
                    slot_us_before=s.get("slot_us_before"), slot_us_after=after, slot_base_us=base_attr.get(slot),
                    best_before=b0, best_after=best[slot], new_best=int(new_best), delta_best_us=delta,
                    model_effect_us=(delta * scale) if (scale and phase == "opt") else (0.0 if phase == "opt" else None),
                    fused_away_frac=fused_away_frac, ts=(min(c.get("ts", 0) for c in cs) if cs else None)))
            # orphan calls (no committed step)
            for (ro, st), cs in bykey.items():
                if (ro, st) in seen: continue
                pt = sum((c.get("usage") or {}).get("prompt_tokens", 0) or 0 for c in cs)
                out = sum((c.get("usage") or {}).get("completion_tokens", 0) or 0 for c in cs)
                cr = sum((c.get("usage") or {}).get("cache_read_input_tokens", 0) or 0 for c in cs)
                cw = sum((c.get("usage") or {}).get("cache_creation_input_tokens", 0) or 0 for c in cs)
                unc = max(pt - cr - cw, 0); cost = usd(unc, cw, cr, out)
                orphan_calls += len(cs); kb_usd += cost; kb_calls += len(cs)
                plan_tech = ""
                try: plan_tech = json.loads(cs[0].get("plan") or "{}").get("tech", "")
                except Exception: pass
                slot = None
                try: slot = json.loads(cs[0].get("plan") or "{}").get("sid")
                except Exception: pass
                kind = op_kind(slot or (slots_in[0] if slots_in else ""))
                rows_kb.append(dict(problem=pid, family=family(name), sample=("stale" if stale else sample), round=r, phase=phase, kb_run=kb,
                                    kb_index=idx, variant_side=side, hint=hint, slot=slot, op_kind=kind, op_class=op_class(kind), share=share,
                                    item_us=item_us, order=None, rollout=ro, step=st, kind=None, technique=plan_tech,
                                    technique_group=("hypothesis" if plan_tech.startswith("h_") else plan_tech), role=cs[0].get("role"),
                                    model=cs[0].get("model"), status="orphan", outcome="orphan", fix_attempts=None, needed_fix=0,
                                    n_calls=len(cs), n_fix_calls=sum(1 for c in cs if str(c.get("role", "")).startswith("fix")),
                                    prompt_tokens=pt, output_tokens=out, cache_read=cr, cache_write=cw, uncached=unc, usd=cost,
                                    truncated=None, tok_src="journal", model_effect_us=(0.0 if phase == "opt" else None), new_best=0,
                                    delta_best_us=0.0, fused_away_frac=fused_away_frac, ts=min(c.get("ts", 0) for c in cs)))
            # mark the step that produced the final best of each slot
            last_nb = {}
            for i, rw in enumerate(rows_kb):
                if rw.get("new_best"): last_nb[rw["slot"]] = i
            for i, rw in enumerate(rows_kb):
                rw["is_final_best"] = int(last_nb.get(rw["slot"]) == i)
            steps_rows.extend(rows_kb)
            fb = {sl: best[sl] for sl in best}
            kb_rows.append(dict(problem=pid, family=family(name), sample=("stale" if stale else sample), round=r, phase=phase, kb_run=kb,
                                kb_index=idx, variant_side=side, hint=hint, decision=(dec or {}).get("status"),
                                slots=";".join(slots_in), op_kind=op_kind(slots_in[0]) if len(slots_in) == 1 else ("multi" if slots_in else ""),
                                op_class=op_class(op_kind(slots_in[0])) if len(slots_in) == 1 else ("multi" if slots_in else ""),
                                item_us=item_us, model_us_round=model_us, share=share, item_count=(item or {}).get("count"),
                                baseline_partition_us=base_part, final_partition_us=fin_part, final_ok=R.get("final_ok") if R else None,
                                has_result=int(bool(R)), n_steps=len(steps), n_calls=kb_calls, usd=kb_usd,
                                base_attr_us=sum(v for v in base_attr.values() if v) if base_attr else None,
                                best_attr_us=sum(v for v in fb.values() if v) if fb else None, scale=scale,
                                fused_away_frac=fused_away_frac, n_ops=len(item_ops)))
    join["per_problem"][pid] = dict(sample=sample, tel_calls_all=tel_calls_all, tel_calls_codegen_fix=tel_calls_cg,
                                    journal_calls=jl_calls, orphan_calls=orphan_calls)

def write(fn, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(os.path.join(OUT, fn), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader()
        for r in rows: wr.writerow(r)

write("steps.csv", steps_rows); write("kbruns.csv", kb_rows); write("rounds.csv", round_rows)
write("problems.csv", prob_rows); write("decisions.csv", dec_rows)
tot = collections.Counter()
for v in join["per_problem"].values():
    for k in ("tel_calls_all", "tel_calls_codegen_fix", "journal_calls", "orphan_calls"): tot[k] += v[k]
join["totals"] = dict(tot)
join["step_rows"] = sum(1 for r in steps_rows if r["status"] != "orphan")
join["step_calls"] = sum(r["n_calls"] or 0 for r in steps_rows)
join["step_usd"] = sum(r["usd"] for r in steps_rows)
join["tel_usd"] = sum(r["usd"] for r in round_rows)
json.dump(join, open(os.path.join(OUT, "join_check.json"), "w"), indent=1)
print(json.dumps({k: join[k] for k in ("totals", "step_rows", "step_calls", "step_usd", "tel_usd")}, indent=1))
