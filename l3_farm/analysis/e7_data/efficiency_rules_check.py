#!/usr/bin/env python3
"""X-129 consistency check: the IMPLEMENTED efficiency rules applied offline to the logged sweep trajectories
must reproduce the E7 replay (e7_analyze.py, policy "patience 8 + min_share 0.005 in rounds >= 1") per problem.

* R1 uses KernelBlaster's own ``patience_trigger`` (agents/opt_multislot.py) on each slot problem's committed steps
  in commit order (steps.csv ``order``); the kept steps are the commits up to and including the trigger.
* R2 uses ModelBlaster's ``kb_efficiency.round_min_share`` on the slot problem's share (kbruns.csv ``share``).
* The $ / loss accounting is e7_analyze.simulate's (skipped steps' journal $, knowledge-role $ pro rata per round,
  loss = best over kept steps - logged best, weighted by 1 - fused_away_frac), re-implemented here so that this
  script does not rewrite e7_results.json / sim_grid.csv / the plots.

Compares with sim_grid.csv (the replay's per-problem numbers) and prints per-problem and total differences.
Stdlib only:  python3 efficiency_rules_check.py [--problems 1_MLP,4_LeNet5,...] [--patience 8 --late 0.005]
"""
import argparse, collections, csv, importlib.util, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
MB = os.path.expanduser("~/Documents/ModelBlaster-clone/ModelBlaster")
sys.path.insert(0, os.path.join(MB, "third_party", "KernelBlaster", "src"))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def patience_fn():
    """KernelBlaster's patience_trigger without importing the whole agent (loguru/telemetry deps):
    the function's source is executed with the outcome constants it uses."""
    import ast
    src = open(os.path.join(MB, "third_party/KernelBlaster/src/kernelblaster/agents/opt_multislot.py")).read()
    tree = ast.parse(src)
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("_sf", "patience_trigger")]
    import typing
    ns = {"IMPROVED": "improved", "Optional": typing.Optional}
    exec(compile(ast.Module(body=fns, type_ignores=[]), "opt_multislot.py", "exec"), ns)
    return ns["patience_trigger"]


rd = lambda fn: list(csv.DictReader(open(os.path.join(HERE, fn))))
F = lambda x, d=None: (float(x) if x not in ("", None) else d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="", help="comma list (default: all 46 replay problems)")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--late", type=float, default=0.005)
    ap.add_argument("--json", default="")
    ap.add_argument("--inflight", action="store_true",
                    help="also keep the steps in flight when patience fires (started before the trigger committed: "
                         "steps.csv ts + seconds), as the live flow does; the replay drops them")
    a = ap.parse_args()
    trig = patience_fn()
    eff = _load(os.path.join(MB, "pipeline", "kb_efficiency.py"), "kb_efficiency")
    steps, kbr, rounds, probs = rd("steps.csv"), rd("kbruns.csv"), rd("rounds.csv"), rd("problems.csv")
    grid = rd("sim_grid.csv")
    want = dict(fc_cap="4", min_share=str(a.late), ms_from="1", patience=str(a.patience), n_ro="4", s_cap="4",
                round_cap="", eps="", v_steps="2")
    ref = {r["problem"]: r for r in grid if all(r[k] == v for k, v in want.items())}
    prob = {p["problem"]: p for p in probs if p["sample"] == "main"}
    replay = [p for p in prob if p not in ("10_ResNet101", "22_EfficientNetB0")]
    sel = [p for p in a.problems.split(",") if p] or sorted(replay, key=lambda n: int(n.split("_")[0]))
    know = {(r["problem"], int(r["round"])): sum(F(r.get(f"usd_{x}"), 0) for x in ("reflect", "hypothesis", "discovery"))
            for r in rounds}
    by_kb = collections.defaultdict(list)
    for s in steps:
        if s["status"] != "orphan":
            by_kb[(s["problem"], s["round"], s["kb_run"])].append(s)
    rows, tot = [], collections.Counter()
    for p in sel:
        nR = int(prob[p]["n_rounds"])
        kbs = [k for k in kbr if k["problem"] == p and k["sample"] == "main" and int(k["round"]) < nR]
        cg_round, cg_saved = collections.Counter(), collections.Counter()
        loss, n_stop, n_skip = 0.0, 0, 0
        ptot = sum(F(k["usd"], 0) for k in kbs) + sum(know.get((p, r), 0) for r in range(nR))
        for k in kbs:
            r = int(k["round"])
            cg_round[r] += F(k["usd"], 0)
            if k["phase"] != "opt":
                continue
            st = sorted(by_kb.get((p, k["round"], k["kb_run"]), []), key=lambda s: int(s["order"]))
            rec = [dict(slot=s["slot"], step=int(s["step"]), status=s["status"], slot_us_before=F(s["slot_us_before"]),
                        slot_us_after=F(s["slot_us_after"])) for s in st]
            base = {}
            for s in st:
                base.setdefault(s["slot"], F(s["slot_base_us"]))
            fa = F(k["fused_away_frac"])
            w = (1 - fa) if fa is not None else 1.0

            def best_of(lst):
                b = dict(base)
                for s in lst:
                    if s["status"] == "improved" and s["slot_us_after"] is not None and b.get(s["slot"]) is not None \
                            and s["slot_us_after"] < b[s["slot"]]:
                        b[s["slot"]] = s["slot_us_after"]
                return b
            full = best_of(rec)
            share = F(k["share"], 0.0)
            if share < eff.round_min_share(0.0, a.late, r):          # R2: skipped slot problem
                cg_saved[r] += F(k["usd"], 0)
                loss += w * sum(max(base[s] - full[s], 0) for s in base if base[s] is not None)
                n_skip += 1
                continue
            t = trig(rec, a.patience)                                # R1: commits up to the trigger
            keep_idx = list(range(len(rec))) if t is None else list(range(t["trigger_commit"]))
            if t is not None and a.inflight:
                # live flow: steps of the other rollouts already started when the trigger committed still finish
                tr = st[t["trigger_commit"] - 1]
                t_commit = (F(tr["ts"]) or 0) + (F(tr["seconds"]) or 0)
                keep_idx += [i for i in range(t["trigger_commit"], len(st))
                             if F(st[i]["ts"]) is not None and F(st[i]["ts"]) < t_commit]
            kept = [rec[i] for i in keep_idx]
            n_stop += t is not None
            cg_saved[r] += sum(F(s["usd"], 0) for s in st) - sum(F(st[i]["usd"], 0) for i in keep_idx)
            bk = best_of(kept)
            loss += w * sum(max(bk[s] - full[s], 0) for s in base if base[s] is not None)
        saved = sum(cg_saved[r] + (know.get((p, r), 0) * cg_saved[r] / cg_round[r] if cg_round[r] > 0 else 0)
                    for r in range(nR))
        g = ref.get(p)
        rs, rl = (F(g["saved_usd"]), F(g["loss_us"])) if g else (None, None)
        final = F(prob[p]["final_us"])
        rows.append(dict(problem=p, saved_usd=round(saved, 4), replay_saved_usd=rs, total_usd=round(ptot, 4),
                         loss_us=round(loss, 4), replay_loss_us=rl, loss_pct=round(100 * (1 - final / (final + loss)), 3),
                         slot_problems_stopped=n_stop, slot_problems_skipped=n_skip))
        tot["saved"] += saved; tot["ref"] += rs or 0; tot["total"] += ptot
        tot["logret"] += math.log(final / (final + loss)); tot["n"] += 1
        if rl is not None:
            tot["logret_ref"] += math.log(final / (final + rl))
    w = max(len(r["problem"]) for r in rows)
    print(f"{'problem':{w}}  saved$  replay$   diff$  loss_us replay_us loss%  stopped skipped")
    worst = 0.0
    for r in rows:
        d = r["saved_usd"] - (r["replay_saved_usd"] or 0)
        worst = max(worst, abs(d) / max(r["total_usd"], 1e-9))
        print(f"{r['problem']:{w}} {r['saved_usd']:7.2f} {r['replay_saved_usd'] or 0:7.2f} {d:7.3f} "
              f"{r['loss_us']:8.2f} {r['replay_loss_us'] or 0:9.2f} {r['loss_pct']:5.2f} {r['slot_problems_stopped']:8d} "
              f"{r['slot_problems_skipped']:7d}")
    print(f"total: implemented saves {tot['saved']:.2f} $ = {100 * tot['saved'] / tot['total']:.2f}% of {tot['total']:.2f} $; "
          f"replay {tot['ref']:.2f} $ = {100 * tot['ref'] / tot['total']:.2f}%; worst per-problem |diff| "
          f"{100 * worst:.3f}% of the problem's spend")
    print(f"geomean speedup lost: implemented {100 * (1 - math.exp(tot['logret'] / tot['n'])):.2f}%, replay "
          f"{100 * (1 - math.exp(tot['logret_ref'] / tot['n'])):.2f}%")
    if a.json:
        json.dump(dict(params=vars(a), rows=rows, totals=dict(tot)), open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
