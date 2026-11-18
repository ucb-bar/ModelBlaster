#!/usr/bin/env python3
"""X-137: fused-slot winners dropped at ingest (`fusion-not-accepted`), sweep-wide table + regression assertion.

  check_fusion_drop.py table [--results DIR] [--out PREFIX]
        Offline, free. Reads results/<p>/work/r*/round.json (ingest statuses) and r*/winners/winners.json (the winners
        behind them, same order), for every sweep problem. Per problem: ingest entries, `fusion-not-accepted` entries,
        how many of those are fused kinds that are SLOTS OF THE FINAL MODEL (final manifest) and whose deployed kernel is not the
        dropped winner (sha1 of the final slot file != winner code_sha), and an estimate of the lost time
        (sum over such winners of baseline_us - slot_us, the winner's own measured gain over the incumbent it was
        compared with; best winner per (op, signature) only). Writes PREFIX.json / PREFIX.md.
  check_fusion_drop.py assert --work DIR [--problem P ...] [--min-fused N]
        For a NEW run (a flow work dir, or a collected pack dir with work/): counts fusion-not-accepted per round and
        exits 1 if any is found. With --min-fused N it also requires >= N fused ingest entries that were added/replaced/
        kept (so a run that never produced a fused winner does not pass vacuously).
  check_fusion_drop.py reference
        The sweep's reference counts for 19_MobileNetV1 and 8_ResNetBasicBlock (the numbers a post-merge run must beat).

Estimate caveats: slot_us is the winner's slot time as measured in its KernelBlaster problem (per slot, not multiplied by the
number of instances); `baseline_us` is the incumbent at the start of that run. The estimate is an indicative upper bound on
what ingesting the dropped winner would have saved per call of the slot; it is not a re-measurement. The exact figure needs
the E8/E1 recomposer on a node (zero LLM): see RUN_AFTER_MERGE.md ("recompose dropped winners").
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mc_common as C  # noqa: E402

DROP = "fusion-not-accepted"
REF = ["19_MobileNetV1", "8_ResNetBasicBlock"]


def jl(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def final_model(work: Path):
    """(model_e2e_us, seed_e2e_us, {slot id: sha1 of final slot file}, {slot id: n instances})."""
    mr = jl(work / "model_runtime.json") or {}
    mu = next(iter(mr.values()), {}) if mr else {}
    slots, inst = {}, {}
    for mf in sorted((work / "final").glob("*/problem/manifest.json")):
        m = jl(mf) or {}
        for s in m.get("slots", []):
            f = mf.parent / s.get("file", "")
            if f.is_file():
                slots[s["id"]] = hashlib.sha1(f.read_bytes()).hexdigest()
            inst[s["id"]] = len(s.get("instances") or [])
    return mu.get("model_e2e_us"), mu.get("seed_e2e_us"), slots, inst


def scan_problem(pdir: Path) -> dict:
    work = pdir / "work"
    model_us, seed_us, fslots, finst = final_model(work)
    rows, tot = [], {"ingest": 0, "dropped": 0, "fused_ingested": 0}
    est_by_op = {}
    seed_based = []
    for rj in sorted(work.glob("r[0-9][0-9]/round.json")):
        r = jl(rj) or {}
        ing = r.get("ingest") or []
        wj = jl(rj.parent / "winners" / "winners.json") or {}
        ws = wj.get("winners") or []
        aligned = len(ws) == len(ing)
        dropped = [(i, x) for i, x in enumerate(ing) if x.get("status") == DROP]
        fused_ok = [x for x in ing if str(x.get("op", "")).startswith("fused__") and x.get("status") != DROP]
        rel = []
        for i, x in dropped:
            w = ws[i] if aligned and i < len(ws) else None
            op = x.get("op")
            if w is not None and w.get("op") != op:
                w = None
            in_final = op in fslots
            deployed_is_winner = bool(w and in_final and fslots[op] == w.get("code_sha"))
            gain = None
            if w and w.get("slot_us") is not None and w.get("baseline_us") is not None:
                gain = max(0.0, float(w["baseline_us"]) - float(w["slot_us"]))
                # baseline_us far above slot_us (> 3x) means the run started from the unoptimised fused seed, not from the kernel the
                # final model deploys: the difference says nothing about the loss, so it is listed but not summed
                if float(w["baseline_us"]) > 3.0 * float(w["slot_us"]):
                    seed_based.append(op)
                    gain = None
            if in_final and not deployed_is_winner and gain:
                key = (op, w.get("signature_sha"))
                if gain > est_by_op.get(key, (0.0,))[0]:
                    est_by_op[key] = (gain, rj.parent.name, w.get("code_sha", "")[:8], w.get("slot_us"), w.get("baseline_us"),
                                      finst.get(op))
            rel.append({"op": op, "in_final_model": in_final, "deployed_is_winner": deployed_is_winner, "gain_us": gain,
                        "slot_us": (w or {}).get("slot_us"), "baseline_us": (w or {}).get("baseline_us")})
        rows.append({"round": rj.parent.name, "ingest": len(ing), "dropped": len(dropped), "fused_kept": len(fused_ok),
                     "aligned": aligned, "dropped_detail": rel})
        tot["ingest"] += len(ing)
        tot["dropped"] += len(dropped)
        tot["fused_ingested"] += len(fused_ok)
    est = round(sum(v[0] for v in est_by_op.values()), 2)
    return {"problem": pdir.name, "rounds": rows, **tot, "model_e2e_us": model_us, "seed_e2e_us": seed_us,
            "n_relevant": len(est_by_op), "n_in_final_vs_seed_baseline": len(seed_based), "est_lost_us": est,
            "est_lost_pct_of_model": round(100 * est / model_us, 2) if (model_us and est) else 0.0,
            "relevant": [{"op": k[0], "round": v[1], "winner_sha": v[2], "slot_us": v[3], "baseline_us": v[4],
                          "gain_us": round(v[0], 2), "final_instances": v[5]} for k, v in est_by_op.items()]}


def table(results: Path, out_prefix: Path):
    ps = sorted((d for d in results.iterdir() if d.is_dir() and d.name[0].isdigit() and (d / "work").is_dir()),
                key=lambda d: int(d.name.split("_")[0]))
    data = [scan_problem(d) for d in ps]
    meta = {"generated": C.now(), "results": str(results), "tree": "n/a (offline, reads sweep results; sweep tree stamp 20261025-123252)",
            "flags": "sweep defaults (worker4.sh)", "job": "n/a", "host": "n/a", "mem_clock_mhz": "n/a"}
    for r in data:
        r["all_dropped"] = r["dropped"]
    tot_i = sum(r["ingest"] for r in data)
    tot_d = sum(r["dropped"] for r in data)
    summary = {"problems": len(data), "problems_with_drops": sum(1 for r in data if r["dropped"]),
               "ingest_entries": tot_i, "dropped_entries": tot_d,
               "dropped_relevant_to_final_model": sum(r["n_relevant"] for r in data)}
    C.write_json(out_prefix.with_suffix(".json"), {"meta": meta, "summary": summary, "problems": data})
    rows = []
    for r in data:
        rd = "; ".join(f"{x['round']}:{x['dropped']}/{x['ingest']}" for x in r["rounds"] if x["dropped"])
        rows.append([r["problem"], r["ingest"], r["dropped"], r["fused_ingested"], rd or "-", f"{r['n_relevant']}" + (f" (+{r['n_in_final_vs_seed_baseline']} seed-based, not estimable)" if r["n_in_final_vs_seed_baseline"] else ""),
                     r["est_lost_us"] or "-", f"{r['est_lost_pct_of_model']}%" if r["est_lost_us"] else "-",
                     r["model_e2e_us"]])
    md = C.md_head("X-137 fused-slot winners dropped at ingest, Level 3 sweep (48+ problems)", meta)
    md += (f"\n{summary['problems_with_drops']} of {summary['problems']} problems dropped at least one winner; "
           f"{tot_d} of {tot_i} ingest entries were `{DROP}`; {summary['dropped_relevant_to_final_model']} (op, signature) pairs are fused "
           f"slots of the final model whose deployed kernel is not the dropped winner.\n\n")
    md += C.md_table(rows, ["problem", "ingest entries", "dropped", "fused kept", "dropped by round (dropped/ingest)",
                            "dropped ops in final model (with incremental estimate)", "est. lost us (upper bound)", "% of final model", "final model us"])
    md += ("\n\nEstimate: sum of (baseline_us - slot_us) of the best dropped winner per (op, signature) whose fused slot is in the final "
           "model with a different kernel, only where the winner's baseline is within 3x of its result (a run that started from the unoptimised "
           "fused seed says nothing about the loss; those are counted as 'seed-based'). It is per slot call, not multiplied by instances, "
           "and not re-measured, so it is an indicative bound, not a measurement. Dropped winners of intermediate fused kinds that a later "
           "chain superseded (e.g. 19_MobileNetV1: conv+bn, bn+relu -> conv+bn+relu) are not in the final model and cost nothing directly, "
           "but they were the seeds for the chain.\n")
    out_prefix.with_suffix(".md").write_text(md)
    print(md[:3000])
    print("\nwritten", out_prefix.with_suffix(".json"), out_prefix.with_suffix(".md"))
    return data


def count_work(work: Path) -> dict:
    per, dropped, fused = {}, 0, 0
    for rj in sorted(work.glob("r[0-9][0-9]/round.json")):
        ing = (jl(rj) or {}).get("ingest") or []
        d = sum(1 for x in ing if x.get("status") == DROP)
        f = sum(1 for x in ing if str(x.get("op", "")).startswith("fused__") and x.get("status") != DROP)
        per[rj.parent.name] = {"ingest": len(ing), "dropped": d, "fused_kept": f}
        dropped += d
        fused += f
    return {"work": str(work), "rounds": per, "dropped": dropped, "fused_kept": fused}


def assert_run(work: Path, min_fused: int):
    w = work / "work" if (work / "work").is_dir() else work
    c = count_work(w)
    ok = c["dropped"] == 0 and c["fused_kept"] >= min_fused
    print(json.dumps(c, indent=1))
    print("PASS" if ok else f"FAIL (dropped={c['dropped']}, fused_kept={c['fused_kept']}, need >= {min_fused})")
    return ok, c


def reference(results: Path):
    out = {}
    for p in REF:
        s = scan_problem(results / p)
        out[p] = {"ingest": s["ingest"], "dropped": s["dropped"], "by_round": {r["round"]: f"{r['dropped']}/{r['ingest']}" for r in s["rounds"]},
                  "model_e2e_us": s["model_e2e_us"], "est_lost_us": s["est_lost_us"]}
    print(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table")
    t.add_argument("--results", default=str(C.RES))
    t.add_argument("--out", default=str(HERE / "fusion_drop_table"))
    a = sub.add_parser("assert")
    a.add_argument("--work", required=True)
    a.add_argument("--min-fused", type=int, default=1)
    a.add_argument("--json-out")
    sub.add_parser("reference")
    x = ap.parse_args()
    if x.cmd == "table":
        table(Path(x.results), Path(x.out))
    elif x.cmd == "assert":
        ok, c = assert_run(Path(x.work), x.min_fused)
        if x.json_out:
            C.write_json(x.json_out, {"pass": ok, **c})
        sys.exit(0 if ok else 1)
    else:
        reference(C.RES)


if __name__ == "__main__":
    main()
