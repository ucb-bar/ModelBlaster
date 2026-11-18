#!/usr/bin/env python3
"""X-136: flag matrix on 43_MinGPTCausalAttention (stable, CV 0.1%) and 41_GRUBidirectional (CV 17%: NOT a single-problem testbed).

Arms (every flag set only if the tree has the Config field; names stay stable across trees):
  control   no flag                     reuse    --reuse-winners siblings        conc     --graph-concurrency auto
  fusext    --fusion-extended all       fixes    --final-guard repair --seed-nearest on --rollout-accept lineage
  gsearch   --graph-search interleaved  a1       reuse + conc + fusext           a1_a3    a1 + fixes
  a1_a4     a1 + gsearch                all      a1 + fixes + gsearch
Each arm runs in two flavours:
  live_<arm>   the real flow (LLM), farm budget (3 rounds x 4 rollouts x 4 steps), global knowledge off, fixed --seed
  stub_<arm>   zero LLM (`stub:graphopt`, every slot returned unchanged), small budget: exercises the flag's code path, round keys,
               graph decisions and COMPOSITION effects (concurrency, fusion proposals, deploy re-timing) on the seed kernels, for $0
Plus `recompose` (zero LLM, E1/E8 recomposer): rebuild the sweep's final programs (and the reuse / concurrency variants) on a node with
the tree under test, to separate composition effects from search effects.

  check_flag_matrix.py plan       [--tarball T] [--problems ...] [--arms ...] [--seeds 1]   commands + cost estimate, nothing runs
  check_flag_matrix.py run        JOB TARBALL --flavour stub|live [--problems ...] [--arms ...] [--seeds 1]
  check_flag_matrix.py remeasure  JOB TARBALL_TAG_LIST...     re-time every collected final of the listed labels on ONE node, interleaved
                                  (2 passes), so node effects cancel; writes rem_<tag>.json in each run dir
  check_flag_matrix.py analyze    [--labels post ...] [--out PREFIX]    table: final / quiet / re-measured us, vs control, LLM calls, $, keys
  check_flag_matrix.py recompose  JOB TARBALL [--problems 41_GRUBidirectional 43_MinGPTCausalAttention]   (see docstring of recompose())

Memory clock, node job id, host, tree stamp / sha256 and the exact command line are in every run's meta.json and in the report.
Compare arms on the SAME node or use `remeasure`: a node effect of ~10% (E0: 43 control C) is larger than most flag effects.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mc_common as C  # noqa: E402
import mc_runner as R  # noqa: E402
import check_default_unchanged as DU  # noqa: E402  (supported())

PROBLEMS = ["43_MinGPTCausalAttention", "41_GRUBidirectional"]
FLAGSETS = {
    "reuse": [("reuse_winners", "--reuse-winners siblings")],
    "conc": [("graph_concurrency", "--graph-concurrency auto")],
    "fusext": [("fusion_extended", "--fusion-extended all")],
    "fixes": [("final_guard", "--final-guard repair"), ("seed_nearest", "--seed-nearest on"), ("rollout_accept", "--rollout-accept lineage")],
    "gsearch": [("graph_search", "--graph-search interleaved")],
}
COMBOS = {"control": [], "reuse": ["reuse"], "conc": ["conc"], "fusext": ["fusext"], "fixes": ["fixes"], "gsearch": ["gsearch"],
          "a1": ["reuse", "conc", "fusext"], "a1_a3": ["reuse", "conc", "fusext", "fixes"],
          "a1_a4": ["reuse", "conc", "fusext", "gsearch"], "all": ["reuse", "conc", "fusext", "fixes", "gsearch"]}
STUB_BUDGET = "--rounds 2 --rollouts 1 --steps 1"
# rough $ per live run at the farm budget (E0: 43 ~58, 41 ~38), used only by `plan`
EST_USD = {"43_MinGPTCausalAttention": 58, "41_GRUBidirectional": 38}
TIERS = {"1": ["control", "a1", "fixes", "gsearch", "all"], "2": list(COMBOS)}


def tree_fields(tarball) -> set:
    txt = ""
    with tarfile.open(tarball) as t:
        for m in t:
            if m.name.endswith("pipeline/kb_graph_opt.py"):
                txt = t.extractfile(m).read().decode()
                break
    cfg = txt.split("class Config", 1)[-1].split("\ndef ", 1)[0]
    return set(re.findall(r"^\s+([a-z_0-9]+)\s*:", cfg, re.M))


def arm_flags(tarball, arm) -> tuple[str, list]:
    fields = tree_fields(tarball)
    flags, missing = [], []
    for fs in COMBOS[arm]:
        for field_, flag in FLAGSETS[fs]:
            if field_ in fields or field_ == "rollout_accept" and "rollout_accept" in fields:
                flags.append(flag)
            else:
                missing.append(flag)
    return " ".join(flags), missing


def plan(a):
    rows, tot = [], 0.0
    tb = a.tarball
    for p in a.problems:
        for arm in a.arms:
            fl, miss = arm_flags(tb, arm) if tb else (" ".join(f for fs in COMBOS[arm] for _, f in FLAGSETS[fs]), [])
            for s in a.seeds:
                est = EST_USD.get(p, 50)
                tot += est
                rows.append([p, arm, s, fl or "(none)", f"~${est}", ", ".join(miss) or ""])
    print(C.md_table(rows, ["problem", "arm", "seed", "flags", "live cost", "flags missing in this tree"]))
    print(f"\nlive total ~${tot:.0f} (flow cost per run; stub arms are $0). Per-arm node time 1-2 h; one node per arm, or run arms back to back"
          " on one node and use `remeasure`. Tier 1 = " + ",".join(TIERS["1"]) + ".")


def run(a):
    for p in a.problems:
        for arm in a.arms:
            fl, miss = arm_flags(a.tarball, arm)
            if miss:
                print(f"NOTE {arm}: flags not in this tree, skipped: {miss}")
            stub = a.flavour == "stub"
            if stub:
                fl = (fl + " --torch-baseline off").strip()      # zero-LLM arms skip the slow torch.compile comparison
            R.launch(a.job, a.tarball, f"{a.flavour}_{arm}", p, a.seeds[0], stub=stub, flags=fl, envs="",
                     budget=STUB_BUDGET if stub else R.STD_BUDGET, th=a.th)
            time.sleep(1)


def runs_of(labels):
    out = []
    for lab in labels:
        for m in sorted((C.RUNS / lab).glob("*/*/meta.json")):
            out.append(m.parent)
    return out


def remeasure(a):
    """Upload each collected final (cache + hints + problem.py) to this node and re-time them in `passes` interleaved passes with the
    tree each was produced with (tarballs of all labels must be installed on the node: `mc_runner.py install`)."""
    dirs = [d for d in runs_of(a.labels) if d.parent.name.startswith("live_") or a.include_stub]
    tasks = []
    for d in dirs:
        meta = C.load_json(d / "meta.json", {})
        lab = meta["tree"]["label"]
        remote = f"{C.NODE_ROOT}/rem/{d.relative_to(C.RUNS)}"
        # upload the work cache + hints + problem
        import io
        b = io.BytesIO()
        with tarfile.open(fileobj=b, mode="w:gz") as t:
            for rel in ("work/cache", "work/graph_hints.json"):
                if (d / rel).exists():
                    t.add(d / rel, arcname=rel)
        C.put(a.job, b.getvalue(), f"{remote}/final.tar.gz")
        C.put(a.job, C.L3 / "problems" / f"{meta['problem']}.py", f"{remote}/problem.py")
        rc, o, e = C.rsh(a.job, f"cd {remote} && tar xzf final.tar.gz && ls work | head -3", timeout=300)
        conc = "on" if "--graph-concurrency" in (meta.get("flags") or "") else "off"
        rs = C.load_json(d / "work" / "summary.json", {}) or {}
        cl = rs.get("cuda_learnings") or {}
        conc = cl.get("conc_auto_choice") or ("on" if cl.get("graph_concurrency") in ("on", "auto") else "off")
        tasks.append((d, meta, lab, remote, conc))
    res = {}
    for ps in range(a.passes):
        for d, meta, lab, remote, conc in tasks:
            script = (f"export CUDA_N={C.CUDA_N}; export ENVS={shlex.quote(meta.get('envs') or '')}\n" +
                      (C.HERE / "remote" / "mc_remeasure.sh").read_text())
            rc, o, e = C.rsh(a.job, script, [lab, remote, meta["problem"], conc, f"p{ps}"], timeout=2400)
            print(d.relative_to(C.RUNS), f"pass {ps}:", o.strip().replace("\n", " | "))
            m = re.search(r"model_e2e_us (\S+)", o)
            val = float(m.group(1)) if m and m.group(1) != "None" else None
            res.setdefault(str(d.relative_to(C.RUNS)), []).append(val)
            C.write_json(d / f"rem_p{ps}.json", {"model_e2e_us": val, "job": a.job, "tree": lab, "pass": ps, "generated": C.now()})
    rc, o, e = C.rsh(a.job, "hostname; nvidia-smi --query-gpu=clocks.max.mem,vbios_version --format=csv,noheader")
    out = {"job": a.job, "node": o.strip(), "passes": a.passes, "generated": C.now(), "model_e2e_us": res}
    C.write_json(HERE / f"remeasure_{a.job}_{int(time.time())}.json", out)
    for d, *_ in tasks:
        pass
    print(json.dumps(out, indent=1))


def analyze(a):
    rows, data = [], []
    for d in runs_of(a.labels):
        meta = C.load_json(d / "meta.json", {})
        rr = C.read_run(d)
        arm = d.parent.name
        flavour, _, armname = arm.partition("_")
        rem = [v for f in sorted(Path(d).glob("rem_p*.json")) for v in [C.load_json(f, {}).get("model_e2e_us")] if v]
        data.append({"label": d.parts[-3], "flavour": flavour, "arm": armname, "problem": meta.get("problem"), "seed": meta.get("seed"),
                     "final_us": rr["model_e2e_us"], "quiet_us": rr["quiet_e2e_us"], "rem_us": rem, "seed_us": rr["seed_e2e_us"],
                     "calls": meta.get("llm_calls_ledger"), "usd": meta.get("llm_usd"), "wall_s": meta.get("flow_wall_s"),
                     "keys": rr["keys"], "hints": rr["hints"], "job": meta.get("job"), "host": meta.get("host"),
                     "mem_clock": meta.get("mem_clock_mhz"), "status": rr["flow_status"], "dir": str(d)})
    base = {(x["label"], x["flavour"], x["problem"]): x for x in data if x["arm"] == "control"}
    for x in data:
        b = base.get((x["label"], x["flavour"], x["problem"]))
        x["vs_control"] = (x["final_us"] / b["final_us"] - 1) if (b and b["final_us"] and x["final_us"]) else None
        rows.append([x["label"], x["flavour"], x["problem"], x["arm"], x["seed"], x["status"], x["final_us"], x["quiet_us"],
                     (sum(x["rem_us"]) / len(x["rem_us"])) if x["rem_us"] else "", f"{100 * x['vs_control']:+.1f}%" if x["vs_control"] is not None else "",
                     x["calls"], x["usd"], x["wall_s"], x["job"], x["mem_clock"], ",".join(map(str, x["keys"]))[:40]])
    md = C.md_head("X-136 flag matrix", {"generated": C.now(), "tree": ", ".join(sorted({d["label"] for d in data})),
                                          "flags": "see arms", "job": ", ".join(sorted({str(d["job"]) for d in data})),
                                          "host": ", ".join(sorted({str(d["host"]) for d in data})),
                                          "mem_clock_mhz": ", ".join(sorted({str(d["mem_clock"]) for d in data}))})
    md += "\n" + C.md_table(rows, ["tree", "flavour", "problem", "arm", "seed", "flow", "final us", "quiet us", "re-measured us (mean)", "vs control",
                                   "LLM calls", "$", "wall s", "job", "mem MHz", "round keys"])
    md += ("\n\nvs control compares the flow's own final and is only meaningful on the same node (or use the re-measured column). "
           "41_GRUBidirectional varies ~17% between identical runs (E0): do not read single-run differences there as effects.\n")
    out = Path(a.out or HERE / "flag_matrix_report")
    C.write_json(out.with_suffix(".json"), {"generated": C.now(), "runs": data})
    out.with_suffix(".md").write_text(md)
    print(md)


def recompose(a):
    """Zero-LLM composition arm: the E1 recomposer (analysis/e1_tools/e1_node.py) rebuilds a finished sweep problem from its kept winners
    and times the shipped program, the reuse=siblings and reuse=global recompositions, each sequential and with --graph-concurrency on,
    interleaved on one GPU. Here it is run against the tree under test (so a post-merge tree's kb_reuse / kb_concurrency / graph_prune are
    exercised) on one of my nodes. Needs e1_data/stage/data.tar.gz (e1_prep.py data 41_GRUBidirectional 43_MinGPTCausalAttention).
    NOTE: e1_node.py sets MB_PRUNE_DEAD=0 itself (the sweep manifests have the dead ops); that is correct for composition checks."""
    ti = C.tree_info(a.tarball)
    stage = C.L3 / "analysis" / "e1_data" / "stage" / "data.tar.gz"
    if not stage.exists():
        raise SystemExit(f"{stage} missing: python3 analysis/e1_tools/e1_prep.py data {' '.join(a.problems)}")
    base = f"{C.NODE_ROOT}/e1/{ti['label']}"
    C.put(a.job, stage, f"{base}/data.tar.gz")
    C.put(a.job, C.L3 / "analysis" / "e1_tools" / "e1_node.py", f"{base}/bin/e1_node.py")
    script = f"""set -e
export PATH={C.CUDA_N}/bin:$PATH CUDA_HOME={C.CUDA_N} TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=16 MB_CUPTI_LIB={C.CUDA_N}/extras/CUPTI/lib64/libcupti.so.12
. /tmp/krisd/venv/bin/activate
cd {base}; [ -d data ] || tar xzf data.tar.gz
export PYTHONPATH={C.NODE_ROOT}/{ti['label']} TMPDIR={base}/tmp; mkdir -p $TMPDIR
for p in {' '.join(a.problems)}; do
  mkdir -p runs/$p; timeout 5400 python bin/e1_node.py $p data/$p runs/$p --reps 6 > runs/$p/node.log 2>&1; echo "$p rc=$?"
done
"""
    cmd = (f"nohup setsid srun --jobid={a.job} --overlap bash -c {shlex.quote(script)} </dev/null >{base}/recompose.log 2>&1 & sleep 3; echo started")
    C.check_mine(a.job)
    rc, out, err = C.farm().ssh(f"mkdir -p {base} 2>/dev/null; " + cmd, b"", timeout=120)
    print(out.decode(), err.decode()[-300:], "-> results in", f"{base}/runs/<problem>/e1.json (fetch with: mc_runner-style srun cat)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--problems", nargs="+", default=PROBLEMS)
        p.add_argument("--arms", nargs="+", default=TIERS["1"])
        p.add_argument("--seeds", nargs="+", type=int, default=[1])
    p = sub.add_parser("plan")
    p.add_argument("--tarball")
    common(p)
    r = sub.add_parser("run")
    r.add_argument("job", type=int)
    r.add_argument("tarball")
    r.add_argument("--flavour", choices=["stub", "live"], required=True)
    r.add_argument("--th", type=float, default=6)
    common(r)
    m = sub.add_parser("remeasure")
    m.add_argument("job", type=int)
    m.add_argument("labels", nargs="+")
    m.add_argument("--passes", type=int, default=2)
    m.add_argument("--include-stub", action="store_true")
    n = sub.add_parser("analyze")
    n.add_argument("--labels", nargs="+", default=["post"])
    n.add_argument("--out")
    c = sub.add_parser("recompose")
    c.add_argument("job", type=int)
    c.add_argument("tarball")
    c.add_argument("--problems", nargs="+", default=PROBLEMS[::-1])
    a = ap.parse_args()
    {"plan": plan, "run": run, "remeasure": remeasure, "analyze": analyze, "recompose": recompose}[a.cmd](a)


if __name__ == "__main__":
    main()
