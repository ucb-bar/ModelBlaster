#!/usr/bin/env python3
"""E8 recomposition tool (local side, stdlib only): rebuild the flow's final program for a Level-3 sweep problem from
the artifacts kept under l3_farm/results/<p>/work, check what is recoverable, and stage a bundle that the node side
(node_side.py) turns into a timed program.

What the flow ships for a problem (kb_model_run.model_runtime): the final IR (hints applied, split variants) exported
as ONE whole-graph KernelBlaster problem (kb_export.export_problem) whose slots are the cached winners
(kb_ingest.cache_plan); timed by the problem's driver.cpp (persistent host, MB_GRAPH=1: graph replay, warm-up 2,
>= 3 reps until 200 ms per body or 200 reps, upper median of per-replay cudaEvent pairs) in 3 requests;
model_e2e_us = median of the 3 request medians.

Recovery levels (per problem, `inventory`):
  L0 shipped      work/final/<p>/problem has manifest + walker head/tail + every slot file; the program is these
                  sources assembled (kb_export.assemble, reimplemented here byte-for-byte; checked against the kept
                  build/kernels.cu when present). data.bin (weights + reference values) was NOT kept: it is
                  regenerated on the node from the KernelBench source (extract_graph --bench-target-mb 0 --quant fp16,
                  as worker4.sh) + the kept final IR graph.json, and verified against manifest data_sha1.
  L1 winners      no final problem, but the kept final IR graph.json + per-round winners (r*/winners/code, winners.json
                  shape guards) + knowledge/impl store cover every non-seed op: recomposable with cache_plan
                  (not needed for any problem of this sweep; reported only).
  L2 partial      some ops' winning sources missing (the winners CACHE index work/cache was mostly not copied back):
                  only the hand seed or the last round's profile program could stand in.
Provenance per final slot: the sha1 of its source is looked up in r*/winners/code/<sha>.cu (the round winners kept
locally), knowledge/impl/code/<sha>.cu (the per-run implementation store) and the manifest's seed_source tag.

  recompose.py inventory [P ...]          -> analysis/e8_data/inventory.json (+ table on stdout); default: all problems
  recompose.py bundle P OUTDIR [--override SLOT=FILE ...]
                                          -> OUTDIR/{problem/, ir/graph.json, problem.py, meta.json}; --override swaps a
                                             slot's source (the hook E1 uses to recompose winners: same buffers / ops /
                                             walker, other slot code). The schedule (walker_tail.cu) can be replaced
                                             the same way with --override walker_tail.cu=FILE.

Library use (E1): `artifacts(p)`, `assemble(problem_dir, sources)`, `provenance(p)`, `bundle(p, out, overrides)`.
"""
import argparse, hashlib, json, re, shutil, sys
from pathlib import Path

L3 = Path(__file__).resolve().parents[2]
RES = L3 / "results"
PROBS = L3 / "problems"
DATA = L3 / "analysis" / "e8_data"
NS_PREFIX = "mbslot_"


# ---------------------------------------------------------------- kb_export.assemble, byte-for-byte
def _namespaced(ns, code):
    code = code.replace('extern "C"', "")
    includes = [l for l in code.splitlines() if l.strip().startswith("#include")]
    body = "\n".join(l for l in code.splitlines() if not l.strip().startswith("#include"))
    defines = sorted(set(re.findall(r"^\s*#\s*define\s+([A-Za-z_]\w*)", body, re.M)))
    undefs = "".join(f"#undef {d}\n" for d in defines)
    return includes, f"namespace {ns} {{\n{body}\n}}  // namespace {ns}\n{undefs}"


def wrap_slot(slot_id, code):
    includes, body = _namespaced(f"{NS_PREFIX}{slot_id}", code)
    return ("\n".join(includes) + "\n" if includes else "") + f"// ==== slot {slot_id} ====\n{body}"


def assemble(problem_dir, sources=None):
    """head + wrapped slots (manifest order; `sources` {slot id: text} overrides) + tail = the kernel TU."""
    problem_dir = Path(problem_dir)
    man = json.loads((problem_dir / "manifest.json").read_text())
    asm = man["assembly"]
    parts = [(problem_dir / asm["head"]).read_text()]
    for s in man["slots"]:
        code = (sources or {}).get(s["id"]) or (problem_dir / s["file"]).read_text()
        parts.append(wrap_slot(s["id"], code))
    parts.append(((sources or {}).get(asm["tail"]) or (problem_dir / asm["tail"]).read_text()))
    return "\n".join(parts)


# ---------------------------------------------------------------- artifacts / provenance
def sha1(t):
    return hashlib.sha1(t.encode() if isinstance(t, str) else t).hexdigest()


def _num(p):
    m = re.match(r"(\d+)_", p)
    return int(m.group(1)) if m else 999


def problems():
    return sorted((d.name for d in RES.iterdir() if d.is_dir() and (d / "work").is_dir() and d.name[0].isdigit()), key=_num)


def artifacts(p):
    w = RES / p / "work"
    fin = w / "final" / p
    prob = fin / "problem"
    a = {"problem": p, "work": str(w), "final_problem": str(prob) if (prob / "manifest.json").exists() else None,
         "final_ir": str(fin / "ir" / "graph.json") if (fin / "ir" / "graph.json").exists() else None,
         "kernels_cu": str(prob / "build" / "kernels.cu") if (prob / "build" / "kernels.cu").exists() else None,
         "rounds": sorted(d.name for d in w.glob("r0*") if d.is_dir()),
         "graph_hints": (w / "graph_hints.json").exists(), "state": (w / "state.json").exists(),
         "kb_source": str(PROBS / f"{p}.py") if (PROBS / f"{p}.py").exists() else None,
         "work_cache": (w / "cache").is_dir()}
    s = w / "summary.json"
    if s.exists():
        v = next(iter(json.loads(s.read_text()).get("irs", {}).values()), {}) or {}
        a.update(model_e2e_us=v.get("model_e2e_us"), model_passed=v.get("model_passed"), model_basis=v.get("model_basis"),
                 model_gpu_uuid=None)
    mr = w / "model_runtime.json"
    if mr.exists():
        v = next(iter(json.loads(mr.read_text()).values()), {}) or {}
        a["model_gpu_uuid"] = v.get("gpu_uuid")
        a["model_eager_us"] = v.get("model_eager_us")
    kt = w / "final_model" / p / "kb_time" / "model.result.json"
    if kt.exists():
        j = json.loads(kt.read_text()).get("jobs", {}).get("model", {})
        a["model_e2e_runs"] = [r.get("e2e_us") for r in j.get("runs", [])]
        a["model_graph_nodes"] = next((r.get("graph_nodes") for r in j.get("runs", [])), None)
        a["model_runtime"] = next((r.get("runtime") for r in j.get("runs", [])), None)
    return a


def _store(p):
    """sha -> where, for every kept winner / impl source of the run."""
    w = RES / p / "work"
    idx = {}
    for f in w.glob("r0*/winners/code/*.cu"):
        idx.setdefault(sha1(f.read_text()), set()).add("winners:" + f.parts[-4])
    for f in w.glob("knowledge/impl/code/*.cu"):
        idx.setdefault(sha1(f.read_text()), set()).add("impl")
    for f in w.glob("r0*/run/*/best/slots/*.cu"):
        idx.setdefault(sha1(f.read_text()), set()).add("kb_best:" + f.parts[-6])
    return idx


def provenance(p):
    a = artifacts(p)
    if not a["final_problem"]:
        return []
    prob = Path(a["final_problem"])
    man = json.loads((prob / "manifest.json").read_text())
    idx = _store(p)
    out = []
    for s in man["slots"]:
        f = prob / s["file"]
        txt = f.read_text() if f.exists() else None
        h = sha1(txt) if txt is not None else None
        where = sorted(idx.get(h, ())) if h else []
        out.append({"slot": s["id"], "op": s["op"], "instances": len(s.get("instances", [])),
                    "seed_source": s.get("seed_source"), "present": txt is not None, "sha1": h, "found_in": where})
    return out


def inventory(p):
    a = artifacts(p)
    r = {**a, "level": None, "notes": []}
    if not a["final_problem"]:
        # L1/L2 check: winners per op shape of the final IR
        r["level"] = "L2 partial" if a["final_ir"] else "none"
        r["notes"].append("no final problem dir kept")
        return r
    prob = Path(a["final_problem"])
    man = json.loads((prob / "manifest.json").read_text())
    missing = [s["id"] for s in man["slots"] if not (prob / s["file"]).exists()]
    heads = all((prob / man["assembly"][k]).exists() for k in ("head", "tail"))
    r.update(n_slots=len(man["slots"]), n_ops=len(man["ops"]), n_buffers=len(man["buffers"]), data_sha1=man.get("data_sha1"),
             data_bytes=max((b["ref"][0] + b["ref"][1]) for b in man["buffers"] if b.get("ref")),
             missing_slots=missing)
    if missing or not heads:
        r["level"] = "L2 partial"
        r["notes"].append(f"missing slot files {missing}" if missing else "missing walker head/tail")
        return r
    asm = assemble(prob)
    if a["kernels_cu"]:
        same = Path(a["kernels_cu"]).read_text() == asm
        r["assemble_matches_kernels_cu"] = same
        if not same:
            r["notes"].append("assembled TU differs from kept build/kernels.cu")
    else:
        r["assemble_matches_kernels_cu"] = None
        r["notes"].append("no build/kernels.cu kept (assembled from slots)")
    pv = provenance(p)
    hand = [x["slot"] for x in pv if (x["seed_source"] or "").startswith("hand") and not x["found_in"]]
    covered = [x["slot"] for x in pv if x["found_in"]]
    unknown = [x["slot"] for x in pv if not x["found_in"] and x not in hand and x["slot"] not in hand]
    r.update(slots_in_store=len(covered), slots_hand=len(hand), slots_unattributed=unknown)
    r["level"] = "L0 shipped"
    r["l1_from_winners"] = not unknown        # would be recomposable even without the final problem dir
    if unknown:
        r["notes"].append(f"{len(unknown)} final slot sources not in kept winners/impl store (cache-only): {unknown[:6]}")
    return r


# ---------------------------------------------------------------- bundle for the node
def bundle(p, out, overrides=None):
    """Self-contained dir the node side rebuilds + times: problem/ (kept sources, with overrides), ir/graph.json,
    problem.py (KernelBench source), meta.json (recorded numbers + override list)."""
    a = artifacts(p)
    if not a["final_problem"] or not a["final_ir"]:
        raise SystemExit(f"{p}: no final problem/IR kept")
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(a["final_problem"], out / "problem", ignore=shutil.ignore_patterns("build"))
    (out / "ir").mkdir(parents=True)
    shutil.copyfile(a["final_ir"], out / "ir" / "graph.json")
    shutil.copyfile(a["kb_source"], out / "problem.py")
    man = json.loads((out / "problem" / "manifest.json").read_text())
    files = {s["id"]: s["file"] for s in man["slots"]}
    files[man["assembly"]["tail"]] = man["assembly"]["tail"]
    files[man["assembly"]["head"]] = man["assembly"]["head"]
    ov = {}
    for k, src in (overrides or {}).items():
        if k not in files:
            raise SystemExit(f"{p}: no slot / walker file {k}")
        txt = Path(src).read_text()
        (out / "problem" / files[k]).write_text(txt)
        ov[k] = {"from": str(src), "sha1": sha1(txt)}
    meta = {"problem": p, "recorded": {k: a.get(k) for k in ("model_e2e_us", "model_e2e_runs", "model_eager_us",
                                                                "model_graph_nodes", "model_gpu_uuid", "model_runtime")},
            "data_sha1": man.get("data_sha1"), "overrides": ov,
            "kernels_cu_sha1": sha1(Path(a["kernels_cu"]).read_text()) if a["kernels_cu"] else None,
            "assembled_sha1": sha1(assemble(out / "problem"))}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["inventory", "bundle"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--override", action="append", default=[])
    a = ap.parse_args()
    if a.cmd == "inventory":
        ps = a.args or problems()
        rows = [inventory(p) for p in ps]
        DATA.mkdir(parents=True, exist_ok=True)
        if not a.args:
            (DATA / "inventory.json").write_text(json.dumps(rows, indent=1))
        for r in rows:
            print(f"{r['problem']:34s} {r['level']:11s} slots={r.get('n_slots', '-'):>3} ops={r.get('n_ops', '-'):>4} "
                  f"asm==kernels.cu:{str(r.get('assemble_matches_kernels_cu')):5s} store={r.get('slots_in_store', '-')} "
                  f"hand={r.get('slots_hand', '-')} unattr={len(r.get('slots_unattributed', []) or [])} "
                  f"e2e={r.get('model_e2e_us')} {'; '.join(r['notes'])}")
    else:
        p, out = a.args
        ov = dict(x.split("=", 1) for x in a.override)
        print(bundle(p, out, ov))


if __name__ == "__main__":
    main()
