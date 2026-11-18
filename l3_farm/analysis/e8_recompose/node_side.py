#!/usr/bin/env python3
"""E8 recomposition tool, node side. Runs ON the GPU node with a private copy of the ModelBlaster tree on PYTHONPATH
(/tmp/krisd/e8/code, the staged l3_farm/state/mb.tar.gz; never the farm's tree) and the farm venv.

  node_side.py rebuild BUNDLE WORK      regenerate the final program's data and verify it:
      1. extract_graph on BUNDLE/problem.py exactly as worker4.sh (--bench-target-mb 0 --quant fp16) -> WORK/ir_src
      2. WORK/ir_final = the kept final IR graph.json + ir_src's weights.npz / io.npz (= kb_proposals.write_variant_ir)
      3. kb_export.export_problem(ir_final, WORK/export, slot_sources = the bundle's slot files, strategy whole)
      4. checks: export manifest (slots, ops, buffers) and walker head/tail == kept ones; data_sha1 == kept data_sha1
         (identical reference data => identical inputs/weights as the sweep's measurement)
      5. WORK/problem = the bundle's problem dir (kept sources, overrides included) + the regenerated data.bin +
         driver.cpp  -> printed as @@REBUILD@@{json}
  node_side.py flowtime WORK [--reps 3]  time WORK/problem the flow's way: kb_model_run._time_builtin (persistent host
      driver.cpp -DMB_HOST + per-slot-unit kernels .so, MB_GRAPH=1, `reps` requests, medians) -> @@FLOWTIME@@{json}
      (each request's own graph-replay distribution min/med/max is kept too)
  node_side.py buildso WORK              the same kernels .so (kb_export.build_so: per-slot units, -O3 -std=c++17
      -arch=sm_89 --default-stream per-thread, -shared) for the common interleaved harness -> @@SO@@<path>
"""
import hashlib, json, os, shutil, subprocess, sys, time
os.environ.setdefault("MB_PRUNE_DEAD", "0")   # merged trees prune dead ops by default: the sweep manifests (35-42, 49) have them; 0 = legacy export
from pathlib import Path


def sh1(b):
    return hashlib.sha1(b).hexdigest()


def rebuild(bundle, work):
    bundle, work = Path(bundle), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    out = {"bundle": str(bundle), "work": str(work), "t": {}}
    t0 = time.time()
    src = work / "ir_src"
    if not (src / ".done").exists():
        shutil.rmtree(src, ignore_errors=True)
        cp = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.extract_graph", "--bench-file", str(bundle / "problem.py"),
                             "--bench-target-mb", "0", "--quant", "fp16", "--out-dir", str(src)], capture_output=True, text=True)
        (work / "extract.log").write_text(cp.stdout + cp.stderr)
        if cp.returncode:
            out.update(ok=False, stage="extract", log=(cp.stdout + cp.stderr)[-3000:])
            return out
        (src / ".done").touch()
    out["t"]["extract_s"] = round(time.time() - t0, 1)
    fin = work / "ir_final"
    fin.mkdir(exist_ok=True)
    shutil.copyfile(bundle / "ir" / "graph.json", fin / "graph.json")
    for f in ("weights.npz", "io.npz"):
        if (fin / f).exists() or (fin / f).is_symlink():
            (fin / f).unlink()
        os.symlink(src / f, fin / f)
    kept = bundle / "problem"
    kman = json.loads((kept / "manifest.json").read_text())
    sources = {s["id"]: (kept / s["file"]).read_text() for s in kman["slots"]}
    from modelblaster.pipeline.kb_export import export_problem
    from modelblaster.pipeline.kb_proposals import _all_specs
    t1 = time.time()
    exp = work / "export"
    shutil.rmtree(exp, ignore_errors=True)
    try:
        export_problem(fin, exp, slot_sources=sources, specs=dict(_all_specs()), strategy="whole", name=kman.get("name"))
    except Exception as e:
        import traceback
        out.update(ok=False, stage="export", log=traceback.format_exc()[-3000:])
        return out
    out["t"]["export_s"] = round(time.time() - t1, 1)
    eman = json.loads((exp / "manifest.json").read_text())
    strip = lambda bs: [{k: v for k, v in b.items()} for b in bs]
    chk = {
        "slots_same": [(s["id"], s["signature"]) for s in eman["slots"]] == [(s["id"], s["signature"]) for s in kman["slots"]],
        "ops_same": eman["ops"] == kman["ops"],
        "buffers_same": strip(eman["buffers"]) == strip(kman["buffers"]),
        "tail_same": (exp / eman["assembly"]["tail"]).read_text() == (kept / kman["assembly"]["tail"]).read_text(),
        "head_same": (exp / eman["assembly"]["head"]).read_text() == (kept / kman["assembly"]["head"]).read_text(),
        "data_sha1_same": eman.get("data_sha1") == kman.get("data_sha1"),
        "data_sha1": eman.get("data_sha1"), "kept_data_sha1": kman.get("data_sha1"),
        "data_bytes": (exp / "data.bin").stat().st_size,
    }
    if not chk["buffers_same"]:   # first differing buffer, for the report
        for a, b in zip(eman["buffers"], kman["buffers"]):
            if a != b:
                chk["first_buffer_diff"] = [a, b]
                break
    out["checks"] = chk
    # the program to time: the KEPT sources (incl. any E1 overrides) + regenerated data and driver
    prob = work / "problem"
    shutil.rmtree(prob, ignore_errors=True)
    shutil.copytree(kept, prob)
    os.replace(exp / "data.bin", prob / "data.bin")
    drv = (exp / "driver.cpp").read_text().replace(str(exp / "data.bin"), str(prob / "data.bin"))
    (prob / "driver.cpp").write_text(drv)
    if not chk["buffers_same"] or not chk["ops_same"]:
        out.update(ok=False, stage="verify", why="regenerated export's buffers/ops differ from the kept manifest: the "
                   "regenerated driver tables would not match the kept walker")
        return out
    out.update(ok=True, problem=str(prob))
    return out


def flowtime(work, reps=3):
    from modelblaster.pipeline.kb_model_run import _time_builtin
    prob = Path(work) / "problem"
    t0 = time.time()
    r = _time_builtin(prob, arch="89", work=Path(work) / "build_flowtime", reps=reps)
    r["wall_s"] = round(time.time() - t0, 1)
    r.pop("log", None) if r.get("passed") else None
    return r


def buildso(work):
    from modelblaster.pipeline.kb_export import build_so
    so, log = build_so(Path(work) / "problem", arch="89", work=Path(work) / "build_so", name="flow.so")
    return so, log


def main():
    cmd = sys.argv[1]
    if cmd == "rebuild":
        print("@@REBUILD@@" + json.dumps(rebuild(sys.argv[2], sys.argv[3])), flush=True)
    elif cmd == "flowtime":
        reps = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[3] == "--reps" else 3
        print("@@FLOWTIME@@" + json.dumps(flowtime(sys.argv[2], reps), default=str), flush=True)
    elif cmd == "buildso":
        so, log = buildso(sys.argv[2])
        print("@@SO@@" + (str(so) if so else "FAILED " + log[-2000:]), flush=True)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
