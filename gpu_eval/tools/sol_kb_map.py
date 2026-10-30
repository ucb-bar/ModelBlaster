#!/usr/bin/env python3
"""Map the SOL-ExecBench problems (original repo) to KernelBlaster's earlier KernelBench-layout
port (``sol-level1`` / ``sol-level2``) and its CUDA results, for old-KB vs new-flow comparisons.

For every SOL problem of L1 / L2: the KB-layout id (``sol-level1/<name>``), which SOL workload
the KB file's ``get_inputs()`` reproduces (input shapes, evaluated on the meta device, matched
against every workload's resolved shapes), whether its dtypes are SOL's, the KB result rows
(``kernel_status_by_problem.csv``, ``speedup_l2.csv``, the eager CSVs) and the stems this port
converted (``manifest.<variant>.json``).

    sol_kb_map.py BENCH KB_SRC KB_RESULTS_DIR --manifest run/manifest.fp32.json --out map.json [--csv map.csv]

KB_SRC holds ``sol-level1/*.py`` and ``sol-level2/*.py``; KB_RESULTS_DIR the CSVs.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sol_upstream as su  # noqa: E402

LEVEL = {"L1": "sol-level1", "L2": "sol-level2"}
CREATE = ["rand", "randn", "zeros", "ones", "empty", "full", "randint", "arange", "tensor", "eye", "linspace",
          "randperm", "normal", "rand_like", "randn_like", "zeros_like", "ones_like", "empty_like", "full_like"]


def kb_shapes(path: Path):
    """(tensor input shapes, dtypes) of a KB-layout file's get_inputs(): on the meta device, or on
    the CPU when its generator needs values (``.item()``, ``unique``)."""
    try:
        return _kb_shapes(path, "meta")
    except (NotImplementedError, RuntimeError):
        return _kb_shapes(path, "cpu")


def _kb_shapes(path: Path, device: str):
    import torch
    import torch._dynamo  # noqa: F401 -- imported lazily otherwise, while the creation ops are patched
    spec = importlib.util.spec_from_file_location("_kbsol", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    saved = {n: getattr(torch, n) for n in CREATE if hasattr(torch, n)}

    def wrap(fn):
        def inner(*a, **k):
            k["device"] = device
            return fn(*a, **k)
        return inner
    try:
        for n, fn in saved.items():
            if not n.endswith("_like") and n != "tensor":
                setattr(torch, n, wrap(fn))
        ins = mod.get_inputs()
    finally:
        for n, fn in saved.items():
            setattr(torch, n, fn)
    ins = list(ins) if isinstance(ins, (list, tuple)) else [ins]
    return ([list(t.shape) for t in ins if torch.is_tensor(t)],
            [str(t.dtype).replace("torch.", "") for t in ins if torch.is_tensor(t)])


def read_csv(p: Path) -> dict:
    if not p.exists():
        return {}
    with open(p) as f:
        return {r["problem"]: r for r in csv.DictReader(f)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("kb_src")
    ap.add_argument("kb_results")
    ap.add_argument("--manifest", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv")
    a = ap.parse_args(argv)
    bench, src, res = Path(a.bench), Path(a.kb_src), Path(a.kb_results)
    status = read_csv(res / "kernel_status_by_problem.csv")
    l2 = read_csv(res / "speedup_l2.csv")
    eager1 = read_csv(res / "sol-level1_eager-nsys.csv")
    eager2 = read_csv(res / "sol-level2_eager-nsys.csv")
    stems = {}
    for m in a.manifest:
        man = json.loads(Path(m).read_text())
        for w in man["workloads"]:
            stems.setdefault((w["set"], w["problem"]), []).append(
                {"variant": man["variant"], "stem": w["stem"], "pick": w["pick"], "uuid": w["uuid"]})
    rows = []
    for s in ("L1", "L2"):
        for pdir in sorted((bench / s).iterdir()):
            defn, wls = su.load_problem(pdir)
            name = pdir.name
            kbf = src / LEVEL[s] / f"{name}.py"
            r = {"set": s, "problem": name, "kb_id": f"{LEVEL[s]}/{name}" if kbf.exists() else None,
                 "sol_dtypes": sorted({v["dtype"] for v in defn["inputs"].values() if v["shape"] is not None})}
            if kbf.exists():
                try:
                    shp, dts = kb_shapes(kbf)
                    r["kb_input_shapes"], r["kb_input_dtypes"] = shp, dts
                    # KB passes SOL's Python scalars as 0-D tensors: compare the non-0-D tensors
                    kb_t = [(x, d) for x, d in zip(shp, dts) if x]
                    match = None
                    for i, w in enumerate(wls):
                        ax = su.resolve_axes(defn, w["axes"])
                        want = [su.shape_of(v, ax) for v in defn["inputs"].values() if v["shape"]]
                        if want == [x for x, _ in kb_t]:
                            match = (i, w["uuid"])
                            break
                    r["kb_workload_index"], r["kb_workload_uuid"] = (match or (None, None))
                    r["shapes_match_a_workload"] = match is not None
                    want_dt = [v["dtype"] for v in defn["inputs"].values() if v["shape"]]
                    r["dtypes_match_sol"] = [d for _, d in kb_t] == want_dt
                except Exception as e:  # noqa: BLE001
                    r["kb_shape_error"] = f"{type(e).__name__}: {str(e)[:200]}"
            kb_row = status.get(name) if s == "L1" else l2.get(name)
            if kb_row:
                r["kb_result"] = kb_row
            eg = (eager1 if s == "L1" else eager2).get(name)
            if eg:
                r["kb_pytorch_eager_ns"] = {k: eg[k] for k in ("mean_ns", "p50_ns", "min_ns") if k in eg}
            r["this_port"] = stems.get((s, name), [])
            # the converted stem at KB's own workload, when one was selected
            r["this_port_same_workload"] = [x["stem"] for x in r["this_port"]
                                            if r.get("kb_workload_uuid") and x["uuid"] == r["kb_workload_uuid"]]
            rows.append(r)
    Path(a.out).write_text(json.dumps(rows, indent=1))
    if a.csv:
        cols = ["set", "problem", "kb_id", "kb_workload_index", "kb_workload_uuid", "shapes_match_a_workload",
                "dtypes_match_sol", "kb_status", "kb_speedup_vs_init", "kb_speedup_vs_pytorch_eager",
                "kb_l2_status", "kb_l2_speedup", "this_port_stems", "this_port_same_workload"]
        with open(a.csv, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(cols)
            for r in rows:
                k = r.get("kb_result") or {}
                wr.writerow([r["set"], r["problem"], r["kb_id"], r.get("kb_workload_index"),
                             r.get("kb_workload_uuid"), r.get("shapes_match_a_workload"), r.get("dtypes_match_sol"),
                             k.get("any_success"), k.get("speedup_vs_init"), k.get("speedup_vs_pytorch_eager"),
                             k.get("status"), k.get("speedup"),
                             ";".join(x["stem"] for x in r["this_port"]), ";".join(r["this_port_same_workload"])])
    n = len(rows)
    print(f"{n} problems; KB id {sum(1 for r in rows if r['kb_id'])}; shapes = a SOL workload "
          f"{sum(1 for r in rows if r.get('shapes_match_a_workload'))}; dtypes = SOL "
          f"{sum(1 for r in rows if r.get('dtypes_match_sol'))}; KB result row "
          f"{sum(1 for r in rows if r.get('kb_result'))}; this port at KB's workload "
          f"{sum(1 for r in rows if r['this_port_same_workload'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
