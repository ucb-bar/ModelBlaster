#!/usr/bin/env python3
"""Native (stock) tensor sizes of the KernelBench L1 / L2 problems, without allocating: each problem module is executed with torch's default device = meta; sizes = inputs + parameters at fp16.
usage: native_sizes.py  -> native_sizes.json {level: {name: {inputs_mb, params_mb, io_mb}}}"""
import importlib.util, json, sys, os
from pathlib import Path
import torch
Q = Path(__file__).resolve().parents[2] / "qrb_a6"
out = {}
for lvl in (1, 2):
    out[lvl] = {}
    for f in sorted((Q / f"problems_l{lvl}").glob("*.py"), key=lambda p: int(p.name.split("_")[0])):
        try:
            torch.set_default_device("meta")
            spec = importlib.util.spec_from_file_location("p", f); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
            init = m.get_init_inputs(); init = init if isinstance(init, (list, tuple)) else [init]
            model = m.Model(*init); xs = m.get_inputs(); xs = xs if isinstance(xs, (list, tuple)) else [xs]
            inp = sum(x.numel() * 2 for x in xs if torch.is_tensor(x)) / 2**20; par = sum(p.numel() * 2 for p in model.parameters()) / 2**20
            out[lvl][f.stem] = {"inputs_mb": round(inp, 1), "params_mb": round(par, 1), "io_mb": round(inp + par, 1)}
        except Exception as e:
            out[lvl][f.stem] = {"error": repr(e)[:120]}
        finally:
            torch.set_default_device("cpu")
json.dump(out, open("native_sizes.json", "w"), indent=1)
for lvl in (1, 2):
    v = [x["io_mb"] for x in out[lvl].values() if "io_mb" in x]; err = [k for k, x in out[lvl].items() if "error" in x]
    import statistics as st
    print(f"L{lvl}: {len(v)} sized, {len(err)} errors; io MiB median {st.median(v):.0f}, p90 {sorted(v)[int(.9*len(v))]:.0f}, max {max(v):.0f}; >256: {sum(1 for x in v if x>256)}, >1024: {sum(1 for x in v if x>1024)}, >4096: {sum(1 for x in v if x>4096)}", err[:4])
