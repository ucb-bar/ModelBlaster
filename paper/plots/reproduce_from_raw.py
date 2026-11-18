#!/usr/bin/env python3
"""Reproduce the CUDA L1/L2 summary statistics and fast_p data from the raw run directories.

Mirrors results/plot_speedup_over_pytorch_combined.py in tachyon-rl-fresh:
  * parse_cycles_per_iter.compute_cumulative_best_cycles(dir, max_iter=10, require_checker=True)
    reads rl_ncu/0_init_annotated.cu, rl_iter_*_best.cu and checker_passed.txt per problem.
  * Our cycles = min over Best_1..Best_10 (Best_0 excluded), speedup = baseline cycles / ours,
    over valid problems only (checker passed, baseline present).
Baselines: torch.compile (as in the original series definitions) and better of eager/compile.
Usage:  python3 reproduce_from_raw.py   (writes repro_stats.csv, repro_speedups.npz)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from parse_cycles_per_iter import compute_cumulative_best_cycles  # noqa: E402

RAW = HERE / "raw"
BASE = HERE / "inputs" / "baselines"

RUNS = {  # (label, level) -> (raw run dir, gpu key)
    ("H100", 1): ("rl_baseline_debug_computelab_H100_header", "h100"),
    ("A6000", 1): ("rl_baseline_debug_computelab_gpt5_header", "a6000"),
    ("L40S", 1): ("rl_baseline_debug_computelab_l40s_level1_best", "l40s"),
    ("A6000", 2): ("rl_baseline_debug_computelab_gpt5_level2", "a6000"),
    ("H100", 2): ("rl_baseline_debug_computelab_H100_header_level2", "h100"),
    ("L40S", 2): ("rl_baseline_debug_computelab_l40s_level2_best", "l40s"),
}


def base_csv(gpu: str, level: int, kind: str) -> Path:
    stem = "baseline_pytorch" + ("_level2" if level == 2 else "")
    return BASE / (f"{stem}_compile_{gpu}.csv" if kind == "compile" else f"{stem}_{gpu}.csv")


def load_base(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    low = {c.lower(): c for c in df.columns}
    p = low["problem"]
    c = low.get("avg_elapsed_cycles") or next(o for k, o in low.items() if "cycle" in k)
    out = df[[p, c]].rename(columns={p: "problem", c: "cycles"})
    out["cycles"] = pd.to_numeric(out["cycles"], errors="coerce")
    return out


def ours_table(run: str, level: int) -> pd.DataFrame:
    d = RAW / run / "gpt-5-20250807" / f"level{level}"
    header, rows = compute_cumulative_best_cycles(d, max_iter=10, require_checker=True)
    df = pd.DataFrame(rows, columns=header)
    cols = [f"Best_{i}" for i in range(1, 11)]
    vals = np.vstack([pd.to_numeric(df[c], errors="coerce").to_numpy(float) for c in cols])
    vals = np.where(vals > 0, vals, np.nan)
    with np.errstate(all="ignore"):
        best = np.nanmin(vals, axis=0)
    return pd.DataFrame({"problem": df["Kernel ID"].astype(str), "ours": best})


def speedups(base: pd.DataFrame, ours: pd.DataFrame) -> np.ndarray:
    m = pd.merge(base, ours, on="problem", how="inner")
    b, o = m["cycles"].to_numpy(float), m["ours"].to_numpy(float)
    ok = (b > 0) & (o > 0) & np.isfinite(b) & np.isfinite(o)
    return (b[ok] / o[ok])


def stats(sp: np.ndarray, pool: int) -> dict:
    return dict(valid_pct=100 * sp.size / pool, n=sp.size, mean=sp.mean(), geomean=float(np.exp(np.log(sp).mean())),
                median=float(np.median(sp)), min=sp.min(), max=sp.max(),
                pct_gt1=100 * (sp > 1).mean(), pct_lt1=100 * (sp < 1).mean())


def main() -> None:
    out_rows, store = [], {}
    for (label, level), (run, gpu) in RUNS.items():
        ours = ours_table(run, level)
        comp = load_base(base_csv(gpu, level, "compile"))
        eag = load_base(base_csv(gpu, level, "eager"))
        m = pd.merge(eag, comp, on="problem", how="outer", suffixes=("_e", "_c"))
        best = pd.DataFrame({"problem": m["problem"], "cycles": np.fmin(m["cycles_e"], m["cycles_c"])})
        for kind, base in (("compile", comp), ("best(eager,compile)", best)):
            sp = speedups(base, ours)
            pool = len(base)
            s = stats(sp, pool)
            out_rows.append(dict(gpu=label, level=level, baseline=kind, pool=pool, **s))
            store[f"{label}_L{level}_{kind}"] = sp
    df = pd.DataFrame(out_rows)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))
    df.to_csv(HERE / "repro_stats.csv", index=False)
    np.savez(HERE / "repro_speedups.npz", **store)


if __name__ == "__main__":
    main()
