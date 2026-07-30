#!/usr/bin/env python3
"""DroNet int8 FireSim comparison: ModelBlaster vs ExecuTorch.

Two panels:
  (A) full-model execute cycles (CLEAN, profiling-off runs).
  (B) per-operator-category breakdown.

Data sources
------------
MB per-op : modelblaster/examples/dronet/int8/generated/profile.csv  (clean,
            in-binary rdcycle per dispatch; sums to the clean total).
ET per-op : /tmp/et_fs_dronet_prof.uartlog  — an XNNPACK-profiling FireSim run
            (`MB_XNN_PROFILE=ON`). The per-op `>>` lines are clean rdcycle
            deltas around each XNNPACK op (the profiling build's *total* is
            polluted by HTIF logging, which is why the full-model total comes
            from the separate profiling-off run — see FULL_* below).
"""
import csv
import re
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = "/scratch2/dima/misc_sw/FreshScheduler/zephyr-chipyard-sw"
MB_CSV = f"{REPO}/modelblaster/examples/dronet/int8/generated/profile.csv"
ET_LOG = "/tmp/et_fs_dronet_prof.uartlog"
OUT = f"{REPO}/plots/dronet_firesim_compare.png"

# Clean full-model execute cycles (profiling-OFF FireSim runs).
FULL_MB = 15_868_769          # bit-exact PASS
FULL_ET_WARM = 13_671_203     # warm (steady-state)
FULL_ET_COLD = 15_257_209     # first (cold) iteration

# ---- category taxonomy (union of both runtimes) ----
CATS = ["Convolution", "MaxPool", "Residual Add", "ReLU / Clamp",
        "Linear / FC", "BatchNorm", "Sigmoid",
        "Transpose", "Convert (quant)", "Reshape / Setup",
        "Dispatch / other runtime"]

# ---------------------------------------------------------------- MB parse
def parse_mb():
    cat = {c: 0 for c in CATS}
    m = {"conv2d_s8": "Convolution", "maxpool2d_s8": "MaxPool",
         "add_s8": "Residual Add", "relu_s8": "ReLU / Clamp",
         "linear_s8": "Linear / FC", "batchnorm2d_s8": "BatchNorm",
         "sigmoid_s8": "Sigmoid"}
    total = 0
    with open(MB_CSV) as f:
        for row in csv.DictReader(f):
            c = int(row["cycles"]); total += c
            cat[m[row["op"]]] += c
    return cat, total


# ---------------------------------------------------------------- ET parse
def et_category(name):
    if "Convolution" in name:       return "Convolution"
    if "Max Pooling" in name:       return "MaxPool"
    if name.startswith("Add"):      return "Residual Add"
    if "Clamp" in name:             return "ReLU / Clamp"
    if "Fully Connected" in name:   return "Linear / FC"
    if "Sigmoid" in name:           return "Sigmoid"
    if "Transpose" in name:         return "Transpose"
    if "Convert" in name:           return "Convert (quant)"
    return None


def parse_et():
    lines = open(ET_LOG, errors="replace").read().splitlines()
    # iteration boundaries: EXECUTORCH_EXECUTE_CYCLES[i] marks the END of iter i.
    marks = [i for i, l in enumerate(lines) if "EXECUTORCH_EXECUTE_CYCLES[" in l]
    op_re = re.compile(r">>,\s*(.+?),\s*(\d+)\s*\(")
    rs_re = re.compile(r"MB_XNN_RESHAPE_CYCLES=(\d+)")
    su_re = re.compile(r"MB_XNN_SETUP_CYCLES=(\d+)")
    # warm iters = those bounded by two consecutive markers (skip cold iter 0).
    iters = []
    for k in range(1, len(marks)):
        seg = lines[marks[k - 1] + 1: marks[k] + 1]
        cat = {c: 0 for c in CATS}
        for l in seg:
            mo = op_re.search(l)
            if mo:
                c = et_category(mo.group(1).strip())
                if c:
                    cat[c] += int(mo.group(2))
                continue
            mr = rs_re.search(l);  ms = su_re.search(l)
            if mr: cat["Reshape / Setup"] += int(mr.group(1))
            if ms: cat["Reshape / Setup"] += int(ms.group(1))
        iters.append(cat)
    if not iters:
        sys.exit("no warm ET iterations parsed")
    avg = {c: float(np.mean([it[c] for it in iters])) for c in CATS}
    # The XNNPACK profiler only times ops *inside* the delegate segments. The
    # gap to the clean profiling-off execute total is ExecuTorch interpreter /
    # delegate-dispatch overhead between the 6 XNNPACK segments (memory
    # planning, quant/dequant portable ops at boundaries, invoke dispatch).
    op_sum = sum(avg.values())
    avg["Dispatch / other runtime"] = max(0.0, FULL_ET_WARM - op_sum)
    total = sum(avg.values())
    return avg, total, len(iters)


def main():
    mb, mb_total = parse_mb()
    et, et_total, n = parse_et()
    print(f"MB total (sum per-op) = {mb_total:,}  (clean full = {FULL_MB:,})")
    print(f"ET per-op sum (avg of {n} warm iters) = {et_total:,.0f}  "
          f"(clean full warm = {FULL_ET_WARM:,})")
    for c in CATS:
        print(f"  {c:16s}  MB {mb[c]:>11,}   ET {et[c]:>12,.0f}")

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1, 2.3]})
    c_mb, c_et = "#2b6cb0", "#dd6b20"

    # ---- Panel A: full model ----
    xs = [0, 1, 1.75]
    vals = [FULL_MB, FULL_ET_WARM, FULL_ET_COLD]
    cols = [c_mb, c_et, c_et]
    alph = [1.0, 1.0, 0.4]
    hatch = [None, None, "//"]
    for xi, v, cc, aa, hh in zip(xs, vals, cols, alph, hatch):
        axA.bar(xi, v, 0.7, color=cc, alpha=aa, hatch=hh,
                edgecolor="white" if hh else None)
        axA.text(xi, v, f"{v/1e6:.2f}M", ha="center", va="bottom",
                 fontsize=10, fontweight="bold",
                 color="#333" if aa == 1 else "#7b341e")
    axA.set_xticks(xs)
    axA.set_xticklabels(["ModelBlaster\nint8 RVV\n(curated)",
                         "ExecuTorch\nint8 XNNPACK\n(warm)",
                         "ExecuTorch\n(cold,\n1st iter)"], fontsize=9)
    axA.set_ylabel("execute cycles")
    axA.set_title("(A) Full-model — DroNet int8, FireSim\n"
                  "(clean, profiling-off, 1 core)", fontsize=11)
    axA.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1e6:.0f}M"))
    axA.grid(axis="y", ls=":", alpha=0.5)
    axA.set_ylim(0, FULL_MB * 1.18)
    speed = FULL_MB / FULL_ET_WARM
    axA.annotate(f"ET {speed:.2f}× faster\n(warm)",
                 xy=(1, FULL_ET_WARM), xytext=(0.5, FULL_MB * 1.08),
                 ha="center", fontsize=10, color="#7b341e", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#7b341e"))

    # ---- Panel B: per-operator ----
    y = np.arange(len(CATS))[::-1]
    mb_v = [mb[c] for c in CATS]
    et_v = [et[c] for c in CATS]
    h = 0.38
    axB.barh(y + h / 2 + 0.02, mb_v, h, color=c_mb, label="ModelBlaster")
    axB.barh(y - h / 2 - 0.02, et_v, h, color=c_et, label="ExecuTorch")
    for yi, v in zip(y + h / 2 + 0.02, mb_v):
        if v > 0:
            axB.text(v + 5e4, yi, f"{v/1e6:.2f}M" if v >= 1e5 else f"{v/1e3:.0f}k",
                     va="center", fontsize=8, color=c_mb)
    for yi, v in zip(y - h / 2 - 0.02, et_v):
        if v > 0:
            axB.text(v + 5e4, yi, f"{v/1e6:.2f}M" if v >= 1e5 else f"{v/1e3:.0f}k",
                     va="center", fontsize=8, color="#7b341e")
    axB.set_yticks(y)
    axB.set_yticklabels(CATS)
    axB.set_xlabel("cycles  (ET per-op from XNNPACK profiler; MB from in-binary rdcycle)")
    axB.set_title("(B) Per-operator category — DroNet int8, FireSim\n"
                  "MB stays NCHW int8; ET (XNNPACK) fuses conv+BN+ReLU but pays "
                  "layout Transpose / Convert / Reshape", fontsize=11)
    axB.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1e6:.0f}M"))
    axB.grid(axis="x", ls=":", alpha=0.5)
    axB.legend(loc="lower right")
    axB.margins(x=0.12)

    fig.suptitle("DroNet int8 — ModelBlaster vs ExecuTorch on FireSim "
                 "(real RTL, dual-Rocket Saturn V256D128 RVV, 1 core)",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT, dpi=140)
    print("wrote", OUT)


if __name__ == "__main__":
    import matplotlib.ticker  # noqa
    main()
