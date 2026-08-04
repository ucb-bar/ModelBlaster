"""Pick best placement variant per DSP segment + emit expanded dispatch graph
and per-backend results.csv files.

For each `dsp_seg_XX`, four variants are evaluated:
  - DSP-mono       : 1 dispatch on DSP, cost = full-segment DSP time
  - CPU-mono       : 1 dispatch on CPU, cost = full-segment CPU time
  - HTA-bundle-CPU : 5 dispatches  [tramp_p0 CPU → conv1 HTA → tramp_p1 CPU
                                    → conv2 HTA → tramp_p2 CPU]
  - HTA-bundle-DSP : 5 dispatches  [tramp_p0 DSP → conv1 HTA → tramp_p1 DSP
                                    → conv2 HTA → tramp_p2 DSP]
                     trampolines on int8-quantized DSP graphs instead of CPU.
                     Reduces trampoline cost ~2.5x (B-type) / ~1.7x (A-type)
                     because the Hexagon HVX vector units accelerate the
                     Q*K^T MatMul, LayerNorm, and Mul/Pow/Add work.

The variant with minimum serial cost wins. We then build an expanded
dispatch graph where HTA-bundle segments contribute 5 dispatches and mono
segments contribute 1 (cpu_seg_XX always contributes 1, CPU-only).

Per-backend results.csv files are emitted so each dispatch has its real
runtime on its intended backend and a prohibitive cost (1e9 us) on the
others — the scheduler then naturally picks the right HW per dispatch and
the makespan reflects the chosen placement variant.

Outputs:
  gen/qnn_vmfb/smolvlm_vision_v3_bundles/qrb5165_v66/<HW>/
      smolvlm_vision_v3_bundles.int8/smolvlm_vision_v3_bundles.int8_dispatch_graph.json

  gen/profile/<HW>/qrb5165_v66/smolvlm_vision_v3_bundles/
      smolvlm_vision_v3_bundles.int8/topo_0/results.csv

  qnn_models/smolVLA/v3_placement_plan.json  -- per-segment variant choice
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parent.parent
_PROFILE_DIR = _REPO / "qnn_models/boards/qrb5165_v66/profiles/smolvlm_vision_v3"
_INFEASIBLE_US = 1_000_000_000.0   # 1000 seconds; effectively "do not place here"
MODEL_NAME = "smolvlm_vision_v3_bundles"


def _load_perf():
    with open(_PROFILE_DIR / "segment_perf.json") as f:
        perf = json.load(f)
    with open(_PROFILE_DIR / "trampolines_perf.json") as f:
        cpu_tramp = json.load(f)
    dsp_tramp_path = _PROFILE_DIR / "trampolines_dsp_perf.json"
    dsp_tramp = json.load(open(dsp_tramp_path)) if dsp_tramp_path.exists() else {}
    return perf, cpu_tramp, dsp_tramp


def _build_bundle(hta_conv_us, hta_convs, tramp_seg, tramp_hw_label):
    """Build a placement bundle from per-phase trampoline timings."""
    if not (tramp_seg and hta_conv_us is not None and hta_convs):
        return None
    phases = []
    for p in ("p0", "p1", "p2"):
        pd = tramp_seg.get(p, {})
        phases.append(pd.get("mean_us") if pd.get("status") == "ok" else None)
    conv_costs = [c["mean_us"] for c in hta_convs]
    if len(conv_costs) < 2 or phases[1] is None:
        return None
    seq = []
    if phases[0] is not None: seq.append(("tramp_p0", tramp_hw_label, phases[0]))
    seq.append(("conv1",      "HTA",          conv_costs[0]))
    seq.append(("tramp_p1",   tramp_hw_label, phases[1]))
    seq.append(("conv2",      "HTA",          conv_costs[1]))
    if phases[2] is not None: seq.append(("tramp_p2", tramp_hw_label, phases[2]))
    return {"total_us": sum(s[2] for s in seq), "sequence": seq}


def _seg_variant_cost(perf_seg: dict, cpu_tramp_seg: dict | None,
                       dsp_tramp_seg: dict | None):
    """Return dict of variant -> total_us and detailed sub-costs."""
    dsp_us = perf_seg.get("Dsp", {}).get("mean_us")
    cpu_us = perf_seg.get("Cpu", {}).get("mean_us")
    hta_conv_us = perf_seg.get("Hta", {}).get("mean_us")  # already sum-of-convs
    hta_convs = perf_seg.get("Hta", {}).get("convs", [])

    bundle_cpu_tramp = _build_bundle(hta_conv_us, hta_convs, cpu_tramp_seg, "CPU")
    bundle_dsp_tramp = _build_bundle(hta_conv_us, hta_convs, dsp_tramp_seg, "DSP")

    return {
        "dsp_mono_us":     dsp_us,
        "cpu_mono_us":     cpu_us,
        "hta_bundle_cpu":  bundle_cpu_tramp,
        "hta_bundle_dsp":  bundle_dsp_tramp,
    }


def _decide_variant(costs: dict) -> tuple[str, float]:
    """Return (variant_name, total_us) for the cheapest viable option."""
    options = []
    if costs.get("dsp_mono_us") is not None:
        options.append(("DSP-mono", costs["dsp_mono_us"]))
    if costs.get("cpu_mono_us") is not None:
        options.append(("CPU-mono", costs["cpu_mono_us"]))
    if costs.get("hta_bundle_cpu") is not None:
        options.append(("HTA-bundle-CPU", costs["hta_bundle_cpu"]["total_us"]))
    if costs.get("hta_bundle_dsp") is not None:
        options.append(("HTA-bundle-DSP", costs["hta_bundle_dsp"]["total_us"]))
    if not options:
        raise ValueError("no viable variant — missing profile data")
    options.sort(key=lambda x: x[1])
    return options[0]


def build_plan(dsp_tramp_budget: int | None = None):
    """Pick best variant per segment, optionally with a hard cap on the
    number of segments allowed to use HTA-bundle-DSP (each consumes 3
    simultaneous DSP contexts; the QRB5165 DSP firmware tops out around
    ~30 contexts total). When the budget is exceeded, fall back to the
    next-best variant for the segment with the smallest DSP-tramp savings
    (greedy demotion).
    """
    perf, cpu_tramp, dsp_tramp = _load_perf()
    plan = []
    for i in range(25):
        seg = f"dsp_seg_{i:02d}"
        costs = _seg_variant_cost(perf[seg], cpu_tramp.get(seg), dsp_tramp.get(seg))
        variant, cost_us = _decide_variant(costs)
        plan.append({
            "segment": seg,
            "variant": variant,
            "total_us": cost_us,
            "costs": costs,
        })

    if dsp_tramp_budget is None:
        return plan

    # Demote DSP-tramp picks to the next-best variant where the savings vs
    # fallback are smallest, until we fit the budget. This keeps the biggest
    # DSP wins (the A-type ones with heavy Q*K^T) on DSP.
    def dsp_savings(p):
        if p["variant"] != "HTA-bundle-DSP":
            return float("inf")
        c = p["costs"]
        # Next-best non-DSP variant cost
        alts = []
        if c.get("dsp_mono_us"):     alts.append(c["dsp_mono_us"])
        if c.get("cpu_mono_us"):     alts.append(c["cpu_mono_us"])
        if c.get("hta_bundle_cpu"):  alts.append(c["hta_bundle_cpu"]["total_us"])
        return min(alts) - c["hta_bundle_dsp"]["total_us"]

    n_dsp = sum(1 for p in plan if p["variant"] == "HTA-bundle-DSP")
    while n_dsp > dsp_tramp_budget:
        # Find DSP-tramp pick with smallest savings; demote it
        ranked = sorted([p for p in plan if p["variant"] == "HTA-bundle-DSP"],
                         key=dsp_savings)
        p = ranked[0]
        c = p["costs"]
        opts = []
        if c.get("dsp_mono_us") is not None:
            opts.append(("DSP-mono",        c["dsp_mono_us"]))
        if c.get("cpu_mono_us") is not None:
            opts.append(("CPU-mono",        c["cpu_mono_us"]))
        if c.get("hta_bundle_cpu") is not None:
            opts.append(("HTA-bundle-CPU",  c["hta_bundle_cpu"]["total_us"]))
        opts.sort(key=lambda x: x[1])
        p["variant"]  = opts[0][0]
        p["total_us"] = opts[0][1]
        n_dsp -= 1

    return plan


def emit_dispatch_graph(plan: list[dict]):
    """Expanded dispatch graph: HTA-bundle segments contribute multiple dispatches."""
    dispatches = {}
    did = 0
    prev_did = None
    out_metadata = []   # per-dispatch source info (for results.csv emission)

    def add(name: str, hw: str, cost_us: float, kind: str, dlc: str, src_seg: str):
        """kind: 'tramp_p0' | 'tramp_p1' | 'tramp_p2' | 'conv1' | 'conv2'
                 | 'dsp_mono' | 'cpu_mono' | 'cpu_seg'."""
        nonlocal did, prev_did
        key = f"dispatch_{did}"
        deps = [f"dispatch_{prev_did}"] if prev_did is not None else []
        dispatches[key] = {
            "id": did,
            "ordinal": 1,
            "total": 1,
            "dependencies": deps,
            "vmfb_path": f"slices/{dlc}",
            "segment_name": name,
            "segment_type": kind,
            "source_segment": src_seg,
            "preferred_hw": hw,
        }
        out_metadata.append({
            "dispatch_id": did,
            "module_name": name,
            "preferred_hw": hw,
            "cost_us": cost_us,
        })
        prev_did = did
        did += 1

    for i in range(25):
        p = plan[i]
        seg = p["segment"]
        variant = p["variant"]
        if variant == "DSP-mono":
            add(seg, "DSP", p["costs"]["dsp_mono_us"], "dsp_mono",
                f"{seg}_quantized.dlc", seg)
        elif variant == "CPU-mono":
            add(seg, "CPU", p["costs"]["cpu_mono_us"], "cpu_mono",
                f"{seg}.dlc", seg)
        elif variant in ("HTA-bundle-CPU", "HTA-bundle-DSP"):
            bundle_key = "hta_bundle_cpu" if variant == "HTA-bundle-CPU" else "hta_bundle_dsp"
            for step_name, step_hw, step_us in p["costs"][bundle_key]["sequence"]:
                if step_name.startswith("tramp_p"):
                    pi = step_name[-1]   # '0' / '1' / '2'
                    # step_hw is "CPU" or "DSP" — encoded in the bundle's sequence.
                    add(f"{seg}_tramp_p{pi}", step_hw, step_us,
                        f"tramp_p{pi}",
                        f"{seg}_tramp_p{pi}.dlc", seg)
                elif step_name == "conv1":
                    add(f"{seg}_conv1", "HTA", step_us, "conv1",
                        f"{seg}_conv1.dlc", seg)
                elif step_name == "conv2":
                    add(f"{seg}_conv2", "HTA", step_us, "conv2",
                        f"{seg}_conv2.dlc", seg)

        # cpu_seg_XX comes after each dsp_seg (except after dsp_seg_24)
        if i < 24:
            cseg = f"cpu_seg_{i:02d}"
            cpu_seg_perf = json.load(open(_PROFILE_DIR / "segment_perf.json"))[cseg]
            cpu_cost = cpu_seg_perf.get("Cpu", {}).get("mean_us")
            add(cseg, "CPU", cpu_cost, "cpu_seg", f"{cseg}.dlc", cseg)

    graph = {
        "dot_file": "",
        "dispatch_vmfb_dir": "slices",
        "_comment": (
            "Expanded v3 vision dispatch graph with placement variants. "
            "Each DSP segment maps to either 1 dispatch (DSP-mono or CPU-mono) "
            "or 5 dispatches (HTA-bundle: tramp_p0 CPU → conv1 HTA → tramp_p1 "
            "CPU → conv2 HTA → tramp_p2 CPU). Variants chosen per-segment from "
            "honest profile data (segment_perf.json + trampolines_perf.json). "
            "CPU segments stay as 1 dispatch on CPU."
        ),
        "dispatches": dispatches,
    }

    # Write graph for all three backend "slots" (the schedule uses whichever
    # slot's path resolves; the dispatches inside have their own preferred_hw).
    for hw in ("CPU", "DSP", "HTA"):
        out_dir = _REPO / "gen/qnn_vmfb" / MODEL_NAME / "qrb5165_v66" / hw / f"{MODEL_NAME}.int8"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{MODEL_NAME}.int8_dispatch_graph.json"
        with open(out_path, "w") as f:
            json.dump(graph, f, indent=2)
    return graph, out_metadata


def emit_results_csvs(metadata: list[dict]):
    """For each backend (CPU, DSP, HTA), emit results.csv with realistic
    cost on the dispatch's preferred HW and prohibitive cost elsewhere."""
    hw_to_label = {"CPU": "CPU", "DSP": "DSP", "HTA": "HTA"}
    for csv_hw in ("CPU", "DSP", "HTA"):
        out_dir = (_REPO / "gen/profile" / csv_hw / "qrb5165_v66"
                   / MODEL_NAME / f"{MODEL_NAME}.int8" / "topo_0")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "results.csv"
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["dispatch_id", "module_name",
                                                "mean_time", "mean_unit"])
            w.writeheader()
            for m in metadata:
                cost = m["cost_us"] if m["preferred_hw"] == csv_hw else _INFEASIBLE_US
                w.writerow({
                    "dispatch_id": m["dispatch_id"],
                    "module_name": m["module_name"],
                    "mean_time": f"{cost:.2f}",
                    "mean_unit": "us",
                })
        print(f"  wrote {out_path} ({len(metadata)} rows)")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsp-tramp-budget", type=int, default=None,
                    help="cap on number of segments using HTA-bundle-DSP. "
                         "Each uses 3 DSP contexts; QRB5165 DSP firmware "
                         "tops out near 30 simultaneous contexts.")
    args = ap.parse_args()
    plan = build_plan(dsp_tramp_budget=args.dsp_tramp_budget)
    print("=== Placement plan per DSP segment ===")
    print(f'{"Segment":<14} {"Variant":<16} {"Cost ms":>10} {"DSP-mono":>10} '
          f'{"CPU-mono":>10} {"HTA+Cpu":>10} {"HTA+Dsp":>10}')
    total_dispatches = 0
    by_variant = {"DSP-mono": 0, "CPU-mono": 0,
                   "HTA-bundle-CPU": 0, "HTA-bundle-DSP": 0}
    for p in plan:
        c = p["costs"]
        dsp_str = f'{c["dsp_mono_us"]/1000:.1f}' if c.get("dsp_mono_us") else "—"
        cpu_str = f'{c["cpu_mono_us"]/1000:.1f}' if c.get("cpu_mono_us") else "—"
        bcpu_str = f'{c["hta_bundle_cpu"]["total_us"]/1000:.1f}' if c.get("hta_bundle_cpu") else "—"
        bdsp_str = f'{c["hta_bundle_dsp"]["total_us"]/1000:.1f}' if c.get("hta_bundle_dsp") else "—"
        print(f'{p["segment"]:<14} {p["variant"]:<16} {p["total_us"]/1000:>10.1f} '
              f'{dsp_str:>10} {cpu_str:>10} {bcpu_str:>10} {bdsp_str:>10}')
        by_variant[p["variant"]] += 1
        total_dispatches += (5 if p["variant"].startswith("HTA-bundle") else 1)
    total_dispatches += 24   # cpu_seg dispatches
    print()
    print(f'Variant counts: {by_variant}')
    print(f'Total dispatches in expanded graph: {total_dispatches} '
          f'(was 49 in monolithic v3)')
    print()

    print("=== Emit expanded dispatch graph ===")
    graph, metadata = emit_dispatch_graph(plan)
    print(f'  emitted graph with {len(graph["dispatches"])} dispatches')
    print()

    print("=== Emit per-backend results.csv ===")
    emit_results_csvs(metadata)
    print()

    # Save the plan for downstream tools / debugging
    plan_path = _HERE / "v3_placement_plan.json"
    with open(plan_path, "w") as f:
        json.dump({"plan": plan, "dispatch_metadata": metadata}, f, indent=2)
    print(f"Placement plan written to {plan_path}")


if __name__ == "__main__":
    main()
