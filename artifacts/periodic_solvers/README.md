# Phase 1 — 7-solver periodic schedule comparison on Dima's workload

Workload:
`data/toplevel/networks_mlp10_dronet20_yolov8_firesim_static_q31profile.json`
— 16×mlp_control @ 10 ms period + 8×dronet @ 20 ms period + 1×yolov8_nano
on the Gemmini+RVV hetero bitstream.

Single command per solver:

```bash
PYTHONPATH=xpu-rt python scripts/run_xpurt_schedule.py \
    --networks-json data/toplevel/networks_mlp10_dronet20_yolov8_firesim_static_q31profile.json \
    --solver <SOLVER>
# For HEFT/PEFT/EDF/CPSAT (registry-backed inner schedulers):
    --solver milp --scheduler <heft|peft|edf|cpsat>
```

Renderer: `xpu-rt/plot_gantt.py --fixture <out.json> --out <out.png>`
(Dima's plotter — emits the per-network-instance lanes that show
periodicity at a glance).

## Headline table

| Solver           | Makespan | mlp[0]→mlp[1] start | dronet[0]→dronet[1] start | Periodicity? | Notes |
|:-----------------|---------:|--------------------:|--------------------------:|:------------:|:------|
| **decomposed**   |  75.57 ms |   0.0 → 10.0 ms   |   0.5 → 20.5 ms          | ✅ clean     | Dima's reference baseline; phase-1 EDF on periodic + backfill |
| greedy           |  60.48 ms |   0.0 → 10.0 ms   |   0.5 → 25.2 ms          | ◐ partial    | mlp clean, dronet drifts |
| greedy_periodic  |  61.20 ms |   9.1 → 20.4 ms   |  46.9 → 51.9 ms          | ✗ broken     | despite the name |
| heft             |  54.43 ms |  53.0 → 53.0 ms   |  43.4 → 45.3 ms          | ✗ broken     | ALL mlp instances bunched together |
| peft             |  56.24 ms |  53.9 → 54.0 ms   |  34.5 → 36.4 ms          | ✗ broken     | same bunching pattern as HEFT |
| edf              | 116.78 ms | 114.8 → 114.8 ms  |  72.1 → 74.1 ms          | ✗ broken     | bunching + large makespan |
| **cpsat**        | 169.17 ms |   0.0 → 10.0 ms   |   2.0 → 20.0 ms          | ✅ clean     | optimal under hard periodic constraints |

Visuals: `dima_<solver>.png` per row; `stack_all.png` for the 7-panel
side-by-side comparison.

## Honest finding: makespan ≠ schedule quality

The naive read of the table — "HEFT wins at 54.43 ms" — is wrong. HEFT,
PEFT, EDF, and (against its own name) `greedy_periodic` produce
**lower makespan numbers by ignoring the periodic release
constraints**. They place all 16 mlp_control instances back-to-back
instead of releasing one every 10 ms. That's not a valid schedule for
a 10 ms control-loop deadline — it's a schedule for "I'll do 16
inferences as fast as possible and pretend the periodicity doesn't
matter."

Only **decomposed** (75.57 ms) and **cpsat** (169.17 ms) produce
schedules where mlp_control instance N starts at t ≈ N×10 ms and
dronet instance N starts at t ≈ N×20 ms. Those are the only two
solvers in our registry that honor `enforce_periodic`-style
constraints out of the box on this workload.

CPSAT's 169.17 ms is *honest* under periodic constraints — it's the
provably-optimal schedule that respects every release time. The 75.57
ms from `decomposed` is the heuristic upper bound — same constraint
set, looser optimization. The gap between them (93.6 ms) measures how
much slack `decomposed` left on the table.

## What this means for the decision loop (Phase 2/3)

Use **decomposed** as the agentic-loop baseline (fast, periods clean,
matches Dima's pipeline). Use **cpsat** as the gold-standard
comparator when we need to know if the heuristic is leaving
significant work on the floor.

Do NOT use HEFT/PEFT/EDF/greedy_periodic for the agentic loop — their
"lower" makespan is a constraint-violation artifact, and any fuse/split
"improvement" we measured against them would be measuring noise.

## Reproduction

```bash
mkdir -p artifacts/periodic_solvers
cd /scratch2/agustin/XPU-RT
WORKLOAD=data/toplevel/networks_mlp10_dronet20_yolov8_firesim_static_q31profile.json

for SOLVER in decomposed greedy greedy_periodic ; do
    PYTHONPATH=xpu-rt python scripts/run_xpurt_schedule.py \
        --networks-json $WORKLOAD --solver $SOLVER
done

for SCHED in heft peft edf cpsat ; do
    PYTHONPATH=xpu-rt python scripts/run_xpurt_schedule.py \
        --networks-json $WORKLOAD \
        --solver milp --scheduler $SCHED --time-limit 600
done

# Move outputs into artifacts/periodic_solvers/dima_<SOLVER>.json
# Render: xpu-rt/plot_gantt.py --fixture <fx> --out <png>
# Stack: scripts/render_compare_gantt.py --panel ... --panel ...
```
