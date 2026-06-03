# Agentic fuse/split demo — yolov8_nano_64 + mlp_control + dronet

## What this directory contains

1. **`AGENTIC_LOOP.md`** — the full end-to-end run: schedule → analyze
   → act → re-schedule, with the agent's actual decision trace. Read this
   first.
2. **`agentic_loop_before_after.png`** — 2-panel Gantt showing the BEFORE
   (170 disp, 83.08 ms) and AFTER (157 disp, 82.67 ms) schedules from the
   loop, with the fused mlp_control1 chain visible as hatched blocks.
3. **`stack_yolo.png`** — 4-scheduler comparison (CPSAT/HEFT/EDF/PEFT) on
   the same 1+4+2 workload, baseline (no fuse/split applied).
4. **`granularity_hint.json`** + **`granularity/granularity_result.json`** —
   the agent's raw output: 330 candidates considered, scored top picks
   with predicted Δmakespan for each.
5. **`fuse_hint.json`** — the Contract-2 hint that was actually applied
   (top fuse candidate; the top split was conv2d_s8 which Phase 1e doesn't
   realize yet — see honest-scope in AGENTIC_LOOP.md).
6. **`mlp_control_fused.graph.json`** — the rewritten IR after
   `apply_fusion_hint.py --pairwise` collapsed mlp_control's 7
   dispatches into 4 fused linear+elu chains.
7. **`before_HEFT.json` / `after_HEFT.json`** — the two scheduler fixtures
   that feed the before/after PNG.

## Reproduction

```bash
cd /scratch2/agustin/ModelBlaster

# Step 1: baseline 4-scheduler scan
for s in CPSAT HEFT PEFT EDF ; do
    /tmp/run_one_solver_v2.sh "$s" 600 configs/agentic_fuse_split_demo.yaml fuse_split_yolo
done

# Step 2-3: agentic candidate generation + scoring
PYTHONPATH=/scratch2/agustin/XPU-RT/xpu-rt \
    /scratch2/agustin/miniforge3/envs/merlin-dev/bin/python \
    /scratch2/agustin/XPU-RT/scripts/granularity_loop.py \
    --networks-json /scratch2/agustin/XPU-RT/data/toplevel/networks_1yolo_4mlp_2dronet_firesim.json \
    --baseline-solver greedy --max-per-type 3 \
    --out-dir artifacts/agentic_fuse_split/granularity \
    --emit-hint artifacts/agentic_fuse_split/granularity_hint.json

# Step 4: realize the hint
uv run python -m modelblaster.pipeline.apply_fusion_hint \
    --hint artifacts/agentic_fuse_split/fuse_hint.json \
    --model mlp_control \
    --ir examples/mlp_control/int8/generated/graph.json \
    --out artifacts/agentic_fuse_split/mlp_control_fused.graph.json \
    --pairwise

# Step 5: re-schedule (swap IR, run, restore)
cp examples/mlp_control/int8/generated/graph.json /tmp/orig_mlp.json
cp artifacts/agentic_fuse_split/mlp_control_fused.graph.json \
   examples/mlp_control/int8/generated/graph.json
/tmp/run_one_solver_v2.sh HEFT 60 configs/agentic_fuse_split_demo.yaml after
cp /tmp/orig_mlp.json examples/mlp_control/int8/generated/graph.json

# Step 6: render before/after
uv run python scripts/render_compare_gantt.py \
    --out artifacts/agentic_fuse_split/agentic_loop_before_after.png \
    --title "Agentic loop: schedule → analyze → fuse → re-schedule" \
    --deadline-ms 75 --x-max-ms 95 \
    --panel "BEFORE (83.08 ms, 170 disp)" \
            artifacts/agentic_fuse_split/before_HEFT.json "baseline" \
    --panel "AFTER (82.67 ms, 157 disp)" \
            artifacts/agentic_fuse_split/after_HEFT.json "fused mlp_control1"
```

## Headline numbers

| | Makespan | Dispatches | Notes |
|:--|---------:|-----------:|:------|
| CPSAT baseline | **77.72 ms** | 233 | best axis-A; still misses 75 ms target by 3.6% |
| HEFT baseline  | 83.08 ms | 170 | baseline before agentic acting |
| HEFT after fuse | **82.67 ms** | 157 | **after** agent applied `mlp_control1[0..5]` fuse |
| Δ from loop    | **−0.41 ms** | **−13 disp** | measurable improvement from a real agentic decision |
