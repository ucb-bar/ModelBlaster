# XPU-RT ⇄ ModelBlaster loop — round 1

- bundle: `/scratch2/agustin/XPU-RT/artifacts/iterate/firesim_batch.json`
- workload: `/scratch2/agustin/XPU-RT/data/toplevel/networks_1yolo_4mlp_2dronet_firesim.json`
- runner: `firesim`
- deadline_us: 65.0

## Per-candidate

| id | axis | label | predicted µs | measured µs | Δ% | advisor verdict |
|----|------|-------|-------------:|------------:|---:|-----------------|
| `baseline` | baseline | decomposed | 75570.5 | 6552.2 | -91.3% | — [coarsen] |

## Notes

- Predicted µs = XPU-RT scheduler's makespan estimate from the schedule fixture (start_time + duration of latest dispatch).
- Measured µs = harness `xpurt_trace.csv` last `actual_end_cycles` / `clock_mhz`.
- Advisor verdict run on the **measured** report so remedies (rebalance / coarsen / finer) reflect what FireSim actually showed.
- `predicted_vs_actual.png` per candidate dir overlays the two timelines on the same axes.
