# FireSim sweep — LLM kernels, modelblaster_pool vs pthreadpool

Multi-model pool sweep (`modelblaster/examples/multi_demo/run.sh` with
`POOL_SIZES=1,2,4`) on FireSim quad-rocket-saturn. Same LLM-cached
kernels in both columns; only the parallel-for pool implementation
changed. Cycle counts are profile-sum (per-op `rdcycle` deltas) at the
1 GHz target clock.

## Headline

| model | target | p | pthreadpool baseline | modelblaster_pool | speedup |
|---|---|---|---|---|---|
| dronet | RVV | 1 | 39.273 ms | 38.212 ms | 1.03× |
| dronet | RVV | 2 | 167.397 ms | **24.869 ms** | **6.73×** |
| dronet | RVV | 4 | 321.448 ms | **CRASHED** (mcause=5 Load access fault) | — |
| dronet | scalar | 1 | 297.716 ms | 286.997 ms | 1.04× |
| dronet | scalar | 2 | 362.784 ms | **146.923 ms** | **2.47×** |
| dronet | scalar | 4 | 535.896 ms | **76.861 ms** | **6.97×** |
| mlp_control | RVV | 1 | 0.197 ms | 0.192 ms | 1.02× |
| mlp_control | RVV | 2 | 46.805 ms | **0.156 ms** | **300.0×** |
| mlp_control | RVV | 4 | 107.828 ms | **0.154 ms** | **701.3×** |
| mlp_control | scalar | 1 | 0.585 ms | 0.585 ms | 1.00× |
| mlp_control | scalar | 2 | 52.642 ms | **0.353 ms** | **149.3×** |
| mlp_control | scalar | 4 | 108.237 ms | **0.244 ms** | **443.3×** |

## Reading the table

- **p=1 sequential** is parity in every row (1.00–1.04×). modelblaster_pool
  doesn't engage when there's only one worker, so this is the noise
  floor. Confirms no regression on the sequential path.
- **dronet** wins are bounded by the actual kernel work: at p=4
  scalar where the pool engages but each conv slice is still 100 k–
  10 M cycles, modelblaster_pool reclaims ~7×. The pthreadpool dispatch
  was meaningful but not the whole story for a heavy network.
- **mlp_control** wins are extreme (300–700×) because the kernels are
  tiny — `linear M=1;K=64;N=4` is ~3 k cycles of work. pthreadpool's
  ~13 M-cycle dispatch had been 99.8 %+ of total wall time. With
  modelblaster_pool's ~20 k-cycle dispatch, the work and the dispatch
  finally have the same order of magnitude.

The agent's prediction at the end of the modelblaster_pool refactor — "(1)
Re-run the multi_demo sweep with the LLM-optimized cached kernels to
double-check the per-op ratios at p=4 (likely closer to 4-5x for
dronet since the kernel cost dominates more)" — lands within ballistic
range: 6.97× at p=4 scalar, where the kernel cost actually dominates
(scalar dronet conv2d is ~10 ms per op, not 1-2 ms like RVV).

## dronet @ p=4 RVV — reproducible crash

`dronet@p4` with the LLM-optimized RVV kernel **reliably crashes** on
FireSim with:

```
mcause: 5, Load access fault
mtval:  0xcc747057
```

Pattern:

- works on p=1 (sequential, full OC range to one worker)
- works on p=2 (OC split in two)
- crashes on p=4 (OC split in four)

The fault address `0xcc747057` is wild — outside any sensible RAM
range. Most likely the LLM's RVV conv kernel makes an alignment or
boundary assumption that holds when the OC tile is large but breaks
once `oc_per` shrinks to 1/4 of OC. This is the same class of error
the firesim re-rank gate caught earlier in Plan B
(`firesim_eval_design.md`); the kernel currently in cache passed
spike correctness AND firesim re-rank for shapes the optimizer
sampled, but those shapes didn't include "tiny OC slice." The full
uartlog trace is at `/tmp/firesim_sweep/uartlog_dronet_p4_rvv_crash_repro.txt`
(also reproduced on a fresh run at `rvv_p4_repro.log`).

**Mitigation paths**:
1. Add a sanity-check dispatch with a deliberately-tiny OC slice to
   the optimize-loop's firesim re-rank set, so the gate catches this
   class of bug going forward.
2. The wrapper in `generate_skeleton.py`'s `parallel_conv2d`
   short-circuits to a sequential kernel call when `(size_t)OC < T` —
   we could similarly short-circuit when `oc_per` would be below some
   minimum (e.g. `oc_per < 4`). That would mask the bug at the cost
   of giving up the parallelism win at the network's smallest OC
   shapes.
3. Re-run the LLM optimize loop for dronet's conv kernels with the
   memory-aware prompt + `oc_blocked` seed already merged
   (`f4a6dc14007`) but force-invalidate the existing cache first, so
   the optimizer considers fresh candidates including the
   tile-boundary-correct ones.

For now, the dronet@p4 RVV slot in the schedule profile is empty.
Schedules built against the LLM RVV firesim profile must use
`topo_0` or `topo_0_1` for dronet, or fall back to scalar at p=4.

## Where the data lives

```
gen/profile_firesim_sweep_llm_v3/RVV/firesim_rocket_saturn/
    dronet/.../topo_0/results.csv            # 38.21 ms
    dronet/.../topo_0_1/results.csv          # 24.87 ms
    (dronet/.../topo_0_1_2_3 — missing, see crash above)
    mlp_control/.../topo_0/results.csv       # 0.19 ms
    mlp_control/.../topo_0_1/results.csv     # 0.15 ms
    mlp_control/.../topo_0_1_2_3/results.csv # 0.15 ms

gen/profile_firesim_sweep_llm_v3_scalar/scalar/firesim_rocket_saturn/
    dronet/.../topo_0/results.csv            # 287.00 ms
    dronet/.../topo_0_1/results.csv          # 146.92 ms
    dronet/.../topo_0_1_2_3/results.csv      # 76.86 ms
    mlp_control/.../topo_0/results.csv       # 0.59 ms
    mlp_control/.../topo_0_1/results.csv     # 0.35 ms
    mlp_control/.../topo_0_1_2_3/results.csv # 0.24 ms
```

11/12 cells. The single missing cell is the known-broken kernel above.
