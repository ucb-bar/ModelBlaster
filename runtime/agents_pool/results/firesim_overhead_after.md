# pthreadpool vs agents_pool — FireSim per-(model, pool) totals

Per-row: sum of `mean_time_ns` across every dispatch in the results.csv. Lower is better; ratio = after / baseline.

Both runs use the same generated kernels.c — the only variable is the parallel-for pool implementation: pthreadpool (xnnpack-vendored) vs agents_pool (raw pthreads + k_sem).

| profile_dir | baseline_ns (pthreadpool) | after_ns (agents_pool) | ratio |
|---|---:|---:|---:|
| `RVV/firesim_rocket_saturn/dronet/dronet.fp32/dronet_firesim_rocket_saturn_RVV_dronet.fp32/topo_0/results.csv` | 297,640,163 | 298,386,647 | 1.003x |
| `RVV/firesim_rocket_saturn/dronet/dronet.fp32/dronet_firesim_rocket_saturn_RVV_dronet.fp32/topo_0_1/results.csv` | 385,112,938 | 152,624,345 | 0.396x |
| `RVV/firesim_rocket_saturn/dronet/dronet.fp32/dronet_firesim_rocket_saturn_RVV_dronet.fp32/topo_0_1_2_3/results.csv` | 535,887,500 | 79,718,496 | 0.149x |
| `RVV/firesim_rocket_saturn/mlp_control/mlp_control.fp32/mlp_control_firesim_rocket_saturn_RVV_mlp_control.fp32/topo_0/results.csv` | 583,741 | 583,910 | 1.000x |
| `RVV/firesim_rocket_saturn/mlp_control/mlp_control.fp32/mlp_control_firesim_rocket_saturn_RVV_mlp_control.fp32/topo_0_1/results.csv` | 52,641,749 | 351,711 | 0.007x |
| `RVV/firesim_rocket_saturn/mlp_control/mlp_control.fp32/mlp_control_firesim_rocket_saturn_RVV_mlp_control.fp32/topo_0_1_2_3/results.csv` | 108,229,352 | 242,587 | 0.002x |
| `scalar/firesim_rocket_saturn/dronet/dronet.fp32/dronet_firesim_rocket_saturn_scalar_dronet.fp32/topo_0/results.csv` | 297,753,377 | 298,354,376 | 1.002x |
| `scalar/firesim_rocket_saturn/dronet/dronet.fp32/dronet_firesim_rocket_saturn_scalar_dronet.fp32/topo_0_1/results.csv` | 385,188,696 | 152,611,343 | 0.396x |
| `scalar/firesim_rocket_saturn/dronet/dronet.fp32/dronet_firesim_rocket_saturn_scalar_dronet.fp32/topo_0_1_2_3/results.csv` | 535,897,689 | 79,726,640 | 0.149x |
| `scalar/firesim_rocket_saturn/mlp_control/mlp_control.fp32/mlp_control_firesim_rocket_saturn_scalar_mlp_control.fp32/topo_0/results.csv` | 584,770 | 585,215 | 1.001x |
| `scalar/firesim_rocket_saturn/mlp_control/mlp_control.fp32/mlp_control_firesim_rocket_saturn_scalar_mlp_control.fp32/topo_0_1/results.csv` | 52,643,207 | 353,339 | 0.007x |
| `scalar/firesim_rocket_saturn/mlp_control/mlp_control.fp32/mlp_control_firesim_rocket_saturn_scalar_mlp_control.fp32/topo_0_1_2_3/results.csv` | 108,232,132 | 244,818 | 0.002x |

## Aggregate per topo (mean ratio across both backends and models)

| topo | n | mean ratio |
|---|---:|---:|
| topo_0 | 4 | 1.001x |
| topo_0_1 | 4 | 0.201x |
| topo_0_1_2_3 | 4 | 0.076x |

## Headline takeaways

- `topo_0` (pool=1, sequential): exact match (1.00x). Both backends fall
  back to the no-fanout path; only the reference kernels run, no
  parallel-for dispatch. Confirms we didn't perturb the per-kernel
  cost.
- `topo_0_1` (pool=2): mean 5x faster (1/0.20).
- `topo_0_1_2_3` (pool=4): mean 13x faster (1/0.076).
- Smaller-op model (mlp_control) sees the largest ratio (~500x) because
  its per-op work is on the order of pthreadpool's per-call overhead,
  so dispatch cost dominated the original numbers. agents_pool's
  ~20k-cycle overhead leaves the mlp_control kernels essentially
  bottlenecked on the kernel itself again.
- Larger-op model (dronet) sees ~6.7x at p=4, where the conv kernels
  are large enough that pthreadpool's overhead was a smaller fraction
  of total. Still a substantial reduction.

These match the bench microbench's ratio of 27,018,896 / 20,066 = 1346x
peak overhead reduction at the dispatch level — but not all of that
shows up as wall-time speedup since real kernels do real work between
dispatches.

## Het schedule (xpurt) round-trip — same kernels

Sched: `schedules/scheduled_networks_mlp_control_dronet_firesim_het_profiled.json`,
backends rvv+scalar, registry chipyard_hetero_example.json. Both
networks PASS goldens.

Per-instance wall times (mtime ticks; 1 tick = 1 µs at 1 GHz target):

| network    | pthreadpool | agents_pool | ratio |
|---|---:|---:|---:|
| dronet     | 281,060     | 280,936     | 1.000x |
| mlp_control| 503         | 503         | 1.000x |

The xpurt het schedule's pool registration uses pool_size=0 (NULL pool)
for both kinds because each backend has only one hart in the
chipyard_hetero_example registry. With NULL pool, parallel_<op>
wrappers fall back to the synchronous-on-scheduler-worker path — no
pool calls are made, so there's nothing for either backend to differ
on. This is the expected outcome: the xpurt single-hart-per-kind
scheduling pattern is orthogonal to the pthreadpool overhead. The
multi_demo sweep above is where the win shows up.

The historical post-defer-printf baseline at commit c02b3a29605
reported dronet wall=38,358 mtime ticks. Our 281k here uses the
reference (non-LLM) dronet kernels — that's expected; the
LLM-optimized cache shrinks per-op cycles ~7x but is orthogonal to
this refactor.
