# Band-compliance audit — headline workload (4 MLP + 2 Dronet + 1 Yolo)

## What this measures

Every scheduler in `xpu-rt`'s registry (excluding ones that crash before
producing a fixture — MOSEK, ML/RL placement, MILP-backend variants)
is run against the user's headline workload at the canonical
frequencies:

- mlp_control @ 10 ms × 4 instances
- dronet     @ 20 ms × 2 instances
- yolov8_nano (aperiodic, 1 instance)

For each scheduler's output fixture the band invariant is checked:

> Every periodic dispatch belonging to instance `k` of network `N`
> with period `P_N` and window `W_N` and base release `R_0` must
> satisfy `R_k = R_0 + k·P_N ≤ start AND start + duration ≤ R_k + W_N`.

Non-periodic ops (yolov8) are counted but not band-checked.

Workload JSON:
`/scratch2/agustin/XPU-RT/data/toplevel/networks_1yolo_4mlp_2dronet_firesim.json`.

Source code: `xpu-rt/diagnostics/band_invariant.py` +
`scripts/audit_band_compliance.py`.

Raw CSV: `band_compliance.csv`.

## Findings

Sorted by deadline-violation count.

| solver           | n_ops | rel_viol | dl_viol | worst_dl (ms) | makespan (ms) | status      |
|:-----------------|------:|---------:|--------:|--------------:|--------------:|:------------|
| **decomposed**   |   248 |        0 |   **0** |         0.000 |        75.571 | clean       |
| **greedy**       |   233 |        0 |      10 |         0.476 |        60.476 | near-clean  |
| max_min          |   227 |        0 |      40 |        22.450 |        76.896 |             |
| cpsat            |   198 |        0 |      40 |        69.548 |       121.551 | unexpected  |
| min_min          |   209 |        0 |      42 |        20.075 |        90.714 |             |
| random_list      |   203 |        0 |      55 |        58.986 |       148.546 |             |
| greedy_periodic  |   236 |        0 |      67 |        31.885 |        61.196 |             |
| fifo             |   241 |        0 |      70 |        44.854 |       166.519 |             |
| fastest_device   |   265 |        0 |      75 |        31.895 |        85.642 |             |
| round_robin      |   287 |        0 |      78 |       117.007 |       408.812 | worst       |
| peft             |   234 |        0 |      84 |        46.237 |        56.242 |             |
| critical_path    |   234 |        0 |      88 |       106.691 |       116.785 |             |
| edf              |   234 |        0 |      88 |       106.691 |       116.785 |             |
| heft             |   245 |        0 |      88 |        44.424 |        54.430 | smallest mksp, worst dl |

### Headline takeaways

1. **Only `decomposed` honors the band invariant fully.** Every other
   scheduler in the registry produces deadline misses on this
   workload, ranging from 10 (greedy) to 88 (heft / critical_path /
   edf).

2. **None of the schedulers violate the release lower bound.** This
   confirms `min_start_t` is universally honored — the gap is on the
   deadline side. Release-time logic in `scheduler_heft.py:154` plus
   the periodic-aware code in `greedy_scheduler.py:150-164` covers
   the lower bound, but the upper bound (`max_end_t`) is not actively
   enforced by the heuristic family. This is Phase A2's target.

3. **CPSAT producing 40 deadline violations is unexpected.** CPSAT
   encodes `max_end_t` as a hard constraint in `scheduler_cpsat.py:168`,
   so seeing violations indicates either (a) the time limit (60 s)
   pushed it to an infeasible-but-best-effort solution, or (b) the
   workload as configured is intrinsically infeasible at this
   frequency mix. Worth a binary-search of `time_limit` and a
   `frequency_feasibility` (Phase B1) check on the workload.

4. **Heft has the smallest makespan (54.4 ms) but the most deadline
   violations (88).** This is the classic accuracy/throughput trade
   the band-aware audit is meant to expose: makespan-only schedulers
   trade deadlines for compaction.

5. **Critical_path and EDF emit identical fixtures.** Same ops count,
   same makespan, same violation count — they're behaving as a single
   solver here. Worth confirming in the registry that they aren't
   aliased.

### Why `n_ops` varies (198–287)

The fixture writer trims periodic instances that occur entirely after
the non-periodic makespan
(`postprocessing.trim_periodic_after_nonperiodic_makespan`). A
scheduler that finishes yolov8 earlier keeps fewer periodic instances
in the fixture, so `n_ops` shrinks. This is a fixture-trimming
artifact, not a solver capability difference — the band check is
applied only to ops the fixture contains.

## What's next

- **Phase A2** — patch the HEFT family to mark `deadline_miss=True`
  on overrunning ops instead of silently allowing them. Already
  honest-marking, not strict-rejecting (user-confirmed).
- **Phase A3** — make compaction and automerge band-safe (no
  shift past `max_end_t`, no merge across instance boundaries).
- **Phase A4** — Gantt overlay showing the band rectangles and red
  overruns so a human reader can immediately spot the violations
  this CSV counts.
- **Phase B1** — `frequency_feasibility` formula will tell us
  whether CPSAT's 40 violations are caused by an infeasible
  configuration (workload-side) or a solver-time issue (solver-side).
- **Phase F1** — diagnose why CPSAT, which should enforce
  `max_end_t` as a hard constraint, doesn't on this workload.
