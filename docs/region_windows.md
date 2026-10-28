# Region windows (`--region-windows off|auto`)

Larger / chunked `fuse_region` candidates for graphs the flow's finders cannot regionise. Default **off**:
nothing runs, nothing is written, the member cap stays 256, every output is byte-identical to a tree without
the feature (evidence in `l3_farm/region_loop/SUMMARY.txt`).

## Why

The graph-agent pilot (`l3_farm/graph_agent_pilot/SUMMARY.txt`) started a Claude Code agent from the flow's own
final state on 8 branchy problems. The arm that only had the flow's existing hint kinds, but `fuse_region` with
**agent-chosen members** plus a megakernel the agent wrote for the region slot, reached a 2.38x geomean over the
flow headline; kernel-only edits reached 1.27x. The flow had tried `region:all` on five of those problems and
rejected it, and could not even **propose** a region for 34_VanillaRNNHidden (1062 IR ops, 805 executed) or
15_DenseNet121 (305 ops, 246 executed): `region_slots.MAX_MEMBERS = 256` and only three whole-graph patterns
(`rnn_stack`, `attention_block`, `all`). In the pilot 34 used four 64-step recurrence windows (+ a 37-op stack
region) and 15 two whole dense blocks.

## What exists / what is new

| | exists | new |
|---|---|---|
| region IR op, contract, composite seed, export | `region_slots.py` (`fuse_region`, `RegionKernelSpec`, `region_seed`), `kb_export` | unchanged |
| member cap | `MAX_MEMBERS = 256` constant | `region_slots.max_members()`: `$MB_REGION_MAX_MEMBERS`, default 256 (`MAX_MEMBERS` kept) |
| explicit-member region hint | `fuse_region` hint with `members` | `windows`: several member lists, one region op each (`graph_hints._apply_fuse_region_cuda`) |
| proposal | `Caps._propose_regions` (3 patterns) | `region_windows.py` + `Caps._propose_windows` |
| decision | paired variant-vs-original, `region_cap` survival budget | unchanged (window hints are ordinary `fuse_region` hints, ids `region:win:<label>`) |

## Flags

| flag | env | default | meaning |
|---|---|---|---|
| `--region-windows off\|auto` | `MB_REGION_WINDOWS` | off | the feature |
| `--region-window-sizes "256,128"` | `MB_REGION_WINDOW_SIZES` | `256` | the window sizes M (ops, views included) of the `chunk<M>` / `motif<M>` candidates; `blocks<M>` uses the largest |
| `--region-window-candidates N` | `MB_REGION_WINDOW_CANDIDATES` | 3 | at most N candidate hints per IR |
| `--region-window-hard-cap N` | `MB_REGION_WINDOW_HARD_CAP` | 1100 | `whole` is proposed only up to N members |
| `--region-window-min-ops N` | `MB_REGION_WINDOW_MIN` | 4 | a window needs this many non-alias ops |
| - | `MB_REGION_MAX_MEMBERS` | 256 | the member cap of `fuse_region` (set by `Caps.apply_env` to `max(256, hard cap, sizes)` when windows are on) |

Round key: with the flag on, `caps_region_windows` and `caps_region_windows_cfg` (sizes, candidates, cap, min) enter
the round key and `summary.json` `capabilities`; off, neither key exists (the summary of a run with other
capabilities is unchanged).

## The proposer (`pipeline/region_windows.py`)

Per IR per round, on the profiled IR (accepted hints applied):

1. **Eligible runs.** Maximal runs of consecutive ops (IR order) that are live (`graph_prune`) and not already region ops.
2. **Cut costs.** `cost[j]` = number of live tensors crossing the cut before op j. A cut with cost 1 is a *single-frontier* cut
   (one tensor enters the next part): between dense blocks, residual blocks, inception modules.
3. **Alias legality.** A view (`view`, `chunk2_c1`) is pointer arithmetic that must stay inside its region
   (`fuse_region`: "an alias inside the region is read outside it"). `alias_reach` records the last reader of each view; a
   window `[a,b)` is legal iff no view produced in it is read at or after b. Inside a DenseNet block every later concat reads
   the earlier layers' views, so no cut inside a block is legal: blocks are atomic units.
4. **Motifs.** `detect_motifs` finds periodic segments of the *coarse* op-kind sequence (dtype suffix, concat arity and slot
   variant ignored) by the longest runs of `t[i] == t[i+p]`, primitive period first, at least 3 repeats and 32 ops. 34: period 4
   (view, cat, linear+tanh, linear) x 256. Motif ends move to legal cuts.
5. **Candidates** (a window set each; all windows are contiguous, hence convex by construction):
   * `whole`: one window over the largest run, up to the hard cap;
   * `chunk<M>`: windows of at most M ops; inside a motif balanced windows of whole periods (34, M=256: 4 x 64 steps), elsewhere cuts in
     the second half of the piece at a single-frontier cut when there is one, else the cheapest, always legal; unsplittable blocks
     larger than M stay plain ops; adjacent fragments coalesce;
   * `blocks<M>`: the largest single-frontier cut that fits (only when the graph has single-frontier cuts: 15's dense blocks);
   * `motif<M>`: only the periodic parts.
   Identical sets keep the first label.
6. **Validation.** Every window is applied to a working copy with `region_slots.fuse_region` (the one authority; the next window sees the
   previous ones fused). A window it refuses is dropped and reported, never proposed.
7. **Ranking.** `est_gain_pct` = 8% (the flow's region estimate) x the covered ops' time share (profile; op-count share without);
   ties favour fewer windows. Top `N` candidates become hints. A candidate id already in the registry (any status) is not
   proposed again.

Members are stored as **atom names** (`graph_hints.atoms`), so a window survives fusions accepted later; `_locate` maps atoms to the current
ops. The hint key is `windows`, not `groups`: `groups` belongs to the split hints and `record_decision` moves it to `tried_groups` when a
hint is accepted.

## Measurements (`l3_farm/region_loop/results/windows_*.txt`; CPU, real final IRs of the headline runs, synthetic weights of the right shapes)

See SUMMARY.txt: windows / ops proposed, largest exported region (source KB, signature pointers), proposer and `export_problem` seconds.

## Invariants

* Flag off: `Caps.propose` / `decide_kinds` / `round_key` / `summary` return exactly what they did; `max_members() == 256`; the `windows` branch of the
  applier is only reached by hints that carry it. Proven by the in-repo loop test and by the base-vs-new identity runs.
* A proposed hint applies on the graph it was proposed on (validated window by window with the real `fuse_region`); on a graph where it no longer
  matches it is skipped softly (`prov["skipped"]`).
* No GPU, no LLM, no new file in the work dir.

## Limits / known risks

* Window hints are decided by the existing slot optimiser (2 decision steps by default for regions, `MB_REGION_DECISION_STEPS`): that optimiser did not
  exploit `region:all` in the earlier runs. The windows lift the proposal limit; making the region slot pay needs the region loop
  (`docs/region_loop.md`).
* A `whole` region of 1062 ops has a 1067-pointer entry signature and a 318 KB composite seed (export is fast; builds were not tried: no GPU).
* Overlapping candidates conflict (same ops): the existing conflict waves decide them one after the other; once one is accepted the others no longer match.
