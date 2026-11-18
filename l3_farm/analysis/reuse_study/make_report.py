#!/usr/bin/env python3
"""Assemble REPORT.md from data/*.md tables + the hand-written text below (numbers come from the data files; rerun after analyze_runs.py)."""
import json
from pathlib import Path
H = Path(__file__).resolve().parent
D = H / "data"
up = (D / "table_uplift.md").read_text()
cov = (D / "table_compact.md").read_text()
fix = (D / "table_seedfix.md").read_text().splitlines()[:12]
mat = (D / "table_matrix.md").read_text()
avail = [l for l in (D / "table_avail.md").read_text().splitlines() if l.startswith("| MEAN")]
txt = f"""# Reuse-uplift study: how much better could the headline Level 3 sweep have been with wider / earlier kernel sharing?
Offline, zero LLM. Scripts and data: `analysis/reuse_study/` (`build_data.py`, `coverage.py`, `summarize.py`, `tiers.py`, `timing.py`, `prev_cov.py`, `potential.py`,
`cost.py`, node side `reuse_node.py` + `reuse_run.py`/`rs.py`, `analyze_runs.py`, `make_tables.py`; raw runs `data/runs/`). Nothing under `results*` or the stores was modified.
Population: 42 finished headline problems (`model_runtime.json`; 18 and 44 from `results_headline_heavy`); skipped: 7, 10, 14, 15, 19, 22 (reruns), 16, 48. Eight of them
(1, 6, 35, 36, 37, 39, 40, 49) ship one whole-model `region__` slot: a problem-specific kernel, nothing shareable at op-kind level, so they are 19% of the mean model time.
Donors = other problems' verified kernels (origin problem from `problem_id`): `store` = gk_impl of knowledge_global_headline v0258 (441 codes), `win` = store + every problem's
cache winners (1,251 rows, used below), `all` = every verified record >= 1.02x (8,382). Compatible = same op kind (= entry + signature + dtype). Exact = the donor was verified on the op's
exact shape; near = nearest by `search_fixes.shape_distance` (<= 2.5, nearest K=4 codes).

## Answer
* **The sweep already shares a lot and the remaining headroom is small.** 70% of model time runs a winner found by the problem's own search, 6% runs a kernel shared in from another
  problem (store seed / refresh deliver 303..388 records at start, 2..53 refreshes), 19% is region slots, and only **4.4% of the mean time is still on a seed or a <1.25x kernel**
  (official `seed_left`: only 17 `cat` 61%, 18 `cat_2` 11%, 46 `bmm_strided` 36%, 47 `reduce` 15%). Compatible foreign kernels exist for most of the rest: **81% of non-region time has
  a donor in the `win` pool** (71% store, 71% first-sweep records; exact shape 13%), but for almost all ops the donor's gain is a few percent at best.
* **Measured uplift (12 problems chosen for the largest foreign coverage of seed / low-gain ops, 6 fresh-process CUDA-graph runs per arm, same node and session as the shipped arm):**
  +own-wide reuse and nearest-4 foreign kernels move the geomean e2e by **+3.6%** (mean +3.7%, best-arm-per-problem upper bound +4.0%); exact-shape donors alone +2.3%.
  Real wins: **29_SwinMLP +8%** (30_SwinTransformerV2's linear / gelu / strided_copy kernels on identical shapes; they only existed ~11 h later, after 29 had finished), **47_NetVladNoGhostClusters +12-14%** (46's `reduce_f16` 232 -> 92 us and `ewise2` kernels, same shapes) and **3_DeepNarrowMLP +11-14%** (28's fused linear+relu, 5's linear); everything
  else <= +4% and 17, 23, 25, 31, 46 exactly 0%. A control (own siblings only, same wide kind limits) gives +0.5% geomean (n=5; 29 +1.4%, 30 +1.2%) and is the whole of the 3-4% on 20/26/30 (those arms picked
  `cache:` candidates of the problem itself), so **foreign sharing proper explains ~+2.5% geomean on this selected set, concentrated in three problems (3, 29, 47).**
* **Sweep-wide extrapolation:** the 12 problems are the most favourable ones; the other 30 non-measured have lower seed/weak shares and mostly near-only donors. Expect about **+1.5% geomean
  (range 0..+3%) and +2% mean** over the 42 if kernels had been shared everywhere from the start, i.e. **no change in the headline conclusion**; three problems (47, 3, 29) account for most of it.
  Pure sharing without search ("cold", below) does **not** replace the search: 0.02-0.4x of shipped except where few ops matter (17: 0.99x, 3: 1.12x).
* **Earlier sharing matters per consumer, not on average:** 29's +8% needed kernels from 30 that were created ~11 h after 29 ended (only reachable by running 30 first or a second pass); 47 started 3.6 h after its sibling 46 and the donor was already in its seed (ewise2) or existed before start but was not delivered (reduce): +12%.
  The mirror case is 46 (started first): its seed-left `bmm_strided` (36%) has an exact donor only from 47 (1.05x record speedup) and measurably nothing to gain (0%).
  The sweep ran 13 h; later-starting problems had more at seed (store v0003 -> v0234) but also fewer unoptimised ops, so early-start problems (1-5, recurrent, MLPs) had the least coverage and the most to gain.

## 1. Coverage per problem (share of summed final-model op time, %; donor pool `win`)
own = winner found by the problem; shared-in = winner whose origin (resolved through mb-reuse / route chains) is another problem; seed/weak = on hand/fused seed or < 1.25x vs the
first-profile seed time; S: = how that seed/weak share splits into exact-shape donor / near donor (<= 2.5) / none; any donor = exact or near (all ops); the next column splits the covered
time by when the first compatible donor reached the problem: in its seed (round 0) / via a refresh / created before the round-0 seed but never delivered / created during its run (not delivered)
/ only after it finished; start = hours from sweep start; store ver = knowledge-store snapshot in its seed; last = zero-GPU prediction (record speedups on exactly the same shape) of exact-donor gain.
{cov}

Mean over 42: own 70.3, shared-in 6.2, seed/weak 4.4 (exact 1.2, near 3.0, none 0.2), region 19.0. Covered time, mean over 42: any donor {avail[0].split('|')[2].strip() if avail else ''}%
({avail[0] if avail else ''}).

Donor pool sensitivity (mean over 42, % of time; S = seed/weak ops): `store`: A exact 5.2 / near 47.6 / none 23.8, S 1.2/2.9/0.2; `win`: 12.0/48.8/15.7, S 1.2/3.0/0.2; `all`: 13.9/47.6/15.0, S 1.3/3.0/0.2;
first-sweep records (other problems, available at t=0): 12.3/42.3/21.9, S 0.5/2.7/1.3. Widening the pool beyond the winners adds almost nothing for the seed/weak ops.

Seed-left / weak ops that foreign kernels could in principle fix (`data/table_seedfix.md`, `data/seedfix.json`):
{chr(10).join(fix)}
Only 46/47 have exact-shape donors; the rest are near donors (distance 0.1-1.4) with record speedups of 1.0-3x (cat, batchnorm, gather) and share < 4% each except 17's `cat`.
In 17 the donor (45's `cat2_c1`, 2.3x on its shape) was delivered by refresh, and a nearest-K pass measured it at no gain (cat is memory bound; the sweep's 3 rounds also found none).

## 2. Availability timing (when did compatible kernels exist vs when the consumer planned)
By start order (14 problems each; covered share of non-region time / delivered in seed / seed or refresh / existed before round 0 but not delivered):
first third 54% / 21% / 45% / 0%, middle 89% / 11% / 41% / 3%, last third 88% / 32% / 49% / 31%. Mean over 42 (`{avail[0] if avail else ''}`).
Reading: only ~1/6 of the coverage was in the seed of the consuming problem (round 0); about as much arrived by refresh in rounds 1-2; ~10% existed before round 0 but was never
delivered (store keeps winners only; e.g. 27, 33, 43, 30 had 30-99% undelivered), ~13% was created while the consumer ran and ~6% only after it finished. Donors that were
actually chosen in the measured arms (`data/donors.json`, slot-level gain in us x instances, nk4/ex/gbest, foreign only): refresh 208, created-during-run-not-delivered 93, pre-start-not-delivered 539 (47's reduce),
seed 75, only after the consumer finished 337 (29, from 30); versus 2,275 from the problem's own cache (wider sibling reuse). The early/late arms (`reuse_node.py --arms early,late`, donors filtered by creation time) were implemented but not run (time).
Donor matrix (nearest donor of covered seed/weak time; consumers in parentheses; `data/matrix_S_top1.json`, all ops `data/matrix_all_top1.json`):
{mat}
Whole-model: 12_VGG19/18/30/10/44/43/29/23 are the biggest nearest donors (convs, linear, fused conv+bn); typical pairs 29<->30, 11->12, 46<->47, 25<->26, 23->20/21, 43->44, 41<-42.

## 3. Measured uplift (GPU, L40S jobs 5325505 / 5325510; sheet `data/table_uplift.md`, raw `data/runs/<p>/ru.json`)
Arms (all kb_reuse cross-application: every candidate on every shape, per-shape best kept if >3% faster, whole-model guard; max 6 kinds, share >= 1%; `reuse_node.py`):
sibw = own winners only, wide limits (control, only on later problems); ex = + exact-shape foreign; nk4 = + 4 nearest foreign per shape; gbest = + 4 best foreign by speedup of the kind;
cold = empty cache (hand seeds) + foreign nearest-4 only (no own winners). Shipped schedule (concurrency) kept for all arms. Per-slot and output correctness checked in a 7th run of every arm
(0 failures unless marked FAIL). Torch = CUDA-graph replay of the eager model, 3 invocations, same node. x vs shipped = shipped_now / arm. The shipped arm is rebuilt from the sweep's
winners and was slot-identical for 3, 31, 46 (others differ by variant bookkeeping, within 0-4% of the recorded sweep e2e; 30 is 11% slower than recorded, 1922 -> 2143, session/node effect).
{up}

Reading the table: 'first sweep us' is the earlier non-headline sweep (other nodes, ~3% noise). cold is limited to the 6 biggest kinds, so remaining kinds stay on hand seeds (25: 214 ms); it is
a lower bound of pure sharing, but shows the search is what makes most kernels; 17 (0.99x of shipped) and 3 (1.12x) show it can match when a few kinds dominate.
Zero-GPU predictor check: exact-shape record speedups predicted 14.4% for 47 (measured 12.2-13.8%) but 5% for 30 and 4% for 29 (measured <= 1.5%), so it is an optimistic ceiling.

## 4. Costs
Headline LLM spend of the 42 problems (state telemetry, assumed Opus list prices, lower bound): **$1,559** (6,205 calls). Spend attributed (by tokens) to op kinds whose final ops had an exact donor
created before round 0: **$87 (5.6%)**; with a near donor: **$426 (27%)** (30_SwinV2 $112, 27 $40, 26 $40, 29 $38, 44 $29, 23 $26). Because the measured near-donor gains are
<= 4% and the kinds' own search remained necessary (cold arms), treat the $426 as an upper bound of the spend that wider sharing could have replaced; the exact-donor $87 is the safer figure.
Zero-LLM cost of the offline pass itself: 4 reuse passes x ~6 kinds, 5-40 min GPU per problem (`wall_s` in `ru.json`).

## Caveats
Single measurement per arm (6 runs, CV < 0.5% typically; wins < 2% are noise; the 31/23/25/46/17 zeros are exact slot equality). 12 problems selected for high coverage, so averages
are an upper bound for the 42. Compositions are checked per slot and per output in a final run but not by a separate full-model correctness suite. Donor kernels come from the store under other shapes
(near arm): verified per slot on the consumer's shapes by the matrix pass, never assumed. `win` pool includes winners not in the store (not shareable by the real mechanism); store-only coverage is
70% vs 80%. Availability classes use `created` = earliest foreign record timestamp of the code vs the consumer's seed snapshot time; a code found independently by two problems counts as available
from the earlier one. Cost attribution is by kind token share, not per shape. 29 and 30 were run last without cold; 18 (heavy) was still running at the end and 44 was not run (time). Not run: early/late arms, sibw on
3/9/17/20/26/46/47, 44, cold with more than 6 kinds, 14/10/15 etc. (reruns pending).
"""
(H / "REPORT.md").write_text(txt)
print(len(txt.splitlines()))
