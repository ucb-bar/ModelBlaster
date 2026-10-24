# ModelBlaster as the graph optimiser (CUDA, Phase 2)

Phase 1 (`docs/kernelblaster_multislot.md`) made a graph partition a KernelBlaster problem
and let KernelBlaster propose one kind of graph change (fuse an elementwise consumer into its
producer), decided once per partition. This phase makes ModelBlaster the graph optimiser of
the flow: it chooses what to send to KernelBlaster by measured cost, applies graph-level
changes from every source through one hint layer, decides each change by measurement, and
iterates until nothing more is accepted. As in Phase 1, the "Design decisions" section lists
every non-obvious choice with the alternatives and why.

## Flow

```
IRs ──► profile (accepted hints applied, cached winners)          pipeline/kb_plan.profile_ir
         │ per-op times (driver MB_OP_US)
         ├─► MB proposals (fuse elementwise consumers, grow chains)  graph_hints.propose_from_profile
         ▼
        rank by (C entry, signature, shape) across IRs, select       kb_plan.rank / select / problems
         │ plan (modelblaster.kb_plan/v1)
         ▼
        KernelBlaster on the plan's problems ──► winners ──► cache   kb_ingest.ingest
         │                               └──► proposals (fuse, split) ──► hints
         ▼
        for the top undecided hints: original + variant problems of the affected ops,
        same seeds, same KB budget, both bests re-timed back to back ──► accept / reject
         │                                                            kb_graph_opt.evaluate_hint
         ▼
        accepted hints apply from the next round (fusions grow into chains); their new
        kernels go into the cache; stop when a round accepts nothing and sends nothing
         ▼
        model-level verification of every IR with everything applied  kb_ingest.verify_ir
```

`python -m modelblaster.pipeline.kb_graph_opt IR_DIR [...] --work W --kb KB_CHECKOUT` runs the
rounds; `gpu_eval/tools/kb_loop.sh` is still one round (now on the hints file, with
`STRATEGY=plan` for the cost-ranked selection and variants seeded from the original's best).

## Code map

| repo | file | role |
|---|---|---|
| MB | `pipeline/graph_hints.py` | hint schema, load/merge/decisions, producers (KB, MB profile, XPU-RT, Phase 1 registry), `apply_hints` (cuda / riscv), `verify_rewrite`, `unique_names` |
| MB | `pipeline/kb_proposals.py` | fused-op chains (`fuse_pair`, `fuse_chain`, `find_chains`, `parse_kind`, `make_chain_spec`, `bind_fused`, seeds from best stage kernels), `build_variant`, `remeasure`, `decide` |
| MB | `pipeline/kb_plan.py` | profile, rank/dedup, select, problems, `export_plan` |
| MB | `pipeline/kb_ingest.py` | `cache_plan` (per-slot lookups, coverage splits, cached stage kernels for fused seeds), `verify_ir` with hints |
| MB | `pipeline/kb_export.py` | `MB_OP_US`, slot variants (`slot_variant`), fused specs rebuilt from their names, `build_and_run(runs=)`, `--hints`, `--plan` |
| MB | `pipeline/kb_graph_opt.py` | rounds to a fixpoint; KernelBlaster as a subprocess |
| KB | `agents/opt_multislot.py` | `propose_splits` (kind `split`) next to `propose_fusions`; fused shapes in the roofline cost |
| KB | `knowledge/slots.py`, `impl_store.py` | `EvalResult.op_us`; fused shapes bucket by their first stage |
| KB | `scripts/run_multislot.py` | `--plugin FILE` (register `stub:` models from another process) |

## Hint schema (`modelblaster.graph_hints/v1`)

A hints file (`W/graph_hints.json`; the old `mb-fusion-registry/1` is read as one):

```json
{"format": "modelblaster.graph_hints/v1",
 "hints": [
  {"id": "fuse:fused__conv2d__batchnorm2d__relu", "kind": "fuse", "target": "cuda",
   "fused_kind": "fused__conv2d__batchnorm2d__relu",
   "chains": [["conv1", "bn1", "relu"]],
   "status": "proposed | accepted | rejected",
   "source": [{"by": "kb | mb-cost | mb-cache | xpurt | registry | user", "ir": "...",
               "est_gain_pct": 7.4, "rationale": "...", "ts": "..."}],
   "evidence": [{"problem": "...", "ir": "...", "original_us": 29.1, "variant_us": 26.9,
                 "gain": 0.074, "decision": "accepted",
                 "remeasured": {"original_us": [...], "variant_us": [...]}, "applied": [...]}]},
  {"id": "split:conv2d", "kind": "split_slot", "target": "cuda", "slot": "conv2d",
   "by": "shape_bucket", "groups": [{"bucket": "conv:N1,IC1,...", "ops": ["c1"], "us": 10.0,
   "eff_pct": 3.1}, ...]},
  {"id": "unfuse:...", "kind": "unfuse", "target": "cuda", "fused_kind": "...", "ops": ["..."]},
  {"id": "xpurt:fuse:mlp_control", "kind": "fuse", "target": "riscv", "network": "mlp_control",
   "dispatch_groups": [[0, 1, 2, 3, 4, 5]]},
  {"id": "xpurt:tile_split:yolov8_nano", "kind": "tile_split", "target": "riscv", "split_ops": [...]},
  {"id": "xpurt:shard:yolov8_nano", "kind": "shard", "target": "riscv", "shard_ops": [...]}]}
```

- **id** is the identity across problems, rounds and models: a fusion is decided per fused
  kind, a split per slot. Evidence from every problem that tried it accumulates on the record
  and the status is the majority of its decisions (a tie rejects).
- **scope**: `chains` (fuse) / `groups` (split) name the instances a proposal was made for.
  A variant applies only those; once accepted they move to `tried_chains` / `tried_groups` and
  the hint applies wherever its pattern matches (split: by shape bucket).
- **producers**: `from_kb_proposals` (KB's `kb-proposal/1`; pairs are grouped by the fused kind
  they produce, found by a dry-run fusion that also flattens fused producers), `propose_from_profile`
  (MB, from a whole-model per-op profile), `kb_ingest.cache_plan` (MB, from the cache's shape
  guards), `from_xpurt` (`fusion_hints/v1`, `split_hints/v1`, `shard_hints/v1`, `unfuse_hints/v1`),
  `load` (Phase 1 registry), and the `import` CLI.

### Applying

`apply_hints(graph, hints, specs, target=, statuses=("accepted",))` returns the rewritten graph,
the fused specs it needs and provenance (`applied`, `skipped` with reasons, `renamed`). The
graph records every application in `graph["graph_hints"]["rounds"]`, fused and split ops carry
`hint`, dispatch ids are renumbered and `depends_on` re-derived from tensor producers, and
`id_remap` maps old to new ids (list-valued, via the original op names each op is made of).

| kind | cuda | riscv |
|---|---|---|
| `fuse` (fused kind; chains or everywhere) | fused op (`kb_proposals.fuse_chain`), longest chains first | — |
| `fuse` with `dispatch_groups` (XPU-RT) | each group translated to an op-name chain (validated by `apply_fusion_hint`'s closure check, then required to be an exclusive producer→consumer chain) | `apply_fusion_hint.apply_hint` |
| `split_slot` | `slot_variant` annotations: one slot per shape group; ops/tensors unchanged | — |
| `unfuse` | stages restored (fp32 fused ops keep their tensors registered) | `apply_unfuse_hint` |
| `tile_split` | skipped: OC/N tiles run one op on several cores; a CUDA kernel already spans every SM and the driver runs one stream, so tiles would serialise (`split_slot` is the CUDA analogue for heterogeneous shapes) | `apply_split_hint` |
| `shard` | skipped: an RVV codegen width for composite-hart dispatch | `apply_shard_hint` |

Why the RISC-V appliers are not used for CUDA: `apply_fusion_hint` names the fused op
`__fused__<k0>__...` (or maps a few int8 pairs to RVV KernelSpecs) for the RISC-V codegen's
chained-call fallback, keeps `internal_tensors` and needs no signature; `apply_unfuse_hint`
requires int8 quant blocks and dtype suffixes; `apply_split_hint` and `apply_shard_hint` only
know `_s8` kinds and exist for multi-core placement. The CUDA flow needs a KernelSpec per fused
op (the slot contract) and fp32 tensors, so it has its own fusion mechanics; the hint layer is
shared and dispatches by target.

### Verifying

`verify_rewrite(ir_dir, graph, build=)` walks the rewritten graph with the torch walker (which
runs fused ops' stages) and requires it to match the original golden (the Level2/3 bound) and
the original walk exactly; with `build` it also exports the whole rewritten graph with seed
kernels and runs the multi-slot driver (partition + every op in isolation). Every variant in
the flow is a driver-checked problem, and the final model-level verification runs the driver
on the whole IR with every accepted hint and cached kernel.

## Mechanisms

### Fused-op chains

`fused__<k0>__<k1>[__<p1>]__<k2>[__<p2>]...`: stage kinds in order; stage i's param that reads
stage i-1's output is spelled out only when that stage has more than one activation input
(`add`), so two-stage names are unchanged from Phase 1 and `parse_kind` is unambiguous. The op
has `sub_ops` (stages), `internal` (stage i's output, i < n-1), `shape` `{"a": ..., "b": ...}`,
and its name joins the original names with `+`. The signature is stage 0's parameters (`a_`),
then stage 1's minus the intermediate (`b_`), ...; every intermediate stays as scratch.
`fuse_pair` flattens either side, so conv→(bn→relu) and (conv→bn)→relu are the same op.
`find_chains` finds every exclusive chain of a kind (each intermediate read once, by the next
stage through the kind's param, and not a graph output). The seed calls the stages back to back;
`cuda_seed(lib, sub_sources)` calls the current best kernel of each stage, each wrapped in its
own namespace (`stage_<kind>`, includes hoisted, macros undefined), else the hand kernels.

### Shape-specialised slots (`split_slot`)

KernelBlaster's `propose_splits` groups a slot's instances by shape bucket and, from per-op
times, computes each group's roofline efficiency; gain = Σ t_g (1 − e_g / max e). ModelBlaster
annotates the instances with `slot_variant` (`kb_export` then makes slot `conv2d__v0`,
`conv2d__v1`, ... with `variant_of`), seeds each variant with the current kernel, and decides
as for fusion. The cache needs nothing new: each variant's winner is guarded on its own shapes.
ModelBlaster makes the same move itself in `cache_plan`: when only some instances of a slot are
covered by shape-guarded winners (problems tuned on a selection of shapes), the slot is split
so each winner serves exactly its shapes and the rest keeps the hand kernel.

### Cost-ranked selection

`profile_ir` exports an IR in its current state as one problem and runs its driver (median of
`runs`); `rank` groups op instances by (C entry, signature hash, shape) across IRs (the cache's
key, so a winner for one serves all); `select` takes items by total time until `coverage`, at
most `top_k`, each ≥ `min_share`, skipping items sent before; `problems` puts each item in the IR
where it costs most, one problem per IR with every instance of its items (partitions need not be
contiguous: inputs come from the reference walk). On ResNetBasicBlock + ResNet18 with hand
kernels, `coverage=0.8 top_k=6` selects 5 of 21 items (87 % of the 81.6 ms total): one problem
of 13 ResNet18 3×3 convs (four shapes) and one of the block's 64-channel 224² conv.

### Decisions

For a hint, `evaluate_hint` exports the affected ops twice from the current state (accepted
hints, cache): the *original* and, with the hint applied, the *variant*; both seeded with the
same current-best kernels (fused seeds composed from them). KernelBlaster optimises both with
the same budget in one run; then both composite bests are built once and run round-robin
`remeasure` times (`kb_proposals.remeasure`), and the medians decide (`accept_gain`, 3 %).
Accepting caches the variant's new slots (the fused kernel, the split variants) on their shapes,
whatever their speedup over their seed, since the decision already measured them against the
unchanged graph.

### Rounds and the fixpoint

`kb_graph_opt.run` keeps `graph_hints.json`, `cache/`, KernelBlaster's `knowledge/` and
`state.json` (items sent, rounds) in the work dir, so a later call resumes. A round: profile →
MB proposals → plan → KB → ingest → KB proposals → decide the `max_variants` undecided hints with
the largest estimated gain. Accepted fusions are applied in the next round's profile, which makes
the fused op's consumer a new candidate (chains grow: conv→bn, then (conv+bn)→relu) and the fused
instances new plan items. It stops when a round accepts nothing, sends nothing and has no more
undecided hints than it can evaluate, or after `rounds`.

## Running

```bash
# rounds to a fixpoint over two IRs (KernelBlaster checkout and its python)
python -m modelblaster.pipeline.kb_graph_opt $L3/8_ResNetBasicBlock $L3/9_ResNet18 --work W \
    --kb $KB --python $KB_PY --model us.anthropic.claude-haiku-5-5 \
    --rounds 4 --rollouts 2 --steps 4 --variant-steps 2 --max-variants 3 --top-k 6 --remeasure 5
# scripted runs: --model stub:<name> --plugin file.py (the file registers the stub)

# pieces
python -m modelblaster.pipeline.kb_plan profile IR... --out P [--hints H --cache C]
python -m modelblaster.pipeline.kb_plan select P/*/profile.json --out plan.json --coverage 0.8 --top-k 6
python -m modelblaster.pipeline.kb_export --plan plan.json OUT [--cache C]
python -m modelblaster.pipeline.graph_hints apply IR H OUT_IR [--all] [--verify --build]
python -m modelblaster.pipeline.graph_hints import {kb,xpurt,registry} FILE... --into H
python -m modelblaster.pipeline.graph_hints show H
STRATEGY=plan bash gpu_eval/tools/kb_loop.sh IR W        # one round
```

## Validation (local TITAN RTX sm_75, shared with another process: timings vary ~±10 %)

- **Unit tests** (pure python): `pipeline/tests/test_graph_hints.py` (15: chain names and
  signatures, binding, flattening, chain matching, composed seeds, longest-first application,
  provenance and ids, scoped chains, unfuse round trip, split annotations, XPU-RT translation and
  delegation, registry import, decisions and merge, MB proposals growing chains, KB proposal
  import, shape buckets, torch-walk verification), `test_kb_plan.py` (7: dedup across IRs,
  budgets, representative IRs, cache coverage splits incl. fused slots, cached stage kernels). KB
  `tests/test_multislot_units.py` (+2: split proposals, `MB_OP_US`, fused buckets).
- **GPU flow** (`pipeline/tests/test_kb_graph_opt_flow.py`, KernelBlaster subprocess with a
  scripted `stub:` LLM that answers fused slots with real fused kernels, ~2-4 min): a tiny
  conv-bn-relu ×2 model. With the 3 % bar, one run went 31.0 → 26.0 µs (1.19×): round 0
  accepted conv→bn (+16 % re-measured) and rejected KB's conv split (−3 %), round 1 accepted the
  extension to conv→bn→relu (+7 %), round 3 was the fixpoint, and the final model verified with
  the cached three-stage kernel. Another run built the chain the other way (conv→bn measured
  +2.6 %, below the bar; bn→relu, then conv→(bn+relu)); a third rejected everything -- these are
  few-µs kernels on a GPU another process keeps ~60 % busy, where one kernel measured 17 µs in
  one run and 27 µs in the next. The test therefore accepts every verified variant
  (`accept_gain=-1`) and checks the mechanics: the chain is accepted in a later round than the
  first fusion, every decision carries five interleaved re-measurements and a gain, the rounds
  stop on their own, and the final model verifies with the cached chain kernel. A second test
  takes a split hint through original/variant problems (`conv2d__v0`, `conv2d__v1`), KB and a
  re-measured decision. `kb_loop.sh STRATEGY=plan` with the same stub: conv→bn accepted at
  +12.7 %, ingested, model verified.
- Suites: MB `pipeline/tests tests benchmarks/tests` 223 passed, 11 skipped, 1 pre-existing
  failure (`test_profile_db.py::test_hetero_runs_are_skipped`) with `MB_KB_ROOT`/`MB_KB_PYTHON`
  set (221 passed, 13 skipped without a KernelBlaster checkout); KB `tests` 83 passed.

### Live run (Bedrock `us.anthropic.claude-haiku-5-5`, single runs)

`kb_graph_opt` on Level3 `8_ResNetBasicBlock` + `9_ResNet18` together, one round per call
(the work dir resumes), `--rollouts 1 --steps 3 (2 in round 2) --variant-steps 1
--max-variants 2 (1) --max-fix 1 --top-k 3 --remeasure 5`, no reflection model:

| round | profile at start (block / R18, ms) | sent to KB | decisions (re-measured gain) | LLM calls |
|---|---|---|---|---|
| 0 | 28.6 / 52.5 (hand kernels) | the block's 224² conv (26.0 → 0.75 ms, 34.7×); 7 of R18's 3×3 convs, three shapes (24.1 → 1.63 ms, 14.7×), deduplicated across the two IRs | `split:conv2d` (KB) rejected, +2.4 % | 8 |
| 1 | 2.65 / 30.3 | R18: 3 more conv shapes (14.9×) | `fuse:fused__batchnorm2d__add__a` accepted +41 %; `fuse:fused__conv2d__batchnorm2d` accepted +29 % | 8 |
| 2 | 1.13 / 6.03 | R18: 3 items (2.8×) | `fuse:fused__conv2d__batchnorm2d__add__a` (the chain extending conv→bn) rejected, −28 % | 5 |

Final model-level verification (both pass, every op within the isolated-op bound): block
28.6 → 1.30 ms (22×; 1.05 ms in a re-verification run), ResNet18 52.5 → 5.75 ms (9.1×), with
3 + 17 fused ops and ResNet18's conv slots split by which winner covers them. 21 LLM calls;
tokens 70.1k prompt / 66.3k completion for rounds 1-2 (round 0's were not recorded: the
Bedrock path names them `input_tokens`/`output_tokens`, fixed in KB `4104d97`); no cost is
reported by that path. In round 0 no fusion was proposed: next to the hand convs BN was under
the 2 % threshold -- it became worth deciding only after the convs were fast, which is what the
rounds are for. The run also exposed a bug, fixed in `0629f6a`: after conv→bn was accepted the
fused slot's seed needed one conv winner covering all its instances, found none (winners are
per shape) and fell back to the hand conv (R18 27.6 → 30.2 ms); the coverage split now also
splits fused slots by first-stage winner (30.2 → 6.5 ms on the same cache, no LLM calls).
Single runs on a shared GPU; the chain rejection at a one-step budget says little about the
chain itself.

## Design decisions

| # | Decision | Alternatives considered | Why |
|---|---|---|---|
| 1 | One hint record per change pattern with a stable id (`fuse:<kind>`, `split:<slot>`), sources and accumulated evidence; status = majority of decisions | one file per producer (registry, XPU-RT hints, proposals); per-problem decisions | Every producer feeds one decision process and one apply path; evidence from different problems and models accumulates on one record instead of the last problem overwriting it. |
| 2 | Accepted hints apply wherever their pattern matches; the tried instances are kept as evidence | apply only to the instances measured | Phase 1's registry policy, and what makes reuse across models work; a fusion that loses elsewhere collects rejecting evidence and flips. |
| 3 | The hint layer dispatches by target; RISC-V hints go through the existing appliers unchanged, CUDA has its own fusion mechanics; `tile_split`/`shard` are skipped on CUDA with the reason | adapt `apply_fusion_hint` for CUDA; drop RISC-V hints | The RISC-V appliers encode int8/RVV assumptions (quant blocks, `__fused__` chained-call names, multi-core tiles) that are right there and wrong for a KernelSpec-per-slot fp32 flow; one schema still covers both, and an XPU-RT fusion group is translated to a CUDA chain when it is one. |
| 4 | Fused kinds spell a stage's param only for multi-input stages; two-stage names unchanged | a new separator or an encoded kind; nested `fused__fused__...` | Kinds are C identifiers (`kernel_<kind>`), Phase 1 registry and cache entries keep working, and parsing is unambiguous given the specs. |
| 5 | Chains are flat: fusing a fused op flattens the stages | nested fused ops | One signature form, one seed, one walker rule; the kind is independent of the order the chain was built in, so conv→(bn+relu) and (conv+bn)→relu meet on one hint (seen in testing). |
| 6 | Variants start from the current-best kernels: unchanged slots from the original's best/cache, fused slots composed of the best stage kernels (namespaced) | hand-kernel seeds (Phase 1) | Phase 1's variant restarted the fused conv from the hand kernel, so the comparison measured optimisation progress, not the graph change; now the variant starts at the unfused optimum and only the change is in question. |
| 7 | Decide on the affected ops only: original and variant problems, same seeds, same KB budget, both bests re-timed interleaved (median of 5) | whole-partition variant vs the original run's number (Phase 1) | Unaffected ops add noise and cost; equal budgets keep optimisation from being credited to the graph change; on the shared GPU single timings moved ±10 % (a fused kernel "regressed" 3 % in its KB step and tied in the re-measurement). |
| 8 | Op names made unique (`name#k`) wherever a graph enters the flow | rename in extract; key on output tensors | 43 of the Level2/3 IRs reuse a module (two ops named `relu`), and name-based fusion silently picked the first; IRs are inputs, the rule is deterministic and idempotent so every reader agrees, tensor names stay. |
| 9 | Rank by (C entry, signature, shape) across IRs; each item optimised in the IR where it costs most | per-IR partitions; per slot | The same key as the cache's shape guard, so an item optimised once serves every model that has it; per-slot ranking would mix shapes a winner is not guarded for. |
| 10 | Plan partitions hold only the selected ops (not contiguous) | windows around them | The driver feeds every boundary input from the reference walk, so nothing else needs to run; the budget goes only where the time is. |
| 11 | The cache splits a slot by coverage when winners cover only some instances (an `mb-cache` split hint) | require a winner covering every instance (Phase 1) | Keeps the strict shape guard (Phase 1 decision 12) while partial-selection winners are still used, each on exactly its shapes. |
| 12 | CUDA "split" is a slot-variant annotation, not a graph rewrite | tile the op (RISC-V `split_hints`) | On one GPU the useful split is per shape regime (one kernel tuned per shape group), which needs no graph change: same golden, same tensors; the cache guard already keys the variants. |
| 13 | KB split proposals are deterministic (bucket groups, per-op times, roofline efficiency spread) | an LLM proposal action | Cheap and reproducible like fusion proposals; the decision is measured anyway. The efficiency spread is a proxy -- in the tests KB's splits were proposed and rejected on measurement. |
| 14 | Items are sent once; fixpoint = a round accepts nothing, sends nothing and has no undecided hints left over | resend until no improvement | Bounded cost; a graph change creates new items (fused kinds) which are then sent, so accepted changes still get kernel work. |
| 15 | Orchestrator in Python, KernelBlaster as a subprocess; `--plugin` registers `stub:` models in it | bash (Phase 1 `kb_loop.sh`); KB in-process | State, provenance and decisions need structure; process isolation keeps KB's dependencies and GPU state apart; tests exercise the real CLI. |
| 16 | Per-op times are an additive driver line (`MB_OP_US`) in `mb-kb-problem/1` | a new format version; walk_cuda timing | Older parsers ignore it; it is the same CUDA-event timing as the per-slot numbers and covers fused ops and cached sources, which walk_cuda (hand/seed kernels only) does not. |

## Further proposal kinds (designs, not implemented)

- **Weight pre-pack / layout.** KB proposes when a conv/GEMM slot is memory-bound on weight
  reads (roofline bytes dominated by weights, low coalescing in its access pattern): a
  `prepack` hint `{entry, layout: "ohwi" | "ihwo" | "blocked<k>"}`. ModelBlaster applies it as a
  new spec variant whose weight parameter is documented in that layout and permutes the weight
  in `data.bin` at export, as `generate_skeleton._backend_pack_weight` does for RVV (layout per
  algorithm, not per backend). The reference stays OIHW; the driver checks the op against the
  unpacked reference. Decided like fusion; cached per (entry, layout).
- **Algorithm-parameter exposure.** For slots whose best kernel is a template over tile sizes,
  expose the parameters as part of the slot (a `params` hint: names and ranges) so ModelBlaster
  can bind per shape variant (with `split_slot`) and KernelBlaster tunes values instead of
  rewriting code. Needs a parameter channel in the manifest and seed.
- **Algorithm choice per shape** (direct / implicit-GEMM / Winograd): a `split_slot` whose
  variants get different seed algorithms; the mechanism exists, the seeds do not.
- **Unfuse on CUDA**: implemented in the apply path; the producer is missing -- a fusion accepted
  on one model whose evidence turns negative on another could be unfused there instead of
  globally rejected (per-IR status).
- **Streams for independent branches** (the CUDA analogue of `tile_split`/`shard`: concurrency
  across ops): an annotation for the walker to launch independent ops on separate streams;
  needs a multi-stream driver.
- **DAG fusion** (residual add reading two internal tensors), and an **LLM proposal action** in
  KernelBlaster's bandit for the kinds above.

## Known limitations / next steps

- Decisions are single measurements of single KB runs per side; LLM run-to-run variance is large
  (Phase 1) and near the 3 % bar the shared GPU decides (seen in testing). More KB seeds per side
  or a significance test on the interleaved timings would make decisions robust.
- Evidence is per hint id, not per IR: a fusion that helps one model and hurts another gets one
  majority status.
- `propose_from_profile` and KB's `propose_fusions` share the elementwise-consumer rule; MB's sees
  the whole model, KB's only its partition. Only linear chains are fused.
- `profile_ir` exports the whole model with its data (Level3 IRs up to a few GB) each round; it
  deletes data.bin afterwards.
- The flow's per-hint KB runs are sequential; independent hints could run in parallel within the
  GPU-memory and process limits.
