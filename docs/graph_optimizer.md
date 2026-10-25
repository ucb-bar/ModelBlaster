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
         │ (with --ssh-host every GPU step runs on the remote pool: "Where the GPU steps run")
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
        independent hints in waves: one KB run per wave, re-timed on leased GPUs
         │                                          kb_graph_opt.schedule / evaluate_hints
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
| MB | `pipeline/kb_graph_opt.py` | rounds to a fixpoint; KernelBlaster as a subprocess; hint footprints, conflict waves, GPU leases, work-dir lock |
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
`remeasure` times on one leased GPU (`kb_proposals.remeasure`), and the medians decide
(`accept_gain`, 3 %). With `--ssh-host` that GPU is one of the KernelBlaster runs' remote GPUs
(see "Flags" below).
Accepting caches the variant's new slots (the fused kernel, the split variants) on their shapes,
whatever their speedup over their seed, since the decision already measured them against the
unchanged graph.

### Rounds and the fixpoint

`kb_graph_opt.run` keeps `graph_hints.json`, `cache/`, KernelBlaster's `knowledge/` and
`state.json` (items sent, rounds) in the work dir, so a later call resumes. A round: profile →
MB proposals → plan → KB → ingest → KB proposals → decide the `max_variants` undecided hints with
the largest estimated gain, independent ones together (next section). Accepted fusions are applied in the next round's profile, which makes
the fused op's consumer a new candidate (chains grow: conv→bn, then (conv+bn)→relu) and the fused
instances new plan items. It stops when a round accepts nothing, sends nothing and has no more
undecided hints than it can evaluate, or after `rounds`.

### Parallel hint decisions

A round's candidates (the `max_variants` undecided hints, in rank order) are decided in
*waves* of independent hints instead of one KernelBlaster run after another
(`kb_graph_opt.footprint` / `conflicts` / `schedule` / `evaluate_hints`).

**What deciding a hint touches.** `footprint(h)` on the hint's IR, in its round-start state:

- `ops`: (IR, original op name) of every op its original/variant problems hold;
- `reads`: the cache entries (C entries, not op kinds: `sum_dim`/`max_dim` share `kernel_reduce`)
  its problems are seeded from -- those ops', their stages' (fused ops), and its own new slot;
- `writes`: the entries an acceptance ingests -- the fused kind (`fuse`), or the base slot's entry,
  guarded on its shapes (`split_slot`).

Entries, not (entry, shape): a slot's cache lookup and coverage split look at every instance of
the slot in the IR, so a split winner for some conv shapes can re-seed (and re-split) every conv
slot of that IR.

**Conflict rule.** Two hints conflict when they share an op, or when one writes an entry the
other reads. Sharing an op covers the dependent cases: a chain and the fusion it extends
(`conv→bn→relu` vs `conv→bn`), two fusions of one op (`conv→bn` vs `bn→relu`), a split and any
fusion of its instances. Writing what the other reads covers composition through the cache: an
accepted split caches conv kernels that a later conv fusion is seeded from. Two fusions of
different ops never conflict -- an accepted fusion changes the graph only from the next round on
(the next profile), and the fused kernel it caches is read only by ops of its own kind.

**Ordering rule.** In rank order (estimated gain, ties in hints-file order), hint *i* goes to the
first wave after every earlier hint it conflicts with that still has room (`--parallel-hints`).
Every conflicting pair is therefore decided in rank order, after the earlier one's decision and
cache ingest, which is exactly what deciding the hints one by one does; independent hints share a
wave. The waves depend only on the order and the footprints (deterministic), and are recorded in
`round.json` (`waves`). Decisions, evidence and ingests are made in rank order in the main thread.
Example (unit test): split `conv2d` > `conv→bn` > `conv→sigmoid` > `bn→relu` > `conv→tanh` gives
`[split] [conv→bn, conv→sigmoid, conv→tanh] [bn→relu]`.

**What is still different from one-by-one.** KernelBlaster's knowledge (implementation store,
outcome log, claims) is shared context, not an input of the comparison: a hint does not see the
store records of hints in its own wave, as it would have when they ran before it; it does see
every earlier wave's. Both sides of a hint read the same store state. Within one hint the original
and variant problems now run concurrently (before, on one GPU, the variant ran after the original
and could pick up its kernels from the store; with several GPUs that was already concurrent).

**Mechanism.** One `run_multislot.py` per wave with all its problems and `--parallel-problems` =
their number, rather than one process per hint:

- KernelBlaster's GPU pool gives each timed run one GPU and its adaptive build limiter bounds the
  compiles, so the wave width does not oversubscribe the GPUs; it only sets how much LLM latency and
  compile time overlaps other hints' runs.
- The knowledge dir has one writer: v1 `claims.json` is single-writer, and the store, outcome log
  and claims writes are synchronous within one process.
- A hint's original and variant run in the same process at the same time.

A hint is re-timed as soon as both its `result.json` exist (KernelBlaster writes the best slot
sources first), while KernelBlaster works on the others. The two composite bests are built, then
run interleaved on **one** leased GPU (`GpuLeases`), so the decision always compares the two
sides on the same device, close in time, whatever the pool's mix of models. A lease is exclusive
within the process and holds KernelBlaster's per-GPU lock file
(`/dev/shm/kernelblaster_gpu_<uuid>.lock`), so KernelBlaster runs of this flow or of any other
process wait instead of timing on that GPU at the same time. There are at most two re-timing
workers per GPU: one builds while the other times.

**Shared state.**

| state | writers |
|---|---|
| `graph_hints.json` | main thread only, after each wave; `graph_hints.save` is atomic (temp file + rename) |
| `cache/index.json` | main thread only (`finish_hint`, rank order); sources are content-addressed |
| `knowledge/` | one KernelBlaster process at a time (the plan run, then one per wave) |
| `state.json`, `round.json` | main thread |
| the work dir as a whole | one `kb_graph_opt` process: `run` holds an flock on `W/.lock` and fails fast if another run holds it |

**Flags.**

- `--parallel-hints auto|N`: auto is `max(4, 2 × GPUs)`; 1 decides one by one.
- `--gpus`, or `--ssh-host` (repeatable) with `--cuda-visible-devices` (`--ssh-key`,
  `--remote-dir`), are passed to every KernelBlaster run.
- `--runtime auto|host|process` and `--gpu-model M` are passed to every KernelBlaster run as well.
  `--gpu-model` limits the runs to GPUs of model M (none there is an error). Without it, the model
  the first profile ran on is passed (next section). Every `result.json` records the GPU it was
  timed on (`gpu`: name, UUID, host, arch).
- The GPU count comes from KernelBlaster's own `select_gpus`: auto takes the idle GPUs, else the
  least busy one, bounded by an externally set `CUDA_VISIBLE_DEVICES`.
- Re-timing leases from the same pool as the KernelBlaster runs, restricted to the flow's GPU model:
  - locally, one of this host's selected GPUs (ModelBlaster's own runs, the GPU lock held);
  - with `--ssh-host` (or `--mb-evaluator kb`), one of the pool's GPUs. The two bests are timed
    through KernelBlaster's evaluator pinned to that GPU (`scripts/time_problems.py`, persistent
    host, the GPU lock taken per run by KernelBlaster, not by the lease), on the GPU model the two
    runs recorded.

  The decision records the device it was timed on (`remeasured.device`, and `remeasure_device`
  in `round.json`'s decisions), locally too.
- `--mb-evaluator auto|kb|builtin`: where the profiles and the verification run (next section).
- The model's result (`kb_model_run`, after the final verification) goes through the same
  evaluator: local pool or the `--ssh-host` GPUs. It times on the GPU model the rounds'
  KernelBlaster runs recorded, or `--gpu-model`. Where no GPU of that model is available it is an
  error (`model_error` in `summary.json`); `--allow-gpu-mismatch` times anyway, with warnings.
  `summary.json` gets `model_gpu`, `model_device` and `model_hardware_match` per IR
  ([gpu_harness_perf.md](gpu_harness_perf.md), "The model's result").

**Measured** (`gpu_eval/work/graphopt_par`, scratch). The model has four independent
conv → {bn, sigmoid, tanh, silu} branches with mixed 1×1 / 3×3 convs (`cat` output). The run is
`--rounds 3 --rollouts 1 --steps 1 --variant-steps 2 --max-variants 4 --top-k 2 --remeasure 5
--accept-gain -1`, so every verified variant is accepted and all runs make the same decisions:
round 0 decides the four fusions, round 1 the `conv→bn→relu` chain, and round 2 is the fixpoint.
The `stub:` LLM answers instantly (lat 0) or after an `asyncio.sleep(5)` (lat 5 s, standing in for
LLM latency). Each cell is the median whole-run wall-clock with the range of N; "before" is
`feat/kb-multislot` @2eb2a7d.

| host | GPUs | lat | before | width 1 | width 2 | width 4 | before → 4 |
|---|---|---|---|---|---|---|---|
| local TITAN RTX (load 8–27, shared GPU) | 1 | 0 | 172 s (152–174), N=3 | — | 114 s (63–116), N=3 | 97 s (69–124), N=2 | 1.8× |
| local | 1 | 5 s | 274 s (251–298), N=2 | — | 160 s (158–161), N=2 | 128 s (116–140), N=2 | 2.1× |
| firesim2 L40S (GPU 2) | 1 | 0 | 108 s (100–116), N=2 | — | 80 s (76–83), N=2 | 82 s (67–96), N=2 | 1.3× |
| firesim2 | 1 | 5 s | 231 s (230–232), N=2 | — | 127 s (118–136), N=2 | 100 s (100–101), N=2 | 2.3× |
| firesim2 L40S (GPUs 2, 3) | 2 | 0 | 106 s (95–117), N=2 | 100 s (90–109), N=2 | 79 s (73–85), N=2 | 74 s (65–82), N=2 | 1.4× |
| firesim2 | 2 | 5 s | 219 s (201–236), N=2 | 173 s (166–180), N=2 | 129 s (128–131), N=2 | 109 s (106–112), N=2 | 2.0× |

- Round 0 alone (profile, plan run, four decisions) on two L40S at lat 5 s: 151 s before vs 48 s at
  width 4 (3.1×).
- Width 1 (this code, one by one) is a little faster than before, because a hint's two problems
  now always run concurrently.
- Width 4 is faster than or level with width 2 in every cell, by 15–21 % when there is LLM latency.
  Hence the auto default of max(4, 2 × GPUs).
- What is left in a round is the profile, the plan run and the chain round, which are serial.
- Every run accepted the same five hints and the final model verified.
- Firesim2 GPUs 0 and 1 had other users' processes and were not used (`CUDA_VISIBLE_DEVICES`
  bounded both before and after to GPUs 2 and 3).
- Locally the box had a load of 8–27 and, in the later runs, another user's process on the GPU, hence
  the ranges.

Live, Haiku (`us.anthropic.claude-haiku-5-5`), local TITAN RTX shared with another user's process,
same model, `--rounds 2 --rollouts 1 --steps 2 --variant-steps 1 --max-variants 4 --top-k 2
--remeasure 5`, default 3 % bar, one run each:

| | wall | round 0 (4 hints) | round 1 | LLM calls | tokens in / out | decisions |
|---|---|---|---|---|---|---|
| before | 351 s | 267 s | 81 s | 15 | 58.9k / 38.8k | 4 accepted, `conv→bn` rejected |
| width 4 (one wave) | 188 s (1.9×) | 99 s (2.7×) | 85 s | 15 | 51.9k / 39.8k | 3 accepted, `conv→bn` and `conv→tanh` rejected |

The decisions differ on `conv→tanh` (+5.7 % in one run, −14 % in the other). These are few-µs
kernels on a GPU another process kept busy, single LLM runs: the same noise as in the single-run
caveats above, not an effect of the waves (the stub runs, with fixed decisions, are the
equivalence check). Both final models verified. About $0.25 per run.

## Where the GPU steps run

A round has five GPU steps: the profile of every IR, the KernelBlaster runs, the re-measurement
of the decisions, and at the end the model-level verification and the model's result. With
`--ssh-host` all five go through KernelBlaster's evaluators on the same pool, so a run can use
nothing but remote GPUs (the local host only builds). Without remote flags, the profiles, the
verification and the re-measurement are ModelBlaster's own runs on this host, as before.

| step | function | no remote flags (default) | `--ssh-host` (or `--mb-evaluator kb`) |
|---|---|---|---|
| profile (per-op times, CUPTI attribution, launch overhead: the planning inputs) | `kb_plan.profile_ir` | `build_and_run` on the default GPU, median of `--profile-runs` | `kb_eval.run_problem`: one job, `reps` = `--profile-runs`, median by partition time as `build_and_run` |
| KernelBlaster runs | `KBRunner` (`run_multislot.py`) | local pool (`--gpus`) | ssh evaluator |
| re-measurement | `measure_hint` → `kb_proposals.remeasure` | a leased local GPU, its lock held | KernelBlaster's evaluator pinned to a leased pool GPU |
| verification | `kb_ingest.verify_ir` | `build_and_run` on the default GPU | `kb_eval.run_problem`, one full run |
| model's result | `kb_model_run.model_runtime` | KernelBlaster's local evaluator | ssh evaluator |

`--mb-evaluator kb` without `--ssh-host` sends the profiles, the verification and the
re-measurement to KernelBlaster's local evaluator, on the `--gpus` pool, with its locks.
`--mb-evaluator builtin` with `--ssh-host` keeps them local. When a KernelBlaster evaluator is
asked for and the checkout has no `scripts/time_problems.py`, the run fails; it never falls back to
the local GPU. `state.json` `gpus.mb_evaluator` records the choice.

**What runs where.** The profile and the verification export the IR here, as before, and the
reference walk (`walk_torch`) runs here on the CPU at export. The inputs, the weights and every
op's reference output are linked into the problem (`data.bin`; above 1 GiB it is copied to each
remote host at run time). The driver checks the partition outputs and every op in isolation
against them. So the problem is self-contained and checked: the evaluator runs only the composed
CUDA side, and the reference never needs the GPU. A full run of KernelBlaster's evaluator
(`only_slots` unset, no screening) prints the same lines as `build_and_run`: `MB_OP_US`,
`MB_ATTR_*`, `MB_LAUNCH_OVERHEAD_US`, `MB_SLOT_CHECK`, `MB_CHECK`. `kb_eval.driver_result` maps its
`EvalResult` to the keys `profile_ir` / `verify_ir` read (`slot_checks` → `slots`, `ok` →
`passed`), so the planning inputs and the cache marks (`model_verified` / `rejected`) come out the
same.

**One GPU model.** `FlowHardware` holds the model every step must use: `--gpu-model`, else the
model the first profile ran on, kept in `state.json` (`gpu_model`) so a resumed run keeps it. From
then on:

- it is the required model of the later profiles and of the verification (`time_problems
  --gpu-model`);
- it is passed as `--gpu-model` to every KernelBlaster run;
- it is the model the re-timing leases are restricted to, and the required model of the
  re-measurement and of the model's result;
- every step's recorded device is checked against it (`kb_eval.check_device`): the profile, every
  KernelBlaster run's `result.json` `gpu`, each re-measurement, the verification and the model run.

Another model is a `HardwareMismatch` that stops the run. With `--allow-gpu-mismatch` it is a
warning instead, and only an explicit `--gpu-model` pins the KernelBlaster runs. No GPU of the
model in the pool is the same error. `summary.json` `gpu` has the model, `models` (every model a
step recorded), `same_model`, `hosts`, `local_gpu_used`, the device of every step (`steps`:
profile, kb, remeasure, verify, model_run) and the warnings. Per IR it adds `verify_device` and
`verify_gpu`. `profile.json` and the verification result carry `device` and `evaluator`.

**Telemetry.** `mb.profile.run` (per IR, inside the profile lap) and `mb.verify.run` record the
evaluator, the device, the host and whether the run passed. With a KernelBlaster evaluator, each
has an `mb.kb_time` child, and KernelBlaster's own spans go to
`rNN/profile/<ir>/kb_time/profile.telemetry.jsonl` and `final/<ir>/kb_time/verify.telemetry.jsonl`.

The pieces take the same flags: `kb_plan profile ... --ssh-host H --remote-dir D
[--cuda-visible-devices ids] [--gpu-model M]`, and `kb_ingest --verify IR ... --ssh-host H
--remote-dir D`. In both, `auto` without `--ssh-host` keeps ModelBlaster's own run.

**Live run: every GPU step on firesim2 L40S** (`gpu_eval/work/graphopt_remote/live1`, scratch).
This is one round on the four-branch conv model from "Parallel hint decisions". The run used
`--ssh-host chrisdong@firesim2... --cuda-visible-devices 2,3 --remote-dir ... --arch 89 --rounds 1
--rollouts 1 --steps 1 --variant-steps 2 --max-variants 4 --remeasure 5 --profile-runs 3
--accept-gain -1` with the scripted `stub:` LLM, so it made no paid LLM calls. Locally it ran with
`CUDA_VISIBLE_DEVICES=""`, and a 1 s `nvidia-smi --query-compute-apps` sampler saw no process on
the local GPU during the run.

- `summary.json` `gpu`: `gpu_model` "NVIDIA L40S", taken from the first profile. `same_model` is
  true, `hosts` is firesim2 only, `local_gpu_used` is false, and there are no warnings.
- Devices per step:
  - profile: GPU 2;
  - the 9 KernelBlaster runs (the plan run, then 4 hints × original/variant in one wave): GPUs 2
    and 3;
  - the 4 re-measurements: GPUs 2 and 3, two each;
  - verification: GPU 2;
  - model's result: GPU 2.

  All of them are L40S.
- Telemetry, inside the 91 s wall clock:
  - `mb.profile.run`: 7.9 s, `evaluator` ssh, device firesim2:2, passed. The profile has 3 runs
    with CUPTI per-op times and 13 launches.
  - `mb.verify.run`: 7.5 s, firesim2:2, passed. All 4 cached fused slots were checked.
  - `mb.model_run.time`: 12.2 s.
  - The decide phase took 51 s.
- The decisions were 4 fusions, all accepted (+6.7 %, +12.5 %, +22 %, 0 % re-measured). The model
  verified, at 49.5 → 49.7 µs eager and 40.0 → 41.0 µs graph replay. The stub's fused kernels are
  naive: this checks the mechanics, not the speed.
- `kb_plan profile ... --ssh-host ... --gpu-model "NVIDIA TITAN RTX"` fails with `NoDevice: no
  NVIDIA TITAN RTX GPU among the selected devices (...:2 (NVIDIA L40S), ...:3 (NVIDIA L40S))`.
  With `--gpu-model l40s` it profiles on firesim2:2. `kb_ingest --verify` with the same flags
  verifies the final cache on firesim2:2.

## Running

```bash
# rounds to a fixpoint over two IRs (KernelBlaster checkout and its python)
python -m modelblaster.pipeline.kb_graph_opt $L3/8_ResNetBasicBlock $L3/9_ResNet18 --work W \
    --kb $KB --python $KB_PY --model us.anthropic.claude-haiku-5-5 \
    --rounds 4 --rollouts 2 --steps 4 --variant-steps 2 --max-variants 3 --top-k 6 --remeasure 5
# scripted runs: --model stub:<name> --plugin file.py (the file registers the stub)
# GPU pool: --parallel-hints auto|N  --gpus auto|all|0,2  or  --ssh-host H --cuda-visible-devices auto
# everything on remote GPUs (the local host only builds):
#   --ssh-host u@gpuhost --ssh-key K --cuda-visible-devices 2,3 --remote-dir /scratch/u/mb --arch 89

# pieces
python -m modelblaster.pipeline.kb_plan profile IR... --out P [--hints H --cache C]
python -m modelblaster.pipeline.kb_plan select P/*/profile.json --out plan.json --coverage 0.8 --top-k 6
python -m modelblaster.pipeline.kb_export --plan plan.json OUT [--cache C]
python -m modelblaster.pipeline.graph_hints apply IR H OUT_IR [--all] [--verify --build]
python -m modelblaster.pipeline.graph_hints import {kb,xpurt,registry} FILE... --into H
python -m modelblaster.pipeline.graph_hints show H
STRATEGY=plan bash gpu_eval/tools/kb_loop.sh IR W        # one round
```

To compare configurations of this flow, use `python -m modelblaster.pipeline.kb_ablate run SPEC`
(`entry: "graph_opt"`). It runs the arms x repeats in isolated work dirs with frozen code,
interleaves them, enforces a budget and writes a paired-statistics report; see
[ablations.md](ablations.md).

To make the LLM calls on another machine, set `KB_LLM_SERVER_URL=http://127.0.0.1:18080` and
`KB_LLM_SERVER_TOKEN_FILE=...` before starting the run. This uses an `ssh -L` tunnel to a
KernelBlaster `llm_server` running there. The KB subprocesses inherit the variables (KernelBlaster
`docs/llm_server.md`).

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
- **Parallel decisions**: `pipeline/tests/test_kb_graph_opt_parallel.py` (7, no GPU) covers
  footprints, conflicts and waves for every width, with conflicting pairs kept in rank order. On a
  real exported IR with a fake KernelBlaster and a deterministic fake re-timing that depends on
  every seed, the decide phase gives the same decisions, evidence, statuses and cache in waves as one
  by one, including a conv fusion seeded from the conv kernel that the earlier split's acceptance
  cached. If the conflict check is disabled, two of the five gains change. It also covers lease
  exclusivity per GPU, a lease waiting for another process that holds KernelBlaster's GPU lock file,
  the work-dir lock across processes, atomic hints-file writes under concurrent writers, and the
  pool flags. In `test_kb_graph_opt_flow.py`, `test_parallel_hints_same_decisions` runs four
  independent branches with the KernelBlaster subprocess: width 4 decides them in one wave, width 1
  one by one, with the same accepted hints and the same fused model.
- **Remote profile / verification** (`pipeline/tests/test_graphopt_remote.py`, 20):
  - mapping and parity: the `EvalResult` → driver-result mapping and `build_and_run`'s median
    choice;
  - `profile_ir` / `verify_ir` with a fake evaluator: the job sent, the median run, the CUPTI
    basis, the device, the cache marks on pass and fail, mismatch errors and
    `--allow-gpu-mismatch`;
  - `FlowHardware`: the first device fixes the model, explicit and resumed models, pinning, and
    the summary;
  - re-timing leases restricted to the model; a local re-measurement records its device; a
    KernelBlaster-local re-measurement leases without holding the GPU lock;
  - the `--mb-evaluator` selection and the CLI flags;
  - `kb_graph_opt._run` with every GPU step faked: what each step is given, the KernelBlaster
    runs pinned on resume, a profile on another model refused, and the builtin default without
    remote flags;
  - on the local GPU, the builtin path and KernelBlaster's local evaluator give the same ops,
    attribution basis, launches, checks and device for the profile and the verification.
- Suites: MB `pipeline/tests tests benchmarks/tests` 223 passed, 11 skipped, 1 pre-existing
  failure (`test_profile_db.py::test_hetero_runs_are_skipped`) with `MB_KB_ROOT`/`MB_KB_PYTHON`
  set (221 passed, 13 skipped without a KernelBlaster checkout); KB `tests` 83 passed.
  With the parallel decisions (KB 5fbcfb1, `gpu_eval/venv_cu126`): `pipeline/tests` 182 passed,
  4 skipped; `tests` 58 passed, 7 skipped.

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
| 17 | Independent hints are decided in waves; conflict = shared op, or one's cache writes are the other's seeds (by C entry); wave = first after every conflicting earlier hint | sequential (as before); a dependency graph executed as soon as inputs are ready; conflicts by (entry, shape) | Waves keep every conflicting pair in rank order, so the outcome is the one-by-one outcome; whole-entry conflicts are conservative because a slot's lookup and coverage split see every instance in the IR. A ready-as-soon-as scheduler would gain little with `max_variants` ≤ a few hints and complicate the determinism argument. |
| 18 | One KernelBlaster process per wave with every problem (`--parallel-problems` = all), not one process per hint | a bounded pool of `run_multislot` processes | The knowledge dir stays single-writer (v1 claims.json); KernelBlaster's pool and build limiter already bound GPU and CPU use across problems; a hint's two sides run in the same process at the same time. |
| 19 | Decisions are re-timed by ModelBlaster on one leased GPU per hint, holding KernelBlaster's per-GPU lock file; started as soon as the hint's two results exist | trust KB's final numbers; re-time after the whole wave | The two sides are compared on the same device, interleaved, regardless of where KB ran them (KB pins a problem's runs to one GPU model, but the two sides are two problems); overlapping with the rest of the wave hides the builds. |
| 20 | The verification (and the profile) export a self-contained, checked problem here (reference walk on the CPU at export, references linked in) and only the composed CUDA side runs on the evaluator's GPU | run the torch reference on the remote GPU too (`walk_cuda` with torch there); a new `time_problems` job kind for checks | The driver already checks every output and op against linked references, so the reference never needs the GPU and stays valid computed on the CPU (the GPU torch walk is ~0.6 % off for conv-transpose with cuDNN); the remote host needs no torch or ModelBlaster, only the binary; a full `EvalResult` already carries the checks, per-op times and attribution, so no new job kind was needed. |
| 21 | One GPU model per run: `--gpu-model`, else the first profile's, persisted, enforced on every step and passed to the KernelBlaster runs | infer it from the first KernelBlaster run; check only the model's result (before) | The first profile is the flow's first GPU step, so a mixed pool cannot split the plan from the runs it plans; persisting it keeps a resumed run on the same model; refusing (unless `--allow-gpu-mismatch`) is the same rule as `kb_model_run`. |

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
- Within a wave, a hint does not see KernelBlaster store records of the other hints in that wave
  (it would one by one); `--parallel-hints 1` restores that order.
- With `--ssh-host`, each round's profile is one `time_problems.py` call. It copies a whole-model
  `data.bin` above 1 GiB to the host again each round, because the evaluator removes its copies
  when it closes.
- With `--remeasure 0` the decision uses KernelBlaster's two final numbers, which in a pool of mixed
  GPU models can come from different models (each problem is pinned to one model, the pair is not).
