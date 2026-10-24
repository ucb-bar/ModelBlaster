# ModelBlaster ↔ KernelBlaster: multi-slot CUDA optimisation (Phase 1)

ModelBlaster owns the graph: it decides what a problem is (a partition), writes the driver
that checks it, and decides graph-level changes such as fusion. KernelBlaster owns the
kernels: it optimises the N kernels of a problem one slot at a time and shares what it
learns across kernels and problems. The contract between the two is ModelBlaster's per-op C
ABI (`KernelSpec.signature`), which already is the CUDA entry point.

This document covers the Phase 1 CUDA implementation. It is written for review: the
"Design decisions" section lists each non-obvious choice, the alternatives and why.
KernelBlaster's semantic update pass has its own document
(`third_party/KernelBlaster/docs/semantic_update_pass.md`).

## Flow

```
IR (graph.json, weights.npz, io.npz)
  │ pipeline/kb_export.py      partition → problem dir (manifest mb-kb-problem/1)
  ▼
problem: manifest, slots/<id>.cu, walker head/tail, driver.cpp, data.bin
  │ KernelBlaster scripts/run_multislot.py → agents/opt_multislot.MultiSlotAgent
  │   per step: pick slot → pick technique → LLM rewrites that slot → build + driver
  │   → reward, OutcomeRecord, ImplStore record, (semantic reflection every N records)
  ▼
result.json, best/slots/*.cu, proposals.json (kb-proposal/1), winners.json (kb-winners/1)
  │                                          │
  │ pipeline/kb_proposals.py variant/decide  │ pipeline/kb_ingest.py
  ▼                                          ▼
fused variant problem → KernelBlaster → accept/reject (fusion_registry.json)
                                             CUDA kernel cache (shape-guarded) → model-level verify
```

`gpu_eval/tools/kb_loop.sh IR_DIR WORK_DIR` runs one full round.

## Code map

| repo | file | role |
|---|---|---|
| MB | `pipeline/kb_export.py` | partitioning, slot extraction from the hand kernels, walker + driver codegen, local build/run |
| MB | `pipeline/kb_ingest.py` | winners → CUDA kernel cache (shape guard), model-level verification |
| MB | `pipeline/kb_proposals.py` | fusion: fused op + `FusedKernelSpec`, variant export, accept/reject registry |
| MB | `pipeline/op_classes.py` | op → class (gemm, conv, reduction, elementwise, data_movement, attention, scan, other) |
| MB | `gpu_eval/harness/walk_torch.py` | reference values (torch IR walk; runs fused ops' `sub_ops`) |
| MB | `gpu_eval/tools/kb_loop.sh` | the round above |
| KB | `data/sources/modelblaster_source.py` | `ModelBlasterSource` ("modelblaster"); `Problem.slots` |
| KB | `knowledge/slots.py` | `SlotProblem`, assembly, driver-output parsing, `LocalEvaluator`, `ServerEvaluator` |
| KB | `knowledge/impl_store.py` | implementation knowledgebase |
| KB | `knowledge/outcomes.py`, `claims.py`, `reflect.py`, `semantic.py` | outcome records + semantic update pass |
| KB | `agents/opt_multislot.py` | the slot-wise RL agent, roofline state, technique bandit, fusion proposals |
| KB | `agents/utils/agentic_llm.py` | `claude-code:<model>` / `stub:<name>` LLM providers |
| KB | `scripts/run_multislot.py` | runner |

## Problem format (`mb-kb-problem/1`)

`manifest.json` fields:

- `name`, `source_ir`, `ir_name`, `partition` (`strategy`, `op_names`, `op_indices`, `is_whole_graph`)
- `target` (`cuda`), `dtype` (`fp32`), `tolerance` (`output_rel`, `output_abs`, `slot_rel`)
- `assembly`: `head`, `tail`, `namespace_prefix` (`mbslot_`), `order` (slot ids)
- `slots[]`: `id`, `op` (first op kind), `ops` (all kinds using it), `op_class`, `entry`,
  `signature`, `file`, `reference_file`, `semantics`, `seed_source`
  (`hand` / `cache:<impl_id>` / `fused-seed` / `override`), `instances[]` (`op_name`, `op`,
  `index`, `shape`)
- `ops[]` (call order): `index`, `name`, `op`, `slot`, `inputs`/`outputs` (buffer indices)
- `buffers[]`: `name`, `kind` (input, const, weight, intermediate, output), `dtype` (f32/i32),
  `numel`, `alias` (`base`, `offset` — views), `ref` (`[offset, nbytes]` into `data.bin`)
- `inputs`, `outputs` (buffer indices), `driver`, `data`, `data_sha1`

**Assembly** (implemented identically in `kb_export.assemble` and `SlotProblem.assemble`):
walker head, then each slot in `order` with its `#include` lines hoisted and `extern "C"`
stripped, wrapped as `namespace mbslot_<id> { ... }`, followed by `#undef` for every macro the
slot defines; then the walker tail (`mb_run_op(i, bufs)` with one generated call per op,
`mb_num_ops()`, `launch_gpu_implementation(void **bufs)`).

**Driver** (`driver.cpp`, generated; no libtorch): runs with no arguments and prints

```
MB_CHECK output <name> max_abs_err=<e> bound=<b> ok=<0|1>    every partition output vs reference
MB_SLOT_CHECK <slot> rel_err=<e> ok=<0|1>                      every op alone on the reference's inputs
MB_PARTITION_US <us> reps=<n>                                  CUDA events, whole partition
MB_SLOT_US <slot> <us>                                         CUDA events per op, summed per slot
MB_OP_US <op name> <us>                                        CUDA events per op instance
passed | failed
```

Output bound: `err <= output_rel * max|finite ref| + output_abs` (1e-3, 1e-5, as the Level2/3
gates). Isolated bound: `max err / max(|finite inputs|, |finite ref|) <= slot_rel` (1e-3, as
`walk_cuda`'s per-kernel check). Matching infinities/NaNs are exact. `MB_ERROR` lines report
CUDA errors.

## KernelBlaster agent (one slot per step)

1. **Slot** sampled with weight `share × 0.6^stalls × (1 + 0.5·sqrt(ln T / (1+tries)))`, where
   `share` is the slot's fraction of measured time and `stalls` counts attempts since its last
   best. Slots under 1 % share are skipped.
2. **State** of the slot: a roofline class from the manifest (bytes moved and FLOPs per op
   instance) and the measured slot time against the arch's peaks: `latency_bound` (< 8 µs),
   `underutilized_{compute,memory}` (< 15 % of both peaks), `memory_bound`, `compute_bound`. The
   numbers (achieved GB/s, GFLOP/s, % of peak, arithmetic intensity) go into the prompt.
3. **Technique** from a per-class catalogue (`TECHNIQUES`) by UCB1 keyed `(op_class, state)`.
   An arm never tried under a key starts from that technique's mean reward under every other
   key, so a lesson on one slot transfers to slots of the same or other classes.
4. **Prompt**: the slot's signature, semantics, instance shapes, C reference, measured time and
   state, neighbouring slots' signatures (read-only), implementation-store context (seed / adapt
   / examples, below), semantic claims and "already tried" notes, the technique, the current
   source. The LLM returns the complete slot source.
5. **Evaluate**: assemble, build, run the driver; up to `max_fix_attempts` fix rounds with the
   error, the failing slot and known pitfalls. A win of more than 3 % is re-measured.
6. **Reward** = slot time saved / partition time (clipped to ±1); −0.05 for a compile error,
   −0.1 for wrong output or a runtime error.
7. **Record**: an `OutcomeRecord` for every attempt; every verified source goes to the
   `ImplStore`; regressions are not carried forward in the rollout.
8. **Composite best**: the fastest verified source per slot, shared by all rollouts, rebuilt
   and re-verified as a whole at the end (falling back to the seed for any slot that breaks).

`--evaluator local` builds with nvcc on the host; `--evaluator server` goes through
KernelBlaster's compile and exec servers (`compile_cu`, `run_gpu_executable`), the same path
the existing flows use. Both were run in Phase 1.

## Implementation knowledgebase (`knowledge/impl_store.py`)

A record per verified slot source: `target`, `arch`, `op`, `op_class`, `signature_sha`,
`shapes` (instance shapes it was verified on), `buckets` (`shape_bucket`), `slot_us`,
parent time and speedup, `seed_us` and `speedup_vs_seed`, `techniques` (lineage),
`parent_id`, `problem_id`, `run_id`, `slot`, `max_rel_err`. Code is content-addressed
(`code/<sha1>.cu`), the index is append-only JSONL.

Retrieval for a slot (the asking problem's own records are excluded — during a run they are
its current state): **seed** (same op, signature, arch and every instance in the record's shape
buckets — the agent's baseline swaps it in if it verifies and is faster), **adapt** (same op and
signature, at least one shared shape bucket — shown with the instruction to adapt and re-tune),
**examples** (the best record of up to two other ops of the same class, > 1.05× over their
seed). Same-op records whose buckets don't overlap at all are not offered by default
(`far_same_op=True` shows them as ideas-only examples) — see decision 16. `shape_bucket` uses
powers of 4 of the work dimensions (GEMM M/N/K/batch, conv N/IC/OC/OH/OW/KH/KW/G, attention
BH/S/D) or of the element count.

## Write-back (`pipeline/kb_ingest.py`)

`ImplStore.export_winners` writes `winners.json` (`kb-winners/1`): per (op, signature, arch,
shape set) the fastest record with ≥ 1.05× over its seed. Ingest checks each against the
current `KernelSpec` signature (hash) and adds it to `kernels/cuda/kb_cache/index.json`
(`mb-cuda-cache/1`) with its **shape guard** (the exact instance shapes it was verified on).
`CudaCache.lookup(entry, signature, arch, shapes)` returns an entry only if every requested
shape is in its guard. `verify_ir` exports the whole IR with the cached sources and runs its
driver; slots that pass are marked `model_verified` for that IR, failing ones `rejected`.
`kb_export.export_problem(slot_sources=...)` seeds problems from the cache.

## KernelSpec proposals: fusion (`pipeline/kb_proposals.py`)

KernelBlaster proposes (`kb-proposal/1`) fusing producer → consumer pairs where the consumer
is an elementwise op that is the only reader of the producer's output, with the estimated
gain (the consumer instance's measured time) and evidence. ModelBlaster:

1. builds the fused op `fused__<A>__<B>[__<param>]` with `sub_ops` [A, B] and `internal` [t];
2. generates its `FusedKernelSpec`: signature = A's parameters (`a_`), then B's minus the one
   that read t (`b_`); semantics from both; reference = both references;
3. seeds the slot with both hand kernels called back to back (`a_<out>` is scratch) — correct
   by construction; the LLM may fuse and stop writing the scratch;
4. exports the variant partition (references from the torch walk, which runs `sub_ops`);
5. after KernelBlaster optimises both, `decide` accepts a fused kind if its fused slot is faster
   than the two ops it replaced (measured per op, `MB_OP_US`) by ≥ 3 % of the original
   partition — only the touched ops are compared, because whole-partition bests also move with
   unrelated slots — and records the evidence in
   `kernels/cuda/fusion_registry.json` (`mb-fusion-registry/1`). A fusion's status is the
   majority of its decisions. Promotion of an accepted fusion into `KERNEL_SPECS` is a reviewed
   change.

## LLM access

Every LLM call goes through KernelBlaster's `generate_code`. Two keyless prefixes were added:
`claude-code:<model>` runs the Claude Code CLI headless (`claude -p`, tools disabled) on the
logged-in account, and `stub:<name>` calls a registered Python function (tests). Moving to API
keys is a model-name change (`--model`, `--reflect-model`).

## Design decisions

| # | Decision | Alternatives considered | Why |
|---|---|---|---|
| 1 | The per-op C signature is the slot contract; slots are keyed by C entry point | op kind; a new KB-side kernel interface | It is already MB's ABI on every backend, so a winner drops into MB unchanged. Op kinds that share an entry (`sum_dim`/`max_dim` → `kernel_reduce`) share code, so they must share a slot. |
| 2 | ModelBlaster generates the driver (no libtorch) with the reference data linked in | KB's libtorch drivers; reading data at run time | One oracle (the torch IR walk) for partition and per-op checks; the binary is self-contained, which KB's exec server needs (it only receives the executable). Above 1 GiB the data is read at start-up (relocation limits). |
| 3 | Every candidate is checked as a partition **and** op by op on the reference's inputs | partition check only | Attribution: a failure names the slot; and ill-conditioned partitions (attention) can amplify fp32 rounding past the bound even when each kernel is right (seen in SOL-ExecBench). |
| 4 | Slots wrapped in C++ namespaces, assembled into one TU | separate TUs per slot | Fits KB's existing one-driver/one-`.cu` compile path unchanged; helper names can't collide; every kernel name carries its slot. |
| 5 | Timing by CUDA events inside the driver, per op and per partition | nsys (KB's CUDA reward), NCU | Works without profiler permissions (NCU counters are unavailable on the dev box), gives per-slot times directly, no trace parsing. nsys/NCU remain possible as optional detail. |
| 6 | One slot per step, sampled by time share with stall decay | whole-file rewrite; argmax slot | Bounded context and clean attribution per edit (agreed). Sampling spreads parallel rollouts; stall decay stops a big slot that keeps failing from starving the rest (found in testing). |
| 7 | Roofline state from the manifest instead of NCU-derived states | the technique DB's LLM state analysis on NCU reports | No counters needed, no extra LLM call per step, deterministic; the numbers are still shown to the LLM. The technique DB's states can be layered on when NCU is available. |
| 8 | Bandit keyed (op class, state) with cross-key priors | per-problem bandit; (state) only | Transfers technique value across slots and problems of the same kind without a separate sharing mechanism. |
| 9 | Partition-level reward (slot time saved / partition time) | slot speedup | Amdahl-aware: a 2× win on a 1 % slot is worth little; keeps the bandit's values comparable across slots. |
| 10 | Composite best = per-slot winners, re-verified together | per-trajectory best only | Slots are separate kernels, so improvements found by different rollouts on different slots compose; the final re-verification guards the assumption. |
| 11 | Implementation store keyed by op + signature hash + arch + shape bucket, retrieval as seed/adapt/examples | technique stats only; embedding search | Exact reuse is the common case across models (same layer shapes); the bucket allows transfer across nearby shapes; examples share structure across ops of a class. Deterministic and cheap. |
| 12 | Strict exact-shape guard in MB's cache, plus model-level verification | bucket-level guard | A kernel tuned or tested only at some shapes is never silently used at others; relaxing to buckets is a later, measured step. |
| 13 | Fused op keeps the intermediate as a scratch parameter | compute its size inside the kernel; drop it | The sequential seed is then trivially correct and needs no size formula; a fused kernel can ignore it. Cost: the buffer is still allocated. |
| 14 | Fusion proposals are deterministic (structure + measured times) in Phase 1 | an LLM proposal action in the bandit | Cheap, reproducible, and the decision is measured anyway. An LLM proposal action (layouts, splits, algorithm parameters) is the next step. |
| 15 | Keyless `claude-code:` provider for development | API keys now | Requested: our own agentic generation until keys are set up; switching is a model-name change. |
| 16 | Same-op code from other problems only when shape buckets overlap | always offer the best same-op winner to adapt | Measured (ResNet18 seeded from a ResNet-block conv, same 10-call budget, one run each): 25× with no cross-problem conv code, 5.4× when it was offered to adapt, 7.5× as an ideas-only example. A winner's blocking is shape-specific and anchors the LLM. Needs a multi-seed ablation before relaxing. |

## Validation (Phase 1, local TITAN RTX sm_75)

- **Export coverage**: all 150 Level2 + Level3 IRs export as whole-graph problems, build and
  pass every output and isolated-op check with the hand-kernel seeds
  (`gpu_eval/work/kbx_sweep/`). Fixed along the way: op kinds sharing a C entry, > 2 GiB data.
- **Tests**: MB `pipeline/tests/test_kb_export.py` (7, incl. extract → export → build → run and
  a deliberately wrong slot caught and attributed); KB `tests/` 77 (scripted-LLM end-to-end
  agent run with improve / no-op / broken-then-fixed candidates, store reuse across problems,
  the semantic pass).
- **Live run** (`claude-code:sonnet`, 2 rollouts × 4 steps, haiku reflection) on Level3
  `8_ResNetBasicBlock` as one 4-slot partition: 27.8 ms → 0.79 ms (35×) in 8 LLM calls /
  142 s; conv2d 30.9 ms → 0.47 ms; all slots within 3e-7 of the reference; 4 claims
  reflected. Winners ingested, shape guard kept them off ResNet18's different shapes, and the
  model-level verification of the ResNet block passed with the cached kernels.
- **Server path**: the same agent through KB's compile and exec servers on the LeNet fixture.
- **Cross-problem** (`9_ResNet18`, 7 slots / 68 op instances, knowledge carried from the ResNet
  block): 25.4× (48.9 → 1.93 ms) in 10 calls on the first run. That run exposed a retrieval bug
  (the problem's own baseline records always won the seed tier, hiding other problems' work),
  fixed; the reruns then measured decision 16 (5.4× / 7.5×). Within a run, "already tried"
  tallies, claims and same-class examples (relu's winner offered to batchnorm/add) reached the
  prompts as intended.
- **Full round** (`kb_loop.sh` on `8_ResNetBasicBlock`, sonnet, 2 × 4 steps per problem,
  ~4 min): original partition 27.2 → 0.89 ms (30.6×); KernelBlaster proposed conv→batchnorm
  (2 pairs) and batchnorm→add; the variant was built, its sequential seeds passed, and it was
  optimised to 0.93 ms; `decide` rejected the fusions (−4.6 % vs the unfused best at this budget)
  and recorded the evidence; the conv2d winner (48.6× over its seed) and relu were ingested and
  the model-level verification passed; 9 claims were reflected (e.g. "register blocking stacks
  after shared-memory tiling", "implicit GEMM on top of a tiled 1×1 conv regressed — don't
  repeat"). With a registry that accepts conv→batchnorm, the export applies it to all three
  pairs and passes; the fused winner is ingested but correctly not used for the downsample pair,
  whose shapes are outside its guard.

## Known limitations / next steps

- fp32 only; fp16/bf16 and int8 slots need dtype-aware drivers and tolerances.
- The exec server refuses a shared GPU; `KERNELBLASTER_ALLOW_SHARED_GPU=1` relaxes that on
  single-GPU dev boxes (timings get noisier, hence the re-measure of wins).
- Proposals: only elementwise-consumer fusion, decided per partition; chains (conv → bn → relu)
  take two rounds. LLM-generated proposals (layouts, splits, algorithm parameters) are next.
- Partitioning strategies are structural (whole / per_op / window); cost-ranked selection of
  which partitions to send is not implemented yet.
- `claims.json` is single-writer: concurrent processes sharing a knowledge dir need a lock.
- Cross-problem reuse is only validated for exact/overlapping shapes; whether, and how, far-shape
  same-op code helps needs a multi-seed ablation (LLM run-to-run variance is large: see the
  ResNet18 numbers). The live results above are single runs.
- RISC-V: the same manifest and agent apply once the Zephyr harness emits per-op PROFILE rows
  and the bridge builds per-slot files; open RISC-V issues are listed in the KB bug-fix notes.
- The KB submodule tracks `kris-dong/KernelBlaster` `feat/multislot`; merge it into
  `backend-refactor` (and upstream) after review.
