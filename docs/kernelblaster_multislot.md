# ModelBlaster ↔ KernelBlaster: multi-slot CUDA optimisation (Phase 1)

ModelBlaster owns the graph: it decides what a problem is (a partition), writes the driver
that checks it, and decides graph-level changes such as fusion. KernelBlaster owns the
kernels: it optimises the N kernels of a problem one slot at a time and shares what it
learns across kernels and problems. The contract between the two is ModelBlaster's per-op C
ABI (`KernelSpec.signature`), which already is the CUDA entry point.

This document covers the Phase 1 CUDA implementation. It is written for review: the
"Design decisions" section lists each non-obvious choice, the alternatives and why.
KernelBlaster's semantic update pass has its own document
(`third_party/KernelBlaster/docs/semantic_update_pass.md`). Phase 2 -- ModelBlaster as the
graph optimiser (unified graph hints, cost-ranked selection, fusion chains, split proposals,
rounds to a fixpoint) -- is in `docs/graph_optimizer.md`. The OpenCL / Qualcomm Adreno target
(`kb_export --target opencl`: OpenCL slots, runtime and driver, Level1-3 coverage, fp16) is in
[opencl_adreno.md](opencl_adreno.md). SOL-ExecBench, ported from the original repository at native workload shapes and run
through this flow, is in [sol_execbench.md](sol_execbench.md).

## Flow

```
IR (graph.json, weights.npz, io.npz)
  │ pipeline/kb_export.py      partition → problem dir (manifest mb-kb-problem/1)
  ▼
problem: manifest, slots/<id>.cu, walker head/tail, driver.cpp, data.bin
  │ KernelBlaster scripts/run_multislot.py → agents/opt_multislot.MultiSlotAgent
  │   per step: pick slot → pick technique → LLM rewrites that slot → build the slot
  │   as a shared object → persistent driver host (CUDA graph replay, paired with the
  │   incumbent) → end-to-end reward, OutcomeRecord, ImplStore record, (reflection)
  ▼
result.json, best/slots/*.cu, proposals.json (kb-proposal/1), winners.json (kb-winners/1)
  │                                          │
  │ pipeline/kb_proposals.py variant/decide  │ pipeline/kb_ingest.py
  ▼                                          ▼
fused variant problem → KernelBlaster → accept/reject (fusion_registry.json)
                                             CUDA kernel cache (shape-guarded) → model-level verify
                                             │ pipeline/kb_model_run.py
                                             ▼
                          the model's result: every partition's winners in ONE whole-model
                          runtime, checked and timed by graph replay vs the hand-seeded model,
                          through KernelBlaster's evaluators (pipeline/kb_eval.py ->
                          scripts/time_problems.py) on the GPU model the runs recorded
```

`gpu_eval/tools/kb_loop.sh IR_DIR WORK_DIR` runs one full round.

**Resume.** Every stage survives a kill (SIGKILL, OOM, reboot, ssh drop, Ctrl-C): rerun the
same command.

- **KernelBlaster.** `run_multislot.py` keeps a journal per problem: an atomic snapshot after the
  baseline and every step, plus each LLM response, fsynced before its candidate is evaluated. A
  rerun skips finished problems and continues the others from their last completed step. An
  in-flight step reuses its journaled responses, so only a call in flight at the kill is paid
  again. Store, outcome-log and claims records are idempotent. A different configuration is
  refused (exit 3; `--force-restart`). SIGTERM / SIGINT stop it cleanly with exit 75.
- **`kb_graph_opt`.** It continues inside the interrupted round (`rNN/progress.json`), and
  `--rounds` is the total for the work dir.
- **`kb_loop.sh`.** It skips its finished steps (`W/.kb_loop/`).

Details: [graph_optimizer.md](graph_optimizer.md), "Resume", and KernelBlaster
`docs/multislot_optimization.md`, "Resume".

**One hardware for the model and its components.** KernelBlaster records the GPU every
result was timed on (`result.json` `gpu`: model name, UUID, host, arch). The model's result
(`kb_model_run`) and, with `--ssh-host`, the graph optimiser's per-round profiles, re-measurement
and final verification run through the same KernelBlaster evaluators: the local GPU pool with its
locks, or SSH to the remote GPUs ([graph_optimizer.md](graph_optimizer.md), "Where the GPU steps
run").
They require that GPU model and arch, so a model number and its partition numbers always come
from the same GPU model, by default from one GPU. No GPU of that model is an error unless
`--allow-gpu-mismatch` is given. Details and the L40S measurements are in
[gpu_harness_perf.md](gpu_harness_perf.md), "The model's result".

## Code map

| repo | file | role |
|---|---|---|
| MB | `pipeline/kb_export.py` | partitioning, slot extraction from the hand kernels, walker + driver codegen, local build/run |
| MB | `pipeline/kb_ingest.py` | winners → CUDA kernel cache (shape guard), model-level verification |
| MB | `pipeline/kb_proposals.py` | fusion: fused op + `FusedKernelSpec`, variant export, accept/reject registry |
| MB | `pipeline/kb_model_run.py` | the model's result: winners composed into one whole-model runtime, checked, timed by graph replay against the seeds (and the sum of the partitions) on the components' GPU model |
| MB | `pipeline/torch_baseline.py`, `pipeline/torch_bench.py` | the model's result against the original PyTorch module (eager, CUDA graph, `torch.compile`; TF32 off / on) on the same GPU, interleaved, GPU and wall-clock time ([gpu_harness_perf.md](gpu_harness_perf.md), "Against PyTorch") |
| MB | `pipeline/kb_eval.py` | ModelBlaster's GPU runs (model result; with `--ssh-host` the graph optimiser's profiles, re-measurement and verification) through KernelBlaster's local / ssh evaluators (`KBEval`, `run_problem`); the GPU model and arch the runs recorded (`required_hardware`, `check_device`); the evaluator flags |
| MB | `pipeline/op_classes.py` | op → class (gemm, conv, reduction, elementwise, data_movement, attention, scan, other) |
| MB | `gpu_eval/harness/walk_torch.py` | reference values (torch IR walk; runs fused ops' `sub_ops`) |
| MB | `gpu_eval/tools/kb_loop.sh` | the round above |
| MB | `pipeline/kb_telemetry.py` | graph-optimiser telemetry: loads KB's `telemetry` package, rolls up KB runs |
| KB | `data/sources/modelblaster_source.py` | `ModelBlasterSource` ("modelblaster"); `Problem.slots` |
| KB | `knowledge/slots.py` | `SlotProblem`, assembly, driver-output parsing, `LocalEvaluator`, `SSHEvaluator` (`gpu_model` / `max_devices` pinning, `EvalResult.gpu`), `ServerEvaluator` |
| KB | `scripts/time_problems.py` | check + time problems outside the agent loop (paired with an incumbent, rounds or jobs order), for ModelBlaster's model result, re-measurement, profiles and verification |
| KB | `knowledge/runtime_host.py` | client for the persistent driver host (protocol, crash recovery, host pool) |
| KB | `knowledge/impl_store.py` | implementation knowledgebase |
| KB | `knowledge/outcomes.py`, `claims.py`, `reflect.py`, `semantic.py` | outcome records + semantic update pass |
| KB | `agents/opt_multislot.py` | the slot-wise RL agent, roofline state, technique bandit, fusion proposals |
| KB | `agents/utils/agentic_llm.py` | `claude-code:<model>` / `stub:<name>` LLM providers |
| KB | `scripts/run_multislot.py` | runner |
| KB | `telemetry/`, `scripts/telemetry_report.py` | token / time telemetry, `telemetry.jsonl` report |

## Problem format (`mb-kb-problem/1`)

`manifest.json` fields:

- `name`, `source_ir`, `ir_name`, `partition` (`strategy`, `op_names`, `op_indices`, `is_whole_graph`)
- `target` (`cuda`), `dtype` (`fp32` / `fp16`), `tolerance` (`output_rel`, `output_abs`, `slot_rel`,
  `slot_abs`; per dtype, [fp16_cuda.md](fp16_cuda.md); `output_cond`, 2)
- `assembly`: `head`, `tail`, `namespace_prefix` (`mbslot_`), `order` (slot ids)
- `slots[]`: `id`, `op` (first op kind), `ops` (all kinds using it), `op_class`, `entry`,
  `signature`, `file`, `reference_file`, `semantics`, `seed_source`
  (`hand` / `cache:<impl_id>` / `fused-seed` / `override`), `instances[]` (`op_name`, `op`,
  `index`, `shape`)
- `ops[]` (call order): `index`, `name`, `op`, `slot`, `inputs`/`outputs` (buffer indices)
- `buffers[]`: `name`, `kind` (input, const, weight, intermediate, output), `dtype` (f32/f16/i32),
  `numel`, `alias` (`base`, `offset` — views), `ref` (`[offset, nbytes]` into `data.bin`),
  `ref_err` (outputs: max |reference − exact|, below; null when not computed)
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
MB_PHASE_MS <phase> <ms>                                       wall time per driver phase
passed | failed
```

Run modes (environment; [gpu_harness_perf.md](gpu_harness_perf.md)): `MB_ONLY_SLOTS=a,b`
limits the isolated checks to those slots' ops (KernelBlaster's one-slot candidates; the
partition is always checked and timed), `MB_SCREEN=1` shortens the timing budgets, and
`MB_GRAPH=1` (kernels built with `--default-stream per-thread`) adds `MB_PARTITION_GRAPH_US`,
the partition replayed as a CUDA graph.

Output bound: `err <= output_rel * max|finite ref| + output_abs + output_cond * ref_err` (1e-3,
1e-5, as the Level2/3 gates; `output_cond` 2). `ref_err` is how far the reference itself is from
the exact value: a second walk at export, in float64 without the fp16/bf16 store rounding, holding
only its live set (`exact_values`, run before the reference walk so the two are never held together; `export_problem(conditioning=False)` /
`kb_export --no-conditioning` skip it, the term is then 0). A candidate as close to the exact value
as the reference differs from it by up to `2 * ref_err`. For a well-conditioned output the term is
noise next to the relative bound; it matters for an output that cancels to ~0, whose fp32
reference is rounding noise of its summands' scale: the gradient of a key bias through a softmax
(SOL L2_021 `sum_4`: reference max 1.45e-5, exact 3e-14, plain bound 1.0e-5) or a routing
gradient (L2_080 `matmul_7`: 6.8e-5 against an exact 1e-13). Without it the reference would fail
its own check. Isolated bound: `max err <= slot_rel * max(|finite inputs|, |finite ref|) + slot_abs`
(1e-3 and 0, as `walk_cuda`'s per-kernel check; fp16: 2e-3 and 2^-23, see
[fp16_cuda.md](fp16_cuda.md)); `rel_err` reports `max err / max(...)`. Matching infinities/NaNs are exact. `MB_ERROR` lines report
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

## NCU profiles (`knowledge/ncu.py`, opt-in `--ncu best|all`)

The driver's `MB_MODE=profile` runs every op exactly once on the reference inputs. Every
kernel lives in its slot's namespace, so one `ncu --kernel-name-base demangled
--print-kernel-base demangled -k regex:'mbslot_' --page details --csv` run (SpeedOfLight,
Occupancy, LaunchStats, MemoryWorkloadAnalysis) profiles all slots, and each kernel row is
attributed to its slot by name. Per slot: duration-weighted SM / memory / DRAM / L1 / L2
throughput, achieved and theoretical occupancy, hit rates, registers, waves, the measured bound
(`compute_bound`, `memory_bound`, `latency_bound`, `underutilized_*`), and NCU's top rule hints
with their estimated speedups. With `--ncu best` the baseline and every new slot best are
profiled (`all`: every verified candidate); the measured bound replaces the roofline state in
the prompt and the bandit key, the metrics and hints go into the prompt, and outcome records get
`state_after` and `metrics_delta` (input to the semantic pass). Profiles are cached by
(slot, code hash). Without counter access (`ERR_NVGPUCTRPERM`, as on the dev box) the first
failure disables NCU for the run and the roofline state is used. `SSHEvaluator` builds locally
and runs the self-contained binary on a remote GPU host (used to validate on firesim2's L40S).
Problems whose `data.bin` is read at run time (above 1 GiB) also run there. The data is copied
to each host once and passed as `MB_DATA_PATH`.

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

To make the calls on another machine (VPN, or keys kept only there), run KernelBlaster's relay
server (`kernelblaster.servers.llm_server`) on that machine. Then set `KB_LLM_SERVER_URL` and
`KB_LLM_SERVER_TOKEN_FILE` in the environment that starts the pipeline, typically through an
`ssh -L` tunnel. `kb_graph_opt` starts `run_multislot.py` with the inherited environment, so no
flags are needed. Accounting (`llm_usage`, telemetry) is unchanged. Only the calls that go through
KernelBlaster are relayed; ModelBlaster's own kernel-generation clients (`pipeline/llm_client.py`
and friends) are not. See KernelBlaster `docs/llm_server.md`.

## Telemetry

KernelBlaster records, for every LLM call:

- input / cache / output tokens;
- the stop reason, flagging truncation at `max_tokens`;
- latency, retries and cost, by role (`codegen_simple`, `codegen_hard`, `fix`, `reflect`, `upkeep`, `hypothesis`)
  and model.

It also records where each step's wall clock goes:

- prompt building and waiting for the LLM;
- the compile-slot queue vs nvcc, build-cache hits;
- the GPU-lease wait;
- the driver's own phases from `MB_PHASE_MS` (CUDA init, partition / op checks, timing loops), NCU;
- knowledge lookup / record / reflection, I/O.

The results go to these places:

- each step's `context.telemetry` in `result.json`;
- the problem's `telemetry`;
- a run-level `{"telemetry": ...}` entry in `summary.json`;
- `OUT/telemetry.jsonl`.

`KB_TELEMETRY=0` turns it off. The details are in KernelBlaster's `docs/telemetry.md`.

ModelBlaster's graph optimiser (`pipeline/kb_graph_opt.py`, via `pipeline/kb_telemetry.py`) adds its own spans:

| span | what |
|---|---|
| `mb.run` | the whole run |
| `mb.round` | a round, with laps `mb.profile`, `mb.propose`, `mb.plan`, `mb.export`, `mb.kb_wait` (the plan's KernelBlaster run), `mb.ingest`, `mb.decide` |
| `mb.profile.run` | inside the profile lap: one IR's profile runs (`evaluator` builtin / local / ssh, `device`, `gpu`, `host`, `passed`); with a KernelBlaster evaluator its child is `mb.kb_time` |
| `mb.prepare` | inside decide: exporting each hint's original and variant problems |
| `mb.kb_wait` | inside decide: waiting for a wave's KernelBlaster run |
| `mb.remeasure` | inside decide: re-timing, in the worker threads |
| `mb.finish` | inside decide: decisions and ingest |
| `mb.verify` | the final model-level verification (`evaluator`) |
| `mb.verify.run` | one IR's verification run (`evaluator`, `device`, `gpu`, `host`, `passed`); child `mb.kb_time` with a KernelBlaster evaluator |
| `mb.model_run` | the model's result per IR; children `mb.model_run.compose` (cache, plan, export), `mb.model_run.time` |
| `mb.kb_time` | one call of KernelBlaster's `time_problems.py` (evaluator, GPU model, jobs, seconds); its own spans go to `<work>/kb_time/<name>.telemetry.jsonl` |

These go to `WORK/telemetry.jsonl`.

`round.json` and `summary.json` get a `telemetry` entry with two parts:

- `mb`: ModelBlaster's own breakdown and derived metrics;
- `kb`: the sum over that round's (or the whole run's) KernelBlaster runs, read from each run's `summary.json`.
  It has wall clocks, time by category, LLM totals and per role/model, truncations, `usd_per_win` and
  `tokens_per_win`.

Running and reading it:

- **Report.** `python <KB>/scripts/telemetry_report.py WORK` prints the tables for ModelBlaster and every
  KernelBlaster run under `WORK`.
- **Where the code comes from.** ModelBlaster loads the span machinery from the KernelBlaster checkout it drives
  (`--kb`): a stdlib-only package, under a private module name.
- **Older checkouts.** A checkout without it, or `KB_TELEMETRY=0`, gives a no-op and the flow is unchanged.

One live round on the ResNet basic block (Haiku 5.5, local TITAN RTX, 2 rollouts × 4 steps) took 202 s:

- **ModelBlaster.** `mb.kb_wait` 164 s, `mb.profile` 21 s, `mb.verify` 16 s.
- **Inside the 158 s KernelBlaster run.** 69 % waiting for the LLM, 25 % nvcc, under 2 % each for CUDA init,
  checks, timing loops and GPU queueing.
- **Cost.** 9 calls, 36 k input / 46 k output tokens, est. $0.27, $0.13 per accepted win.

## Ablations

`python -m modelblaster.pipeline.kb_ablate` compares configurations of `run_multislot.py`
(`entry: "multislot"`) or of the graph optimiser. Examples: knowledge v1 vs v2, mined evidence on
vs off, reward, model policies, NCU. Each arm x repeat runs in its own directory with a per-run
knowledge seed and snapshot copy, and the arms are interleaved on one GPU model. The report gives
paired geo-mean ratios with bootstrap CIs and permutation / Wilcoxon p-values. See
[ablations.md](ablations.md); example specs are in `gpu_eval/ablations/`.

## Design decisions

| # | Decision | Alternatives considered | Why |
|---|---|---|---|
| 1 | The per-op C signature is the slot contract; slots are keyed by C entry point | op kind; a new KB-side kernel interface | It is already MB's ABI on every backend, so a winner drops into MB unchanged. Op kinds that share an entry (`sum_dim`/`max_dim` → `kernel_reduce`) share code, so they must share a slot. |
| 2 | ModelBlaster generates the driver (no libtorch) with the reference data linked in | KB's libtorch drivers; reading data at run time | One oracle (the torch IR walk) for partition and per-op checks; the binary is self-contained, which KB's exec server needs (it only receives the executable). Above 1 GiB the data is read at start-up (relocation limits) from `$MB_DATA_PATH`; KB's `SSHEvaluator` copies it to each GPU host once. |
| 3 | Every candidate is checked as a partition **and** op by op on the reference's inputs | partition check only | Attribution: a failure names the slot; and ill-conditioned partitions (attention) can amplify fp32 rounding past the bound even when each kernel is right (seen in SOL-ExecBench). |
| 4 | Slots wrapped in C++ namespaces, assembled into one TU | separate TUs per slot | Fits KB's existing one-driver/one-`.cu` compile path unchanged; helper names can't collide; every kernel name carries its slot. |
| 5 | Timing inside the driver: the partition by CUDA events around a graph replay (eager as fallback and diagnostic), per-op attribution from CUPTI activity records | events around each op (the original choice); nsys; NCU | No profiler permissions needed, no trace parsing. Events around each op charged tiny ops 2–3× their kernel time; CUPTI's kernel durations match nsys (`gpu_harness_perf.md`). |
| 6 | One slot per step, sampled by time share with stall decay | whole-file rewrite; argmax slot | Bounded context and clean attribution per edit (agreed). Sampling spreads parallel rollouts; stall decay stops a big slot that keeps failing from starving the rest (found in testing). |
| 7 | Roofline state from the manifest instead of NCU-derived states | the technique DB's LLM state analysis on NCU reports | No counters needed, no extra LLM call per step, deterministic; the numbers are still shown to the LLM. The technique DB's states can be layered on when NCU is available. |
| 8 | Bandit keyed (op class, state) with cross-key priors | per-problem bandit; (state) only | Transfers technique value across slots and problems of the same kind without a separate sharing mechanism. |
| 9 | End-to-end reward: fraction of the deployed partition time saved (CUDA graph replay, paired with the incumbent in one process), accepted at 1 % after a full re-measure against the composite | slot time saved / sum of slot times (the original choice, `--reward slot`); slot speedup | What is deployed is the whole forward pass: fusion, fewer launches, capture-ability and cross-op cache effects count, and per-slot event times misjudge tiny slots. Same unit as before, so bandit and v2 statistics stay comparable. |
| 10 | Composite best = per-slot winners, re-verified together | per-trajectory best only | Slots are separate kernels, so improvements found by different rollouts on different slots compose; the final re-verification guards the assumption. |
| 11 | Implementation store keyed by op + signature hash + arch + shape bucket, retrieval as seed/adapt/examples | technique stats only; embedding search | Exact reuse is the common case across models (same layer shapes); the bucket allows transfer across nearby shapes; examples share structure across ops of a class. Deterministic and cheap. |
| 12 | Strict exact-shape guard in MB's cache, plus model-level verification | bucket-level guard | A kernel tuned or tested only at some shapes is never silently used at others; relaxing to buckets is a later, measured step. |
| 13 | Fused op keeps the intermediate as a scratch parameter | compute its size inside the kernel; drop it | The sequential seed is then trivially correct and needs no size formula; a fused kernel can ignore it. Cost: the buffer is still allocated. |
| 14 | Fusion proposals are deterministic (structure + measured times) in Phase 1 | an LLM proposal action in the bandit | Cheap, reproducible, and the decision is measured anyway. An LLM proposal action (layouts, splits, algorithm parameters) is the next step. |
| 15 | Keyless `claude-code:` provider for development | API keys now | Requested: our own agentic generation until keys are set up; switching is a model-name change. |
| 17 | NCU: one run per candidate over all slots, attributed by namespace, opt-in | NCU per slot (KB native: per bottleneck kernel); always on | One run is cheaper than N; the namespace makes attribution exact; opt-in because counters need permissions and each run adds replay time. |
| 18 | Fusion decisions compare only the touched ops (per-op `MB_OP_US`) | compare whole-partition bests | Partition bests also move with unrelated slots: on MiniGPT a lucky GEMM made a 2.3 % epilogue fusion look like +29 %. |
| 19 | Persistent driver host per (problem, GPU); candidates as dlopen'ed shared objects, dlclosed after each request | a process per candidate (as before); a process per N candidates | Removes CUDA init and process start per candidate and the relink of the linked data. A candidate .so is about 1 MB. dlclose of CUDA modules measured clean: 200 cycles, no growth. Crashes, hangs and sticky CUDA errors kill and restart the host, so they never reach the next candidate. |
| 20 | The model's result is one whole-model runtime with every partition's winners, timed like a partition | the sum of partition times; per-problem speedups | Partition boundaries hide and add launches and cache effects. What ships is the model. |
| 16 | Same-op code from other problems only when shape buckets overlap | always offer the best same-op winner to adapt | Measured (ResNet18 seeded from a ResNet-block conv, same 10-call budget, one run each): 25× with no cross-problem conv code, 5.4× when it was offered to adapt, 7.5× as an ideas-only example. A winner's blocking is shape-specific and anchors the LLM. Needs a multi-seed ablation before relaxing. |

## Validation (Phase 1, local TITAN RTX sm_75)

- **Export coverage**: every KernelBench Level1 (100), Level2 (100) and Level3 (50) IR exports
  as a whole-graph problem that builds and passes every output and isolated-op check with the
  hand-kernel seeds; SOL-ExecBench 139/144 (#021/#028 fp32 conditioning, #080 a 1.5e-5 miss on a
  near-zero output, #072/#073 out of memory on the shared GPU). Level1 needed 24 more hand
  kernels and reference ops (losses, norms, argmax/argmin with int64 index buffers, triu/tril,
  flip, exclusive cumsum, ...) and a conv1d reshape fix in the torch walk.
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
- **API key (Bedrock, `us.anthropic.claude-haiku-5-5`)**: full rounds on `8_ResNetBasicBlock`
  (35.0 → 0.97 ms, 36×, 9 calls) and `44_MiniGPTBlock` (430 → 9.5 ms, 45×, 8 calls; fusion
  variant 6.7 ms); per touched-op decision: batchnorm→add accepted (+4.6 %), mul_scalar→masked
  fill rejected (2.3 %), conv→batchnorm rejected.
- **NCU on L40S** (`--evaluator ssh` to firesim2, `--ncu best`, Haiku): Level2
  `1_Conv2D_ReLU_BiasAdd` 3.79 → 0.14 ms (26.6×) in 8 calls; 5 NCU runs; measured bounds in the
  prompts (conv compute-bound, relu memory-bound at 75 % DRAM throughput) and metric deltas in the
  outcome records.
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

(Phase 2, `docs/graph_optimizer.md`, addresses the proposal, chain and selection items below.)

- fp32 and fp16 ([fp16_cuda.md](fp16_cuda.md): contract, tolerances, generated `_f16` kernels, Level3
  validation); bf16 and int8 slots are not implemented.
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

## Trace store

Every KernelBlaster run writes `OUT/trace/` (KernelBlaster `docs/trace.md`). For each step
it records:

- the exact prompts and raw responses
- every candidate source, winning and losing, and every fix attempt
- the build commands and compiler output
- the raw driver and NCU output
- the decision, with links to `result.json` `steps[]`, the outcome id and the telemetry
  span

The graph optimiser adds ModelBlaster's own steps in `WORK/trace/` (`pipeline/kb_trace.py`,
`docs/graph_optimizer.md`, "Trace"). Use `scripts/trace_view.py` in the KernelBlaster
checkout to view a trace.
