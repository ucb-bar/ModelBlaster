# Region loop (`--region-loop off|on`)

A bounded, tool-using agent loop on the REGION slots of the exported slot problems, driven through the flow's own LLM path (the relay),
replacing the generic per-slot rollout steps on those slots. Default **off**: nothing runs, no key is written, the KernelBlaster command line
and every output are what they were.

## Why

The graph-agent pilot (`l3_farm/graph_agent_pilot/SUMMARY.txt`): a Claude Code agent with typed tools, started from the flow's own final state,
moved 7 of 8 branchy programs 1.75x-3.6x past the flow headline (geomean 2.2x; hints-only arm 2.38x; kernel-only control 1.27x) at $2-7 and
10-60 min per problem. Almost all of the gain was `fuse_region` with agent-chosen members plus a megakernel the agent wrote for the region slot.
The flow tried `region:all` on five of those problems and rejected it: its slot optimiser (one prompt, one reply, one evaluation per step) did not
exploit the region. The pilot's recommendation 4(b): "give region slots the agentic kernel loop and a larger budget than the generic slot loop".
This is that loop: the pilot's TOOL SEMANTICS (profile, scratch experiments, paired timing, verify, commit / revert) ported onto the flow's
LLM path, without the Claude Code scaffolding (permission lockdown, CLI isolation, transcripts).

## The relay carries plain chat only: the protocol

Determined first (`servers/llm_server.py`, `agents/utils/llm_relay.py`, `agents/utils/query.py`): the relay's `POST /v1/generate` takes
`{messages, n_tasks, model, role?, client?, run_id?, request_id?}` with `messages` a list of `{role, content: str}`. The provider paths
(`generate_code_bedrock`: Converse with the system messages folded into the provider's system prompt and tool blocks never built;
`generate_code_openai`: `chat.completions.create(model, messages, n, stream)`; `agentic_llm` stubs / `claude-code:` CLI) forward nothing else.
There is **no tool / function-calling pass-through** (no `tools`, `tool_choice`, `tool_calls`, `function_call` anywhere on the path; the client
`generate_code_retry(messages, model, ...)` has no parameter for them). So the loop uses a **structured-text protocol** over plain chat
completions, which also works through `stub:` models and any provider:

* every assistant turn = reasoning + exactly ONE action block `{"tool": "<name>", "args": {...}}` in an ```` ```action ```` fence, plus (for
  `verify` / `time` / `scratch`) one ```` ```cuda ```` block with the COMPLETE candidate source / standalone program, or (for `edit`) SEARCH / REPLACE
  blocks in an ```` ```edit ```` fence;
* the harness executes the action and answers with the result as the next user message (strict user / assistant alternation: the Converse API
  requires it); budgets are appended to every result;
* a reply that cannot be parsed costs a turn and gets the error text back; 4 in a row end the loop; a reply cut by the provider's output limit
  (`stop_reason` max_tokens / length) is never executed, the model is told to use `edit`;
* `parse_reply` / `render_result` are the only protocol code: native tool calling would replace those two functions and nothing else.

Output-token caps matter: the relay process uses `BEDROCK_MAX_TOKENS` / `ANTHROPIC_MAX_TOKENS` (default 8192) and, for OpenAI-compatible
proxies, `OPENAI_MAX_TOKENS` (unset: the proxy's default, 4096 for Anthropic models). A megakernel needs 10k+ output tokens per full resend: set
them to >= 32000 on the relay host (RUNBOOK). The `edit` tool (SEARCH / REPLACE on the pending or committed source) keeps later turns short.

## What exists / what is new

| | exists | new |
|---|---|---|
| region slots, contract, parity seed, export | `region_slots.py`, `kb_export` | unchanged |
| slot problem, evaluators (local GPU pool / ssh), paired timing (`incumbent=`), per-slot attribution | `knowledge/slots.py` | used through a `Backend` |
| experiment programs | `agents/experiment_tool.py` | `scratch` tool |
| LLM path, roles, usage, pricing | `generate_code_retry`, `model_router`, `cost_tracker` | `default_llm`, `usage_usd` |
| knowledge: claims, technique menu, store examples, outcome rows | `opt_multislot` (`_knowledge`, `_retrieve`, `_record_outcome`, `_store`) | `_region_knowledge`, `_region_event`, region records |
| loop, tools, budgets, protocol, prompts | - | `kernelblaster/agents/region_loop.py` |
| agent hook | - | `MultiSlotAgent(region_loop=LoopConfig)`: `_region_phase` before the rollouts |
| flags | - | MB `--region-loop`, `--region-loop-budget`, `--region-loop-gate`; KB `--region-loop`, `--region-loop-budget` |
| pre-loop gate | `graph_verify` (CPU) | `pipeline/region_loop.py` `gate` in `prepare_hint` |

## Touch points

* KernelBlaster `agents/region_loop.py` (new, ~900 lines): `LoopConfig`, `Ledger`, protocol (`parse_reply`), `lint_source`, `Backend` /
  `EvaluatorBackend`, `RegionLoop` (tools `t_show` ... `t_finish`), `build_system_prompt`, `default_llm`, `region_slots`.
* `agents/opt_multislot.py`: constructor args `region_loop`, `region_backend`, `region_llm`; `_pick_slot` skips slots the loop ran on;
  `_run` calls `_region_phase` once before the rollouts (phase "rollouts", resume-safe: finished slots are in the snapshot); `_snapshot` /
  `_restore` / `_write_result` carry a `region_loop` key ONLY when on; methods `_region_knowledge`, `_region_event`, `_region_one`,
  `_region_phase`, `_region_summary`.
* `scripts/run_multislot.py`: `--region-loop on|off` (`$KERNELBLASTER_REGION_LOOP`), `--region-loop-budget` (`$KERNELBLASTER_REGION_LOOP_BUDGET`);
  `run_config` carries the setting only when on (a resumed run with another setting is refused).
* ModelBlaster `pipeline/region_loop.py` (new): `resolve_mode`, `kb_args`, `RegionLoopHook` (round key / summary), `gate`.
  `kb_graph_opt.py`: Config fields `region_loop`, `region_loop_budget`, `region_loop_gate`; `KBRunner.command` appends `kb_args`; `_run` builds the
  hook (round key + `summary.json` `region_loop` only when on); `prepare_hint` calls `gate` before the variant is exported.

## The loop

`MultiSlotAgent._run`: baseline + store seeds as always; then, with the loop on, for every region slot of the problem (largest time share first) a
`RegionLoop` runs with the problem's budget shared by share (floor: 20%, 6 turns); the model is the **hard role** (`router.models["codegen_hard"]`,
`--model-hard` / `MODEL_CODEGEN_HARD` / `--model`). Slots the loop ran on are not rolled out; all other slots get the generic steps as before.
A committed program is adopted only after a **final re-verification** of the committed composite (a failure reverts the slot to its start); its
source goes into the implementation store, `best/slots`, `result.json` (`region_loop`: per slot stop reason, commits, ledger incl. usd, turns,
per-tool seconds) and from there through the ordinary winners export into MB's cache.

Tools (each call is timed and counted; every result shows what is left):

| tool | semantics |
|---|---|
| `show` | `region` (signature, semantics = members / checked outputs / scratch, instances, call buffers), `source` (paged) / `outline` (a 300 KB composite seed is mapped, not dumped), `candidate`, `reference`, `knowledge`, `history` |
| `profile` | the committed program through the evaluator: end to end (graph / eager), per-slot attributed GPU time and share, launches, largest ops; cached per source; `ncu: true` only where the evaluator can (not on the lab B200 hosts) |
| `scratch` | `ExperimentTool`: compile + run a standalone program on the leased GPU (<= 12 programs / 600 s per loop, refused APIs as in the multi-turn tool); refused on an ssh evaluator |
| `edit` | SEARCH / REPLACE blocks (exact, unique match) on the pending (else committed) source; result = new unverified pending |
| `verify` | static lint (entry defined, no sync / host copy / main / libraries; streams, events and threads are legal in a region) -> evaluator run with every check (compile, the region slot's isolated check vs the reference walk, partition outputs, CUDA-graph capturability) -> backend extra checks (`repeat_run`: a second full build + check) |
| `time` | paired candidate vs committed, 5 alternating reps (the evaluator's interleaved `incumbent=` timing when the runtime supports it, else A B A B ...), median of ratios; a timing on a clock-suspect GPU (X-149 clock guard) is refused as not evidence; `time` is refused without a passing `verify` of the exact source |
| `commit` | gain >= 15% in one timing, or two consecutive timings each >= 3%; refused otherwise; commits are final-verified at the end |
| `revert` | drop the pending candidate, or the last N commits |
| `status` / `finish` | budgets, per-tool accounting, pending state / stop |

Budgets: `max_turns` (LLM calls), `max_tool_calls`, `max_usd` (the relay's usage: provider `cost_usd` when reported, else tokens x the price table;
a model without a price is charged an Opus-class fallback, never 0; `$KERNELBLASTER_REGION_LOOP_PRICE=in,out` overrides), `max_wall_s`. Default
40 / 120 / $8 / 3600 s per problem. A budget stops the loop; committed work stays; a pending candidate is lost. Long sessions are compacted: only
the last turns keep their code blocks and results verbatim (older code is elided with its sha; `show candidate` brings it back).

The prompt: the region contract (inputs / outputs / scratch / weights, dtypes incl. `--quant native`'s mixed pointer types), the flow's dtype rules and
hardware text (`opt_multislot.DTYPE_RULES`, `hw_profile.hardware_text`), the kernel-writing rules, **what a region may do that an ordinary slot may
not and why** (persistent / megakernel patterns; fork / join side streams with events and helper threads inside the call, because the flow's own region
seed does exactly that and a region spans a whole DAG; every fork joined before return, statics created once, nothing synchronises the host,
co-residency via `mb_sm_budget()`), the B200 facts (build line `-gencode=arch=compute_100a,code=sm_100a`, `--default-stream per-thread`, **ncu
disabled** on the lab hosts: measure with `profile`, `scratch` probes and paired `time`), the knowledge section (claims in scope, already-tried
tallies, the class's technique menu with the launch-bound techniques, recurrent advice, implementation-store examples).

Knowledge feeds back like slot steps: every failed `verify` (compile / run / check) and every `time` becomes an `OutcomeRecord`
(technique `region_loop`, e2e timing basis, diff summary), v1 claims observe it, v2 appends it to the outcome store; the committed source is stored
as an implementation; `maybe_reflect` / `maybe_upkeep` run after the phase.

## Pre-loop gate (MB)

`prepare_hint` (flag on, gate on): a variant that contains region ops is checked with `graph_verify.verify` (CPU: structural, then the torch
walkers on one seed of random inputs, outputs and sampled intermediates) against the original IR **before** it is exported to KernelBlaster. A
variant that does not verify is a rejected hypothesis (`job["early"]`: evidence `why`, never timed, no LLM spend).

## Invariants

* Flag off: `region_cfg is None`; `_pick_slot` sees an empty exclusion set (same candidates in the same order); the snapshot, result, run config, MB
  round key, `summary.json` and the KernelBlaster command line have no new key / argument. Evidence: in-repo tests (`test_agent_flag_off_is_unchanged`,
  `LoopTest.test_off_equals_unset_equals_env_off`) and the base-vs-new identity runs in SUMMARY.txt.
* Everything GPU-side is behind `Backend` (the tests use a fake), the LLM behind `llm(messages, role) -> (text, usage)`.
* A loop never raises into the agent: an LLM / tool error is a failed call or a stop reason; the generic rollouts still run on slots the loop did not run on.
* The committed program always passed `verify` and a final re-verification; only paired timings count.

## Cost accounting (what to expect)

Per turn about 3-6k prompt tokens (system 2.5k + task + the last turns; older code elided) and 0.3-2k output tokens (edits) or 6-12k (a full
megakernel): roughly $0.05-0.4 per turn on an Opus-class model; 40 turns ~ $2-10. The pilot's sessions cost $2-7 (25-56 turns). The budget caps it.

## Not done / limits

* GPU-only: nothing here was run on a GPU (no nvcc / device in this environment): the Evaluator-backed backend, `scratch`, the paired `time` on a real
  persistent host, and the real relay are unexercised (RUNBOOK lists the checks).
* The pilot's bench-level runtime checks are NOT in the flow driver: the NaN-poisoned op-by-op run, replay of the timed CUDA graph with new inputs, and
  the in-place weight swap (derived weights follow the parameters). `verify` has `repeat_run` instead. Porting them into `kb_export`'s driver is open.
* `scratch` needs the GPU on the host that runs the loop (the B200 pooled / per-GPU farm modes: yes; KernelBlaster on a controller with
  `--ssh-host`: refused with a message).
* A scripted `stub:` model drives the tests; no real model has seen the prompt: the protocol adherence of an Opus-class model over the relay is the
  first thing the GPU validation measures (`protocol_errors`, `truncated_replies` in the ledger).
* The paired decision still compares the hint's `__orig` (generic rollouts) with the `__variant` (region loop + rollouts): the loop is extra spend on the
  variant side by design.
* A loop killed mid-slot restarts that slot's loop from the problem's best (commits of an unfinished loop are not in the snapshot; the loop's
  `events.jsonl` / `result.json` under `<out>/region_loop/<slot>/` keep the trace).
