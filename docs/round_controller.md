# Round controller (`--round-controller off|fixed|graph-first|auto`)

CPU-only feature; no GPU run has validated it yet (see "Not done"). Code: `pipeline/round_controller.py`, hooks in
`pipeline/kb_graph_opt.py` (one guarded block per site), `pipeline/region_loop.py` (`set_round`). Tests:
`pipeline/tests/test_round_controller.py` (30).

## Why

The flow's graph moves are separate flags: the fusion / split hints (always), `--merge-sibling-convs auto`, `--region-windows auto`,
`--region-slots`, `--region-loop on`, `--graph-rewrite agent`. A round always did the same thing: profile, propose everything, run the
kernel plan (KernelBlaster rollouts), then decide the hints (paired variant-vs-original runs). Whether a round should tune kernels, restructure
the graph, or stop was never a decision. The controller makes it one, per round boundary, and records why.

## Moves

At the start of a round (after the profile and the cheap CPU proposals, before any KernelBlaster / LLM spend) the controller picks a move set:

| component | what runs when it is on | when it is off |
|---|---|---|
| `kernel` | the plan's KernelBlaster run (normal rollouts) | the plan is emptied for this round (`plan.json` still has the full plan; nothing is marked sent, so the items stay eligible) |
| `graph` | the paired decisions of the proposed hints (fuse / split / merge / windows / ...) and the `--graph-rewrite agent` proposer | no candidates this round (`n_todo` 0); the hints stay `proposed` for a later round; the LLM proposer is not called |
| `region_loop` | `--region-loop on` is passed to this round's KernelBlaster runs (plan region slots, variants of region hints decided now) | `--region-loop` is left off the command line of this round's runs |
| `stop` | no round: recorded as a fixpoint, the loop ends | - |

The flags stay the capability switches: the controller never turns a feature on (`--region-loop off` stays off; with `--merge-sibling-convs off`
there are no merge candidates). It only chooses WHEN what is enabled runs. It does not look at dtypes, `--quant`, arch or any gate:
`graph_verify`, the region pre-loop gate, hint verification (`verify_rewrite`), the paired decision and the final verification run exactly as before
on whatever it lets through.

## Policies

* `off` (default; `$MB_ROUND_CONTROLLER` when the flag is absent): nothing below exists. No key, no record, every output byte-identical.
* `fixed`: kernel + graph (+ region loop when its flag is on) every round: the flow's current order. Only the guard and the budget can change it.
* `graph-first`: the first round of a run is graph-only when it has decidable graph candidates (hints with est. gain >= `--min-gain-pct`), so the
  kernel steps start in round 2 on the restructured graph (a graph change alters shapes and slots; kernel tuning done before it is partly wasted).
  Afterwards as `fixed`. Costs one round: use `--rounds N+1`.
* `auto`: the LLM-free rule below.

### The `auto` rule (`RoundController._auto`, ordered, every branch writes its reason)

Inputs (all from state the flow already has; `controller.inputs` in `round.json`):
`small_unfused.share` (share of profiled time in unfused ops of class elementwise / data-movement / reduction that take <= `--round-controller-small-us`,
default 15 us; fused and `region__` ops are excluded), `candidates.{graph,merge,region}` (count and estimated gain: sum of the best 3 `est_gain_pct`
of the proposed hints above `--min-gain-pct`; merge = `merge_sibling_convs`, region = `fuse_region` incl. windows, graph = everything else),
`live_region_ops` (profile ops starting `region__`), `last_gain_pct` (kernel and graph+region gain of the previous round), `prev_round_realized_pct`
(whole partition change since the previous round), `rounds_left`, `spent_usd` / `budget_left_usd`.

1. Guard first: stopped moves (below) are out. Nothing left to run = `stop`.
2. No graph candidate with est. gain >= min-gain-pct: graph out (kernel-only round, not a fixpoint).
3. First round, and (est. graph gain >= 8 % or (small-op share >= 25 % and some candidate)): GRAPH-ONLY round (kernel deferred).
4. Later: last graph gain >= 2 x max(last kernel gain, 3 %) and the previous round was not graph-only: GRAPH-ONLY round (a restructured graph changes
   which kernels are worth tuning). Never two graph-only rounds in a row.
5. Otherwise kernel + graph.
6. Region loop (only with `--region-loop on`, not stopped): on when there are live region ops and the kernel move runs, or region candidates and
   the graph move runs; else off with the reason.

A deferred component (3, 4, and graph-first's first round) blocks the fixpoint test for that round (otherwise an empty graph-only round would end the run).

### The budget guard (all policies except off)

A move is stopped when its last two ROUNDS IN WHICH IT HAD WORK each gained less than `--round-controller-min-gain` percent (default 3) of the model's
profiled partition time. "Had work": kernel = the plan had problems; graph = at least one decision; region loop = region ops live or a region hint decided.
Gain per move (`controller.gain`, percent of the round's profiled partition_us): kernel = sum over the plan's problems of (KernelBlaster baseline -
final) partition_us; graph / region = sum of (original_us - variant_us) of ACCEPTED paired decisions (hints of kind `fuse_region` count as region).
These are per-sub-problem measurements, a lower bound of the whole-model gain; `prev_round_realized_pct` is the whole-model change as a cross-check.
Kernel and graph both stopped = stop. USD budget: `--round-controller-budget-usd X` (default none) stops the run before the next profile once
the KernelBlaster runs' `llm_usage.cost_usd` (all runs of this work dir, plan and hint decisions) reaches X; it does not stop a round already in flight.

## Flags / env

| flag | env | default |
|---|---|---|
| `--round-controller off\|fixed\|graph-first\|auto` | `MB_ROUND_CONTROLLER` | off |
| `--round-controller-min-gain PCT` | `MB_RC_MIN_GAIN` | 3 |
| `--round-controller-budget-usd USD` | `MB_RC_BUDGET_USD` | none |
| `--round-controller-small-us US` | `MB_RC_SMALL_US` | 15 |

A non-off policy puts `round_controller` (and the non-default parameters) into the resume key: a work dir is resumed with the same setting. The move
of a round is saved in `rNN/progress.json` (phase `controller`), so a killed and resumed round keeps it. Not combined with the `--graph-*` search
flags (interleaved decisions / beams schedule themselves): `ValueError` at start.

## Records

* `rNN/round.json` and `state.json` rounds: `controller` = `{policy, move, kernel, graph, region_loop, stop, deferred, reasons[], inputs{...}, gain{...},
  worked{...}, spent_usd}`. A stop before the profile (budget, guard) is a `state.json` round with `move: stop` and no `rNN` dir.
* `summary.json`: `round_controller` = `{policy, min_gain_pct, budget_usd, small_us, stop_reason, rounds[{round, move, reasons, gain, worked}]}`.
* stdout: one JSON line per round: `{"round": r, "controller": {"move": ..., "reasons": [...]}}`.

## Reused hooks

`MergeHook` (msc): proposals run before the choice (CPU); decided only under the graph move. `RegionLoopHook` (rl): the switch is
`region_loop.set_round(bool|None)`, read by `kb_args`; the pre-loop gate is unchanged. `Caps` (windows / region slots): proposals before the choice.
`RewriteHook` (rw): the only proposer with LLM cost; it runs after the choice and only under the graph move (with the controller on it runs after
`msc.propose`, otherwise the order is the legacy one). The hints decision (`kb_resume.decide_hints`): unchanged, fed the empty candidate list on a
kernel-only round.

## SOL problems (`--quant native`, B200 arch 100)

Nothing in the controller reads dtype, quant or arch; native-quant op kinds that `op_class` does not know are class `other` and simply do not
count as small ops (the share is then conservative, and graph-first / candidate-based rules still work). The B200's launch floor is lower than an L40S's:
the 15 us "small" threshold is a flag. Gates (dtype contract, tolerance, graph_verify, region gate) are untouched, so a SOL run's acceptance criteria
are the same with the controller on or off.

## Flag-off identity

`l3_farm/round_controller/results/identity/mb_loop.txt`: 3-round loop, fake GPU steps, real plan with KernelBlaster runs every round, fp16 and fp32 fire
IRs: base (flow-b200 441bd69) vs new default vs `--round-controller off` vs `MB_ROUND_CONTROLLER=off`: summary, state, graph_hints, round.json and
all work files identical (6/6, 86 files each). `fixed` equals base after removing the controller keys (2/2). In-repo: `FlowTest.test_off_equals_unset_equals_env_off`.

## Not done

No GPU / relay run: the thresholds (8 %, 25 %, 2x, 3 %, 15 us) are untuned priors; gains are per-sub-problem lower bounds; the region-loop gain is only
what accepted `fuse_region` decisions show (the loop's own slot gains inside a plan run are counted as kernel gain); `--graph-rewrite agent` is still not
implemented upstream; no mid-round budget stop; not combinable with `--graph-*` search.
