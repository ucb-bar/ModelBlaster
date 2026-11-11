# Ablations of the ModelBlaster <-> KernelBlaster flow

`pipeline/kb_ablate.py` runs controlled comparisons of the optimisation flow with one command.
Examples: knowledge v1 vs v2, step reward e2e vs slot, model-heterogeneity policies, a knowledge
seed with vs without mined evidence, NCU on vs off. You describe the comparison in a spec file.
The runner then:
- executes every arm x repeat in its own isolated run directory;
- interleaves the arms over time;
- stops launching when the LLM budget would be exceeded;
- writes a report with per-problem and per-arm tables and paired statistics.

```
python -m modelblaster.pipeline.kb_ablate run SPEC [--root DIR] [--dry-run] [--max-parallel N]
                                                   [--budget-usd X] [--who NAME] [--ledger FILE]
                                                   [--force] [--retry-failed]
python -m modelblaster.pipeline.kb_ablate status ROOT
python -m modelblaster.pipeline.kb_ablate report ROOT [--force]
```

Run it from the directory that contains the `modelblaster` checkout, or put that directory on
`PYTHONPATH`. The default root is `$MB_ABLATE_ROOT/<name>`, else `./ablations/<name>`.

`run` again on the same root continues the ablation:
- completed runs are skipped;
- runs that were killed are resumed;
- runs stopped by the budget are started.

`run` exit codes:

| code | meaning |
|---|---|
| 0 | all runs done |
| 1 | some runs failed |
| 2 | spec / consistency error |
| 4 | stopped on the budget |
| 75 | stopped by a signal (resumable) |

`report` exits 3 when it refuses to compare.

## Why

Before this runner, each arm was set up by hand. Some inputs could leak between arms:
- the MB kernel cache;
- `data/kernelblaster/knowledge/mined.json`, which holds evidence from earlier runs on the same
  problems. This confounded the v1-vs-v2 result in KernelBlaster `docs/knowledge_architecture.md`
  §16;
- knowledge snapshots copied by hand;
- variables inherited from the calling shell;
- the time-varying load on shared GPUs.

Work dirs also did not record the code commit or the environment.

The runner removes each of these sources by construction and records the rest.

## Spec

The spec is JSON, or YAML if PyYAML is installed. Keys starting with `_` are comments. Relative
paths are relative to the spec file, and `${VAR}` is expanded; an unset variable is an error.

```json
{
 "name": "knowledge_v1_vs_v2",
 "description": "v1 vs v2, both from the seed without mined evidence",
 "entry": "multislot",
 "kb": "${MB_KB_ROOT}",
 "inputs": ["${MB_ABLATE_PROBLEMS}/p_12_Gemm_Multiply_LeakyReLU", "${MB_ABLATE_PROBLEMS}/p_57_Conv2d_ReLU_HardSwish"],
 "flags": {"--model": "us.anthropic.claude-haiku-5-5", "--reflect-model": "us.anthropic.claude-haiku-5-5",
           "--rollouts": 2, "--steps": 3, "--max-fix": 1},
 "env": {"KERNELBLASTER_ALLOW_SHARED_GPU": "1", "BEDROCK_MAX_TOKENS": "16384"},
 "path_exclude": "xilinx", "path_prepend": ["/usr/local/cuda/bin"],
 "gpu_model": "NVIDIA TITAN RTX",
 "knowledge": {"mined": false},
 "seeds": [0, 1, 2], "max_parallel": 2,
 "budget": {"usd": 3.0, "est_usd_per_run": 0.3, "who": "ablate",
            "ledger": "/scratch/dima/BlasterMerge/gpu_eval/work/llm_budget/ledger.jsonl"},
 "arms": {"v1": {"flags": {"--knowledge": "v1"}},
          "v2": {"flags": {"--knowledge": "v2"}}}
}
```

### Base keys (shared by every arm)

| key | meaning |
|---|---|
| `entry` | `multislot`: KernelBlaster `scripts/run_multislot.py` on exported problem dirs. `graph_opt`: `python -m modelblaster.pipeline.kb_graph_opt` on IR dirs |
| `kb`, `mb` | the KernelBlaster checkout (default `$MB_KB_ROOT`) and the ModelBlaster checkout (default: this one; `graph_opt` needs it in a directory named `modelblaster`) |
| `python` | the interpreter for the entry point (default `$MB_KB_PYTHON`, else this one) |
| `script` | replaces the entry point's script, keeping its outputs (tests use a fake) |
| `inputs` | problem dirs (`multislot`) or IR dirs (`graph_opt`); globs allowed; their names must be unique |
| `per_input` | `true`: one run per input instead of one run over all inputs. The inputs then share no knowledge within a run |
| `flags` | `{"--flag": value}`. `true` gives a bare flag, a list repeats the flag, and `null` / `false` in an arm removes a base flag |
| `args` | raw extra arguments, appended |
| `env` | environment variables to set |
| `drop_env` | extra variables to remove from the inherited environment |
| `path_exclude` | regex of `PATH` entries to drop (default `$MB_PATH_EXCLUDE`) |
| `path_prepend` | entries to put first on `PATH` |
| `gpu_model` | passed as `--gpu-model` to every run; see Fairness |
| `knowledge` | `seed_dir` (default `<kb>/data/kernelblaster/knowledge`); `mined` (default `true`, KernelBlaster's default); `snapshot` (a knowledge dir every run starts from, copied); `prebuild: true` (a v2 `knowledge.db` built once with `scripts/seed_knowledge.py` from the seed, honouring `mined`, then copied into every run) |
| `mb_cache` | `graph_opt` only. `{"snapshot": DIR}` copies a kernel cache into each run's work dir. `{"shared": DIR}` links one shared cache and is recorded as not isolated. Without the key, every run starts with no cache |
| `seeds` / `repeats` | seeds per repeat (default `repeats: 2` gives seeds 0, 1); repeat *r* uses the same seed in every arm |
| `interleave` | `counterbalance` (default: ABBA blocks), `abab`, `random` (each block shuffled with `order_seed`) or `sequential` (all of A first; avoid) |
| `max_parallel` | runs at a time (CLI `--max-parallel`); runs share the GPU pool through its leases |
| `budget` | `usd` (the cap); `est_usd_per_run` (estimate until an arm has completed runs); `ledger` (JSONL to append to); `who`; `prices` (`{"haiku": [in, out]}` in $/M tokens, used only when a run reports no cost) |
| `freeze_code` | default `true`; see Reproducibility |
| `trace` | value for `--trace` when the target supports it (default `meta`; `null` = the target's default) |
| `timeout_s` | per-run timeout (SIGTERM) |
| `stop_grace_s` | how long runs get to stop after a signal (default 150) |
| `torch_baseline` | JSON `{problem: us}`; adds a "vs torch" column. A run's own `torch_us` / `torch_e2e_us` key is used if present. A `graph_opt` run whose model run timed torch itself (`kb_model_run --torch-baseline`, [gpu_harness_perf.md](gpu_harness_perf.md), "Against PyTorch") reports its paired `speedup_vs_torch` (same GPU, same session) in "vs torch" and `speedup_vs_torch_wall` in "vs torch wall"; this file is then only the fallback |
| `report` | `baseline` (default: the first arm); `pairs` (`baseline` or `all`); `bootstrap`; `permutations` |

### Arm keys

An arm can set `flags`, `args`, `env`, `knowledge`, `mb_cache` and `description`. Inputs, seeds,
the GPU model and the entry point are base-only, so arms cannot differ in them.

### Rules enforced on the spec

- Every arm must set `--model` explicitly, because both entry points default to a paid model.
- The runner sets these flags itself: `--out` / `--work`, `--knowledge-dir`, `--seed`,
  `--gpu-model`, `--resume`, `--kb`, `--python`.
- Secret-looking variables (`*_TOKEN`, `*_KEY`, `*SECRET*`, ...) are refused in `env`. They must
  come from the calling environment, e.g. `source set_key.sh`, because specs and manifests are
  stored.

Placeholders `{run_dir}`, `{root}`, `{arm}`, `{seed}` and `{repeat}` work in flag values and
`env` values.

The entry points are subprocesses, so a new flag on `run_multislot.py` or `kb_graph_opt` works in
`flags` without any change here. Examples: `--knowledge`, `--ncu`, `--reward`, `--model-hard`,
`--hetero-policy`, `--graph-hints`, `--explore-frac`.

Example specs are in `gpu_eval/ablations/`. All are small by default (2 problems, 2-3 repeats,
2 x 3 steps, Haiku):

| spec | compares |
|---|---|
| `stub_mechanics.json` | $0: the KernelBlaster stub LLM on its LeNet fixture, v1 vs v2 |
| `knowledge_v1_vs_v2.json` | v1 vs v2 knowledge, both without mined evidence |
| `v2_mined_on_off.json` | v2 with vs without mined evidence (prebuilt databases) |
| `reward_e2e_vs_slot.json` | step reward on end-to-end time vs the changed slot's time |
| `hetero_policy.json` | Haiku only vs escalation to Sonnet by category or by stall |
| `graph_opt_ncu.json` | the whole flow with NCU off vs best. Needs `kb_graph_opt` to pass `--ncu` through |

Run `--dry-run` first. It prints the plan order, each arm's command, what the target supports
and the estimated cost, and it creates nothing.

## Isolation

Each arm x repeat (x input) gets `ROOT/runs/<arm>__r<NN>[__<input>]/`:

| what | how |
|---|---|
| work dir | fresh: `out/` (`multislot`) or `work/` (`graph_opt`) |
| knowledge | `knowledge/` (`multislot`) or `work/knowledge/` (`graph_opt`). Empty, or a copy of the arm's snapshot / prebuilt database. A run's learning never reaches another run or the snapshot |
| seed | `seed/`: a per-run copy of the knowledge seed, with or without `mined.json`. `KB_KNOWLEDGE_SEED_DIR` points at it, so a concurrent edit of the repo's `mined.json` cannot leak in. The seed variants are built once under `ROOT/seeds/<hash>` |
| MB kernel cache | none unless the arm opts in (`mb_cache`) |
| temporary files | `TMPDIR=tmp/` (removed after the run) |
| environment | the calling environment minus result-affecting or shared-state variables, then the base `env`, then the arm `env`. The dropped variables include `PYTHONPATH`, `KB_KNOWLEDGE*`, `KB_GRAPH_HINTS`, `KB_REFLECT_MODEL`, `KB_UPKEEP_MODEL`, `KB_HETERO_POLICY`, `MODEL_CODEGEN_*`, `KERNELBLASTER_MULTISLOT_*`, `KERNELBLASTER_BUILD_CACHE_DIR`, `KERNELBLASTER_PERF_LOG_*` and `TMPDIR`; the full list is `DROP_ENV` in the code. The GPU lock dir (`/dev/shm`) stays shared on purpose, so concurrent runs take exclusive per-measurement GPU leases |

Each run writes `manifest.json`, which records:
- the run's identity: arm, repeat, seed;
- the resolved command (secret-looking arguments redacted);
- the environment diff: variables set (secrets redacted), dropped and removed;
- the code identity of the KB and MB checkouts (commit, branch, dirty flag, hash of the
  uncommitted diff; for a frozen copy, where it came from);
- tool versions: python, torch / CUDA, nvcc, the host and its local GPUs (name, UUID, driver);
- the isolation record: seed hash, mined on/off, knowledge snapshot path and hash, MB cache;
- input fingerprints (a hash of each `manifest.json` / `graph.json`);
- the spec's base hash and the arm's configuration hash;
- what the target supports (`--resume`, `--trace`) and the mode (fresh, resume-flag,
  resume-implicit);
- start and finish times, the rc, and earlier segments if the run was resumed;
- the GPU models and UUIDs the run recorded (from `result.json` `gpu`, or `summary.json` `gpu`);
- the run's $.

`run.log` holds the run's output, and `metrics.json` holds the parsed results.

## Reproducibility

With `freeze_code: true` (the default), the KB checkout is copied once per ablation into
`ROOT/code/kb`, and for `graph_opt` the MB checkout into `ROOT/code/mb/modelblaster`. The copy is
taken with `git stash create` + `git archive`: HEAD plus uncommitted changes to tracked files;
untracked files are not included. Every run uses this copy. Other agents' commits to the live
checkouts during a long ablation therefore cannot split the arms across code versions. The copy
is about 7 MB for KB and about 70 MB for MB.

With `freeze_code: false`, the live checkouts are used. Before each launch the runner checks that
their commit and diff hash are unchanged since the ablation started, and refuses otherwise unless
`--force` is given.

`run` on an existing root refuses when:
- the base configuration changed (base hash: everything except bookkeeping keys such as name,
  budget, parallelism, report, interleave and the seed list);
- an existing arm's configuration changed.

Adding arms or repeats to an existing root is allowed. `--force` overrides both refusals.

`report` refuses paired comparisons (exit 3, with the reasons listed in the report) when any of
these hold:
- runs used different KB or MB code;
- runs used different base configurations;
- runs of one arm used different arm configurations;
- runs were timed on different GPU models.

`report --force` compares anyway and marks the report as forced.

## Fairness

- **Same GPU model.** `gpu_model` is passed as `--gpu-model`: KernelBlaster then uses only GPUs
  of that model, and `kb_graph_opt` times and verifies on it. The report lists the GPU models each
  arm actually recorded and refuses to compare across models.
- **Interleaving.** The default is counterbalanced blocks: A B, B A, A B, ... per repeat. With
  `max_parallel` >= the number of arms, the arms of a block run at the same time under the same
  load. All of A then all of B would confound the arm with time-varying load on a shared GPU.
- **Pairing.** Repeat *r* uses the same `--seed` in every arm, so the statistics pair runs by
  (problem, repeat).
- **Concurrency.** Runs share the GPU pool: each timed measurement takes an exclusive lease, and
  `max_parallel` bounds the number of concurrent runs.

## Budget

A run's cost comes from every `telemetry.jsonl` under its directory. Each `llm` event is counted
once:
- `cost_usd` if the provider reported one (claude-code), else KernelBlaster's `est_usd` (Bedrock),
  else the `prices` table;
- responses replayed from a resume journal are not counted again;
- without telemetry, the cost comes from `result.json` `llm_usage_by_model` / `llm_usage`.

Before each launch, the runner adds up:
- the $ already spent, including restarted attempts;
- an estimate for each run in flight;
- an estimate for the next run.

It stops launching when that total would pass the cap. The estimate is the arm's mean over its
completed runs, else `est_usd_per_run`, else a rough per-call figure (`--dry-run` labels it as
rough). Runs it did not start stay pending with reason `budget`; `run --budget-usd` with a larger
cap continues them.

With `budget.ledger` (or `--ledger`), each finished run segment appends one row to the shared
ledger: `{"ts", "who", "what", "model", "calls", "prompt_tokens", "completion_tokens", "est_usd"}`.
These are the fields that `gpu_eval/work/llm_budget/README.txt` defines. Stub-model runs with no
spend are not logged.

## Resumability

`ROOT/state.json` holds each run's status: pending, running, done, failed or interrupted. It also
records attempts, pid, $ and wall time. One runner at a time owns a root (`.lock`).

On SIGINT, SIGTERM or SIGHUP, the runner sends SIGTERM to every run's process group, waits up to
`stop_grace_s` and exits 75. KernelBlaster stops cleanly and exits 75; those runs are marked
interrupted.

On the next `run`:
- An interrupted run continues in place with `--resume` when the target's `--help` lists it
  (`run_multislot.py` with its journal).
- A `graph_opt` run that resumes from its work dir alone (`--more-rounds` / `kb_resume` in its
  help) is rerun with the same command.
- Otherwise the run restarts from a fresh directory. The $ of the abandoned attempt still counts.
- A run left running by a runner that was SIGKILLed is waited for if it is still alive (matched by
  pid and process start time), else accounted and resumed.
- Failed runs are data and are not retried unless you pass `--retry-failed`.

## Report

`ROOT/report.md`, `report.json` and `report.csv` (one row per run x problem) are written after
`run`. `report ROOT` re-renders them from the run directories alone.

### Per-problem result

| field | `multislot` | `graph_opt` |
|---|---|---|
| final time | `final_e2e_us` (deployed partition, the e2e reward's basis), else `final_partition_us` | `model_e2e_us`: the whole model, graph replay on the runs' GPU model (`kb_model_run`), else the partition `final_us` |
| speedup | final time vs the run's own seed baseline | final time vs `seed_e2e_us` / `first_us` |
| other speedups | partition speedup | partition speedup, model speedup |
| other | wins, LLM calls, tokens, $, seconds, GPU | the same |

### Per arm

The per-arm table reports, for each arm:
- runs and failed runs;
- results ok out of all;
- geo-mean speedup and geo-mean final time;
- vs torch, when a baseline is given;
- problems improved by more than 1 %;
- wins (accepted steps, from KernelBlaster's `win` counter);
- LLM calls, tokens in / out, $ and $/win;
- mean wall time per run;
- truncation rate (truncated responses / calls);
- the GPU models recorded.

The per-problem table gives each arm's geo-mean final time and speedup over repeats, with the
per-repeat values.

### Paired comparisons

Paired comparisons are made against `report.baseline`, or between all pairs with
`report.pairs: "all"`:

- **Pairs.** A pair is (problem, repeat) where both runs completed. Pairs where one side failed
  are dropped and counted.
- **Ratio.** The ratio is B/A of the final time (< 1 favours B) and of the speedup (> 1 favours
  B), as a geometric mean of the per-pair ratios.
- **Confidence interval.** 95 % percentile bootstrap over the pairs (10 000 resamples).
- **Permutation test.** Two-sided paired sign-flip test on the log ratios: exact up to 16 pairs,
  else 20 000 random flips. This is the paired form of the within-problem permutation test in
  `docs/knowledge_architecture.md` §16.
- **Wilcoxon test.** Two-sided signed-rank test: exact up to 25 pairs, else the normal
  approximation with tie correction.
- **Sample size.** n (pairs and problems) is always shown. With n < 6, the smallest attainable p is
  2/2^n, and the report says so.

Normalising by the run's own seed baseline (the speedup ratio) absorbs GPU-load drift between
runs. The final-time ratio does not, but it is what deployment sees. Both are reported.

## Validation

| test | what it checks |
|---|---|
| `pipeline/tests/test_kb_ablate.py` (14 tests, about 10 s, no GPU) | Uses a fake entry point that writes plausible `result.json` / `summary.json` / `telemetry.jsonl` with seeded random times and costs. Covers spec parsing and refusals, interleaving orders, isolation, manifest contents and secret scrubbing, the `graph_opt` entry and cache opt-in, budget stop + ledger + continuation, resume after a hard kill, clean SIGTERM stop, statistics and comparability refusals |

Isolation is checked with the fake run: per-run TMPDIR / seed / knowledge dirs, mined.json
present or absent per arm, a snapshot copied rather than shared, inherited `KB_KNOWLEDGE` and
`PYTHONPATH` dropped.

Resume after a hard kill is checked with SIGKILL of both the runner and the run. With a target
that supports `--resume`, the run continues; without one, it restarts.

The statistics tests use known synthetic effects:
- a 25 % effect over 6 problems x 4 repeats: ratio 0.79, CI 0.77-0.82, p < 1e-4;
- a null arm: p 0.26 / 0.10, CI spans 1;
- exact Wilcoxon p checked against R's `wilcox.test` example.

The comparability tests confirm that the report refuses to compare when the code or GPU model
differs, and when a checkout moved or the spec changed between invocations.

End-to-end runs on the local TITAN RTX (shared with another process), 2 arms x 2 repeats each:

| run | what | result |
|---|---|---|
| `graph_opt`, $0 | `kb_graph_opt` on the 4-branch conv model, scripted stub LLM, 1 round; arms differ only in env (`KERNELBLASTER_MULTISLOT_REWARD` e2e / slot) | 4/4 runs done, about 2-3 min each, two at a time. GPU recorded per run: NVIDIA TITAN RTX. Final whole-model times 71.4 / 47.6 us vs 70.1 / 47.3 us. Ratio 0.987 (n = 2), so the two repeats differ far more than the arms do: that is shared-GPU load, which is what the pairing absorbs |
| `multislot`, $0 | `gpu_eval/ablations/stub_mechanics.json`: KernelBlaster's stub LLM on its LeNet fixture, v1 vs v2 | 4/4 runs done; `--resume` detected; per-run seed without `mined.json` |
| `multislot`, Haiku, $0.11 total | v1 vs v2 (no mined evidence), 1 x 2 steps | 4 ledger rows (`who: ablate`), $0.024-0.031 per run. The Bedrock key appears in no file under the root |
