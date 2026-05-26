# `modelblaster/benchmarks/` — arm-agnostic compile-flow scoreboard

Measurement infrastructure for the agentic-flow experiment tracked in
`notes/agentic_baselines_plan.md`. The harness's job is to score any
compile-flow change (current deterministic pipeline, beam-search
optimize, future agentic flow, anything else) on the **same metrics**
so "is X better?" is a measurement, not an argument.

## Concepts

The harness keeps three things separate:

- **Arm** (`config/arms.yaml`) — a treatment. *What we're comparing.*
  One row per compile flow (A: deterministic, B: optimize beam search,
  C: agentic via Claude Code).
- **Workload** (`config/workloads.yaml`) — what's being measured.
  One row per `(model, quant, slice?, target, runner)`. Independent of
  arm; the same workload runs across all arms the matrix allows.
- **Matrix** (`config/matrix.yaml`) — which (arm × workload) pairs to
  populate. Defaults to cross-product; include/exclude rules carve out
  the actual matrix.

Each **run** of `(arm, workload)` produces a directory of artifacts:

```
results/<arm-id>/<workload-id>/<run-id>/
  run.json              env + git_sha + wall_clock + peak_rss
  profile_<runner>.csv  per-region cycles (spike or firesim)
  accuracy.json         L_inf, RMSE, cos vs PyTorch reference
  llm_tokens.json       input_cached/uncached/output (B + C)
  decision_log.jsonl    per-decision agent activity (C only)
  wall_cycles.txt       XPURT WALL_CYCLES markers (heterogeneous targets)
  xpurt_trace.csv       XPURT_TRACE=1 per-tile records
  stdout.log, stderr.log
  env.txt, git_sha.txt
results/<arm-id>/<workload-id>/latest -> <run-id>
```

The `latest` symlink is what the aggregator follows by default. Pin a
specific snapshot with `--run-id <id>` or compute mean+variance across
the N latest with `--runs N`.

## Layout

```
benchmarks/
  config/
    arms.yaml          treatments (the things being compared)
    workloads.yaml     workload + execution environment (the things being measured)
    matrix.yaml        which (arm × workload) pairs to populate
    metrics.yaml       declarative metric extraction (one entry per metric column)
    pricing.yaml       per-LLM-model $/MTok for dollars_equivalent
  arms/
    arm_a_curated.py   Arm A driver
    arm_b_optimize.py  Arm B driver
    arm_c_agentic.py   Arm C driver (stub until Phase 4 gate opens)
  runners/
    spike.py           wraps validation.spike_runner — correctness / accuracy
    firesim.py         wraps validation.firesim_runner — authoritative cycles
  ingest/
    profile_csv.py     gen/profile/.../results.csv → per-region cycles
    accuracy.py        vs PyTorch reference; respects backend atol/rtol overrides
    tokens.py          llm_tokens.json sums (provider-agnostic)
    xpurt_trace.py     XPURT_TRACE markers → utilization + makespan
    run_json.py        wall_clock, peak_rss, cache_hit_rate
  aggregate.py         walks results/ → dashboard.csv + summary.md
  winner_analysis.py   per-cell winner-by-metric; --ablate <piece> for port-back gates
  results/             ignored (see results/.gitignore)
```

## Invariants

- **Same `run.json` schema** across all arms. No per-arm special cases
  in `aggregate.py`. Adding a new arm = one row in `arms.yaml` + one
  driver file. Cross-product with existing workloads is automatic.
- **Cycle source is honest.** `cycles_firesim` is the only primary
  metric on `rvv_opu`, `gemmini`, `gemmini_q31`, `hetero_gemmini_opu`.
  Spike runs the Gemmini and Saturn-OPU extensions atomically — there
  is no microarchitectural pipeline model — so spike cycles for those
  targets are correctness instrumentation only. The aggregator refuses
  to compute Arm-vs-Arm cycle deltas on accelerator targets using
  `cycles_spike`.
- **Token costs apples-to-apples.** Bedrock (Arm B) and Claude Code
  (Arm C) both produce `llm_tokens.json` with the same schema; pricing
  in `pricing.yaml`, one file to re-price.
- **Reproducible per run.** Each run dir captures env vars, git SHA,
  runner type, kernel cache hash, wall clock, peak RSS.
- **Compile via shell, ingest via Python.** Arm drivers shell out to
  the existing `examples/<m>/run.sh` and `examples/xpurt_demo/run.sh`
  so we don't drift from `_run_lib.sh`'s env-var handling (PATH
  ordering for the dev box's stale Vitis cmake, the `_f16`
  auto-promotion when the IR contains f16 ops, the spike-fork env
  vars for the Gemmini and Saturn OPU variants). Ingest is pure
  Python — reads the artifacts those scripts produce.
- **Replicates supported.** N replicates of the same (arm, workload)
  produce N `<run-id>` subdirs; `latest` is the most recent. The
  aggregator can compute mean+stddev across the N latest with
  `--runs N`.

## Quick start

The Arm A driver shells out to `examples/<m>/run.sh`, which in turn
invokes `west build` and `spike`. Activate the Zephyr build env once
per shell first (same prerequisite as a manual `bash examples/<m>/run.sh`):

```bash
source tools/miniforge3/etc/profile.d/conda.sh && conda activate zephyr
source scripts/set_envvars_sdk.sh
```

Then run the harness:

```bash
# Smoke: one workload end-to-end, Arm A.
uv run python -m modelblaster.benchmarks.arms.arm_a_curated \
    --workload dronet_rvv_smoke

# Aggregate everything under results/.
uv run python -m modelblaster.benchmarks.aggregate

# Inspect.
cat benchmarks/results/summary.md
```

The driver checks for `west` (and `spike`, when the workload's
runner is `spike`) on PATH before invoking the shell pipeline; a
missing tool fails fast with the activation instructions above
rather than mid-`west build` with `command not found`.

## Adding a workload, metric, arm, or matrix rule

- **New workload:** append one row in `config/workloads.yaml`. If it's
  blocked on a prereq (e.g. backend not yet registered, model not yet
  admitted), set `blocked_by: <ref>` — the row stays in the matrix and
  renders empty until artifacts appear.
- **New metric:** one entry in `config/metrics.yaml` (source path,
  extractor module:function, unit, arms, nullable_if). One extractor
  function in `ingest/`. No aggregator code change.
- **New arm:** one row in `config/arms.yaml` + one driver file in
  `arms/<id>.py` that emits `run.json` matching the shared schema. The
  matrix automatically cross-products it against existing workloads;
  carve out exceptions in `matrix.yaml`.
- **Matrix carve-out:** an `exclude` rule in `matrix.yaml` like
  `{ workload_tag: smoke, arm: [B, C] }` or
  `{ workload_id_pattern: "smolvla_*", arm: B }`.

## Live cost monitor (`mb-cost`)

A terminal UI that tails every `llm_calls.jsonl` under
`benchmarks/results/` and shows running spend live. Math matches
AWS Bedrock prompt-caching semantics exactly
(`inputTokens` = uncached portion only; `cache_read` and
`cache_write` are reported separately and billed at their own rates).

### Quickstart

```bash
cd <repo root>
uv sync --extra benchmarks    # one-time; pulls rich + pyyaml

# Live TUI (default). Always-on-top alternate-screen mode.
uv run mb-cost

# With a visual budget alarm (no kill -- just colors + bell).
uv run mb-cost --budget-usd 100

# Watch one specific cell only.
uv run mb-cost --paths benchmarks/results/B-bedrock/<workload>/latest/llm_calls.jsonl

# One-shot text report (CI / Slack / cron-friendly; no TUI).
uv run mb-cost report
```

Keyboard inside the TUI:

| Key | Action |
|-----|--------|
| `q` / Ctrl-C | quit (terminal restored) |
| `p` | pause / resume polling |
| `s` | cycle per-model sort: cost → calls → name |
| `j` / `k` | scroll recent calls down / up |
| `r` | reset state + re-scan all files |
| `?` | toggle key-hints overlay |

### What gets tracked

Each call lands in four aggregation windows simultaneously:

- **CUMULATIVE** — lifetime spend across every record on disk
- **THIS MONTH** — calls with `ts` in the current UTC month
- **SESSION** — calls during the currently-active named session
  (see below; only when one is active)
- **PER-MODEL** — cross-cutting table grouped by `model_id`
- **PER-KERNEL** — cross-cutting table grouped by op (conv2d_s8,
  linear_s8, …) parsed from each call's `phase` field
  (`synth:<op>` or `optimize:<op>`, set by `generate_kernels.py`)

### Sessions (named time windows)

A session is a labeled window of wall-clock time. Cost incurred while
a session is active is attributed to it. Single-active model:
starting a new session auto-ends the previous one. State persists at
`benchmarks/results/.sessions.json` (gitignored).

Two ways to scope a session:

**Manual** — for interactive work where you bound the run yourself:

```bash
uv run mb-cost session start baseline-v1 --label "first dronet baseline"
# ... run any number of arm_b_bedrock / arm_b_gemini cells ...
uv run mb-cost session end
uv run mb-cost session list
```

**Command-wrapped** — the right hook for automation (CI, Claude Code,
or any wrapper). The session opens just before exec and closes on the
command's exit (any exit code):

```bash
uv run mb-cost run baseline-v1 --label "first run" -- \
    uv run python -m modelblaster.benchmarks.arms.arm_b_bedrock \
        --workload dronet_scalar_smoke --beam 1 --expansions 2 --iterations 1
```

The label is optional; `mb-cost session list` shows historical
sessions with their windowed totals.

### Hard budget cap (driver-side)

The visual `--budget-usd` only alarms; it doesn't stop spending. For
unattended runs, the Arm B-bedrock driver accepts a hard kill:

```bash
uv run python -m modelblaster.benchmarks.arms.arm_b_bedrock \
    --workload <id> --max-usd 5.00
```

`pipeline/bedrock_client.BedrockClient` tracks cumulative cost using
the same math as the monitor and raises `BudgetExceeded` once the cap
is crossed. The arm driver writes `exit_status=budget_exceeded` to
`run.json` and returns exit code 2.

### Reconciling against AWS Cost Explorer

| Source | Lag | What you'll see |
|---|---|---|
| `llm_calls.jsonl` `request_id` field | instant | CloudTrail event id for each call |
| CloudTrail → Event history (filter `bedrock-runtime` / `Converse`) | ~5-15 min | Per-call audit log |
| CloudWatch → Metrics → Bedrock (`InputTokenCount`, `OutputTokenCount`) | ~1-5 min | Time-series totals |
| Billing → Cost Explorer (service = Amazon Bedrock) | ~24-48 h | Dollar amounts; verifies `dollars_equivalent` math |

Set an AWS Budget alert at the Billing console for the absolute backstop
("don't spend more than $X/month").

## Port-back gate for CompGen integrations

When evaluating whether a CompGen piece (dossier schema, oracle, plan,
pass-card scheduler, etc.) earns its place in ModelBlaster:

```bash
uv run python -m modelblaster.benchmarks.winner_analysis --ablate <piece>
```

A port lands only when disabling `<piece>` in Arm C flips a winner on
at least one (arm, workload) cell. The dashboard is the gate.
