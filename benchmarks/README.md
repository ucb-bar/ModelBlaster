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

```bash
# Smoke: one workload end-to-end, Arm A.
uv run python -m modelblaster.benchmarks.arms.arm_a_curated \
    --workload dronet_rvv_smoke

# Aggregate everything under results/.
uv run python -m modelblaster.benchmarks.aggregate

# Inspect.
cat modelblaster/benchmarks/results/summary.md
```

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

## Port-back gate for CompGen integrations

When evaluating whether a CompGen piece (dossier schema, oracle, plan,
pass-card scheduler, etc.) earns its place in ModelBlaster:

```bash
uv run python -m modelblaster.benchmarks.winner_analysis --ablate <piece>
```

A port lands only when disabling `<piece>` in Arm C flips a winner on
at least one (arm, workload) cell. The dashboard is the gate.
