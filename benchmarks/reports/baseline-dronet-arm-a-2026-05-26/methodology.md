# Methodology: baseline-dronet-arm-a-2026-05-26

The first real Arm A baseline on a non-toy workload (dronet, 32 ops, 8
distinct kinds). Three replicates of `dronet_scalar_smoke` give us an
anchor that every later (model, target, arm) capture and every compiler
modification can diff against on the same metric schema.

## Approach
Arm A driver (`arm_a_curated`) with `BACKEND=reference`, `OPTIMIZE=0`,
`RUNNER=spike` on `dronet_scalar_smoke`. No LLM calls. Three replicates
back-to-back so `aggregate.py --runs 3` produces mean ± stddev on every
cycle / latency / binary-size metric.

The scalar reference path is bit-exact by construction (each reference
kernel's `reference_impl` is the verifier's golden), so this baseline
sets the cycle floor for the curated-reference compile flow on dronet.

## Hypothesis
- Per-replicate cycle stddev should be near zero (spike is deterministic
  given the same inputs); any non-zero value would suggest non-determinism
  in our calibration sampling.
- Wall-clock variance comes entirely from west/cmake/spike startup, not
  the model.

## Knobs

| Knob | This run |
|---|---|
| arm | A (curated) |
| LLM model | — (no LLM calls) |
| beam / expansions / iterations | — |
| FIRESIM_EVAL | 0 (spike-only) |
| max_usd | — (free) |
| replicates | 3 |

## Result interpretation

- **bit_exact: true** in every replicate (linf=0.0 vs PyTorch).
- **wall_cycles ≈ 4.55M** per replicate (deterministic across all 3).
- **wall_clock_s ≈ 8.6s** per replicate, ~99.8 MB peak RSS in the python
  pipeline driver.
- The Arm A scalar floor is what any Arm B-bedrock optimization will
  need to beat to claim a cycle win on dronet.

See `report.md` for the full table; `kernels/dronet/int8/scalar/kernels.c`
for the generated reference C; `per-cell/<cell>/cycles_per_op.json` for
per-op breakdowns.

## Known gaps in this baseline

- **No `dronet_rvv_smoke` rep.** Build fails on
  `CONFIG_RISCV_V_KERNEL_ONLY` (referenced by `harness/backends/rvv.conf`
  but undefined in both Zephyr trees on this host). Needs to be either
  backported into the workspace's `arch/riscv/Kconfig.isa` or gated with
  `if RISCV_ISA_EXT_V` in the conf. Phase B prereq — tracked separately.
- **No FireSim cycle anchor.** Per `tmp/07_gotchas.md` #6, spike cycles
  aren't authoritative for accelerator targets; this baseline holds only
  for the scalar path. FireSim baseline is deferred (Phase C).

## Reproducing this report

```bash
git checkout 6744d45
source scripts/setup_benchmark_env.sh
uv run mb-cost session start baseline-dronet-arm-a-2026-05-26 \
    --label "Arm A curated baseline: dronet_scalar_smoke, 3 reps"
for rep in 1 2 3; do
    uv run python -m modelblaster.benchmarks.arms.arm_a_curated \
        --workload dronet_scalar_smoke
done
uv run mb-cost session end
uv run mb-cost export --full baseline-dronet-arm-a-2026-05-26 \
    --session baseline-dronet-arm-a-2026-05-26
```

## Next steps

- Arm B-bedrock pass on the same cell with `--beam 2 --expansions 3
  --iterations 2 --max-usd 5` (`baseline-dronet-arm-b-2026-05-26`).
- Diff the two via `mb-cost diff baseline-dronet-arm-a-2026-05-26
  baseline-dronet-arm-b-2026-05-26` to see whether LLM-driven kernels
  beat the scalar reference on cycles.
- Fix the RVV Kconfig blocker and add `dronet_rvv_smoke` to this baseline
  (would let us measure RVV vs scalar speedup).
