# SOL-ExecBench's tolerance as an acceptance gate (`--sol-gate off|final|accept`)

Default **off**: with the flag unset nothing in this document runs, every export, manifest, driver, `data.bin`, work dir and
summary is what it was (KernelBench problems never have a SOL tolerance, so they are unaffected in every mode).

## Why

The 24-problem SOL run on B200 (instance `sol24`) passed the flow's own gate on 23 problems. Auditing the final kernels with SOL's
rule (`l3_farm/sol_audit`) found three failures:

| problem | final vs SOL's reference | seed | cause |
|---|---|---|---|
| 92008 strided conv | matched 0.98612 (need 0.99), max abs err 0.0487 vs atol 0.021 | passes (1.0) | one-pass TF32 conv winner |
| 92011 qk-norm SDPA | matched 0.98815, max err 0.00367 vs atol 0.002 | passes (1.0) | four one-pass TF32 matmul/bmm slots |
| 92015 feature pyramid | 0.0044 (outputs to 6.4e11) | **fails (0.0067)** | SOL's own TF32 reference is 6 % off fp64 |

The flow's bound is one scale per tensor (`output_rel * max|golden| + output_abs (+ conditioning)`, `output_rel` 1e-3 fp32 / 8e-2
bf16; per slot `slot_rel * max(scale)`); SOL's is per element, `|out - ref| <= max_atol + max_rtol*|ref|` on >= 99 % of the
elements. For fp32 problems the flow is 1.5x-3.5x looser than SOL's atol (1000x on 92007 / 92010), and 92015's bound was 6.4e8.

## SOL's rule (`pipeline/sol_tolerance.py`: `sol_compare`, and its C++ twin)

Per output tensor, in float64 on the native-dtype tensors: shape and dtype equal; any inf / NaN in the output fails (unless
`allow_negative_inf` and both are -inf at that element); an all-zero output against a non-zero reference fails;
`|out - ref| <= max_atol + max_rtol*|ref|` on at least `required_matched_ratio` (default 0.99) of the elements; `max_error_cap`
(if set) bounds the maximum error. L2's workloads spell the ratio `required_match_ratio`: SOL's pydantic model ignores that key,
so 0.99 applies (the converter records it; `normalize` reproduces this and lists it in `ignored_keys`). Quirks reproduced from the
port's evaluator (`baseline_cc_sol/remote_bench_sol.py`): a NaN reference with a finite output is an unmatched element (not a
"non-finite output"); an inf reference with a finite output matches (`inf <= atol + rtol*inf`); an empty tensor matches.

One implementation, four users: the flow's driver (C++ snippet compiled into `driver.cpp`), `graph_verify` (its SOL hook delegates),
`mbir.compare(..., sol=)` / `mbir.compare_sol` (the walkers' golden checks add it when `MB_SOL_GATE` is on and the IR has a
tolerance), and the CLI.

## Where the tolerance and the reference come from

1. **Tolerance in the IR.** `extract_graph --bench-file F` records `graph.json` `sol_tolerance` =
   `{format, source, bench, outputs: [one normalized tolerance per graph output, in order], ignored_keys}` from the bench's
   module-level `SOL_TOLERANCE` (parsed with `ast`, the bench is not imported). Absent for a KernelBench file (the extraction is
   then identical: tested by extracting one file with and without the line). For an IR extracted earlier: `MB_SOL_BENCH=<bench.py>`
   (or `--sol-bench`) at run time, or `python -m modelblaster.pipeline.sol_tolerance record --ir IR --bench F`. Order of
   precedence: `MB_SOL_TOLERANCE` (JSON / `@file` / bench path; an explicit override) > what the IR recorded > `MB_SOL_BENCH`.
2. **SOL's reference outputs** (`sol_ref.npz` beside the IR): the bench's `sol_reference_outputs(None, device)` -- the original
   reference in its native dtypes with torch defaults on the GPU (matmul TF32 off, **cuDNN TF32 on**), one float32 array per output
   plus meta (device, dtypes, torch / cuDNN versions). Written at extraction with `MB_SOL_REF=1`, or lazily by the gate when the
   IR has a bench (`--sol-bench`, `MB_SOL_BENCH`, or the farm layout `<problem>/problem.py` beside `irs/<name>`), or with the CLI
   `ref`. A reference computed on the CPU is marked as such and treated like the golden (no cuDNN TF32 noise in it). The reference
   rides in `data.bin` (appended after everything else, only for a gated export) so the driver needs nothing else.

## Which reference the gate compares with, and the bias of the golden

The driver has no torch reference: `data.bin` holds the **golden**, `walk_torch.walk(ir)` on the CPU in fp32 with the storage dtype's
rounding applied at every bf16 / f16 tensor store. For an fp32 problem the golden is the ideal math to ~1e-6 (the seed's error vs
fp64 is 1e-5); SOL's reference carries cuDNN's TF32 convolution error (92008: 0.016 vs fp64 at atol 0.021; 92015: 6 % off). A kernel
passes SOL when `e_kernel + e_reference` stays inside the allowance, so a check against the golden **alone** is too lenient exactly
where it matters: the 92008 winner is 0.9948 matched against fp64 (passes) and 0.9861 against SOL's reference (fails). For bf16
problems the golden rounds at every bf16 store whereas SOL's reference keeps its explicit fp32 sections (92004: the final matches
SOL's reference exactly while the golden is 0.031 away), so the golden is the wrong yardstick in the other direction too.

Decision (`sol_gate.decide_policy`, per output, from the **seed's** measured verdicts):

1. SOL's own reference (margin 1.0) when the seed passes it -- exact SOL semantics on this machine's torch / cuDNN (the audit's
   verdicts were bit-identical on two B200 GPUs; SOL's own run may use another cuDNN algorithm, so a borderline verdict -- 92008's
   0.9861 vs 0.99 -- can move);
2. else the **golden** at SOL's tolerance x `--sol-golden-margin` (default 0.5: the halved allowance leaves room for the reference's
   own error; with a GPU reference absent this is the default for every output) when the seed passes it;
3. else the golden at SOL's own tolerance;
4. else (the seed misses SOL's tolerance against even the golden): the golden, SOL's tolerance, required ratio lowered to the seed's
   matched ratio x 0.999 -- **no regression against the seed**.

The flow's own whole-tensor bound stays and is ANDed (`--sol-bound and`, default; `replace` drops it for the SOL-covered outputs): the
stricter of the two decides. Every driver run prints one `MB_SOL <buffer> ref=sol|golden|golden1 matched=... required=... max_abs_err=...
max_rel_err=... max_excess=... nonfinite=.. allzero=.. cap_exceeded=.. n=.. atol=.. rtol=.. ok=.. gate=..` line per reference it
has: SOL's reference, the golden at the margin, and (when the margin is not 1) the golden at SOL's own tolerance (`golden1`, never
gating); `gate=1` marks the one that decides.

### SOL-unreachable problems (92015)

When SOL's reference exists and the **seed** fails it, the problem is recorded `sol_unreachable` (matched ratio, atol, rtol of the
seed vs the reference), a loud line is printed (`"sol_gate": "SOL_UNREACHABLE"`), and the policy falls to step 2 / 3 / 4 above. The
result is then reported `unverified_vs_sol: true` (it passed a gate that is not SOL's own check against SOL's own reference; SOL's
harness would still fail it). It is never silently passed. For 92015 the golden check (the seed is 0.9928 matched against fp64) is
a real gate where the flow's bound (6.4e8) was none: the 7-slot TF32 final (0.0043 vs fp64) fails it and is rolled back.

## The gate

`--sol-gate final|accept` (env `MB_SOL_GATE`; `off` default). The gate acts per IR that has a SOL tolerance; others are untouched.

| | off | final | accept |
|---|---|---|---|
| seed probe (one report-only whole-model run per IR, policy decided, persisted in `work/sol_gate.json`) | - | yes | yes |
| SOL tables in the drivers of exports | no | the gate's own whole-model runs only | every export (partitions that produce a graph output gate it in KernelBlaster's own candidate acceptance) |
| bound on fp32 slots derived from SOL's tolerance | no | no | yes (`--sol-slot-frac`, default 0.5) |
| whole-model SOL check + rollback after each round's ingest and after the hint decisions | - | - | yes |
| final verification (`verify_ir`) and the final model run enforce SOL, with rollback | - | yes | yes |

### Slot level: why the combination (a) + (b) + (c)

A winner inside the flow's slot bound can still break SOL at the output (the TF32 winners). Options weighed:

* **(a) a conservative per-slot bound derived from the output tolerance** (accept): fp32 slots get
  `slot_rel = min(slot_rel, max(2e-5, frac * min_out (max_atol + max_rtol*mean|ref|)/max|ref|))`, `frac` = `--sol-slot-frac` (0.5).
  Computed on the real IRs: 92008 -> 2.0e-4, 92011 -> 2.0e-4, 92022 -> 1.4e-4 (frac 0.5). The isolated-op errors of the sol24 runs:
  seed fp32 kernels 1e-7..7e-6 (the floor 2e-5 keeps them clear); the failing one-pass TF32 winners 7.4e-4..9.2e-4 (92008: 8e-4,
  92011: 9.2e-4 / 7.4e-4 / 3.6e-4) -- all rejected; TF32 slots that passed SOL in 92022 / 92023 / 92006 (2.2e-4..4e-4) are rejected too
  at 0.5 (frac 1.0 keeps most of them; `final` mode keeps all that still pass the end-to-end check): the price of a pre-filter that
  cannot know the output's sensitivity. It is applied to the manifest tolerance, each slot, each op record and the driver tables, so
  KernelBlaster's prompts quote the tighter number and its candidates already respect it. Only fp32 slots (a bf16 / fp16 slot's own
  rounding exceeds this scale), only once the seed was measured. Cheap (no extra runs). **Not a guarantee**: a pre-filter that saves
  rollbacks and LLM effort.
* **(b) whole-model check at acceptance** (accept): after each round's ingest and again after the hint decisions, every IR is verified
  as the final model would be (`verify_ir`: hints applied, cached winners in place) with SOL enforced; a failure rolls back. One run per
  check instead of one per KernelBlaster candidate: the plan's KernelBlaster runs partition the graph (`--plan-mode all`: one
  problem per op instance), their candidates see a partition output, not the graph output, so the per-candidate whole-model check
  would mean exporting and building the whole model around every candidate (~12 per problem at 3 rollouts x 4 steps, ~0.3-0.6 min
  each: 4-8 min per problem for no stronger guarantee than (b) per round). Not implemented; the data is in "Cost".
* **(c) a mandatory final-model gate with automatic fallback** (final and accept): the guarantee.

Rollback (`SolGate.verify`): when the whole-model run fails SOL's rule (and not the flow's own checks: those keep their existing
fallback), the cached winners it points at -- the class of slots whose isolated-op error is within 2x of the largest (`blame`: a one-pass
TF32 slot and an fp32 slot differ by orders of magnitude, so the class that lost precision separates; at most 4 per attempt) -- are
**rejected for that IR** (the cache's per-IR `rejected` list: the next lookup takes the next-best winner, else the seed kernel;
the same winner code is `kept-existing`, never re-enters, a different one is checked again) and the model is verified again, at most
`--sol-max-rollbacks` (8) attempts, after which every used winner is reverted at once. Because the seed passes by construction of the
policy, **if any visited state passes (the seed does) the reported final passes SOL's check on workload #0** against the
policy's reference; if even the all-seed state fails (hints applied, e.g. a fused hand kernel) the result says so (`pass: false`).
Profiles also blame SOL failures (`kb_plan.failed_cached_slots`), so a round's profile never aborts on one.

## Reporting (gate on only)

* `work/sol_gate.json`: per IR the tolerance, reference (path, device), policy, seed verdicts, `sol_unreachable`, every rollback
  (where, slots, impl ids, isolated errors, matched ratios), the final verdict.
* `summary.json`: `sol_gate` (settings + the above) and `irs.<ir>.sol`; `model_runtime.json`: `<ir>.sol` =
  `{mode, status, policy, pass, unverified_vs_sol, outputs: {buffer: {matched_ratio, required_ratio, max_abs_err, max_rel_err, ok, ref,
  sol_ref_ok, golden_ok}}, tolerance, reference, rollbacks, seed, sol_unreachable?, warnings?}`; each `rNN/round.json`:
  `sol_gate` / `sol_gate_decide` (per IR: attempts, rolled-back slots, seconds).
* the raw `sol` per driver run is in `profile.json`, `verify_ir` results and the model run (`model.sol`).

## Cost of the accept gate (measured on the sol24 telemetry, 23 problems)

A whole-model check is an export (CPU walk, median 2.6 s total per problem over all its exports, 75 s for the biggest) plus build and
run through KernelBlaster's evaluator: `mb.verify.run` median 11 s (max 20 s). A check with no failure is therefore ~15-30 s;
accept adds two per round (after ingest, after decisions) plus the seed probe (~1 min once; `golden1` / SOL lines are printed in the same run) and
one run per rollback attempt: 2 rounds ~ 1.5-3 min per problem, +0.3 min per rollback, against 10-40 min of problem wall time
(`mb.kb_wait` is 90 % of it). The driver's SOL check itself is a streaming pass over the outputs (milliseconds) inside runs that exist
anyway. LLM spend is unchanged except through (a): tighter slot bounds mean more candidates fail the isolated check and are fixed
or dropped by KernelBlaster.

## CLI (`python -m modelblaster.pipeline.sol_tolerance`)

`tolerance --bench F|--ir D` print the normalized tolerance; `record --ir D --bench F` add it to an IR; `ref --ir D --bench F
[--device cuda|cpu]` write `sol_ref.npz`; `compare OUT.npz REF.npz [--bench F|--tolerance JSON]` SOL's check of arrays;
`parse LOG` the SOL verdict of a driver log; `check PACK --ir ORIG_IR [--bench F] [--seed] [--arch 100]` re-check a final pack
(`work/final/<name>`: its graph and the winners in `problem/slots/*.cu`) on this host's GPU: re-export with the SOL tables, build, run,
print the verdicts -- the audit, cheaply (`--export-only` stops before the build).

## Not done / limits

* No whole-model check per KernelBlaster candidate (see (b)); a TF32 candidate in a partition whose output is not a graph output is
  caught by the per-slot bound or at the next round check, not at the step.
* One workload (#0) and SOL's reference on this machine's torch / cuDNN; SOL's own run may differ at the margin.
* Multiple outputs share one `SOL_TOLERANCE` (as SOL's workloads have one); a graph hint that renames an output keeps the last
  recorded tolerance.
* GPU-only behaviour (the compiled driver's check on the device, the real farm) is covered by the RUNBOOK
  (`l3_farm/sol_gate/RUNBOOK.txt`), not by the CPU tests (which compile the same C++ with the host compiler).
