# LLM model selection for Arm B (Bedrock) and beyond

Captured 2026-05-25 in us-east-1. Rates from the Anthropic pricing page
and the AWS Bedrock pricing page (cross-referenced where possible). See
`benchmarks/config/pricing.yaml` for the machine-readable rates the
aggregator consumes.

## Context

Kernel synthesis on the beam-search optimize loop has a specific
cost/quality profile:

- **Output size is small.** A generated kernel is typically 500-2000
  tokens of C. Output-token cost ranks below input-token cost for
  total $.
- **Input size is moderate to large.** Each call carries the spec,
  reference_impl, target hints, previous candidates from the beam, and
  shape calibration. Per-op prompts run 1.5k-4k tokens; the beam search
  amplifies this 8-16x per cell.
- **Prompt caching is load-bearing.** The static parts of the prompt
  (spec, reference_impl, target hints) repeat across the K candidate
  expansions per iteration. A model that supports cache reads at ~10%
  of base input cost is worth a meaningful fraction of the per-call
  budget.
- **Failed candidates are expensive twice.** A build_fail or
  verify_fail wastes the LLM tokens AND, on FIRESIM_EVAL runs, a
  FireSim cycle eval. Cheaper-but-worse models can be a false economy.

So the model decision is not "what's cheapest per MTok," it's "what
minimizes total $ for a converged baseline at a tolerable wall-clock."

## Cost / quality table (Bedrock us-east-1, US cross-region profiles)

Rates are $/MTok. "Cell estimate" assumes a dronet-class cell:
~200k input tokens + ~50k output tokens across the optimize loop
(beam=2 × expansions=2 × iterations=2 × ~10 ops × ~2k tokens/call).
Cache hits would lower this further; estimates conservatively assume
zero cache hits.

| Model | Input | Cache read | Output | Code quality (subjective) | Cell estimate | Cell estimate × 5 runs |
|---|---|---|---|---|---|---|
| **Claude Opus 4.5/4.6/4.7** | $5.00 | $0.50 | $25.00 | Excellent — frontier | ~$2.25 | ~$11 |
| **Claude Sonnet 4.5/4.6** | $3.00 | $0.30 | $15.00 | Excellent for C | ~$1.35 | **~$7** |
| **Claude Haiku 4.5** | $1.00 | $0.10 | $5.00 | Good but error-prone on long C | ~$0.45 | ~$2.25 |
| Claude 3.5 Sonnet v2 (Bedrock 2x markup) | $6.00 | $0.60 | $30.00 | Solid but older | ~$2.70 | ~$13 |
| Llama 4 Maverick (current default) | placeholder | placeholder | placeholder | Lower than Claude on C kernels | unknown | unknown |
| Devstral 2 123B (Mistral coding-specialist) | placeholder | n/a | placeholder | Unknown for accelerator C | unknown | unknown |
| Qwen3 Coder 480B | placeholder | n/a | placeholder | Strong on benchmarks; unverified here | unknown | unknown |
| Mistral Large 3 | $0.50 | n/a | $1.50 | Lower on C than Claude | ~$0.18 | ~$0.90 |

The dollar estimates assume the beam loop *converges* — i.e. produces
enough viable candidates to find an improvement. Lower-quality models
with the same beam knobs produce more build_fail / verify_fail
candidates, raising the effective $/improvement.

## Recommendation: Claude Sonnet 4.5 for the baseline

**`us.anthropic.claude-sonnet-4-5-20250929-v1:0`** is the sweet spot
for Arm B-bedrock. Three reasons:

1. **It's where the cost/quality knee is.** ~$1.35/cell is ~3x cheaper
   than Opus and 3x more expensive than Haiku 4.5. Opus delivers
   marginal C-quality gains for accelerator kernels; Haiku trades
   meaningful quality (longer kernels, more rare-intrinsic usage) for
   the cost cut.
2. **First-party Anthropic rates on Bedrock.** Sonnet 4.5 is one of the
   4.5-era models that lost the legacy 2x Bedrock markup. Per
   1M input tokens you pay the same $3 on Bedrock as on the Claude API,
   modulo the +10% premium for single-region endpoints (not applicable
   for the `us.` cross-region profile).
3. **Strong prompt-cache support.** The beam loop re-sends the same
   spec + reference_impl per expansion; Sonnet's $0.30/MTok cache-read
   rate (vs $3.00 uncached) cuts repeated-context costs by 10x once
   the 5-minute cache warms.

For the 6-cell × N=5 dronet baseline, this puts the all-in LLM cost
estimate at roughly **$40-60**, depending on how often the cache
warms. That's the right order of magnitude for a deliberately
reproducible baseline.

## Followups to revisit

These are not blockers for the first baseline but should be the next
experiments after it lands:

### 1. Mixed-model beam (Sonnet drafts, Haiku reranks)

The beam-search loop has two distinct LLM workloads: candidate
generation (creative, needs Sonnet) and candidate filtering / picking
the best (judgment, Haiku-class can suffice). The repo's
`pipeline/generate_kernels.py` already separates these into different
phases tagged in `llm_calls.jsonl` (`phase=kernel_synthesis` vs
`phase=beam_rerank`). Route the rerank phase to Haiku 4.5 and the
synthesis phase to Sonnet 4.5; the per-cell $ should drop by ~30-40%
with similar quality.

### 2. Devstral 2 / Qwen3 Coder spike test

Both are coding-specialist models. If Bedrock surfaces their pricing,
worth a one-cell smoke test on `dronet_rvv_opu_int8` to see whether
domain-specialist tuning beats Sonnet 4.5's generalist quality on
intrinsic-heavy code (`riscv_vector.h`, `saturn_opu.h`). If they win
on quality at lower $, switch.

### 3. Llama 4 Maverick honesty check

The repo's default is still Llama 4 Maverick. Once its Bedrock pricing
is surfaced or invoice-confirmed, run one Arm-B-llama cell on
`dronet_scalar_smoke` so the dashboard can defensibly say "Sonnet 4.5
is N% better than the previous default at M% lower / higher cost."

### 4. Verify the +10% endpoint premium assumption

The `us.` prefix is documented as a cross-region inference profile;
the +10% regional premium documented for Claude 4.5+ applies to
single-region endpoints. If a Bedrock invoice line item for the
baseline run shows a +10% delta vs the `pricing.yaml` rate, flip the
`verified: false` flag and apply the multiplier. The aggregator surfaces
`verified: false` as a `summary.md` note today, so this discrepancy
won't silently corrupt comparisons.

### 5. Batch API for offline cells

For non-interactive cells that can wait, Bedrock supports a Batch API
at a 50% discount on Claude models. Not applicable to the interactive
beam loop, but useful for one-shot offline benchmark captures.

## Why not just use the cheapest model?

For kernel synthesis, "cheaper per token" frequently means "more
build_fail and verify_fail candidates per useful kernel." The cost
metric that matters is **$ per converged baseline**, not $/MTok. A
hypothetical $0.10/MTok model that produces 80% build-fail candidates
at default beam knobs would need its iterations or expansions cranked
up to find a working kernel, multiplying both the LLM cost and the
FireSim eval cost. Sonnet 4.5's higher per-token rate is offset by
fewer wasted candidates.

If the baseline ever needs to be re-captured at substantially lower
cost (e.g. for a long-running CI integration), the followups above
provide the data to make that tradeoff with numbers, not vibes.

## Pricing source freshness

Anthropic publishes per-token rates on
[the Claude pricing page](https://platform.claude.com/docs/en/about-claude/pricing).
The AWS Bedrock pricing page surfaces a subset of these but lags
behind by several model generations. The next refresh of this doc
should re-check both pages and update `pricing.yaml`'s `last_updated`
field.
