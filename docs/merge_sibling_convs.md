ITEM 3: HINT KIND merge_sibling_convs  (sibling convolutions joined by a channel concat -> ONE convolution with derived weights)
=================================================================================================================================
Branch merge-sibling of ModelBlaster (worktree mb_merge_hint, off graph-rewrite 24c787a). Flag --merge-sibling-convs off|auto (env MB_MERGE_SIBLING_CONVS), default OFF.
Same text is committed as docs/merge_sibling_convs.md. EXISTS = in the tree before; NEW = added here; CHANGED = a guarded edit of an existing function.

0. WHY
------
Graph-agent pilot (l3_farm/graph_agent_pilot): the one genuinely new rewrite that paid off was 18_SqueezeNet `merge_expand_fused`, in 8 Fire modules:
    cat(relu(conv1x1(s)), relu(conv3x3,pad1(s)))  ->  relu(conv3x3(s; W = concat_OC(zero_pad_centre(W1x1), W3x3), b = concat(b1, b3)))
+34.4% alone; with the agent's kernel and the follow-up fuse hints 6.2 ms vs 9.8 ms for the hints-only arm. graph_rewrite's `test-merge-expand` proposer already
makes this rewrite deterministically but only for ONE shape of the pattern (two relu-ed stride-1 convs into a cat2_c1) and as an opaque patch hint (kind `rewrite`).
`merge_sibling_convs` is the reusable pattern: a detector with explicit conditions, a cost gate, an applier that works from the hint (no stored patch), and the
existing paired variant-vs-original decision.

1. THE PATTERN AND ITS CONDITIONS  (NEW: pipeline/merge_convs.py; every condition is re-checked by the applier on the graph it is applied to)
-------------------------------------------------------------------------------------------------------------------------------------------
IR DAG: a channel concat (cat2_c1 / cat3_c1 / cat4_c1 / catn, any dtype suffix) whose CONSECUTIVE inputs come from 2-D convolutions (conv2d / conv2d_f16 /
conv2d_bf16), each optionally followed by an elementwise activation, reading the SAME tensor.
  * exclusivity: a conv's output (and activation's) is read only by its successor in the branch and the concat, and is not a graph output;
  * odd kernels (KH, KW each odd; non-square allowed: 1x3 + 3x1 -> 3x3), equal N/IC/IH/IW, equal OH/OW, equal stride, equal dilation;
  * CENTRED padding: c = P - D*(K-1)/2 is the same for every branch (0 = "same" padding; stride-2 1x1 p0 + 3x3 p1 qualifies, so does dilation 2 with p = 2*(K-1)/2);
    merged padding = D*(K_max-1)/2 + c (must be >= 0). Derivation: branch window starts at i*S - P_i; padding the filter by d=(K_max-K_i)/2 taps per side shifts
    the first tap by d*D, so P = P_i + d*D and the centres coincide iff c is common;
  * groups == 1. conv2d_grouped / conv2d_dw / convnd / conv3d / conv_transpose* are SKIPPED with the reason in the proposal report (a grouped merge needs per-group
    interleaving of the output channels, which does not preserve the concat's channel layout; depthwise has IC == 1 per group);
  * one op kind (one dtype suffix), one weight dtype (native IR: weight_dtypes), one output tensor dtype, bias dtype equal across the branches that have one;
  * activation: the SAME op with the same parameters on every branch (relu relu6 silu sigmoid tanh elu leaky_relu hardtanh gelu gelu_exact ewise1; all-or-none).
    Branches with different activations have different keys and are not merged with each other; the activation is applied ONCE to the merged output;
  * biases: all / none / a mix (zeros stand in through a `const` derivation). NOTE the conv kernel signature requires a bias argument, so "none" does not occur in real IRs;
  * the run may be a part of the concat (inception-style cat of 1x1, pool-proj, 3x3 ...): the merged branches must be CONSECUTIVE inputs (channels stay contiguous);
    the concat is kept for the rest (arity reduced: cat4_c1 -> cat2_c1 ...), or removed when the run covers it all. Non-adjacent siblings are not merged (would need a permutation).
  * fused conv+act ops (fused__conv2d__relu) are not matched: the merge applies BEFORE the fusions (apply order below), and proposals on a profile that already contains
    the fusion are a no-match (reported). So the usual order of events is: round 0 proposes both; the merge, if accepted, is applied first from then on.
NOT matched (reported in the per-IR "skipped" list with the reason, evidence for why a candidate does not exist): mismatched stride / dilation / not-centred padding /
output size, grouped or depthwise, even kernels, another reader of an intermediate, a non-conv branch (stays in the concat), different activations.
Inception-style "several 1x1 convs reading the same input but feeding different consumers" is a different rewrite (merge + slice) and is NOT this kind.

2. HINT RECORD (graph_hints v1, like every kind)
------------------------------------------------
  {"id": "merge:conv1x1+conv3x3+relu", "kind": "merge_sibling_convs", "target": "cuda", "pattern": "conv1x1+conv3x3+relu", "policy": {max_waste, compute_share, ...},
   "instances": [{"ir", "cat", "convs": [...], "acts": [...]}], "chains": [[convs + acts + cat]],     # the explicit scope
   "status": "proposed|accepted|rejected", "source": [{"by": "mb-cost", "ir", "est_gain_us", "est_gain_pct", "max_waste", "instances", "cost": [per instance: waste, saved/extra/net us],
                                                      "evidence": "8 x conv1x1+conv3x3+relu: padding waste <= 0.80 FLOP ratio, est. net 1198.4 us", "ts"}],
   "evidence": [ {decision, gain, original_us, variant_us, remeasured, applied: [{kind, op, from, added, derived_weights, waste, tensor_map}], ...} ]}
`source.by` is "mb-cost" (this detector). "agent-rewrite" is what a discovery-mode proposal (ITEM 4) carries until a human promotes it into a detector like this one.
One hint per pattern per run (all instances of that pattern), identity = pattern, so evidence accumulates across problems like fuse:<kind>; a second IR's instances are added to the
known hint (graph_hints.merge extends chains; the hook extends instances).
SCOPE: a PROPOSED hint (decided by prepare_hint, statuses=None) applies to its recorded instances only; an ACCEPTED hint applies wherever its pattern matches in the graph it is applied
to (the rule of `fuse`): the detector is re-run under the policy stored in the hint (waste cap + cost gate) and the other instances of the same pattern join the measured ones (tested
on two chained modules and on the real SqueezeNet: a hint measured on one Fire module, accepted, merges all 8).

3. APPLIER (NEW merge_convs.apply_hint, called from graph_hints.apply_hints; CHANGED graph_hints: CUDA_KINDS += kind, order table, branch, docstring)
-----------------------------------------------------------------------------------------------------------------------------------------
For each recorded instance that still matches (else skipped with the reason in prov["skipped"]; an applied-twice hint is a skip "not in the graph", the graph unchanged):
  ops:      conv2d(K_max, OC = sum OC_i, merged padding) reading the shared input -> [act] ; names merge_<tag>, merge_<tag>_o, merge_<tag>_act (tag = sha1 of the member names);
            the concat's OUTPUT NAME / shape / dtype are preserved for its consumers (the activation, or the conv when there is none, writes it; a partial merge writes a new
            tensor and a rebuilt smaller concat writes it); branch intermediates are removed from tensors; ops are placed where the concat was; dispatch ids / depends_on are
            renumbered by the flow's own _renumber; the first new op carries hint=<id>; prov["renamed"] maps every removed op to the merged conv, prov["added"] lists the new ops.
  weights:  DERIVED WEIGHTS (weight_derivations.py): d.merge.<tag>.w = concat(axis 0) of pad(w_i, centre) nodes (no pad node for the largest branch),
            d.merge.<tag>.b = concat of biases (const zeros for a missing one). Only STRUCTURAL ops, so exact: bit-identical to a hand-written stack. The records are
            declarative and built from the graph's shapes alone (no weight values are needed to APPLY a hint); write_variant_ir / mbir.load_ir / graph_verify replay them from the
            ORIGINAL weights.npz. Original keys are never edited. dtype contract: the derived dtype is the member's dtype (native IR: weight_dtypes[key] declared, bias dtype
            kept when it differs from the weight's); a derivation never changes dtype.
  tensor_map in prov["applied"]: every vanished branch output is a channel slice of the merged tensor ({from, axis 1, start, stop}); the gate compares them.
Order: merge_sibling_convs sorts right after `unfuse` and before `fuse` / fuse_siblings / fuse_attention / fold_layout / split_slot / fuse_region (-0.5 in the order table):
a later pattern-level fusion applies to the merged conv (tested: conv+relu fuses), a chain-scoped fusion naming an op the merge removed is skipped with a reason (tested).

4. DETECTION / PROPOSAL POLICY AND COST MODEL (merge_convs.find / propose; NEW)
--------------------------------------------------------------------------------
waste = (merged FLOPs - original FLOPs) / original FLOPs  (per pixel: sum OC_i*IC*K_i^2 vs sum OC * IC * K_max^2). 1x1(64)+3x3(64): 0.80; 3x3+5x5 equal widths: 0.47;
1x1+3x3+5x5 equal widths: 1.2; 1x1(128)+3x3(16): 3.8. A candidate is proposed only when
  (1) waste <= max_waste (flag --merge-sibling-max-waste / env MB_MERGE_SIBLING_MAX_WASTE, default 1.0: the merged conv at most doubles the FLOPs), and
  (2) the estimated net gain is positive: saved = concat time + launch_us * (ops removed - ops added), minus extra = compute_share * sum(conv time) * waste.
      Concat time / conv times: the round's profile when present, else bytes / bw_gbps (2000) and FLOPs / tflops (f32 20, f16 100, bf16 100). compute_share = 0.25: small-IC convs
      are mostly memory / launch bound, the padded taps ride on loads that are needed anyway. (Policy dataclass: all of these are fields; only max_waste has a flag.)
  (3) the pattern's total est. gain >= min_gain_pct of the model (as every hint).
This is a RANKING AND SANITY gate, not an oracle: real GPU time decides. The decision is the EXISTING paired variant-vs-original timing (prepare_hint exports the original
problem = the removed ops and the variant problem = the merged ops with the same current-best kernels; KernelBlaster optimises both with the same budget; both are re-timed
back to back; accepted at accept_gain, otherwise REJECTED with the evidence record and never re-proposed). Observed gate outputs on real data: fp16 18_SqueezeNet 64x3x512x512:
all 8 Fire modules pass (est. net 1198 us over the byte model); the same net in fp32 (reduced input): only 4 of 8 pass (the model is conservative where fp32 convs are compute bound).

5. NUMERICS AND THE TOLERANCE POLICY (deliberate)
-------------------------------------------------
In real arithmetic the merged conv equals the originals: the padded taps are exact zeros. In floating point it is NOT bit-identical in general: a different kernel and loop
order, a longer reduction (extra 0*x terms: with an infinite or NaN input 0*inf = NaN would poison a finite output), and in fp16 / bf16 a different accumulation split. Policy:
  * the acceptance criterion is the existing graph_verify gate, NOT equality: model output within the dtype's output tolerance + conditioning term, the exits of the edit
    (the concat output) and every vanished branch output (tensor_map slices) and an even sample of downstream tensors within bound (the SqueezeNet lesson: an output-only check
    passes a wrong merge because the output barely depends on the input), plus the differential sensitivity check and the golden;
  * with exact = on (the gate's default "auto" when every derivation is structural) a float64 walk of both graphs must agree to 1e-9: only reordering error is left there;
  * no tolerance is loosened for this kind. Measured: CPU torch walker: error 0.0 on the reduced IR (the zero taps add exact zeros and the accumulation order agrees); real fp16
    SqueezeNet 64x3x512x512: output abs error 6.1e-5 (bound 8.6e-4), worst sampled tensor 4.9e-4 (bound 9.3e-3). On the GPU the per-slot check of the new conv shape is the
    kernel's (slot_rel/slot_abs of its dtype, against the C reference on the merged weights) and the whole-model check against the PyTorch baseline is unchanged.
  * fp16 overflow is not a new risk: each output channel is the same sum plus zeros.

6. TOUCH POINTS
---------------
NEW      pipeline/merge_convs.py           detector (find), cost model (estimate), proposal (propose), applier (apply_hint), scope_ops, MergeHook (round-loop handle)
NEW      pipeline/tests/merge_fixtures.py  synthetic branchy IR builder (branches x dtypes)        NEW pipeline/tests/test_merge_convs.py (34 tests)
CHANGED  pipeline/graph_hints.py           CUDA_KINDS += "merge_sibling_convs"; apply_hints: order -0.5 and the branch; module docstring. Nothing runs unless a hint of the kind exists.
CHANGED  pipeline/kb_graph_opt.py          Config.merge_sibling_convs / merge_sibling_max_waste (CLI flags are generated from Config); MergeHook next to RewriteHook;
                                           round key + summary only when on; propose after rw.propose; decide kinds in both candidate filters; _scope_ops; footprint writes;
                                           the ingest list of "only new ops" kinds. All guarded by the hook being active (flag off: decide_kinds() == (), round_key() == {}).
UNCHANGED (checked) kb_export (derived weights are ordinary buffers), kb_ingest.cache_plan, kb_proposals.write_variant_ir / fuse (derived-weight support came with graph-rewrite),
                    graph_verify / weight_derivations / graph_edit (used as is), extract_graph, farm.py, KernelBlaster (hint sync is kind-generic).
NOT CHECKED (needs the flow on a GPU / out of scope): graph_search interleaved mode and preapply_audit with this kind (unknown kinds are left alone by the audit; the interleaved
engine decides by the generic hint path), beam members, KernelBlaster slot loop on the merged conv shapes (new OC / K: cache misses seeded by the hand kernel), build + driver check.

7. INVARIANTS (and where each is enforced)
------------------------------------------
 weight mapping   derived from original keys only, originals never edited                       gate: weights / derivation_mismatch / weight_not_mapped (tests: tampered array, invented key, rogue array)
 dtype contract   derived dtype = member dtype; native IR: weight_dtypes declared                 wd.validate (tests: fp32 fp16 bf16 native)
 tensor names     the concat output name / shape / dtype preserved; removed intermediates gone    test_op_and_tensor_bookkeeping
 dead ops         none left behind (all five/more removed ops vanish)                             structural gate "dead"
 unique names     merge_<tag>[_o|_act] made unique against ops and tensors                        apply
 slot naming      the merged conv is conv2d(+suffix) with a new shape: cache miss, hand-kernel seed   export test: slots, manifest buffers
 shape guards     OC = sum OC_i, K = K_max, padding recomputed; gate shape_guard                  gate
 stale graphs     a hint whose instances no longer match applies nothing and says why             tests: applied twice, changed stride, removed op
 flag off         inert hook, round key / summary / outputs identical                             test_off_equals_unset_and_leaves_no_trace + identity runs (SUMMARY.txt)

8. NOT DONE (item 3)
-------------------
GPU validation (export + nvcc + driver check of the merged IR in fp16 / native; the paired timing on a real B200; KernelBlaster slot loop on the new shapes): RUNBOOK.txt. Fused
conv+act branches; non-adjacent siblings; merge+slice (inception-style); grouped / depthwise; mixed activations; calibration of the cost model constants against measured B200
times (RUNBOOK asks for the numbers); a numerical gate inside the round loop (the hook runs no gate: the structural + numerical gate is run by the tests and by
squeezenet_merge_gate.py; in the flow the final-model verification against the PyTorch golden and the per-slot checks stand guard, as for every other hint).
