OPEN-ENDED GRAPH REWRITES IN THE MODELBLASTER / KERNELBLASTER FLOW: DESIGN (infrastructure, "option C")
=======================================================================================================
Branch graph-rewrite of ModelBlaster (worktree mb_graph_rewrite, off merge-tensor-dtypes 330a4f3). Same text is committed as docs/graph_rewrite.md.
Scope: the parts needed whatever the Claude-Code graph-agent pilot concludes. The LLM rewrite step and new-kernel-spec edits are NOT part of this work.
Everything below marked EXISTS was in the tree before; NEW was added on this branch; CHANGED = an existing function got a guarded edit.

0. WHY (what the pilot showed, and what it needed)
--------------------------------------------------
The closed hint vocabulary (modelblaster.graph_hints/v1: fuse, split_slot, unfuse, fold_layout, fuse_attention, fuse_siblings, fuse_region, fuse_view)
cannot express "replace these 5 ops by 2 using a weight that is the zero-padded stack of two others". The graph-agent pilot (l3_farm/graph_agent_pilot) gave a
Claude Code agent typed graph-edit tools on the flow's FINAL state and got 2.5-3.6x on 5 branchy problems, but (a) its equivalence walker ran out of memory on
18_SqueezeNet (the flow's walk() keeps every activation), (b) an output-only equivalence is too weak when the output barely depends on the input
(SqueezeNet: < 1e-3 relative across inputs), and (c) a rewrite whose weights are not traceable to the model fails late with "IR weights not mapped to the
Model". This work builds the parts of that machinery that belong IN THE PIPELINE, independent of who proposes the rewrite: derived weights as an IR feature,
a verification gate, a memory-safe walker, and the flag / proposer plumbing that feeds candidates into the flow's existing paired decision.
Ported / adapted from the pilot (its tree is not imported): gt_edit.Surgery + toposort -> pipeline/graph_edit.Editor; gt_edit.Prov (weights with provenance)
-> pipeline/weight_derivations.Recorder (+ pad / stack / scale / cast / const, dtype contract, replay); gt_edit.walk_free / equivalence (sampled intermediates,
input-sensitivity) -> walk_torch.walk_free + graph_verify.numeric. NOT ported: the restricted-script interpreter (belongs to the LLM proposer), node/GPU bench code.

1. WHAT AN OPEN-ENDED REWRITE IS IN THIS IR
-------------------------------------------
IR = graph.json (ops, tensors, input/output, optional weight_dtypes / dispatches / graph_hints / weight_derivations) + weights.npz + io.npz (golden).
A tensor NAME is a value: same name => same shape, dtype and (up to rounding) value. A rewrite replaces a set of ops by another set that computes the same
values for every tensor the rest of the graph (and the golden io) still reads; it may add tensors, add weights (derived), remove ops. Everything else is a
consequence: the graph outputs and inputs are fixed; the walkers (walk_torch/walk_cuda) and the exporter run ops in LIST order, so the edited list must be topological.

Three edit types:
 (1) REWIRE with existing op kinds + derived weights.   [NEW end to end; this is the main deliverable]
     Example (built and tested): SqueezeNet's fire module. conv1x1(s)->relu, conv3x3(s)->relu, cat2_c1  (5 ops)  becomes  conv3x3(s; W = concat([pad(W1x1 -> 3x3), W3x3], axis 0),
     b = concat([b1, b3]))->relu  (2 ops), the cat's output name kept; the three weights are DERIVATIONS of original weights (section 3). On the real
     18_SqueezeNet IR (65 ops, 8 fire modules) the deterministic proposer finds 8 such rewrites (the pilot's 37 -> 21 is after the flow's own fusions).
     Op kinds used are existing ones, so every kernel spec, hand kernel, seed and cache entry exists; new SHAPES (a 3x3 conv with OC = E1+E3) are cache misses seeded by the hand kernel.
 (2) REGION nodes: an arbitrary connected sub-graph collapsed to one region__ op.   [EXISTS: fuse_region with explicit `members`; region_slots.py]
     Semantics = its members (the walkers recurse into sub_ops); the slot is seeded with the members' current kernels back to back (parity) and refined by the
     normal per-slot KernelBlaster loop; limit 256 members (the pilot split 192-op regions into four). NEW here: only the routing: a proposer may return an
     edit_type "region" candidate; the hook turns it into a fuse_region hint with explicit members (source agent-rewrite), the flow's own applier does the rest.
 (3) NEW OP KINDS needing a kernel spec.   [INTERFACE ONLY, LATER PHASE]
     A new kind needs: reference semantics (a walk_torch rule + the C reference), tolerance (cuda_dtypes keys slot_rel/slot_abs), a CUDA seed kernel, an op class
     (op_classes). graph_rewrite.NewOpSpec records exactly that; the gate rejects any edited graph using a kind without a spec ("op_kind", message names this
     requirement) and the hook rejects edit_type "new_op" ("new_op_kind: a later phase"). Nothing consumes NewOpSpec yet. What it would take (not done): register the
     spec in reference_kernels.KERNEL_SPECS for every dtype suffix, walker rule in walk_torch.OPS, hand kernel in the harness .cu files, op_classes entry,
     bind() support, a seed for the slot loop, tolerance check in the gate (per-slot) and GPU build validation.

2. END-TO-END FLOW (flag --graph-rewrite agent; default off)
-----------------------------------------------------------
 export -> per-slot KernelBlaster search -> paired variant-vs-original decision -> knowledge, with the rewrite entering as a hint:
 round r:  profile IRs (accepted hints applied)  ->  [MB proposals (propose_from_profile), caps.propose]  ->  rw.propose(...)  <- NEW hook, right after caps.propose
   RewriteProposer.propose(ir, profile, budget) -> [RewriteCandidate]          (edited IR dir + provenance + est cost + tensor_map)
   per candidate:  graph_verify.verify(current IR, edited IR, orig_weights = SOURCE IR weights)  ->  Verdict
        rejected -> hint record (status rejected, evidence {stage: graph_verify, reasons}) = a REJECTED HYPOTHESIS, never built, never timed, never re-proposed (same content id)
        ok       -> hint (status proposed, source {by: agent-rewrite, proposer, est_gain_pct, est_cost, verify summary}); kind "rewrite" (patch + derived weights) or "fuse_region"
   plan / export / KernelBlaster on the plan problems (unchanged) ... then the existing decide phase picks proposed hints by est_gain (kinds now include rewrite):
   prepare_hint: g0 = current graph; gh.apply_hints(g0, [hint]) -> variant graph (patch applied; derived weights recorded in graph["weight_derivations"]);
        write_variant_ir materialises variant weights.npz = base + derived; export_problem builds the ORIGINAL problem (the ops the patch removes, from g0) and the VARIANT
        problem (the ops it adds); both seeded with the same current-best kernels (new shapes: hand-kernel seeds)
   KernelBlaster optimises both with the same budget (per slot: the normal slot loop, rollouts, knowledge v2), both composite bests are rebuilt and re-timed back to back on one GPU
        (kb_proposals.remeasure); finish_hint / graph_search.decide_checkpoint (interleaved mode) accept at accept_gain, record evidence, ingest the variant's new slots into the cache
   outcome = graph hypothesis: hint id "rewrite:<ir>:<sha10 of the patch>" with evidence list (decision, gain, original_us, variant_us, remeasured, applied incl. derived weight keys).
        KernelBlaster knowledge v2 syncs graph_hints.json (sync_graph_hints: accepted -> promoted, rejected -> retired, keyed by hint id): generic over kinds, no KB change needed.
   next rounds: accepted rewrite hints apply again from the source IR (patch + derived weights), so profile / verify_ir / model_run see the rewritten model.
 Flag off (default): rw is an inert object: propose() returns at its first line, decide_kinds() == (), round_key() == {}, no files, no summary key. Proven in section 9.

 Hint ordering / composition (CHANGED graph_hints.apply_hints): a rewrite is proposed on the graph the then-accepted hints produced. The hint stores `rewrite.after_hints` (ids
 accepted at proposal time); apply_hints places the rewrite right after the last of them and before every hint accepted later. Consequences: a later chain-scoped fusion that names an
 op the rewrite removed is SKIPPED (reason recorded); a later pattern-level fusion applies to the NEW ops (test: the merged conv->relu fuses); a rewrite whose base changed
 (patch digest guard) is SKIPPED with the reason "rewrite does not apply: ...", never applied wrongly. Not supported in v1: stacking a second rewrite on ops an earlier rewrite added
 within the same round (it is proposed against the profiled graph); it works from the next round.

3. DERIVED WEIGHTS (IR extension)  [NEW: pipeline/weight_derivations.py, 14 tests]
---------------------------------------------------------------------------------
graph.json["weight_derivations"] = { "d.<name>": {"dtype": "f16", "shape": [...], "expr": <node>} }  (absent => IR byte-identical to before).
Language (declarative, no code): leaf {"w": key} (base weight or another derived key; acyclic) and nodes {"op": ..., "args": [...], params}:
  pad(pads=[[lo,hi]..], value) | stack(axis) | concat(axis) | transpose(perm) | reshape(shape, one -1) | slice(axis,start,stop) | scale(by) | cast(dtype) | add | mul | const(shape,value,dtype)
Structural ops (all but scale/cast/add/mul) move values without arithmetic: EXACT. Arithmetic is float32 math rounded once to the operand dtype (f16: RNE; bf16 = float32 array holding bf16-exact
values, rounded by cuda_dtypes.round_bf16; ints truncate). concat/stack refuse mixed dtypes (say cast). Replay is deterministic numpy => bit-identical to what the recorder stored.
 (a) extractor/loader rebuild:  weight_derivations.replay / fill_missing / materialize / write_weights; harness mbir.load_ir fills missing derived keys from the base keys of the same weights.npz
     (CHANGED: only when the graph has derivations); kb_proposals.write_variant_ir writes weights.npz = base + derived instead of the symlink when the graph has derivations
     (CHANGED; no derivation => symlink + graph.json text exactly as before, tested); graph_hints.verify_rewrite fills them (CHANGED). The extractor (extract_graph) does not produce derivations today
     (unchanged, extraction proven byte-identical, section 9); Recorder is the API a future extractor-level fold would use.
 (b) verifier recompute:  graph_verify replays every derivation from the ORIGINAL weights.npz and compares bytes with the stored array (code derivation_mismatch), validates the table
     (derivation_unmapped / _unknown_op / _cycle / _shadows_base / _bad_key / _dtype_contract ...), and rejects any weight an op uses that is neither original nor derived.
 (c) export/walkers see ordinary weights: kb_export reads ir.weights[key] and names the buffer "w:<key>" with cuda_dtypes.weight_dtype; verified by export_problem on the rewritten IR (fp16 and native).
 (d) rejection class of E13 ("IR weights not mapped to the Model"): reason code weight_not_mapped with that exact phrase, for (i) an op naming a weight that exists nowhere, (ii) an array in weights.npz
     with no recorded lineage (weights computed by arbitrary code), (iii) base_weight_modified (an original key edited in place; a derived weight needs its own d.* key), (iv) derivation_mismatch.
     Torch-baseline mapping (torch_baseline.map_state): base weights stay in weights.npz, derived ones are ordinary "unused" extras, so the Model mapping of the base weights is unaffected.
 Dtype contract: float derived weights are stored like every float weight of the IR: fp16 IR -> f16 arrays, bf16 IR -> bf16-exact float32, fp32 -> f32; a native (--quant native) IR names the dtype in
 weight_dtypes[key] (Recorder.declare does it; validate() rejects a missing entry: derivation_dtype_contract); a derivation may not silently change dtype (cast is explicit).

4. EDIT HANDLE AND PATCHES  [NEW: pipeline/graph_edit.py]
-------------------------------------------------------
Editor(graph, base_weights): deep copy + add_tensor / add_op / remove_ops / rewire / derive (Recorder) / result() (stable topological sort; cycles and dangling inputs raise PatchError).
diff(base, edited) -> patch "modelblaster.graph_rewrite.patch/v1" {remove_ops, add_ops, add_tensors, remove_tensors, changed_tensors, weight_derivations, weight_dtypes, requires.ops{name: digest}, tensor_map};
apply_patch(graph, patch) refuses (PatchError stale/exists/missing/cycle) when a removed op is missing or differs (op digest; slot_variant / dispatch_id / depends_on / hint are NOT part of the digest:
they are annotations the appliers renumber). apply_patch(base, diff(base, edited)) == edited (gate check patch_replay). The hint stores the patch, not a graph, so it re-applies every round.

5. VERIFICATION GATE  [NEW: pipeline/graph_verify.py (module + CLI), walk_torch.walk_free, 21 tests]
--------------------------------------------------------------------------------------------------
verify(orig_dir, edited_dir, orig_weights=None, specs=None, seeds, variants, shape_variants, tolerance, tensor_map, n_track, exact, conditioning, device, sol, structural_only) -> Verdict
Verdict: ok, reasons [{code, msg, where}], warnings, checks {name: {ok, numbers}}, max_errors {name: {abs, bound, rel_to_ref, tensor}}, info (ops before/after, derived weights, new op kinds, new slots,
new cache keys, dead ops, sampled tensors, sensitivity, seconds). verify never raises (gate_error). verify_or_reject(on_reject=...) / commit_if_ok = the rollback API; the original is never touched.
CLI: python -m modelblaster.pipeline.graph_verify ORIG_IR EDITED_IR [--weights W] [--seeds N] [--variants ..] [--shape-variant O:E] [--sol-bench F] [--exact] [--json OUT]; exit 0 ok / 1 rejected.
Structural: io (declarations + io tensors + golden), names (unique; duplicates the original had are tolerated), dag (list order topological), tensors (registered, positive shapes, known dtypes, same-name => same
shape/dtype), dtype contract (float tensors of an op have the kind's storage dtype; cast_a_to_b reads a writes b; single-dtype IR keeps one float dtype; precision-policy fp32 islands allowed), weights
(section 3), kinds (spec exists; fused__/region__ built from members; bind() works exactly as kb_export binds), shape guards (conv2d / linear weight and tensor sizes vs the op's shape dict), slots (ids valid, one
signature per C entry; informational new slots / new cache keys), dead ops (warning: pruned at export), patch_replay.
Numerical (walk_free, CPU default): golden (stored input), outputs (seeds x input distributions uniform/normal/wide/small, bound = output_rel*max|ref| + output_abs + OUTPUT_COND * reference_error, the
kb_export conditioning term from a float64 walk without store rounding), sampled intermediates, differential (output response to the input vs a noise floor), exact64 (exact derivations => float64 walks agree to 1e-9),
sol (SOL tolerance on the SOL reference outputs where the original passes), shape_variants (same checks on other shapes of the problem: pairs (orig, edited) are supplied by the proposer; a patch is name/shape specific).
SAMPLED INTERMEDIATES (the SqueezeNet lesson): compared = (1) the edit's EXITS (tensors produced by an added/changed op that the original also defines: cat_out here), (2) tensor_map entries (a removed tensor
declared as a slice of a new one: e1r = cat_out[:, 0:2]), (3) an even stratified sample of the other common tensors downstream of the edit, (4) the outputs. A mismatch always rejects. If the output is
input-insensitive (< 1e-3 relative change across inputs) the verdict carries a warning, and if additionally nothing the edit touches can be compared the rewrite is rejected as `unverifiable`.
Evidence: wrong rewrites (filters stacked in the wrong order / 1x1 filter padded off-centre) on a model with a ~100 constant output: outputs check ok, sampled check fails (test); on the REAL fp16
18_SqueezeNet IR (64x3x512x512): wrong rewrite -> output error 6e-5 within bound 8.6e-4 (an output-only check would accept), sampled 'features_4_squeeze' error 0.24 > bound 0.002 -> rejected; the correct
rewrite -> ok (results/squeezenet_gate_*.json).
Memory-safe walker: walk_torch.walk_free(ir, dev, dtype=, store_rounding=, grab=, skip=, cpu_fallback=): same op rules / rounding / float64 option as walk (walk is UNCHANGED), every tensor freed after its
last reader, grabbed tensors copied to host when produced, dead ops skipped, CUDA OOM retried on CPU. Tests: bit-identical outputs and intermediates for fp32 / fp16 / bf16 / float64 exact / fused (sub_ops),
live-tensor count 3 vs 30 on a 30-op chain, OOM fallback simulated. (Real run: SqueezeNet gate peak RSS ~20 GB on this Mac for the 64x3x512x512 input, 35 s: CPU conv temporaries; use a smaller batch /
--n-track or a GPU device for bigger.)

6. REQUIRED INVARIANTS (what a rewrite must preserve) and where each is enforced
------------------------------------------------------------------------------
 weight mapping    every weight an op/const names is an original weight (bit-equal to weights.npz) or a recorded derivation of original weights; originals are never edited in place. gate: weights.
 dtype contract    --quant native / per-tensor dtypes: tensor dtypes per op kind, explicit casts, weight_dtypes entries for derived weights; fp16/bf16/fp32 single-dtype IRs keep their float dtype. gate: dtype, derivation_dtype_contract.
 dead-op pruning   export prunes ops that reach no output (graph_prune, MB_PRUNE_DEAD): the edited graph's dead ops are a warning; walk_free skips them; a rewrite must not rely on a pruned op.
 unique names      op names unique (flow's `name#k` rule, graph_hints.unique_names); patches/hints address ops by (unique) name; new tensor names must not collide. gate: names, patch apply.
 slot naming       slot id = C entry of the kind (+ __<slot_variant>): a new op shape of an existing kind shares the slot with all its instances (cache guard by shape); region/fused kinds hash members. gate: slots.
 shape guards      kernel cache entries are guarded by shape (kb_ingest.cache_plan, per (entry, signature, shape)): a merged conv with a new OC is a miss (hand-kernel seed); split_slot variants are re-derived by cache_plan
                   and are excluded from the patch digest. gate: shape_guard (op shape dict vs weights/tensors) + info.new_cache_keys.
 cache keys        hint identity = content hash of the patch ("rewrite:<ir>:<sha10>"); region = hash of the member set; kernel cache unchanged; build cache keyed on sources (unchanged); round key only changes when the flag is on.
 tensor names      a name is a value (see 1). A removed tensor that downstream knowledge/hints reference is a tensor_map entry or stays gone with its readers.
 io / golden       graph io and io.npz are unchanged (a rewrite is the same function of the same input).

7. TOUCH POINTS (file / function)
---------------------------------
 NEW      pipeline/weight_derivations.py   derivation language, validate / replay / fill_missing / materialize / write_weights / Recorder
 NEW      pipeline/graph_edit.py           Editor, diff, apply_patch, toposort, graph_digest
 NEW      pipeline/graph_verify.py         verify, Verdict, structural(), numeric(), select_tensors, _sol_check, verify_or_reject, commit_if_ok, CLI
 NEW      pipeline/graph_rewrite.py        RewriteCandidate, RewriteProposer, AgentProposer (stub), MergeExpandProposer, RegionProposer, NewOpSpec, RewriteHook, apply_rewrite_hint, make_proposer
 NEW      gpu_eval/harness/walk_torch.py   walk_free (additive; walk untouched)
 CHANGED  gpu_eval/harness/mbir.py         load_ir: fill derived weights (only if graph.weight_derivations)
 CHANGED  pipeline/kb_proposals.py         write_variant_ir: materialise weights when graph.weight_derivations
 CHANGED  pipeline/graph_hints.py          CUDA_KINDS += "rewrite"; apply_hints: rewrite branch + after_hints ordering (only when a rewrite hint exists); verify_rewrite fills derived weights
 CHANGED  pipeline/kb_graph_opt.py         Config.graph_rewrite* fields (CLI flags generated from Config); RewriteHook in _run (propose after caps.propose; decide kinds in both candidate filters;
                                           round key; summary); _scope_ops / footprint for kind rewrite; prepare_hint: variant partition includes the added ops; finish_hint: ingest slots of kind rewrite
 UNCHANGED (checked) kb_export (derived weights are ordinary weights), kb_ingest.cache_plan (JSON round trip keeps weight_derivations), kb_plan.profile_ir / kb_model_run / verify_ir (all go through write_variant_ir),
           graph_search.decide_checkpoint (finish_std = finish_hint), KernelBlaster (hint sync is kind-generic), extract_graph, farm.py, state dirs.
 NOT CHECKED (needs the flow on a GPU) graph_search interleaved mode with a rewrite hint, preapply_audit unapply of kind rewrite (unknown kind: would be left alone), beam members.

8. LLM PROPOSER (NOT IMPLEMENTED): interface
--------------------------------------------
AgentProposer.propose raises NotImplementedError (agent mode with it aborts the run at round 0's propose: loud, early). Intended: one agent session per IR (Claude Code with the pilot's isolation flags, cost cap
budget["llm_usd"], scratch workspace = read-only IR copy + profile.json + a typed tool set modelled on the pilot's gttool.py):
   show / match      overview, ops by GPU time, dataflow levels, weights WITH provenance (original | derived), chain/sibling/fan-out/repeated-window matches          (read-only)
   region NAME OPS   edit type 2 with explicit members (candidate edit_type "region")
   rewrite SCRIPT    def rewrite(ed): on a graph_edit.Editor in a restricted interpreter (AST-checked: no imports/underscore attributes/eval/open; timeout); weights ONLY through ed.derive (Recorder)
   verify            graph_verify.verify on the pending candidate; the Verdict (reasons + max errors) is the tool result so the agent can repair
   commit            writes the verified candidate under budget["work"] and returns RewriteCandidate(est_gain_pct, est_cost, provenance={model, session, script sha, tool-call count})
The agent never times anything; timing and acceptance stay in the flow (paired variant-vs-original at accept_gain; the pilot's 15% commit rule is not used).
Prompt skeleton: the IR overview + profile, the invariants of section 6, the derivation language, "a name is a value", the verdict format, and the budget; ask for at most budget.max_candidates candidates,
cheapest-to-verify first, each with a one-line hypothesis (recorded as description).

9. TEST EVIDENCE (full numbers in SUMMARY.txt)
----------------------------------------------
 new tests: test_weight_derivations 14, test_graph_verify 21, test_graph_rewrite 20 (CPU). Existing ModelBlaster suites before/after, flag-off identity of the round loop (base vs new vs --graph-rewrite off:
 summary, state incl. round key, hints, 44 files), seeded extraction byte-identical for 4 problems x {fp32, fp16}, exports identical, structural gate accepts all 48 real headline final graphs (fused / region / fold_layout).

10. OPEN ITEMS (not done)
------------------------
 LLM proposer; type-3 kernel specs; GPU validation (real build + bench of a rewritten problem with derived weights: kb_export host build, data.bin layout, driver check; KernelBlaster slot loop on the new conv shapes;
 timing); interleaved graph-search / beam / preapply-audit interplay with kind rewrite; shape-generic rewrites (a rewrite re-derived at other shapes: the gate takes pairs, the proposer must regenerate);
 two-file (base + derived) weights to avoid copying multi-GB base weights per variant dir; extractor-level derivations; a per-slot (not per-model) tolerance check for new kinds.
