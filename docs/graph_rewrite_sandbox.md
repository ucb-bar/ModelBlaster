ITEM 4: REWRITE SANDBOX AS AN OFFLINE DISCOVERY MODE  (restricted rewrite interpreter + discovery CLI -> candidate hint patterns)
================================================================================================================================
Branch merge-sibling of ModelBlaster. Same text is committed as docs/graph_rewrite_sandbox.md. NOTHING here is part of the normal sweep or of kb_graph_opt: two new modules,
no change to any existing file for this item. The discovery CLI refuses to run without --enable-discovery (or MB_REWRITE_DISCOVERY=on) and never applies anything.

0. WHAT EXISTS / WHAT IS NEW
----------------------------
EXISTS (graph-rewrite branch): graph_edit.Editor / diff / apply_patch, weight_derivations (Recorder, declarative derivations, replay), graph_verify (gate), graph_rewrite
  (RewriteCandidate, RewriteProposer, AgentProposer stub that raises NotImplementedError, RewriteHook), the pilot's research code (graph_agent_pilot/gt_edit.py: AST check +
  Prov + Surgery, NOT imported).
NEW pipeline/graph_rewrite_sandbox.py   the restricted interpreter: check_script (AST whitelist), run_rewrite -> SandboxResult, handles g / W / P / T, a small CLI.
NEW pipeline/graph_rewrite_discover.py  discovery pipeline + CLI: ToolLoop / PairedEvaluator interfaces, ScriptedToolLoop stub, FakeEvaluator, ToolLoopProposer (adapter to
                                        graph_rewrite.RewriteProposer), describe_pattern, rewrite_proposals.json writer.
NEW tests: test_graph_rewrite_sandbox.py (17 tests), test_graph_rewrite_discover.py (9 tests). CPU only.

1. THE SCRIPT AND ITS HANDLES (semantics ported from the pilot, code rewritten and tested)
-----------------------------------------------------------------------------------------
    def rewrite(g, W, P, T):          # top-level, exactly 4 parameters
        ...
        return {"tensor_map": {old_tensor: {"from": new, "axis": 1, "start": 0, "stop": 64}}, "description": "one-line hypothesis"}      # optional
 g  GraphHandle over graph_edit.Editor (deep copy of the graph; the original is never touched). Reads return COPIES: ops() op(n) producer(t) consumers(t) tensors() has_tensor(t)
    inputs() outputs() unique(base). Mutations (each validated, clear message): add_tensor(name, shape, dtype) add_op(op_dict) update_op(name, **fields) remove_ops(names)
    remove_tensor(t) rewire(old, new[, only_ops]). Fused / region ops cannot be added (the flow's fusion hints make them); graph inputs/outputs cannot be removed.
 W  WeightCatalogue: has / shape / dtype / keys of the model's parameters AND the weights derived so far. W.value / W.get raise weight_not_readable: weight VALUES are not available.
 P  ProvenanceWeights over weight_derivations.Recorder: concat slice reshape transpose pad stack scale cast mul add zeros ones (pilot: concat slice reshape transpose mul add zeros ones;
    pad/stack/scale/cast are the extra declarative ops of weight_derivations). New key first (must start "d."), returns the key. Every call records a DECLARATIVE derivation in
    graph["weight_derivations"] (expression over parameters / earlier derived keys), evaluates it (numpy; structural ops exact, arithmetic float32 rounded once to the operand dtype),
    declares weight_dtypes in a native IR. The result is therefore a REPLAYABLE derivation table: the verifier and the loader recompute every weight from the original weights.npz.
 T  TensorMeta (read-only, live): shape(t) dtype(t) numel(t) has(t) names().      math: ceil floor sqrt prod gcd pi.
SandboxResult: ok, script_sha, graph (edited), patch (graph_edit.diff: remove_ops / add_ops / add_tensors / weight_derivations / requires digests / tensor_map), derivations {key: {dtype,
 shape, expr}}, provenance {key: {leaves = model parameters it traces back to THROUGH intermediate derivations, exact (structural only), shape, dtype}}, tensor_map, description, error
 {code, msg, line}, log (print output), steps. run_rewrite never raises. The patch replays: apply_patch(base, patch) == edited (tested).
After the script: _finish checks every op input is produced / a graph input / a constant (else dangling_tensor with the fix), graph outputs are still produced, the graph is a DAG,
every weight/bias/gamma/beta key is a model parameter or P-derived, every op tensor is registered; orphan tensors are pruned. Then graph_verify decides (below).

2. WHAT IS REJECTED  (SandboxError.code; the message is what the proposer sees, tested for each)
-------------------------------------------------------------------------------------------------
 syntax / too_large (200 KB) / no_rewrite;
 forbidden_construct: import, while, class, global/nonlocal, with, yield/await, del, walrus, decorators, ** and << (giant ints), numeric constants > 1e7, strings > 10k;
 forbidden_name: underscore names, eval exec compile open getattr setattr hasattr type object id iter next format chr ord vars globals locals __import__ os sys np torch ...;
 forbidden_attr: attributes starting with _ and EVERY attribute not on a whitelist (container / str methods without `format`, the handle API): this keeps `__class__`, `__globals__`,
   `'{0.__class__}'.format(g)`, generator/frame attributes out;
 step_budget / timeout: bounded work: a traced-line budget (300,000 default) catches runaway `for` loops over growing lists and deep recursion, `range` is capped at 1e6 elements,
   SIGALRM wall clock (60 s default, main thread only);
 weight_untraceable: an op naming a weight/bias key that is neither a model parameter nor P-created ("w*w computed outside P" cannot even be written: W.value raises
   weight_not_readable; naming an invented key is weight_untraceable with the instruction to use P.*); bad_weight: bad P call (not a d. key, exists, unknown source, shape mismatch);
 dangling_tensor / bad_edit / bad_op / cycle: surgery errors with a message that says what to do, never a raw KeyError;
 script_error: any other exception of the script with its type and LINE.
Honest limit: this is a robustness boundary against LLM mistakes and prompt-injected scripts (no I/O, no imports, no attribute escape, bounded work), NOT a security sandbox against
a determined adversary in the same process: memory is not capped (a variable-sized `[0] * n` inside the 300k-statement budget is possible), and CPython sandboxes are never airtight.
Run the proposer's tool loop in a container / separate process as well.

3. THE GATE
-----------
The sandbox accepts any structurally valid edit; EQUIVALENCE is graph_verify's job: write_result_ir(res, ...) writes graph.json + weights.npz (source + derived) + io.npz and
graph_verify.verify(orig, edited, tensor_map=res.tensor_map) runs structural + numerical checks with sampled intermediates. A structurally perfect but WRONG rewrite (filters stacked in
the wrong order) is rejected by the sampled-intermediate check (tested through the sandbox, the discovery pipeline and the tool-loop adapter).

4. DISCOVERY CLI  python -m modelblaster.pipeline.graph_rewrite_discover IR_DIR... --out DIR --proposer scripted:DIR|module:Class [--evaluator none|fake:o,v|module:Class]
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
Gate: exit 2 without --enable-discovery / MB_REWRITE_DISCOVERY=on. Per IR: ToolLoop.run(session) -> [ScriptProposal]; each: sandbox -> write edited IR -> graph_verify ->
PairedEvaluator.measure(orig, variant) -> accept when gain >= --accept-gain (default 0.03, as the flow's accept_gain); no evaluator: "verified_untimed" is NOT accepted unless
--allow-untimed. Outputs: rewrite_attempts.json (EVERY attempt with stage sandbox | graph_verify | timing | verified_untimed | proposer and the reasons: the rejected hypotheses),
rewrite_proposals.json (accepted only), work/<ir>/<script sha>/ (edited IRs).
Interfaces (the LLM side lives in another agent's tool-loop module, l3_farm/region_loop, branch region-loop, in progress; nothing here starts a relay or reads a key):
  ToolLoop.run(session) -> [ScriptProposal(name, source, description, est_gain_pct, tensor_map, provenance)]; session offers show() (ops by profile time, shapes, weights) and
    try_rewrite(source) -> structured {ok, error{code,msg,line}, ops_removed, ops_added, derivations, structural{ok,reasons}} so a loop can repair; session.budget has max_candidates,
    llm_usd, work. A loop that returns an unverified script costs only a rejected attempt. NotImplementedError from a loop aborts the run (as AgentProposer does); any other
    exception loses only that IR's candidates (recorded).
  PairedEvaluator.measure(orig_ir, variant_ir, tag) -> {"original_us", "variant_us"}: the flow's paired variant-vs-original timing (kb_proposals.remeasure on a GPU) is the real
    implementation (NOT written here: GPU); FakeEvaluator for tests; none = untimed.
  ToolLoopProposer(loop) is a graph_rewrite.RewriteProposer: each verified sandbox result becomes a RewriteCandidate (edited IR, tensor_map, est cost, provenance), so the SAME loop
    plugs into `--graph-rewrite agent --graph-rewrite-proposer module:Class` (no-arg constructor) once someone writes that class; without a loop it raises NotImplementedError.
  ScriptedToolLoop(scripts) / .from_dir(DIR): the stub (every *.py is a script; first-lines "# description: ..." is its hypothesis); CLI scripted:DIR; "--proposer llm" says where the LLM loop belongs.

5. rewrite_proposals.json  (format modelblaster.rewrite_proposals/v1; merged across runs by pattern id, evidence accumulates)
---------------------------------------------------------------------------------------------------------------------------
 {"proposals": [{"id": "rwp:<signature>", "source": "agent-rewrite", "status": "proposed", "suggested_kind": "merge_sibling_convs_cat",
    "already_implemented": "merge_sibling_convs" | null,         # merge_convs.find on the original graph covers exactly the removed ops
    "pattern": {"description", "before": [op signatures], "after": [op signatures], "ops_before", "ops_after", "removed_ops", "added_ops"},
                 # an op signature = {op kind, inputs, outputs (tensors renamed x0.. external inputs, y0.. region outputs, t0.. internals), weight/bias flags, shape_params}
    "recipe": {"script": the rewrite script (replays on other shapes: it reads every shape from g/T/W), "script_sha", "derivations": concrete table, "provenance",
               "leaf_roles": parameter -> role in the removed ops, "pad_formulas": [{derived, pads, formula: "(KH_max - KH[c0]) // 2"}], "exact"},
    "detector_sketch": {"anchor_op", "match": ["conv2d_f16(x0) -> t0", ...], "exclusive_tensors", "shape_constraints": [{op, count, equal_fields, differing_fields}], "notes"},
    "evidence": [{"problem", "original_us", "variant_us", "gain", "decision", "timed", "verify": {ok, checks, max_errors, warnings}, "script_sha", "provenance", "ts"}],
    "proposers": [...]}]}
The signature hashes before/after op kinds and local dataflow (names and shapes do not matter; branch order does). `suggested_kind` is a heuristic name (merge_sibling_convs_cat when a
concat and >= 2 convs become one conv, else rewrite_<kinds>_<hash>). A human / agent REVIEWS a proposal and promotes it: write the detector from detector_sketch + the recipe, an applier
that records the derivations declaratively, the cost gate, tests, and register the kind in graph_hints.CUDA_KINDS - exactly what ITEM 3 is for merge_sibling_convs.

6. EQUIVALENCE WITH ITEM 3 (tested)
-----------------------------------
The fire-module merge written as a script yields the same ops, shapes, edges and BIT-IDENTICAL materialised weights as merge_convs' applier (fp32, fp16): test_equals_the_merge_
sibling_convs_applier; and on a real extracted 18_SqueezeNet (reduced input) the discovery CLI with a saved script + FakeEvaluator produced a proposal marked already_implemented:
merge_sibling_convs (demo/out_small, timing is FAKE).

7. INVARIANTS
-------------
 originals untouched (deep copy); weights only through P with provenance to parameters; every result carries patch + derivations and replays; graph io unchanged; no file outside OUT_DIR
 written; no network, no key, no GPU; flag-gated; accepted = verified AND (timed gain >= accept_gain, or --allow-untimed); a proposal is never applied automatically.

8. NOT DONE (item 4)
--------------------
The LLM ToolLoop (region-loop agent's module) and the real PairedEvaluator (GPU); Python-level memory cap / process isolation of the interpreter; shape-generic re-verification of a
proposal on other shapes of the same problem family (graph_verify accepts shape_variants pairs: the discovery pipeline does not yet regenerate them); automatic promotion; deduplication of
patterns that differ only in branch order; proposals for edit type 3 (new op kinds): rejected (gate: op_kind).
