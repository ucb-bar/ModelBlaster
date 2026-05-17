# `agents/notes/`

Focused design notes captured before or alongside implementation. Each
file is self-contained so it can be picked back up later without the
chat history that produced it. **Not user-facing docs** — promote
stable bits to `agents/README.md` once they've crystallized.

## Pipeline & flow

| file | what it covers |
|---|---|
| `pipeline_overview.md` | Canonical end-to-end pipeline diagram (PyTorch → IR → kernels → build → run) + the multi-model + scheduling extension on top. The reference picture this README's "Pipeline at a glance" is condensed from. |
| `int8_quantization_flow.md` | Symmetric per-tensor int8 PTQ pass: scale derivation, weight/bias quant, the integer-simulator golden, Q0.31 requantize convention. |
| `mixed_precision_plan.md` | Architecture for per-op precision overrides via `get_precision_spec()` + auto-cast IR pass. Eight implementation phases, ordered. |
| `vint_mixed_precision_experiments.md` | Per-configuration accuracy log: which ViNT op sets promoted to fp16 give which navigation-accuracy wins, with cos_sim and wp4 Δpos numbers. |
| `vint_int8_op_coverage.md` | ViNT-specific op inventory: which ops needed implementing for int8 path. |
| `conv_weight_layout_decisions.md` | How OIHW / HWIO / IHWOC layouts are negotiated per-backend (gemmini packs HWIO, rvv packs IHWOC, scalar stays OIHW), where the packing happens, and how universal kernels detect via `-DAGENTS_*_WEIGHTS=1`. |
| `xpurt_walker_semantics.md` | How the IR walker emits dispatches + buffers + scratch — the model of how skeleton.c becomes runnable C. |

## Profiling & runtime

| file | what it covers |
|---|---|
| `profile_emission.md` | The IREE-schema per-dispatch profile CSV format. What spike/firesim runners write, where, and how XPU-RT ingests it. |
| `firesim_eval_design.md` | FireSim re-rank step in the kernel optimize loop. Top-K spike survivors get re-scored on real RTL for cache-locality wins. Plus the cache-aware optimize-prompt stanza. |
| `firesim_eval_design.md`, `firesim_sweep_v8_results.md`, `firesim_co_execution_baseline_plan.md` | FireSim flow design + results capture. |
| `multi_model_threading.md` | Architectural plan for running multiple models in one binary with optional intra-model threading. Phased delivery (1-7) with concrete file/symbol changes. |
| `posix_affinity_investigation.md` | What's needed to add `pthread_setaffinity_np` to Zephyr's POSIX layer (currently absent). ~90 LOC patch shape, Phase A/B/upstream tradeoffs. |
| `dispatch_and_cores.md` | Core-registry model (`agents/cores/*.json`): how core kinds (CPU_P, CPU_E, GEMMINI) map to backends and to physical hart IDs in the harness. |
| `scheduler_investigation.md` | Inspection of XPURT-emitted schedule JSON. Documents the dispatch / time_dependency / hardware_target / module_name format and how `harness_xpurt` consumes it. |

## Hardware-specific

| file | what it covers |
|---|---|
| `saturn_opu_backend.md` | rvv_opu backend status, OPU programming model, what's curated so far, what's still pending. |
| `saturn_opu_spike_support.md` | Scoping + design of the spike OPU extension (`customext/saturn_opu.cc`). Encoding cheat-sheet, per-instruction semantics, build wiring. |
| `saturn_fp_precision_stripping.md` | The five-knob FPGA area study for stripping FP precision out of Saturn (FP32-only, FP16-only variants). |
| `saturn_strided_memop_bug.md` | Root cause of the V256 strided memop GPR-corruption bug that surfaced on yolov8 V512D256. Workaround: scalar gather/scatter + unit-stride loads. |
| `saturn_vrf_lutram_plan.md` | Saturn VRF → LUTRAM mapping for FPGA area reduction. |
| `gemmini_extension_plan.md`, `gemmini_firesim_status.md`, `gemmini_config_validation_plan.md` | Gemmini backend status across each milestone (initial RoCC bring-up, Q0.31 validation, FireSim cycle accounting). |
| `gemmini_lut_optimization.md` | FPGA-side LUT/DSP knobs that cut KU040 build 210K → 82K LUTs. |
| `zephyr_rvv_context_switch_bug.md`, `zephyr_rvv_fix_summary.md`, `zephyr_v_decouple_design.md` | Zephyr V-state save/restore design + the bugs that motivated each iteration. |
| `microros_baseline_status.md`, `microros_local_overview.md` | micro-ROS broker on FireSim status + setup notes. |

## Project & infra

| file | what it covers |
|---|---|
| `freshscheduler_chipyard_port_plan.md` | Plan for moving the FreshScheduler-side chipyard tree from the FireSim shared copy to a self-contained `hw/chipyard/`. |
| `freshscheduler_area_sweep_v1.md` | FPGA area sweep design + first round of results. |
| `ku040_bitstream_plan.md` | VCU118 / KU040 bitstream build plan; relevant when porting a config off Alveo U250. |
| `vint_zephyr_plan.md` | Original plan for getting ViNT through the agents flow (torch.export path motivation, op coverage gap). |

These are working notes — promote bits to `agents/README.md` once
they're implemented, stable, and worth surfacing to a new reader.
