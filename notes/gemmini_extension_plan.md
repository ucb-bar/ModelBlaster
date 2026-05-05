# Gemmini accelerator support — planning notes

Captured for later. Current status: planning only, no code yet.

## Verdict

Feasible. **8–13 weeks for one engineer**, with the "make it work"
milestone at 5–7 weeks. The pipeline doesn't fight you here — Gemmini
is structurally a new `target` that bolts into the existing
heterogeneous-backend abstraction. SmolVLA was novel design at every
layer; this isn't.

## What we already have that's reusable

- **Per-target kernel codegen** in `agents/pipeline/reference_kernels.py`
  — already structured around `(op, target)` seeds. Adding `gemmini`
  is a new column.
- **Per-target Kconfig overlays** at `agents/harness/backends/<bs>.conf`
  — same shape as `rvv.conf` and `scalar.conf`.
- **Per-target build metadata** in `agents/pipeline/backends.py`
  (compile flags, spike args). Adding gemmini = one new entry.
- **Heterogeneous core registry** — `agents/cores/chipyard_hetero_example.json`
  already declares a `gemmini0` core with `kind="gemmini"` and
  capabilities `["linear_s8", "conv2d_s8"]`. So the **scheduling**
  side already knows how to route a dispatch to Gemmini; what's
  missing is just **kernels that emit Gemmini code** for those ops.
- **The xpurt walker** dispatches by `core_kind` — adding `gemmini`
  as a kind in `generate_xpurt_main.py` is purely additive
  (`else if (strcmp(core_kind, "gemmini") == 0) ...`).

## Managing the HW-config matrix

Gemmini's config space is roughly
`(dim, dataflow, dtype_in, dtype_acc, scratchpad_kb, coupling, …)`.
The headers (`gemmini_params.h`, etc.) are auto-generated per-config
by chipyard's gen step. Proposed shape:

```
agents/cores/gemmini/<config>.json
{
  "name":           "saturn_ws_int8_16x16_rocc",
  "dim":            16,
  "dataflow":       "ws",
  "dtype_in":       "int8",
  "dtype_acc":      "int32",
  "coupling":       "rocc",
  "scratchpad_kb":  256,
  "accumulator_kb": 64,
  "headers":        "vendored/gemmini/saturn_ws_int8_16x16_rocc/",
  "extra_cflags":   ["-mext=gemmini"],
  "extra_libs":     ["gemmini_lib"]
}
```

Codegen reads the config at build time, points the include path at
the right header bundle, links the right static lib. Most kernel code
goes through `tiled_matmul_auto()` and `tiled_conv_auto()` —
config-agnostic at the C level, **the headers are what make it
config-aware**. So one `kernels.c` source compiles cleanly against any
Gemmini config; only the include path and linked lib differ.

For the few cases that want explicit tile control (fusing a
pre-scale, picking a non-default orientation), the codegen reads
`dim` + `scratchpad_kb` from the config and parameterizes — same
pattern as the Plan B memory-aware optimizer.

The harness CMake takes `-DGEMMINI_CONFIG=<name>`, the scheduler's
machine map gets `gemmini0 → kind=gemmini, config=<name>`, and the
build picks the right bundle. Each config gets its own build-dir tag
(`build/${TARGET}_${GEMMINI_CONFIG}_${RUNNER}`) so swapping configs
doesn't churn rebuilds.

## RoCC vs ReRoCC

Good news: **kernel source is identical** (same Gemmini library API).
What differs:

| | RoCC (tightly coupled) | ReRoCC (decoupled) |
|---|---|---|
| Build flag | `-DGEMMINI_ROCC` | `-DGEMMINI_REROCC` |
| Linker | gemmini_rocc lib | gemmini_rerocc lib (network driver added) |
| Per-op latency | ~deterministic, 1-cycle dispatch | network round-trip on top |
| Sharing | 1 Gemmini per rocket | N rockets per Gemmini |
| Scheduler model | every gemmini-kind machine is independent | gemmini-kind machines on the same remote tile **share a resource** — scheduler must serialize their ops |

That last row is the only place ReRoCC really changes the higher
levels. The core registry would gain a `shared_with: [<other_core_names>]`
field; the XPU-RT MILP would turn shared-resource constraints into
"no two ops on these machines can overlap." That's a small constraint
addition (cvxpy mutex constraint — few lines).

For a **first cut, support RoCC only** — more common in chipyard
configs, simpler, doesn't touch the scheduler. Add ReRoCC after the
core path is proven.

## Staged plan

**Stage 1 — RoCC scalar Gemmini path through the pipeline (2–3 weeks)**
- Pick one chipyard config: e.g. `saturn_ws_int8_16x16_rocc`. Vendor
  its `gemmini_params.h` + tiled lib into
  `agents/runtime/gemmini/<config>/`.
- Add `target=gemmini_<config>` to `backends.py`; new Kconfig overlay
  at `agents/harness/backends/gemmini.conf`.
- Reference kernels for `linear_s8` and `conv2d_s8` using
  `tiled_matmul_auto` / `tiled_conv_auto`. PyTorch goldens already
  exist (we have int8 dronet + mlp_generic).
- Spike validation — chipyard's Gemmini extends spike's ISA, so
  correctness can be checked there. Per-op cycle counts on spike
  won't match silicon, but goldens will.

**Stage 2 — HW-config abstraction proven across two configs (1–2 weeks)**
- Same kernels.c source, two configs (e.g. 8×8 vs 16×16, both WS
  int8 RoCC). Confirms the include-path / lib-swap is the only delta.
- Add config descriptor schema + loader
  (`agents/pipeline/gemmini_configs.py`).

**Stage 3 — FireSim integration with the existing xpurt schedule (1–2 weeks)**
- Build a chipyard FireSim hwconfig with one Gemmini per rocket
  (RoCC). Mostly bitstream build time (offline, hours).
- Route dronet's `conv2d_s8` dispatches in the xpurt het schedule
  onto `gemmini0`. The xpurt walker already supports per-kind
  dispatch tables; this just adds a `gemmini` core_kind branch in
  `generate_xpurt_main.py`.
- Run the multi_demo pool-sweep (extended with
  `TARGET=gemmini_<config>`) to populate
  `gen/profile/gemmini_<config>/firesim_*/...`. Validates per-op
  cycles on real silicon.
- Run the xpurt het schedule end-to-end with Gemmini-targeted convs
  — measured speedup vs RVV-only baseline.

**Stage 4 — ReRoCC support (2–3 weeks)**
- Build flag + linker delta for ReRoCC. Verify on chipyard's ReRoCC
  reference config.
- `shared_resource` annotation in the core registry; cvxpy mutex
  constraint in `xpu-rt/scheduler.py`. ~50-line patch.
- Validate by running the same dronet+mlp_control xpurt schedule on
  a 4-rockets / 1-shared-Gemmini config — schedule should serialize
  Gemmini ops automatically.

**Stage 5 — LLM optimize loop for Gemmini kernels (2–3 weeks)**
- Extend `generate_kernels` LLM prompt with Gemmini idioms (pre-scale
  fusion, orientation pick, tile-size selection within scratchpad
  budget).
- Seed algorithms: `gemmini_tiled_matmul_basic` (vanilla),
  `gemmini_tiled_matmul_fused_scale` (fused dequant/scale),
  `gemmini_im2col_conv2d`. The cache-aware spike beam-search +
  firesim re-rank from Plan B both transfer unchanged — same gates,
  new target.

| stage | weeks | bite risk |
|---|---|---|
| 1 — RoCC path through pipeline | 2–3 | low — mechanical, mirror RVV |
| 2 — HW-config abstraction | 1–2 | low — config-descriptor schema |
| 3 — FireSim integration | 1–2 | medium — chipyard bitstream build offline; one-time pain |
| 4 — ReRoCC support | 2–3 | medium — scheduler ext + driver build |
| 5 — LLM optimization | 2–3 | low — same Plan B infrastructure |
| **total (1 engineer)** | **8–13 weeks** | |

## What's likely to bite

1. **Quant layout mismatches.** Gemmini wants int8 weights in a
   specific tiled stride. The current int8 path emits flat int8
   tensors. ~1 week of plumbing in `extract_graph` +
   `weights.c` codegen to produce Gemmini-ready layouts. Plan for it;
   not a research problem.

2. **Degenerate shapes.** `mlp_control`'s `linear M=1;K=64;N=4` is
   silly for a 16×16 matrix unit — most of the unit sits idle. The
   scheduler should prefer scalar/RVV for those ops. See "Mesh
   utilization at batch=1" below.

3. **Bitstream churn.** Each Gemmini config = a different FireSim
   bitstream = an offline build that takes hours. Plan to land one
   canonical config first, validate, then add variants.

4. **Spike's Gemmini ISA support is functional, not cycle-accurate.**
   Per-op timing decisions need FireSim. Spike is fine for
   correctness gating in the optimize loop; firesim re-rank already
   built handles perf.

5. **Library version drift.** The Gemmini library headers + tiled-API
   are versioned with chipyard; bumping the chipyard submodule may
   invalidate vendored bundles. Pin a chipyard SHA per config and
   document it.

## Mesh utilization at batch=1 — what these workloads actually exercise

Worth recording so we don't forget which routings are obvious wastes.
Mesh dim assumed = 16; analysis for our current dronet + mlp_control
at batch=1.

### Linear (mlp_control + dronet's two FC tails) — GEMV pattern, bad fit

| dispatch | M | K | N | mesh util |
|---|---|---|---|---|
| mlp.0 | 1 | 16 | 256 | **6.25%** |
| mlp.2 | 1 | 256 | 128 | **6.25%** |
| mlp.4 | 1 | 128 | 64 | **6.25%** |
| mlp.6 | 1 | 64 | 4 | **1.56%** |
| dronet linear1, linear2 | 1 | 2048 | 1 | **0.39%** |

M=1 means 15/16 of mesh M-rows sit idle every cycle. **Never route
these to Gemmini** even with zero dispatch overhead — scalar/RVV will
beat Gemmini just from less wasted work.

### Conv2D at batch=1 — usually fine because spatial dim is M

Post-im2col GEMM dims: `M=OH*OW`, `K=IC*KH*KW`, `N=OC`. M is
typically large because spatial activation maps are big.

| dispatch | M=OH·OW | K=IC·KH·KW | N=OC | mesh util |
|---|---|---|---|---|
| conv_modules.0 | 3136 | 27 | 32 | ~84% (K=27 partial) |
| conv_modules.1 | 196 | 288 | 32 | full |
| conv_modules.2 | 196 | 288 | 32 | full |
| conv_modules.3 (1×1 shortcut) | 196 | 32 | 32 | full |
| conv_modules.4 | 49 | 288 | 64 | full |
| conv_modules.5 | 49 | 576 | 64 | full |
| conv_modules.6 (1×1) | 49 | 32 | 64 | full |
| conv_modules.7 | **16** | 576 | 128 | just barely (M = exactly mesh dim) |
| conv_modules.8 | **16** | 1152 | 128 | just barely |
| conv_modules.9 (1×1) | **16** | 64 | 128 | just barely |

Most of dronet's convs fill the mesh. The last block (.7–.9) is at
the threshold — M=16 fills exactly one M-tile. Below that you'd lose
utilization fast, but dronet doesn't go below.

### Practical implications for routing

1. **Restrict gemmini0's advertised capabilities to `conv2d_s8` only**
   — drop `linear_s8` from the registry. Simplest filter.
2. **Add a shape-threshold filter on conv routing**: if
   `min(M, K, N) < mesh_dim`, prefer a non-Gemmini machine. dronet
   doesn't trip this in practice but mobilenet_v2's depthwise convs
   would (K = KH·KW = 9 only). Easiest implementation: a
   `min_matmul_dim` field in `agents/cores/<core>.json`.
3. **The Plan-B firesim re-rank already handles the perf side.** If
   we keep `linear_s8` capability on gemmini0 and let the planner
   pick by profiled time, the firesim sweep will show `linear@M=1`
   as slower on Gemmini than RVV, and the MILP just won't route
   there. Belt-and-suspenders only.
4. **Mobilenet_v2 depthwise convs** are the workload that exposes
   poor utilization (K=9). Either keep depthwise on RVV, or implement
   a "depthwise-as-GEMM-broadcast" kernel.
5. **Highest-value Gemmini ops in dronet**: conv_modules.0 and
   conv_modules.4–8 (millions of mesh cycles each on RVV today). The
   1×1 shortcuts (.3, .6, .9) full-mesh but represent <10% of dronet's
   compute. The fat conv (conv_modules.0) at 9 ms RVV → likely sub-1ms
   on Gemmini if util holds — that's the demo number.

**Short version**: mlp_control wouldn't see any benefit from Gemmini
(every op is GEMV, route to RVV); dronet's convs would benefit
significantly (~7 of 10 conv ops fully utilize the mesh, including
the heaviest one). Current workloads are a reasonable test bed for
the Gemmini path even at batch=1, just don't expect mlp to win
anywhere.
