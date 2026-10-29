# The A6 flow mechanisms on the OpenCL target (Qualcomm Adreno)

`kb_graph_opt --target opencl` ([opencl_adreno.md](opencl_adreno.md)) runs the graph optimiser with the board as its one
device. This page says which of the A6 mechanisms of the CUDA flow (round controller, merge-sibling convs, region windows,
region loop; the SOL gate is KernelBench-irrelevant) exist on that target, what was adapted, and what refuses.

| mechanism | flag | on `--target opencl` | why |
|---|---|---|---|
| round controller | `--round-controller auto\|fixed\|graph-first`, `--round-controller-budget-usd N` | **works** | flow level: it reads the profile, the proposed hints and the KernelBlaster spend. Its candidate count follows the flow's target (it counted only `cuda`-stamped hints before; now `cfg.target`). `--round-controller-small-us` (default 15 us, a B200 number) is the "small unfused op" threshold of its `auto` rule; Adreno seed ops are ms-scale, so almost nothing is "small" there and `auto` decides on estimated graph gains. |
| merge-sibling convs | `--merge-sibling-convs auto` | **works** | a graph rewrite (concat of sibling convs -> one conv with derived zero-padded weights). The merged conv is an ordinary `conv2d` / `conv2d_f16` shape, so the OpenCL seed and the OpenCL export need nothing new. Adapted: the hook stamps its hints with the flow's target (`opencl`), so they are decided on the board and a CUDA-decided hint is never applied to an OpenCL graph. |
| region slots / region windows | `--region-slots`, `--region-windows auto` | **refused** | a region slot's seed is the CUDA composite (`cuda_seed`: one launcher over every member's CUDA kernel, streams / events); there is no OpenCL composite. |
| region loop | `--region-loop on` | **refused** | KernelBlaster's loop builds and times CUDA programs (nvcc, streams, graphs); no OpenCL backend. Also inert without region slots. |
| SOL gate | `--sol-gate` | **refused** | reads the CUDA driver's `MB_SOL` lines; not used for KernelBench. |
| view fusion, graph-rewrite agent, multiturn | `--view-fusion`, `--graph-rewrite`, `--multiturn` | **refused** | their hints are CUDA-stamped / the experiment tool uses nvcc. |

"Refused" means `kb_graph_opt` stops before doing any work (`opencl_refusals`: exit 2 from the CLI, `ValueError` from
`run`), naming each flag and why, including values taken from the environment (`MB_REGION_LOOP`, `MB_REGION_WINDOWS`,
`MB_SOL_GATE`, `MB_GRAPH_REWRITE`). Nothing runs as a silent no-op.

So **A6 on OpenCL** = `--round-controller auto --round-controller-budget-usd N --merge-sibling-convs auto --rounds 3`
(+ the usual flow: `--plan-mode all`, `--knowledge v2`, `--ncu best` = the OpenCL profile).

Other target-dependent details of this merge:

* the search fixes' cache routing (`fallback_router`, `ingest_requeued`) reads CUDA arch keys (`sm_<N>`) and `.cu` sources: off on
  OpenCL (`Fixes.cuda`); the profile's own fallback (reject a failing cached winner, redo the profile) and the final-guard warning
  are target-independent and stay on;
* `CudaCache.add` / `add_reuse` / `cache_plan` / `usable` filter by the cache's target (entries carry `target` only when not CUDA);
* the walker tail now defines `mb_any` for every target (the CUDA-side per-slot dtype work spells call arguments `mb_any{b[i]}`; the
  OpenCL tail did not define it: first board build failed with `'mb_any' was not declared`).

Tests: `pipeline/tests/test_opencl_a6.py` (refusals, hook stamping, applier per target, merged IR exported for OpenCL, Fixes,
`_run` with the controller on OpenCL hints); KernelBlaster `tests/test_price_sonnet55.py`.
