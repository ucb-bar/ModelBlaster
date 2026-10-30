O# CHANGES: hardware-aware Adreno seeds and prompts (copy of mb_qrb_a6 + its KernelBlaster clone)

Base: ModelBlaster `qrb-a6` a168983, KernelBlaster `qrb-a6` 1598b18. Patches: `qrb_hw.mb.patch` (ModelBlaster, applies on a168983),
`qrb_hw.kb.patch` (KernelBlaster, applies on 1598b18); both `git apply --check` clean on the live trees. Nothing committed.

## Flags and defaults
| flag (environment) | default | effect |
|---|---|---|
| `MB_OPENCL_F16_MATH` = 0 / 1 / auto | auto: ON when the device label (`--arch` / `$MB_OPENCL_ARCH`) is adreno650..750, OFF otherwise (NVIDIA stand-in, no label) | export: fp16-math `_f16` seeds. `0` = the previous storage-only seeds, slot files byte-identical (checked) and the head sets `MB_CL_STD_DEFAULT "CL1.2"` + `MB_NO_FP16_PRELUDE` (legacy build). KernelBlaster: the fp16 rule of the prompt (half arithmetic allowed / encouraged) follows the same flag |
| `MB_OPENCL_HW_INFO` = 1 / 0 (KernelBlaster) | 1 (only for archs with a table entry: adreno650) | the "Target hardware" block of the OpenCL system prompt. Both flags off (or a stand-in arch) = the previous prompt byte for byte |
| `MB_OPENCL_ARCH` | set by `kb_graph_opt`/`kb_export`/`kb_plan` from `--arch` when it is a device label (`setdefault`) | feeds the two flags above |
| `MB_CL_STD` (board run time) | auto = device's OpenCL C level (CL2.0 on Adreno 650, CL1.2 on the stand-in) | replaces the hard-coded `-cl-std=CL1.2`; `MB_CL_STD=CL1.2` or empty restores / drops it |
| `MB_CL_FP16` (board run time) | 1 on a device with cl_khr_fp16 (0 in a legacy-built problem) | prelude enables `cl_khr_fp16`, defines `MB_HAS_FP16` |
| `kConvOpts` in the conv slot (`-DMB_OT=8`, `-DMB_ACC_HALF=1 -DMB_ACC_FLUSH=16`) | `-DMB_OT=8`, fp32 accumulate | conv tile (8 or 4 channels) and the documented fp16-accumulate switch |

## ModelBlaster (`qrb_hw.mb.patch`)
- `gpu_eval/harness/opencl/mb_opencl_kernels_f16_math.cpp` (NEW): fp16-math seeds: `conv2d_f16` (any stride/pad/dilation, N, 4 px x 8 (or 4) oc tile, weights re-laid out `[tap][OC]`, fp32 accumulators, optional half accumulate with flush), `matmul_f16`, `linear_f16` (4x8 tile, weight re-layout `[K][Np]`), `batchnorm2d_f16` (half8, fp32 mad, int math), `ewise2_f16` (int math, per-opcode kernels, contiguous and bias-broadcast half8 paths).
- `pipeline/opencl_seeds.py`: `f16_math_enabled(arch)`, `OpenCLSeeds(..., f16_math, arch)`, fp16-math templates for the 17 unary + 2 binary MAP ops (relu, relu6, add, mul in pure half8; others widen to float8) and for `ewise1_f16` / `ewise_scalar_f16`; storage-only path untouched.
- `pipeline/kb_export_opencl.py`: `head_source` emits the legacy build defines when fp16 math is off.
- `gpu_eval/harness/opencl/mb_opencl_runtime.h`: `-cl-std` from the device (was forced CL1.2, a switch `$MB_CL_STD`); fp16 prelude; `MB_SCRATCH_IMAGE` / `mb_scratch_image` (image2d view of a slot-owned buffer, cl_khr_image2d_from_buffer; slots still cannot call clCreate*/clEnqueue*/clFinish themselves).
- `pipeline/kb_export.py`, `kb_graph_opt.py`, `kb_plan.py`: set `MB_OPENCL_ARCH` from `--arch`.
- `pipeline/tests/test_opencl_f16_math.py` (NEW, 16 tests): the switch, legacy identity, half-math/direct-vector content, signature identity, runtime header logic, numerics-policy emulation, clang `-x cl` syntax check of every generated program (skipped without LLVM clang).
- `docs/opencl_adreno.md`: runtime section (std, prelude, images), fp16 sections, decisions 5 and 8.

## KernelBlaster (`qrb_hw.kb.patch`)
- `src/kernelblaster/agents/multislot_opencl.py`: `opencl_system_prompt(dtype, tolerance, arch)`, `OPENCL_HARDWARE` (measured device facts, public peaks flagged, QNN GPU backend notes + op list from the QAIRT 2.45 docs, calibration incl. the cc_runs findings), fp16 rules with the manifest bound (the old prompt said "fp32 in/out" even for fp16), OpenCL C 2.0, build options, image path; technique catalogue additions (fp16 half arithmetic, half vector I/O, accumulate policy, fast-math options, work-group limits, conv register tile / weight tile / image reads / epilogue fusion, gemm tiles, subgroup reduction, half8 elementwise).
- `src/kernelblaster/agents/opt_multislot.py`: `TargetProfile.prompt_for` uses it for `opencl`.
- `data/kernelblaster/knowledge/techniques/60_adreno.json`: 7 techniques added.
- `docs/multislot_optimization.md`: one table row. `tests/test_opencl_hw_prompt.py` (NEW, 9 tests).

## Tests
MB `pipeline/tests/test_kb_export_opencl.py test_kb_graph_opt_opencl.py test_opencl_a6.py`: 27 passed, 3 skipped before; after the change the same 27 pass (+16 new = 43 passed, 3 skipped).
KB `test_multislot_opencl.py test_multislot_agent.py test_knowledge_seed.py test_knowledge_v2.py test_knowledge_v2_open.py`: before 49 passed, 8 skipped, 1 failed (`test_sqlite_concurrent_writers_union_evidence`, a known failure: `No module named 'third_party'` in a spawned process); after the same + 9 new = 58 passed, 8 skipped, same 1 failure.
