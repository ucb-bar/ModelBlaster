# harness_k3ir: K3 IR runtime

A standalone runtime that executes ModelBlaster IRs (`graph.json` + `weights.npz`, the mbir format of
`gpu_eval/`) on the SpacemiT K3 (8 X100 cores at VLEN 256, 8 A100 AI cores at VLEN 1024 with the IME
matrix extension). It runs int8_mixed IRs (int8 linears with fp32 islands, `pipeline/int8_mixed.py`),
fp32 IRs, and the fp16 / mixed int8 + fp16 builds that `f16_ir.py` derives from them.

## Build

```
make host                     # x86, reference kernels only (the numerics model of the C kernels)
make k3    CROSS=<prefix>     # static riscv64 binary for the K3 (IME + RVV kernels)
make spike CROSS=<prefix>     # static riscv64, reference + RVV kernels, no IME, no thread spawn (Spike + pk)
```

`CROSS` is the prefix of a riscv64 glibc toolchain with static libraries (default
`riscv64-unknown-linux-gnu-`; the SpacemiT toolchain v1.1.2 was used for the K3). The fp16 sources build
with Zfh + Zvfh, which both K3 core kinds implement.

## From a model to a plan

1. IR: `pipeline/extract_graph.py --model M --quant fp32|int8 --out-dir DIR` (`--quant int8` uses the
   model's `get_int8_mixed_policy()`; see docs/vulkan_amd.md, "int8 SmolVLA").
2. Optional fp16 / mixed builds (`f16_ir.py`): `fused` (fp32 IR with the runtime's fusions), `f16`
   (fused fp32 IR -> fp16 build; int8_mixed IR -> mixed build), `cnn` (classic CNN extract -> runtime
   kinds), `check`, `golden-inputs`.
3. `pack_ir.py IR_DIR OUT_PREFIX` writes `OUT_PREFIX.{plan,wblob,cblob,names.json}`.

## Run

```
k3ir run   P.plan P.wblob P.cblob --in INPUT.bin [--iters N --warm W] [--out OUT.bin]
k3ir feed  P.plan P.wblob P.cblob --feed FEED.bin --dump DUMP.bin       # every op on its reference inputs
k3ir chunk V.plan V.wblob V.cblob  PF.plan PF.wblob PF.cblob  S.plan S.wblob S.cblob \
           --obs obs_L6.bin --temb temb.bin --steps 5 [--nobs 1] [--out acts.bin]   # SmolVLA chunk
k3ir ddpm  P.plan P.wblob P.cblob --win WIN.bin --sched SCHED.bin --temb TEMB.bin  # diffusion student loop
k3ir bench ...                                                         # kernel / dispatch microbenchmarks
```

Placement and kernels:

| flag | meaning |
| --- | --- |
| `--cpus 0-7 --backend x100` / `--cpus 8-15 --backend a100` | pool CPUs and kernel family |
| `--cpus2 8-15 --place-graphs vision=1` / `--place-ops` | second pool (two clusters) and graph- or op-level placement |
| `--pool 1`, `--pool2 1` | static-split dispatch (0 = dynamic) for pool 0 / pool 1 |
| `--fuse-rowq` | row op + quantize fusion (byte-identical) |
| `--attn 1` | fused fp32 attention (byte-identical to the unfused chain) |
| `--attn 3`, `--attn 6` | fused int8 IME attention with 16-bit P (6: 16-bit Q too); a numerics change |
| `--attn 5`, `--attn 7` | auto: int8 (3 / 6) when the key set has `--attn-i8-min` (128) or more keys, else fp32 |
| `--f16-epi 1` (default) | fp16 outputs stored from the int8 GEMM epilogue (0: fp32 view + conversion pass) |
| `--spin-log2 N` | idle-worker spin budget (2^N iterations, default 22) |
| `--vstore`, `--prefetch`, `--mtail`, `--no-epilogue`, `--epi-mode`, `--no-cbozero`, `--vperm` | kernel A/B switches |
| `--prof` | per-op profile |

## Verify

- `--check`: every exact kind (int8 kinds, exact fp16 kinds) is recomputed with the reference C and
  compared byte for byte (`check_mismatches` in the RESULT line, `CHECK_MISMATCH` lines).
- `--attn-check`: fused attention compared against the unfused chain (`ATTN_CHECK` line).
- Isolation gate on the host: `ref_walk.py feed` writes a feed from the CPU reference walk,
  `k3ir feed` dumps every op's output, `ref_walk.py compare` requires int8 kinds byte-equal to the walk
  and fp32 ops within the per-slot bound. `tools/host_gate.sh` runs this; `tools/spike_gate.sh` runs the
  Spike build at VLEN 1024 and 256 and compares its dump with the host's (`dumpdiff.py`).
- `chunk_summary.py`: open-loop SmolVLA action error against the fp32 PyTorch chunk.
- `tools/macs.py`, `tools/plandump.py`: MAC counts and op listing of a plan.

The gate scripts read `tools/env.sh` (`PY`, `SPIKE`, `PK`, `SPIKE_ISA`, `K3IR_OUT`; all overridable).
