# KernelBench level1 at STOCK dimensions on spike (RVV, fp32)

First real KernelBench-problem benchmarks at **full stock dimensions** on RISC-V
(`RUNNER=spike TARGET=rvv BENCH_MAX_ELEMENTS=0`, `--isa=rv64gcv_zicntr`). The
256 MB default was only the Zephyr `ram0` DTS region — `SPIKE_RAM_SIZE` (a DTS
overlay bumping ram0) + `SPIKE_MEM_MB` (`spike -m`) lift it. spike is bounded by
host RAM (125 GB here), not 256 MB.

| bench | stock dims | io | wall cycles (spike RVV) | verify |
|---|---|---|---|---|
| 12_Matmul_with_diagonal_matrices | 4096² | 67 MB | 1,510,900 | max_abs_err=0 |
| 40_LayerNorm | 16·64·256·256 | 268 MB | 15,439,550 | 0.0106 |
| 95_CrossEntropyLoss | 32768×4096 | 512 MB | 114,701,250 | 0 |

All three needed the ram0 bump (io > 256 MB) and passed — proving the config
limit was the only barrier for moderate-io benches.

## The hard boundary: baked io ≤ ~2 GB
io is `.incbin`'d into the ELF rodata. Two build-time limits cap it — and
neither is fixable on bare-metal spike (no filesystem to load from at runtime):
- **~2 GB**: RISC-V `R_RISCV_PCREL_HI20` reloc truncation — >2 GB of rodata
  pushes Zephyr's own code sections out of PC-relative range (e.g. printk.c).
- **4 GB**: GNU `ar` archive-member limit.

So benches whose stock io exceeds ~2 GB **cannot build for spike**, regardless of
`ram0`/`spike -m`:
- 47/48/49_reductions (128·4096·4095 = **8.6 GB**), 51/53 argmax/min
- activations 19–32 (~1.6 G-elem = **6.4 GB**)
- softmax/log_softmax (4096·393216 = **6.4 GB**)
- batch-112 norms (34/35/36), MSE/Huber/Hinge losses (32768² = **4 GB**)

## Compute-time boundary (the other limit)
spike is a functional simulator (~100–300 MIPS). O(N) ops above finish in
seconds–minutes at stock. But compute-heavy ops are impractical:
- matmul 2048³ (8.6 GFLOP, naïve kernel) ≈ hours
- conv2d/3d, conv_transpose (100s of GFLOP) ≈ days
- SDPA 97 (100s of TFLOP) ≈ weeks

## Realistically benchmarkable at stock on spike (RVV)
The intersection of io < 2 GB AND light compute: `diag_matmul`, `layer_norm`,
`cross_entropy_loss`, small matmuls / matrix-vector, and any reduction/norm/loss
whose stock io stays under 2 GB. For the big-io or compute-heavy problems, use
native_sim (with runtime io-load) for correctness, and FireSim for the
optimized-kernel stock-dim performance metric (Phase 4).
