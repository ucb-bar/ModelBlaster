# Saturn OPU backend (`rvv_opu`) — design notes

Status: **infrastructure landed, no curated kernels yet** (2026-05-16).

## What is the OPU?

The Outer Product Unit is a Saturn-internal matrix MAC engine added by
the `new_opu_integrated` / `opu-fp8` Saturn branches. It exposes four
architectural matrix registers (m0..m3) and a small custom instruction
set encoded on the V opcode (0x57) with custom funct fields.

The integer mode of interest:

```
VOPACC  md, vs2, vs1      // md += vs2 ⊗ vs1
        ^      ^   ^
        i32    i8  i8     // when configured with opuParams (i8 OPU)
```

`m1` holds an i32 accumulator tile; one VOPACC consumes a vector of i8
inputs and a vector of i8 inputs and outer-product-MACs them into the
tile. The tile size depends on the bitstream's `opuParams` — at
`V256D128 opuParams`, mlmax=16 → m1 is a 16×16 i32 tile, equivalent
to a Gemmini 16×16 = 256 INT8 PE mesh.

See `agents/cores/saturn_opu/include/saturn_opu.h` for the full
encoding table and the asm-macro programming model. The canonical
i8 matmul body, distilled from upstream
`generators/saturn/benchmarks/opu-gemm/kernel.h::i8_mm_bme_sq`:

```c
asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(ml));
asm volatile("vle32.v v0, (%0)" : : "r"(c_bias));
OPMVINBCAST(m1, v0);               // seed m1 with bias

for (k = 0; k < K; k++) {
    asm volatile("vle8.v v18, (%0)" : : "r"(&b[k*N]));
    asm volatile("vle8.v v16, (%0)" : : "r"(&at[k*M]));
    VOPACC(m1, v18, v16);          // m1 += b_col ⊗ at_col
}

asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(vl));
for (r = 0; r < ml; r++) {
    VMV_VR(v0, r, m1);             // pull row r of m1 into v0
    asm volatile("vse32.v v0, (%0)" : : "r"(&c[r*N]));
}
```

Note the asymmetric data layout: `at` is **transposed** (column-major
in M) while `b` is row-major (column-major in N). The outer product
naturally produces a row-of-A times column-of-B fragment — i.e. the
inputs are columns from A^T and from B, written contiguously per K
step.

## What landed (infrastructure only)

| file | purpose |
|---|---|
| `agents/pipeline/backends.py::RVV_OPU` | backend registration |
| `agents/harness/backends/rvv_opu.conf` | Kconfig overlay (V kernel-only, same hygiene as rvv) |
| `agents/cores/saturn_opu/include/saturn_opu.h` | vendored asm macros |
| `agents/kernels/rvv_opu/` | curated-kernel directory (empty + README) |

Plus a few one-line wiring updates: added `rvv_opu` to
`_CACHE_AWARE_TARGETS` in `generate_kernels.py` (so the memory-model
optimize stanza fires for OPU rewrites) and gated the fp16 auto-promote
in `_run_lib.sh` to {`rvv`, `scalar`} so `TARGET=rvv_opu` doesn't
silently rewrite to `rvv_opu_f16` (which doesn't exist).

## What's NOT in yet

1. ~~**Curated kernels.** The first candidate is
   `rvv_opu_linear_s8_outerprod_acc.c`...~~ **First curated kernel
   landed 2026-05-16**: `agents/kernels/rvv_opu/rvv_opu_matmul_s8_outerprod.c`,
   ported from upstream `opu-gemm/kernel.h::i8_mm_bme_sq`. Targets
   the agents-flow `matmul_s8` op (used in ViNT attention's Q·Kᵀ
   and ·V matmuls — M=N=7, K=64..512). Validated 5/5 on the
   OPU-extended spike (`--extension=saturn_opu`); spike -l confirms
   `vopacc`/`opmvinbcast` actually firing in-kernel. Out-of-tile
   shapes (M or N > mlmax = VLEN/8) cleanly fall back to an
   embedded scalar reference.

   Still pending:
   - **linear_s8 curation.** The OPU pattern naturally fits matmul
     more than linear (linear's M=1 in single-batch inference is
     degenerate for outer-product). A linear_s8 curation would need
     either a strided-load variant or to batch multiple linears
     before the OPU. Defer until profiling shows linear_s8 dominates.
   - **Tiled OPU coverage** for M > mlmax or N > mlmax. The upstream
     `i8_mm_bme_sq` has the tiling loop; needs porting alongside
     the matmul_s8 requantize tail. Current curation only handles
     single-tile (which covers ViNT attention).
   - **conv2d_s8 via OPU-im2col**. Conv → matmul reshape; the OPU
     win depends on whether im2col + OPU beats the cache-blocked
     RVV conv2d_s8 already curated. Profile first.

2. **conv2d kernels.** OPU is most natural for matmul; conv2d via
   im2col + OPU gemm is the obvious mapping but needs an im2col
   step (the `opu-fused-gemm-transpose` benchmark on opu-fp8 hints
   at how to keep the im2col data flow vectorized).

3. **Quantization-aware Q0.31 requantize on the int32 drain.**
   `linear_s8_pc` expects the drain to fold (output_multiplier,
   output_shift) → int8 output, not raw int32. The gemmini_q31
   curated kernel has a worked example of this pattern; we can
   crib from it.

4. ~~**Spike verify.** Stock spike doesn't decode `.insn r 0x57, 0x2,
   0x51, ...`.~~ **DONE 2026-05-16.** Custom spike extension landed at
   `hw/chipyard/toolchains/riscv-tools/riscv-isa-sim/customext/saturn_opu.cc`
   (~270 LOC, follows the cflush.cc template). Functional model for
   VOPACC, OPMVINBCAST, VMV_VR, VMV_RV; cross-checked bit-exact against
   a scalar reference on a 4×4×3 i8 matmul and a 16×16×8 randomized
   matmul. Built into `hw/chipyard/.conda-env/riscv-tools/`. Backend's
   `spike_args=("--extension=saturn_opu","--isa=rv64gcv_zicntr")` and
   `_run_lib.sh` auto-routes to that spike via `AGENTS_OPU_SPIKE` env.
   See `agents/notes/saturn_opu_spike_support.md` for the design;
   commit-ready as one atomic change with the agents-side backend code.

5. **FireSim bitstream.** Needs a build with `opuParams`. Chipyard's
   `REFV256D128DualRocketSaturnOPUGemmini32x32Q31WsConfig` (currently
   on master tip of chipyard's SaturnConfigs.scala — same file the
   user is in the middle of swapping for an OPU-fp8 experiment) is
   the closest available OPU+Q31 Rocket config. The Shuttle-side
   `OPUV256D128ShuttleConfig` from `chipyard/OPUConfigs.scala` on
   the saturn `opu-fp8` branch is the cleaner Shuttle target if we
   want OPU without the Q31 gemmini also on the SoC.

## Programming-model gotchas

- **m0..m3 register encoding HACK.** The matrix register operand
  slot in OPU custom instructions is encoded using the standard
  scalar-register field of `.insn r`. We #define `m0` as `"x0"` etc.
  so source reads naturally, but the upstream GNU assembler will
  happily encode `VOPACC(m1, v18, v16)` as if it touched x1, x18,
  x16 — fine for the OPU decoder (it reads the same 5-bit slots),
  but means GCC's register allocator must not see this as a use of
  x1/x18/x16. The `asm volatile` block doesn't list them as clobbers
  today, which is technically a bug. Mitigation: keep OPU sequences
  short, don't expect compiler scheduling around them.

- **`OPMVINBCAST(m1, v0)` is a broadcast.** The intent is "fill m1
  with v0 in every row" — same i32 bias vector replicated. To zero
  the accumulator, pass a v0 that was `vmv.v.i v0, 0`'d first.

- **VOPACC is destructive in md.** The accumulator IS the destination
  (`md += ...`), so any subsequent OPMVINBCAST resets it.

- **No matrix-reg context switch in Zephyr.** Today the agents flow
  enters generated kernels with `irq_lock` held, so a kernel can't
  be preempted mid-OPU-sequence. If we ever drop that mask (e.g. to
  let kernel workers run preemptably), we'll need Zephyr code in
  `arch/riscv/core/` to save/restore m0..m3 on context switch.

## How to test once the first curated kernel lands

```
# Build the model with rvv_opu backend, run on FireSim:
TARGET=rvv_opu BACKEND=reference RUNNER=firesim \
  GLOBAL_CURATED_DIR=$PWD/agents/kernels \
  bash agents/examples/<model>/run.sh
```

For a quick sanity check before the agents-flow integration, the
upstream `opu-gemm` benchmark can run standalone on chipyard sims:

```
cd hw/chipyard/generators/saturn/benchmarks/opu-gemm
make
spike pk opu-gemm   # only on a Saturn-OPU-aware spike fork
```

## References

- Saturn OuterProductUnit: `hw/chipyard/generators/saturn/src/main/scala/exu/OuterProductUnit.scala`
- Saturn OPU sequencer: `hw/chipyard/generators/saturn/src/main/scala/backend/OuterProductSequencer.scala`
- Chipyard OPU configs: `hw/chipyard/generators/chipyard/src/main/scala/config/OPUConfigs.scala` (on `origin/opu-fp8`)
- Upstream programming examples: `hw/chipyard/generators/saturn/benchmarks/opu-*` (on `origin/opu-fp8`)
- Vendored header: `agents/cores/saturn_opu/include/saturn_opu.h`
