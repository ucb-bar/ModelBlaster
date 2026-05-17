/* Saturn OPU (Outer Product Unit) custom-instruction asm macros.
 *
 * Vendored from chipyard's saturn submodule at
 *   generators/saturn/benchmarks/common/bme.h  (branch origin/opu-fp8)
 * with renamed identifiers and additional comments.
 *
 * The OPU is a Saturn-internal matrix engine layered on the V (vector)
 * extension. It exposes four matrix registers (m0..m3) plus a small
 * custom instruction set that share the V opcode (0x57) and use custom
 * funct fields. Programs touch the OPU exclusively through these asm
 * macros; the upstream GNU assembler does NOT have first-class OPU
 * mnemonics, so we encode each op via `.insn r ...` and a HACK: the
 * matrix register operands are passed by *abusing* the scalar register
 * encoding slots in the R-type instruction (the OPU decode in HW reads
 * the same bits as a matrix-register index 0..31). To keep that
 * legible at the source, we #define m0..m3 and v0..v31 to the scalar
 * register strings "x0".."x31".
 *
 * Programming model (i8, the only path validated today):
 *
 *   1. Set vector configuration (e8, m1, ta, ma) via vsetvli.
 *   2. Seed an accumulator matrix register (m1) with a vector via
 *      OPMVINBCAST(m1, v0) — broadcasts the i32 vector v0 across all
 *      rows of m1.
 *   3. Outer-product accumulate:
 *           VOPACC(m1, v_b, v_a)
 *      computes  m1 += v_b ⊗ v_a   (column-by-row outer product).
 *      v_a and v_b hold i8 inputs; m1 holds the i32 accumulator.
 *   4. Drain rows of the matrix back into vectors:
 *           for r in 0..ml:
 *             VMV_VR(v0, r, m1)
 *             vse32.v v0, (out + r*N)
 *
 * Encoding cheat-sheet (matches Saturn's OuterProductUnit.scala decode):
 *
 *   opcode = 0x57 (V opcode)
 *
 *   VMV_RV       md = rs1[xlen-1:0] copied into row vs2 of matrix md
 *                funct3=0x6 funct7=0x55
 *   VMV_VR       vd = row rs1 of matrix ms2 broadcast as a vector
 *                funct3=0x6 funct7=0x5d
 *   OPMVINBCAST  md[row,*] = vs2 for all rows (init / broadcast load)
 *                funct3=0x6 funct7=0x59
 *   VOPACC      (int)  md += vs2 ⊗ vs1 (outer-product MAC)
 *                funct3=0x2 funct7=0x51
 *   OPFMACC     (fp)   md += vs2 ⊗ vs1 (outer-product MAC, fp variant
 *                                       — requires fp-capable OPU
 *                                       config and is NOT validated by
 *                                       the integer-only backend)
 *                funct3=0x1 funct7=0x4b
 *
 * NOTE: This header intentionally does NOT touch matrix register
 * context save/restore. The agents flow enters generated kernels with
 * irq_lock held (see run_graph_b's irq mask path), so a kernel cannot
 * be preempted mid-OPU-sequence — matrix state stays consistent within
 * a single dispatch. Cross-dispatch state is not assumed: each kernel
 * must seed its own m1 accumulator (OPMVINBCAST) before VOPACC.
 *
 * BUILD REQUIREMENTS:
 *   - The Saturn bitstream / verilator must be built with VectorParams
 *     `opuParams` (or `opuMxParams` for the MX variant). Encoding the
 *     OPU custom instructions on a non-OPU Saturn build will trap as
 *     illegal instruction.
 *   - The compiler does NOT need any OPU-specific -march extension —
 *     `.insn r 0x57, ...` works under the standard rv64gcv -march.
 *     OPU is implemented as a Saturn-internal decode hook on V opcode
 *     0x57, not a separate ISA extension.
 */
#ifndef AGENTS_SATURN_OPU_H
#define AGENTS_SATURN_OPU_H

/* HACK: matrix and vector register names piggy-back the assembler's
 * scalar-register parser. The HW decoder reads the same 5-bit slot to
 * pick an architectural matrix/vector register, so encoding mN/vN as
 * "xN" is bit-equivalent at the .insn r layer. */
#define m0  "x0"
#define m1  "x1"
#define m2  "x2"
#define m3  "x3"
/* mc0..mc3: matrix counterparts (e.g. for fused gemm + transpose, the
 * MT variant). Kept here for parity with bme.h but unused by the int
 * gemm/conv kernels at the moment. */
#define mc0 "x16"
#define mc1 "x17"
#define mc2 "x18"
#define mc3 "x19"

#define v0  "x0"
#define v1  "x1"
#define v2  "x2"
#define v3  "x3"
#define v4  "x4"
#define v5  "x5"
#define v6  "x6"
#define v7  "x7"
#define v8  "x8"
#define v9  "x9"
#define v10 "x10"
#define v11 "x11"
#define v12 "x12"
#define v13 "x13"
#define v14 "x14"
#define v15 "x15"
#define v16 "x16"
#define v17 "x17"
#define v18 "x18"
#define v19 "x19"
#define v20 "x20"
#define v21 "x21"
#define v22 "x22"
#define v23 "x23"
#define v24 "x24"
#define v25 "x25"
#define v26 "x26"
#define v27 "x27"
#define v28 "x28"
#define v29 "x29"
#define v30 "x30"
#define v31 "x31"

/* VMV_RV md, rs1, vs2
 *   Move scalar rs1 into row vs2 of matrix register md.
 *   Encoding: opcode=0x57 (V), funct3=0x6 (opmvx), funct7=0x55
 */
#define VMV_RV(md, rs1, vs2) \
    asm volatile(".insn r 0x57, 0x6, 0x55, " md ", %0, " vs2 \
                 : : "r"(rs1));

/* VMV_VR vd, rs1, ms2
 *   Broadcast row rs1 of matrix ms2 into vector register vd.
 *   Encoding: opcode=0x57, funct3=0x6, funct7=0x5d
 */
#define VMV_VR(vd, rs1, ms2) \
    asm volatile(".insn r 0x57, 0x6, 0x5d, " vd ", %0, " ms2 \
                 : : "r"(rs1));

/* OPMVINBCAST md, vs2
 *   Initialize all rows of matrix md with vector vs2 (broadcast-fill).
 *   Used to seed the accumulator with a bias vector before the
 *   outer-product MAC loop.
 *   Encoding: opcode=0x57, funct3=0x6, funct7=0x59
 */
#define OPMVINBCAST(md, vs2) \
    asm volatile(".insn r 0x57, 0x6, 0x59, " md ", x0, " vs2);

/* VOPACC md, vs2, vs1
 *   Integer outer-product accumulate:  md += vs2 ⊗ vs1
 *   md is i32; vs1, vs2 are i8 (or whatever the OPU was configured
 *   for — opuParams sets aWidth=bWidth=8, cWidth=32).
 *   Encoding: opcode=0x57, funct3=0x2 (opmvv), funct7=0x51
 */
#define VOPACC(md, vs2, vs1) \
    asm volatile(".insn r 0x57, 0x2, 0x51, " md ", " vs1 ", " vs2);

/* OPFMACC md, vs2, vs1
 *   FP outer-product MAC. Requires an fp-capable OPU bitstream
 *   (opuMxParams or similar). The integer-only backend does NOT use
 *   this; kept here for forward compat with an rvv_opu_f16 / rvv_opu_fp8
 *   backend variant.
 *   Encoding: opcode=0x57, funct3=0x1 (opfvv), funct7=0x4b
 */
#define OPFMACC(md, vs2, vs1) \
    asm volatile(".insn r 0x57, 0x1, 0x4b, " md ", " vs1 ", " vs2);

#endif /* AGENTS_SATURN_OPU_H */
