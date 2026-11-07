/* RISC-V Zvt (vector-matrix tile extension, draft) encodings for kernels.
 *
 * Mirrors zephyr include/zephyr/arch/riscv/zvt.h (ucb-bar/zephyr branch
 * zvt-matrix-context) but is standalone so kernels also build outside
 * Zephyr. Encodings: riscv-isa-sim 72958551 riscv/encoding.h, verified
 * with that spike's spike-dasm. Zvt is a draft -- re-verify if the spike
 * pin moves.
 *
 * No assembler knows Zvt yet (binutils 2.43, LLVM 19), so:
 *   - GPR-operand ops are emitted with `.insn r` (works in GNU as and
 *     clang; the compiler may allocate the registers);
 *   - vector/tile-operand ops are emitted as `.word` with constant
 *     register numbers (GNU as rejects vector regs inside `.insn`).
 *
 * Use these *inside one asm block per tile computation*: any compiler-
 * emitted vsetvl{i} between tile ops clears mtype and makes the next
 * tile instruction illegal.
 */
#ifndef MODELBLASTER_ZVT_INSN_H
#define MODELBLASTER_ZVT_INSN_H

/* asm-string fragments; operands named per GNU inline asm. */
#define ZVT_S_MSETMTYPE(mt, vt) ".insn r 0x57, 7, 0x42, x0, " mt ", " vt "\n\t"
#define ZVT_S_MSETTN(rd, rs)    ".insn r 0x57, 7, 0x44, " rd ", " rs ", x0\n\t"
#define ZVT_S_MSETTM(rd, rs)    ".insn r 0x57, 7, 0x44, " rd ", " rs ", x1\n\t"
#define ZVT_S_MSETTK(rd, rs)    ".insn r 0x57, 7, 0x44, " rd ", " rs ", x2\n\t"
#define ZVT_S_VTLE32(base, tss) ".insn r 0x07, 7, 0x29, x0, " base ", " tss "\n\t"
#define ZVT_S_VTSE32(base, tss) ".insn r 0x27, 7, 0x29, x0, " base ", " tss "\n\t"

#define ZVT_STR_(x) #x
#define ZVT_STR(x) ZVT_STR_(x)
#define ZVT_S_WORD3(match, rd, rs1, rs2) \
    ".word (" ZVT_STR(match) " | ((" ZVT_STR(rd) ") << 7) | ((" \
    ZVT_STR(rs1) ") << 15) | ((" ZVT_STR(rs2) ") << 20))\n\t"

/* acc tile td (td encoded as rd = 2*td); vs2 = A columns, vs1 = B columns */
#define ZVT_S_VTMMS(td, vs2, vs1) ZVT_S_WORD3(0xf20000f7, (td) * 2, vs1, vs2)
#define ZVT_S_VTMMU(td, vs2, vs1) ZVT_S_WORD3(0xf2000077, (td) * 2, vs1, vs2)
#define ZVT_S_VTFMM(td, vs2, vs1) ZVT_S_WORD3(0xf2001077, (td) * 2, vs1, vs2)
#define ZVT_S_VTZERO(td)          ZVT_S_WORD3(0x43e06057, (td) * 2, 0, 0)

/* mtype: [1:0] log2(widen)+1, [7:5] tk, [23:10] tm */
#define ZVT_MTYPE(widen_code, tk, tm) \
    ((unsigned long)(widen_code) | ((unsigned long)(tk) << 5) | \
     ((unsigned long)(tm) << 10))
#define ZVT_WIDEN4 3 /* i8 -> i32 */
#define ZVT_VTYPE_E8      (0UL << 3)
#define ZVT_VTYPE_E32     (2UL << 3)
#define ZVT_VTYPE_ALTFMT  (1UL << 8) /* vtmms: signed x signed */

/* Tile selector for vtle/vtse: tile [30:27], column-major [24], index */
#define ZVT_TSS(tile, colmajor, idx) \
    (((unsigned long)(tile) << 27) | ((unsigned long)(colmajor) << 24) | \
     (unsigned long)(idx))

#endif
