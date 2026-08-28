/* RVV fixed-point intrinsics: bridge the pre-1.0 and 1.0 argument forms.
 *
 * The curated int8 conv kernels are written against the RVV intrinsics v1.0
 * API, in which every fixed-point instruction takes an explicit rounding mode:
 *
 *     __riscv_vsmul_vx_i32m4 (vs2, rs1, __RISCV_VXRM_RNU, vl)
 *     __riscv_vnclip_wx_i16m2(vs2, shift, __RISCV_VXRM_RNU, vl)
 *
 * GCC 13.2 -- the SpaceMiT cross-toolchain available here -- implements the
 * earlier form, where the rounding mode comes from the `vxrm` CSR and is not an
 * argument. Against it the kernels fail to compile with "too many arguments"
 * and "'__RISCV_VXRM_RNU' undeclared".
 *
 * That is a toolchain-version mismatch, not a kernel defect, and it is worth
 * fixing rather than routing around: silently falling back to a different
 * algorithm because one failed to build is how a run ends up labelled "rvv"
 * while executing scalar reference code. (That exact thing happened here once
 * already -- see backends.curated_aliases.)
 *
 * WHY THE MACRO IS SAFE. A function-like macro is not re-expanded inside its
 * own replacement list, so `#define f(a,b,c,d) f(a,b,d)` rewrites the call and
 * then leaves the inner `f` alone. This is standard C preprocessor behaviour,
 * not a trick that depends on the compiler.
 *
 * WHY DROPPING THE ARGUMENT IS CORRECT HERE. Every use in these kernels passes
 * __RISCV_VXRM_RNU, and RNU is encoding 0 -- the reset value of `vxrm`. The
 * static assertion below pins that, so if a kernel is ever written with a
 * different mode this stops compiling instead of quietly rounding the wrong
 * way. mb_rvv_vxrm_init() additionally writes the CSR once, so the value does
 * not depend on whatever ran before us on the same hart.
 */

#ifndef MB_RVV_VXRM_COMPAT_H_
#define MB_RVV_VXRM_COMPAT_H_

#if defined(__riscv_v_intrinsic) && !defined(__RISCV_VXRM_RNU)

#define __RISCV_VXRM_RNU 0 /* round-to-nearest-up   (vxrm reset value) */
#define __RISCV_VXRM_RNE 1 /* round-to-nearest-even */
#define __RISCV_VXRM_RDN 2 /* round-down / truncate */
#define __RISCV_VXRM_ROD 3 /* round-to-odd */

/* Set vxrm explicitly rather than trusting the reset value: the CSR is
 * per-hart architectural state and another thread's kernel could have left it
 * elsewhere. Cheap -- one csrwi, hoisted out of any loop. */
static inline void mb_rvv_vxrm_init(void)
{
	__asm__ volatile("csrwi vxrm, 0" ::: "memory");
}

/* Compile-time guard: these rewrites are only valid for RNU. */
#define MB_VXRM_ASSERT_RNU(m) \
	((void)sizeof(char[1 - 2 * !((m) == __RISCV_VXRM_RNU)]))

#define __riscv_vsmul_vx_i32m4(vs2, rs1, vxrm, vl) \
	(MB_VXRM_ASSERT_RNU(vxrm), __riscv_vsmul_vx_i32m4(vs2, rs1, vl))
#define __riscv_vsmul_vv_i32m4(vs2, vs1, vxrm, vl) \
	(MB_VXRM_ASSERT_RNU(vxrm), __riscv_vsmul_vv_i32m4(vs2, vs1, vl))
#define __riscv_vsmul_vx_i32m2(vs2, rs1, vxrm, vl) \
	(MB_VXRM_ASSERT_RNU(vxrm), __riscv_vsmul_vx_i32m2(vs2, rs1, vl))
#define __riscv_vsmul_vx_i32m1(vs2, rs1, vxrm, vl) \
	(MB_VXRM_ASSERT_RNU(vxrm), __riscv_vsmul_vx_i32m1(vs2, rs1, vl))

#define __riscv_vnclip_wx_i16m2(vs2, rs1, vxrm, vl) \
	(MB_VXRM_ASSERT_RNU(vxrm), __riscv_vnclip_wx_i16m2(vs2, rs1, vl))
#define __riscv_vnclip_wx_i16m1(vs2, rs1, vxrm, vl) \
	(MB_VXRM_ASSERT_RNU(vxrm), __riscv_vnclip_wx_i16m1(vs2, rs1, vl))
#define __riscv_vnclip_wx_i8m1(vs2, rs1, vxrm, vl) \
	(MB_VXRM_ASSERT_RNU(vxrm), __riscv_vnclip_wx_i8m1(vs2, rs1, vl))
#define __riscv_vnclip_wx_i8mf2(vs2, rs1, vxrm, vl) \
	(MB_VXRM_ASSERT_RNU(vxrm), __riscv_vnclip_wx_i8mf2(vs2, rs1, vl))

#define __riscv_vssra_vx_i32m4(vs2, rs1, vxrm, vl) \
	(MB_VXRM_ASSERT_RNU(vxrm), __riscv_vssra_vx_i32m4(vs2, rs1, vl))
#define __riscv_vssra_vx_i32m2(vs2, rs1, vxrm, vl) \
	(MB_VXRM_ASSERT_RNU(vxrm), __riscv_vssra_vx_i32m2(vs2, rs1, vl))

#define MB_RVV_VXRM_COMPAT_ACTIVE 1

#else /* intrinsics v1.0, or no vector support at all */

static inline void mb_rvv_vxrm_init(void) {}

#endif

#endif /* MB_RVV_VXRM_COMPAT_H_ */
