/* source: curated */
/* algorithm: direct */
/* origin: RVV silu_s8 — register-resident LUT via vrgather.vv instead of a
 *   memory-indexed vluxei8 gather.
 *
 *   Measured on this machine (Saturn V256/D128, fq jobs 35/36): the
 *   __riscv_vluxei8_v_u8m8 gather costs 11.9 cyc/elem while the plain f32
 *   requantize chain costs 1.7 — a memory-indexed gather is ~7x an
 *   arithmetic op stream here, because every one of the 256 lanes issues an
 *   independent byte load. silu_s8 sits at 17.2 cyc/elem for that reason.
 *
 *   The table is exactly 256 bytes and int8 inputs address exactly 256
 *   entries. At VLEN=256 an e8m8 register group holds exactly 256 bytes
 *   (vlmax_e8m8 == 256), so the WHOLE LUT fits in one register group and
 *   vrgather.vv can permute out of it — a register-internal crossbar with
 *   no memory traffic at all.
 *
 *   Guarded: vrgather.vv returns 0 for any index >= vlmax, so if the part
 *   has VLEN < 256 (vlmax_e8m8 < 256) the table would not fit and we fall
 *   back to the original vluxei8 path. Output is bit-identical either way —
 *   same LUT contents, same expf/roundf expression as the stock kernel.
 *
 *   HARDWARE-MEASURED on yolov8_nano (the only model that dispatches this
 *   op; 57 instances, 868,000 elements total, fq job 118 on
 *   f2_dual_small_norose_tacit_q31_60mhz): silu_s8's OWN per-dispatch
 *   hardware cycle counter reads 4,796,189 cycles = 5.53 cyc/elem, vs. the
 *   stock vluxei8 kernel's repeatedly-measured 17.16 cyc/elem (ids
 *   406/425/429 in kernel_opt_log.jsonl) = ~3.1x. Per-op spike verify
 *   against golden at all 13 of yolov8_nano's real silu shapes (plus
 *   extra_shapes) is max_abs_err=0.
 *
 *   RETRACTED 2026-09-02 (was: "IMPORTANT CAVEAT ... yolov8_nano's shared
 *   conv2d_s8/rvv_vsmul_vnclip is CONFIRMED BROKEN on real hardware for
 *   this model", citing a 3-way bisect — oc_blocked err=0, untiled
 *   vsmul_vnclip err=3, tiled vsmul_vnclip Store/AMO access fault,
 *   kernel_opt_log ids 37/904 — and concluding that the FPGA run's
 *   whole-model err=1 was "almost certainly" the conv's fault, with no
 *   yolov8_nano figure to be banked until the conv was fixed).
 *
 *   WHY IT IS RETRACTED. That claim was measured on 2026-08-28 against the
 *   file as it then stood. kernels/rvv/rvv_conv2d_s8_rvv_vsmul_vnclip.c was
 *   REWRITTEN the same week (d201114, then e2304da "shape-conditional 1x1
 *   iGEMM dispatch"): the rvv_vsmul_vnclip *algorithm* is retired, and the
 *   file now carries a shape-conditional tiled-direct / iGEMM production
 *   conv under that name — see that file's own header, which identifies the
 *   original hardware fault as 32-bit index arithmetic wrapping for BSS
 *   buffers above 0x80000000, not anything about this kernel. The claim
 *   above was left pointing at an algorithm NAME whose implementation had
 *   changed underneath it, and it caused a wrong attribution.
 *
 *   WHAT REPLACES IT (measured, 2026-09-02):
 *     * spike, per-op curated verify at all of yolov8_nano's real conv
 *       shapes with MB_DRIFT_ATOL=2: conv2d_s8/rvv_vsmul_vnclip PASS,
 *       max_abs_err=0, and it is the selected pick. Same result on
 *       dronet/rvv. Reproduce:
 *         experiments/kernel_validation/validate_kernels.py --tier1 \
 *             --nets yolov8_nano --backends rvv
 *       log: experiments/kernel_validation/results/regen_yolov8_nano_rvv.log
 *     * FPGA (coordinator's measurement, not re-run here): yolov8_nano rvv
 *       reads max_abs_err=0 both unsplit-1-hart and unsplit-2-hart.
 *
 *   WHAT IS STILL OPEN. yolov8_nano's rvv PAIR arms read max_abs_err=1.
 *   That defect is UNATTRIBUTED. Three candidates have since been cleared
 *   for it and should not be re-investigated: this kernel's memo LUT (all
 *   three rvv-pair arms run 57 silu calls with ZERO concurrent on different
 *   harts — racy but never exercised concurrently), the OH row-tile helper
 *   (its scratch is passed per dispatch, nothing shared), and the conv
 *   itself (zero mutable statics; and the shardec arm has byte-identical
 *   conv split args yet reads 0, so the effect is order-dependent). The
 *   3.1x / 5.53-cyc-per-element figures above are still single-run FPGA
 *   numbers from the pre-fix conv and have NOT been re-measured on a clean
 *   rebuild. */

/* Workspace slot selector -- mirrors gemmini_conv2d_s8_gemmini_tiled_conv.c's
 * MB_GEM_WS_SLOT. Named distinctly because kernels.c concatenates every
 * selected kernel into one translation unit and identically-named macros,
 * enums or struct tags would collide. */
#if defined(CONFIG_SMP) && defined(CONFIG_MP_MAX_NUM_CPUS) && CONFIG_MP_MAX_NUM_CPUS > 1
#include <zephyr/kernel.h>
enum { MB_RVV_SILU_WS_SLOTS = CONFIG_MP_MAX_NUM_CPUS };
#define MB_RVV_SILU_WS_SLOT ((int)arch_proc_id())
#else
enum { MB_RVV_SILU_WS_SLOTS = 1 };
#define MB_RVV_SILU_WS_SLOT 0
#endif

struct mb_rvv_silu_memo {
    int8_t lut[256];
    float  c_si, c_so;
    int    c_lo, c_hi, c_valid;
};

void kernel_silu_s8(const int8_t *input, int8_t *output, int n,
                    float scale_in, float scale_out,
                    int activation_min, int activation_max) {
    /* ONE MEMO SLOT PER HART. lut and the (c_si, c_so, c_lo, c_hi, c_valid)
     * cache key are shared mutable state: two harts running silu with
     * DIFFERENT quant parameters at the same time interleave a table build
     * with a table read, and the reader gathers from a half-rebuilt table.
     * Latent so far -- no measured schedule runs two silu dispatches on
     * different harts at once (yolov8_nano's rvv pair overlaps silu only
     * with conv) -- but the sibling defect in the gemmini im2col conv
     * kernel WAS reached and cost max_abs_err=89.
     *
     * Per-hart slots restore CORRECTNESS, not the memo hit rate: two harts
     * alternating the same op now each build their own table instead of
     * sharing one. Making that back into a shared cache would need real
     * synchronisation and is a separate, measurable change. */
    static struct mb_rvv_silu_memo mb_rvv_silu_memo_all[MB_RVV_SILU_WS_SLOTS];
    struct mb_rvv_silu_memo *const mb_memo = &mb_rvv_silu_memo_all[MB_RVV_SILU_WS_SLOT];
    int8_t *const lut = mb_memo->lut;

    if (!mb_memo->c_valid || scale_in != mb_memo->c_si || scale_out != mb_memo->c_so
                 || activation_min != mb_memo->c_lo || activation_max != mb_memo->c_hi) {
        for (int v = 0; v < 256; v++) {
            int8_t iv = (int8_t)(uint8_t)v;
            float f = (float)iv * scale_in;
            float y = f / (1.0f + expf(-f));
            int32_t q = (int32_t)roundf(y / scale_out);
            if (q < activation_min) q = activation_min;
            if (q > activation_max) q = activation_max;
            lut[v] = (int8_t)q;
        }
        mb_memo->c_si = scale_in; mb_memo->c_so = scale_out;
        mb_memo->c_lo = activation_min; mb_memo->c_hi = activation_max; mb_memo->c_valid = 1;
    }

    size_t vlmax = __riscv_vsetvlmax_e8m8();
    int i = 0;
    size_t vl;

    if (vlmax >= 256) {
        /* Whole 256-byte table resident in one e8m8 register group. */
        size_t tvl = __riscv_vsetvl_e8m8(256);
        vuint8m8_t vlut = __riscv_vle8_v_u8m8((const uint8_t *)lut, tvl);
        for (; i < n; i += vl) {
            vl = __riscv_vsetvl_e8m8(n - i);
            vuint8m8_t vidx = __riscv_vle8_v_u8m8((const uint8_t *)(input + i), vl);
            vuint8m8_t vout = __riscv_vrgather_vv_u8m8(vlut, vidx, vl);
            __riscv_vse8_v_u8m8((uint8_t *)(output + i), vout, vl);
        }
    } else {
        for (; i < n; i += vl) {
            vl = __riscv_vsetvl_e8m8(n - i);
            vuint8m8_t vidx = __riscv_vle8_v_u8m8((const uint8_t *)(input + i), vl);
            vuint8m8_t vout = __riscv_vluxei8_v_u8m8((const uint8_t *)lut, vidx, vl);
            __riscv_vse8_v_u8m8((uint8_t *)(output + i), vout, vl);
        }
    }
}
