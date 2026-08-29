/*
 * Copyright (c) 2026 Dima Nikiforov <vnikiforov@berkeley.edu>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Modelblaster harness entry point. Runs a generated DNN model on a fixed
 * test input, validates the model output IN-BINARY against the baked-in
 * test_golden (already in rodata via test_io.S's .incbin), and prints
 * a single summary line for the host-side runner to parse.
 *
 * The on-device verify replaced an earlier per-element printf dump.
 * That dump was fast on spike (functional sim, fast HTIF UART) but
 * dominated FireSim runtime on real RTL (4k+ floats per bench at one
 * line each, several minutes per run). The summary path keeps spike
 * fast and lets FireSim re-rank candidates in seconds instead of
 * blowing the timeout.
 */

#include <stdio.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/reboot.h>

#include "model.h"
#include "test_io.h"

static model_output_t model_output[MODEL_OUTPUT_SIZE];

int main(void)
{
#if defined(CONFIG_SMP) && defined(CONFIG_RISCV_ISA_EXT_V) && (CONFIG_MP_MAX_NUM_CPUS > 1)
    /* Pin main to hart 1 on hetero bitstreams (GemminiAndOPUShuttleConfig,
     * chipyard_hetero_q31, firesim_rocket_saturn) — hart 0 boots without
     * V hardware on these, so running a V-using generated kernel from
     * the boot hart would trap on the first vsetvli. Hart 1 has V (+
     * Saturn OPU matrix). Pinning is a no-op on single-hart builds
     * (e.g. spike) because the guard requires MP_MAX_NUM_CPUS > 1.
     *
     * Requires CONFIG_SCHED_CPU_MASK_PIN_ONLY=y in the FireSim overlay
     * (harness/backends/firesim_chipyard.conf) — already set. */
    k_thread_cpu_pin(k_current_get(), 1);
#endif

    printf("modelblaster harness: model=%s in=%d out=%d\n",
           MODEL_NAME, MODEL_INPUT_SIZE, MODEL_OUTPUT_SIZE);

    /* Single-model harness has no thread pool — pass NULL. The
     * generated kernel bodies ignore it; only the parallel-for wrapper
     * (when emitted) would dispatch onto a real modelblaster_pool_t.
     * model_run_test() feeds the baked test input(s) — it is arity-agnostic
     * (1 input or N typed inputs), so this call is unchanged across models. */
    /* ------------------------------------------------------------------
     * MODELBLASTER_MASK_IRQ_DURING_RUN: run the whole inference with
     * machine interrupts masked.
     *
     * WHY. On the Saturn RVV configs in this tree (measured on
     * f2_dual_small_norose_tacit_q31_60mhz, VLEN=256/dLen=128) a trap
     * taken while a vector kernel is executing can come back with
     * EXACTLY ONE scalar register corrupted. It is deterministic and
     * bit-reproducible, and it is invisible on spike. Two shipped fp16
     * conv kernels were blamed for it before the trap path was:
     *
     *   ViNT depthwise_conv2d_f16 : mcause 5, mtval 0x7aa61300c on a
     *     `vlse16.v v2,(a0),t6` whose a0 the kernel's own arithmetic
     *     provably cannot compute -- every other register in the frame
     *     is exactly right for that iteration (fq jobs 248/249/271).
     *   ViNT conv2d_f16           : mcause 4, mtval 3 on the same kind
     *     of strided weight gather (fq jobs 248/290/292).
     *
     * Masking MIE for the duration of the run removes both, with the
     * model otherwise byte-identical (fq jobs 283/285). The kernels
     * themselves are clean: all 81 ViNT conv shapes run fault-free in
     * isolation on the same bitstream (fq jobs 270/296).
     *
     * COST. No preemption and no tick servicing for the duration of one
     * model_run_test(). This harness is single-threaded and main() is
     * pinned to one hart, so nothing else is runnable; k_cycle_get_64()
     * reads free-running mtime and stays correct, and the per-op rdcycle
     * profile gets cleaner (no ISR time folded in). A harness that runs
     * a thread pool or several models concurrently must NOT do this --
     * hence the switch rather than an unconditional lock.
     * ------------------------------------------------------------------ */
#if !defined(MODELBLASTER_MASK_IRQ_DURING_RUN)
#define MODELBLASTER_MASK_IRQ_DURING_RUN 1
#endif
#if MODELBLASTER_MASK_IRQ_DURING_RUN
    unsigned int mb_irq_key = irq_lock();
#endif

    model_run_test(model_output, NULL);

#if MODELBLASTER_MASK_IRQ_DURING_RUN
    irq_unlock(mb_irq_key);
#endif

    /* In-binary golden compare.
     *
     * Both `model_output[i]` and `model_test_golden[i]` are widened to
     * float for the comparison so the same loop body works for f32,
     * f16, and integer outputs. We track the global max absolute error
     * and max relative error; the host gates on
     *
     *   (max_abs_err <= atol) || (max_rel_err <= rtol)
     *
     * which is a sufficient PASS condition for numpy.allclose-style
     * tolerance — every element satisfies at least one bound globally,
     * so it satisfies them element-wise too. Conservative vs the
     * per-element threshold but correct, and avoids shipping the full
     * tensor over UART. */
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    for (int i = 0; i < MODEL_TEST_OUTPUT_LEN; i++) {
        float a = (float)model_output[i];
        float g = (float)model_test_golden[i];
        float ae = a > g ? a - g : g - a;
        float ag = g > 0.0f ? g : -g;
        float re = ae / (ag > 1e-12f ? ag : 1e-12f);
        if (ae > max_abs_err) max_abs_err = ae;
        if (re > max_rel_err) max_rel_err = re;
    }
    printf("=== MODELBLASTER_VERIFY === max_abs_err=%.9g max_rel_err=%.9g n=%d\n",
           (double)max_abs_err, (double)max_rel_err, MODEL_TEST_OUTPUT_LEN);

    /* Optional: dump the raw model output buffer so the host can
     * compare in domain-specific units (e.g. waypoint coordinates for
     * a navigation policy, not int8 LSBs). Guarded on output size to
     * avoid blowing past FIRESIM_TIMEOUT on detection-class models
     * (yolov8's ~75k-element output would take ~10s over HTIF). The
     * Per-element format is printf("%.9g\n", ...) — host parses one
     * float per line between BEGIN / END markers. */
#if !defined(MODELBLASTER_DUMP_OUTPUT_MAX_ELEMS)
#define MODELBLASTER_DUMP_OUTPUT_MAX_ELEMS 256
#endif
    if (MODEL_TEST_OUTPUT_LEN <= MODELBLASTER_DUMP_OUTPUT_MAX_ELEMS) {
        printf("=== MODELBLASTER_OUTPUT_BEGIN ===\n");
        for (int i = 0; i < MODEL_TEST_OUTPUT_LEN; i++) {
            printf("%.9g\n", (double)(float)model_output[i]);
        }
        printf("=== MODELBLASTER_OUTPUT_END ===\n");
    }

    /* Per-kernel profile (rdcycle deltas, populated by run_model). */
    int n_records = 0;
    const model_op_record_t *records = model_profile_records(&n_records);
    printf("=== MODELBLASTER_PROFILE_BEGIN ===\n");
    printf("dispatch_id,name,op,shape,cycles\n");
    for (int i = 0; i < n_records; i++) {
        printf("%d,%s,%s,%s,%lu\n",
               records[i].dispatch_id,
               records[i].name, records[i].op, records[i].shape,
               records[i].cycles);
    }
    printf("=== MODELBLASTER_PROFILE_END ===\n");

    /* Wall-clock total for the run (k_cycle_get_64 / mtime delta). The
     * runner reads this line to get the cross-hart-correct number;
     * per-op rdcycle deltas above are used for relative comparisons. */
    printf("=== MODELBLASTER_WALL_CYCLES === %lu\n", model_wall_cycles());

    sys_reboot(SYS_REBOOT_COLD);
    return 0;
}
