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
#include <stdlib.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/reboot.h>

#include "model.h"
#include "test_io.h"

/* Opt-in TACIT / L-Trace of just the model inference. Enable by building with
 * -DMB_TACIT_TRACE_MODEL=1 (see CMakeLists.txt), then run on the TACIT-enabled
 * spike with `--trace=l`. Brackets only run_model() so the trace is the model's
 * control flow (not boot) -- preferable to CONFIG_STARTUP_TACIT for substantial
 * models. See samples/tacit/TACIT_TRACING.md. */
#if defined(MB_TACIT_TRACE_MODEL)
#include <tacit/tacit.h>
#include <zephyr/arch/cpu.h>
#endif

/* Opt-in HARDWARE TACIT capture of the inference via raw MMIO (independent of
 * the spike-oriented MB_TACIT_TRACE_MODEL / l_trace path above). Build with
 * -DMB_TACIT_HW=1. Programs the TACIT encoder controller + DMA trace sink to
 * stream the branch trace into a DDR region clear of the app/data, brackets
 * exactly model_run_test(), then disables + flushes the sink and reports the
 * byte count on the console. Register map matches the validated trace_test
 * recipe (encoder ctrl @0x3000000, DMA sink @0x3010000). */
#if defined(MB_TACIT_HW)
#ifndef MB_TACIT_TRACE_BUF
#define MB_TACIT_TRACE_BUF 0x90000000UL   /* 256 MiB into DDR, clear of app */
#endif
#define MB_TACIT_CTRL    0x3000000UL      /* +0x00 control: bit1=enable bit0=active */
#define MB_TACIT_TARGET  0x3000020UL      /* trace sink target id */
#define MB_TACIT_BPMODE  0x3000024UL      /* branch-predictor mode (0 = branch-target) */
#define MB_TACIT_DMA_FLUSH 0x3010000UL    /* write 1 -> flush FIFO */
#define MB_TACIT_DMA_DONE  0x3010004UL    /* read  1 -> flush drained */
#define MB_TACIT_DMA_START 0x3010008UL    /* 64-bit dma_start_addr */
#define MB_TACIT_DMA_BYTES 0x3010010UL    /* 64-bit addr_counter (bytes written) */
#define MB_MMIO32(a) (*(volatile unsigned int  *)(a))
#define MB_MMIO64(a) (*(volatile unsigned long *)(a))
#endif

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

    /* MODELBLASTER_PROFILE_ITERS: opt-in repeat loop for real-hardware
     * per-op profiling (board mcycle counts). Iteration 0 is a WARMUP
     * (I-cache/scratch/DMA-descriptor cold-start effects) and is tagged
     * as such on every printed line so the host-side consumer can
     * discard it and average the steady-state iterations that follow.
     * Default 1 (single-shot, identical to the original behavior) so
     * every other build of this harness (spike verify, single-run
     * FireSim, etc.) is unaffected unless this is explicitly overridden
     * at configure time (-DMODELBLASTER_PROFILE_ITERS=N). */
#ifndef MODELBLASTER_PROFILE_ITERS
#define MODELBLASTER_PROFILE_ITERS 1
#endif
    for (int mb_prof_iter = 0; mb_prof_iter < MODELBLASTER_PROFILE_ITERS; mb_prof_iter++) {
    printf("=== MODELBLASTER_PROFILE_ITER === %d %s\n", mb_prof_iter,
           mb_prof_iter == 0 ? "WARMUP" : "STEADY");

    /* Single-model harness has no thread pool — pass NULL. The
     * generated kernel bodies ignore it; only the parallel-for wrapper
     * (when emitted) would dispatch onto a real modelblaster_pool_t.
     * model_run_test() feeds the baked test input(s) — it is arity-agnostic
     * (1 input or N typed inputs), so this call is unchanged across models. */
#if defined(MB_TACIT_TRACE_MODEL)
    LTraceEncoderType *_tacit_enc = l_trace_encoder_get(arch_curr_cpu()->id);
    l_trace_encoder_configure_target(_tacit_enc, TARGET_PRINT);
    l_trace_encoder_start(_tacit_enc);
#endif
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

#if defined(MB_TACIT_HW)
    /* Program + enable the encoder immediately before the inference so the
     * FSync lands on a real inference PC and the trace holds only model
     * control flow (boot / verify / print are outside the window). */
    unsigned long mb_tacit_bytes = 0;
    MB_MMIO64(MB_TACIT_DMA_BYTES) = 0UL;               /* reset byte counter */
    MB_MMIO64(MB_TACIT_DMA_START) = MB_TACIT_TRACE_BUF; /* dma start addr */
    MB_MMIO32(MB_TACIT_TARGET)    = 1u;                 /* DMA sink (target id 1) */
    MB_MMIO32(MB_TACIT_BPMODE)    = 0u;                 /* branch-target mode */
    __asm__ volatile("fence" ::: "memory");
    MB_MMIO32(MB_TACIT_CTRL)      = 3u;                 /* enable=1, active=1 */
    __asm__ volatile("fence" ::: "memory");
#endif

    model_run_test(model_output, NULL);

#if defined(MB_TACIT_HW)
    __asm__ volatile("fence" ::: "memory");
    MB_MMIO32(MB_TACIT_CTRL) = 1u;                      /* disable tracing */
    __asm__ volatile("fence" ::: "memory");
    MB_MMIO32(MB_TACIT_DMA_FLUSH) = 1u;                 /* drain FIFO -> DDR */
    /* FIFO is 32 bytes; a few thousand cycles is plenty. Poll done with a
     * bounded spin (done_reg latches and cannot be cleared from SW, so also
     * fall through after the delay regardless). */
    for (volatile int _d = 0; _d < 200000; _d++) {
        if (MB_MMIO32(MB_TACIT_DMA_DONE) & 1u) break;
    }
    __asm__ volatile("fence" ::: "memory");
    mb_tacit_bytes = MB_MMIO64(MB_TACIT_DMA_BYTES);
#endif
#if defined(MB_TACIT_TRACE_MODEL)
    l_trace_encoder_stop(_tacit_enc);
    for (int _i = 0; _i < 16; _i++) { __asm__ volatile("nop"); } /* flush */
#endif
    /* RVV->scalar visibility barrier. The kernels write model_output via RVV
     * vector stores (vse/vsse); the verify loop below reads it with scalar
     * loads. On FireSim/Saturn's weak memory model the vector store buffer may
     * not have drained before the scalar reads -> stale/partial output ->
     * spurious miscompute (correct on spike, wrong on FireSim; worse in
     * complex kernels with more in-flight stores). Same root cause + fix as the
     * ExecuTorch riscv_executor_runner fence. */
    __asm__ volatile("fence rw, rw" ::: "memory");

#if MODELBLASTER_MASK_IRQ_DURING_RUN
    irq_unlock(mb_irq_key);
#endif

#if defined(MB_TACIT_HW)
    printf("=== MB_TACIT_HW === trace_buf=0x%lx bytes=%lu\n",
           (unsigned long)MB_TACIT_TRACE_BUF, mb_tacit_bytes);
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
    } /* end MODELBLASTER_PROFILE_ITERS loop */

#ifdef CONFIG_ARCH_POSIX
    /* native_sim: no HTIF/reboot — terminate the host process cleanly so the
     * native runner gets a clean exit (stdout already flushed above). */
    exit(0);
#else
    sys_reboot(SYS_REBOOT_COLD);
#endif
    return 0;
}
