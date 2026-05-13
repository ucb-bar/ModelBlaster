/*
 * Copyright (c) 2026 Dima Nikiforov <vnikiforov@berkeley.edu>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Agents harness entry point. Runs a generated DNN model on a fixed
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
#include <zephyr/sys/reboot.h>

#include "model.h"
#include "test_io.h"

static model_output_t model_output[MODEL_OUTPUT_SIZE];

int main(void)
{
    printf("agents harness: model=%s in=%d out=%d\n",
           MODEL_NAME, MODEL_INPUT_SIZE, MODEL_OUTPUT_SIZE);

    /* Single-model harness has no thread pool — pass NULL. The
     * generated kernel bodies ignore it; only the parallel-for wrapper
     * (when emitted) would dispatch onto a real agents_pool_t. */
    run_model(model_test_input, model_output, NULL);

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
    printf("=== AGENTS_VERIFY === max_abs_err=%.9g max_rel_err=%.9g n=%d\n",
           (double)max_abs_err, (double)max_rel_err, MODEL_TEST_OUTPUT_LEN);

    /* Optional: dump the raw model output buffer so the host can
     * compare in domain-specific units (e.g. waypoint coordinates for
     * a navigation policy, not int8 LSBs). Guarded on output size to
     * avoid blowing past FIRESIM_TIMEOUT on detection-class models
     * (yolov8's ~75k-element output would take ~10s over HTIF). The
     * Per-element format is printf("%.9g\n", ...) — host parses one
     * float per line between BEGIN / END markers. */
#if !defined(AGENTS_DUMP_OUTPUT_MAX_ELEMS)
#define AGENTS_DUMP_OUTPUT_MAX_ELEMS 256
#endif
    if (MODEL_TEST_OUTPUT_LEN <= AGENTS_DUMP_OUTPUT_MAX_ELEMS) {
        printf("=== AGENTS_OUTPUT_BEGIN ===\n");
        for (int i = 0; i < MODEL_TEST_OUTPUT_LEN; i++) {
            printf("%.9g\n", (double)(float)model_output[i]);
        }
        printf("=== AGENTS_OUTPUT_END ===\n");
    }

    /* Per-kernel profile (rdcycle deltas, populated by run_model). */
    int n_records = 0;
    const model_op_record_t *records = model_profile_records(&n_records);
    printf("=== AGENTS_PROFILE_BEGIN ===\n");
    printf("dispatch_id,name,op,shape,cycles\n");
    for (int i = 0; i < n_records; i++) {
        printf("%d,%s,%s,%s,%lu\n",
               records[i].dispatch_id,
               records[i].name, records[i].op, records[i].shape,
               records[i].cycles);
    }
    printf("=== AGENTS_PROFILE_END ===\n");

    /* Wall-clock total for the run (k_cycle_get_64 / mtime delta). The
     * runner reads this line to get the cross-hart-correct number;
     * per-op rdcycle deltas above are used for relative comparisons. */
    printf("=== AGENTS_WALL_CYCLES === %lu\n", model_wall_cycles());

    sys_reboot(SYS_REBOOT_COLD);
    return 0;
}
