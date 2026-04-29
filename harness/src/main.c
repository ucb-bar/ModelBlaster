/*
 * Copyright (c) 2026 Dima Nikiforov <vnikiforov@berkeley.edu>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Agents harness entry point. Runs a generated DNN model on a fixed test
 * input and prints the output tensor between marker lines for the host-side
 * spike_runner to parse.
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

    /* Print output tensor in a stable, machine-parseable format.
     * Use %.9g to round-trip f32 cleanly. */
    printf("=== AGENTS_OUTPUT_BEGIN ===\n");
    for (int i = 0; i < MODEL_OUTPUT_SIZE; i++) {
        printf("%.9g\n", (double)model_output[i]);
    }
    printf("=== AGENTS_OUTPUT_END ===\n");

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
     * spike_runner reads this line to get the cross-hart-correct number;
     * per-op rdcycle deltas above are used for relative comparisons. */
    printf("=== AGENTS_WALL_CYCLES === %lu\n", model_wall_cycles());

    sys_reboot(SYS_REBOOT_COLD);
    return 0;
}
