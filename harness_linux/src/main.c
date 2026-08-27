/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * ModelBlaster harness, Linux/POSIX flavour. Mirrors harness/src/main.c
 * marker-for-marker so validation/runner_common.py parses both identically --
 * the runner contract is the stdout protocol, and nothing downstream should be
 * able to tell which harness produced a log.
 *
 * Differences from the Zephyr harness, all forced by the platform:
 *
 *   k_thread_cpu_pin  -> sched_setaffinity. Pinning matters more here, not
 *                        less: profiles are per-core, and on the SpaceMiT K1
 *                        cores 0-3 carry the IME extension while 4-7 do not,
 *                        so an unpinned run can silently execute a cluster-0
 *                        kernel on cluster 1.
 *   sys_reboot        -> return. Rebooting a development board that other
 *                        people are logged into would be rude.
 *   rdcycle           -> handled in the generated model.c, not here: reading
 *                        the cycle CSR from userspace raises SIGILL on this
 *                        kernel, so generate_skeleton --platform linux emits
 *                        rdtime instead.
 */

#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "model.h"
#include "test_io.h"

static model_output_t model_output[MODEL_OUTPUT_SIZE];

/* MODELBLASTER_CPU=<id> pins the whole process to one core. Left unset, the
 * scheduler is free to migrate us, which turns a per-core profile into an
 * average over whatever cores were idle. */
static void pin_from_env(void)
{
    const char *s = getenv("MODELBLASTER_CPU");
    if (!s || !*s)
        return;
    char *end = NULL;
    long cpu = strtol(s, &end, 10);
    if (end == s || cpu < 0) {
        fprintf(stderr, "MODELBLASTER_CPU=%s is not a cpu id; not pinning\n", s);
        return;
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET((int)cpu, &set);
    if (sched_setaffinity(0, sizeof set, &set) != 0) {
        perror("sched_setaffinity");
        return;
    }
    sched_yield();
    printf("modelblaster harness: pinned to cpu %ld (running on %d)\n",
           cpu, sched_getcpu());
}

int main(void)
{
    pin_from_env();

    printf("modelblaster harness: model=%s in=%d out=%d\n",
           MODEL_NAME, MODEL_INPUT_SIZE, MODEL_OUTPUT_SIZE);

    run_model(model_test_input, model_output, NULL);

    /* In-binary golden compare, identical arithmetic to the Zephyr harness:
     * widen both sides to float so one loop covers f32/f16/int outputs, and
     * report global max abs / max rel error. The host gates on
     * (max_abs_err <= atol) || (max_rel_err <= rtol). */
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

#if !defined(MODELBLASTER_DUMP_OUTPUT_MAX_ELEMS)
#define MODELBLASTER_DUMP_OUTPUT_MAX_ELEMS 256
#endif
    if (MODEL_TEST_OUTPUT_LEN <= MODELBLASTER_DUMP_OUTPUT_MAX_ELEMS) {
        printf("=== MODELBLASTER_OUTPUT_BEGIN ===\n");
        for (int i = 0; i < MODEL_TEST_OUTPUT_LEN; i++)
            printf("%.9g\n", (double)(float)model_output[i]);
        printf("=== MODELBLASTER_OUTPUT_END ===\n");
    }

    int n_records = 0;
    const model_op_record_t *records = model_profile_records(&n_records);
    printf("=== MODELBLASTER_PROFILE_BEGIN ===\n");
    printf("dispatch_id,name,op,shape,cycles\n");
    for (int i = 0; i < n_records; i++) {
        printf("%d,%s,%s,%s,%lu\n",
               records[i].dispatch_id, records[i].name, records[i].op,
               records[i].shape, records[i].cycles);
    }
    printf("=== MODELBLASTER_PROFILE_END ===\n");

    /* NOTE ON UNITS: on Linux these counts are rdtime ticks, a fixed 24 MHz on
     * the K1, not core cycles -- rdcycle is unavailable from userspace. The
     * field name is kept for protocol compatibility; convert with
     * PROFILE_CLOCK_MHZ=24, not with the core clock. */
    printf("=== MODELBLASTER_WALL_CYCLES === %lu\n", model_wall_cycles());
    return 0;
}
