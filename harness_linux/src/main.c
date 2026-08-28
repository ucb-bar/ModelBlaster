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

    /* Repeat count. Two independent reasons, and the second is not obvious.
     *
     * 1. A single cold measurement is not a profile. The accept/reject
     *    criterion this feeds requires a median over warm repetitions, and two
     *    closed-loop candidates in this project were previously rejected on
     *    n=1 samples. The gaps were large enough that the conclusions stand,
     *    but the stated criterion was not met.
     *
     * 2. It is the only way to OBSERVE recurrent state. A stateful model --
     *    VitFly's LSTM -- keeps h_state/c_state in file-scope arrays that
     *    nothing resets, so invocation k consumes what k-1 wrote. With a
     *    single invocation that claim rests entirely on the C storage class;
     *    an arena change or a zero-init flag would break it silently and no
     *    test would notice. Running twice makes it a measurement: for a
     *    stateful model the outputs MUST differ between iterations, and for a
     *    stateless one they must be identical.
     *
     * Default 1 so every existing caller and golden comparison is unchanged. */
    long iters = 1;
    {
        const char *e = getenv("MODELBLASTER_ITERS");
        if (e && *e) {
            long v = strtol(e, NULL, 10);
            if (v > 0) iters = v;
        }
    }

    for (long it = 0; it < iters; it++) {
        run_model(model_test_input, model_output, NULL);
        if (iters > 1) {
            /* One block per iteration so the host can compare them. A
             * stateful model's outputs diverge across iterations by design;
             * printing only the last would hide exactly that. */
            printf("=== MODELBLASTER_ITER_BEGIN [%ld] ===\n", it);
            int dump = MODEL_TEST_OUTPUT_LEN <= 64 ? MODEL_TEST_OUTPUT_LEN : 64;
            for (int i = 0; i < dump; i++)
                printf("%.9g\n", (double)(float)model_output[i]);
            printf("=== MODELBLASTER_ITER_END [%ld] ===\n", it);

            int n_rec_it = 0;
            const model_op_record_t *rec_it = model_profile_records(&n_rec_it);
            printf("=== MODELBLASTER_ITER_PROFILE_BEGIN [%ld] ===\n", it);
            printf("dispatch_id,name,op,shape,cycles\n");
            for (int i = 0; i < n_rec_it; i++)
                printf("%d,%s,%s,%s,%lu\n",
                       rec_it[i].dispatch_id, rec_it[i].name, rec_it[i].op,
                       rec_it[i].shape, rec_it[i].cycles);
            printf("=== MODELBLASTER_ITER_PROFILE_END [%ld] ===\n", it);
            printf("=== MODELBLASTER_ITER_WALL [%ld] === %lu\n",
                   it, model_wall_cycles());
        }
    }

    /* In-binary golden compare of the LAST iteration, identical arithmetic to
     * the Zephyr harness. For a stateless model every iteration produces the
     * same outputs so this is unambiguous; for a stateful one the golden
     * describes iteration 0, so MODELBLASTER_ITERS>1 is a state-persistence
     * probe rather than a correctness run and the per-iteration blocks above
     * are what to read.
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
