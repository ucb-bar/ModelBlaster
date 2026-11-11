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
 *
 * MULTI-CORE. Built with -DMODELBLASTER_USE_POOL this creates a
 * modelblaster_pool and hands it to run_model, which is what makes the
 * generated `parallel_<op>` wrappers take their pool arm instead of falling
 * through to the serial call. Without the define the third argument stays
 * NULL and the binary is byte-for-byte what it was -- so enabling multi-core
 * cannot silently re-measure the single-core profile with a different build.
 */

#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "model.h"
#include "test_io.h"

#ifdef MODELBLASTER_USE_POOL
#include "modelblaster_pool.h"
#endif
#include "mb_placement.h"

static model_output_t model_output[MODEL_OUTPUT_SIZE];

/* MODELBLASTER_CPU names the CPUs this run uses. Accepts one id ("3"), a list
 * ("0,1,2,3") or an inclusive range ("0-3"); ORDER MATTERS: cpu[0] is the
 * calling (main) thread, which runs slice 0 of every parallel-for, and
 * cpu[i] is pool helper i.
 *
 * WHY AN ORDERED LIST AND NOT A MASK. It used to be a process-wide mask that
 * the pool's helpers inherited, with modelblaster_pool_create(n) then
 * putting helpers on CPUs 1..n-1 regardless of it -- so MB_CORES=8-11 ran
 * its helpers on 1-3 while the topo tag said 8-11. That does not fail; it
 * produces a number. Now the list is handed to
 * modelblaster_pool_create_on_harts() verbatim, and the main thread places
 * itself with mb_placement (which, on a K3 A100 CPU, registers the thread
 * with /proc/set_ai_thread first -- fine here: this single-model profiling
 * harness's main thread IS slice 0, so it is meant to be an A100 thread).
 *
 * Returns how many CPUs are in the list (0 = MODELBLASTER_CPU unset, nothing
 * pinned). A malformed list is fatal for the same reason a failed placement
 * is: running "somewhere" and filing it under the requested CPUs is the
 * bookkeeping fiction this project exists to keep out of the cost database.
 */
#define MAX_CPUS 16

static int parse_cpu_list(const char *s, int *cpus)
{
    int n = 0;
    const char *p = s;
    while (*p) {
        char *end = NULL;
        long lo = strtol(p, &end, 10);
        if (end == p) {
            fprintf(stderr, "MODELBLASTER_CPU=%s is not a cpu list\n", s);
            return -1;
        }
        long hi = lo;
        p = end;
        if (*p == '-') {
            p++;
            hi = strtol(p, &end, 10);
            if (end == p) {
                fprintf(stderr, "MODELBLASTER_CPU=%s: range with no end\n", s);
                return -1;
            }
            p = end;
        }
        if (lo < 0 || hi < lo) {
            fprintf(stderr, "MODELBLASTER_CPU=%s: bad range\n", s);
            return -1;
        }
        for (long c = lo; c <= hi; c++) {
            for (int i = 0; i < n; i++)
                if (cpus[i] == (int)c) goto dup;
            if (n >= MAX_CPUS) {
                fprintf(stderr, "MODELBLASTER_CPU=%s: more than %d cpus\n",
                        s, MAX_CPUS);
                return -1;
            }
            cpus[n++] = (int)c;
dup:        ;
        }
        while (*p == ',' || *p == ' ') p++;
    }
    return n;
}

static int pin_from_env(int *cpus)
{
    const char *s = getenv("MODELBLASTER_CPU");
    if (!s || !*s)
        return 0;
    int n = parse_cpu_list(s, cpus);
    if (n < 0)
        exit(3);
    if (n == 0)
        return 0;
    if (mb_place_current_thread(cpus[0]) != 0) {
        fprintf(stderr, "modelblaster harness: cannot place main thread on "
                        "cpu %d (MODELBLASTER_CPU=%s)\n", cpus[0], s);
        exit(3);
    }
    printf("modelblaster harness: pinned to cpu %s (%d core(s), running on %d)\n",
           s, n, sched_getcpu());
    return n;
}

int main(void)
{
    int cpus[MAX_CPUS];
    int n_pinned = pin_from_env(cpus);

    printf("modelblaster harness: model=%s in=%d out=%d\n",
           MODEL_NAME, MODEL_INPUT_SIZE, MODEL_OUTPUT_SIZE);

    void *pool = NULL;
#ifdef MODELBLASTER_USE_POOL
    /* Thread count comes from the pin mask unless overridden. One source of
     * truth: a separate MODELBLASTER_POOL_THREADS that disagreed with
     * MODELBLASTER_CPU would produce a run whose topo tag says one width and
     * whose pool is another, and the profile tree records the tag. */
    long threads = n_pinned > 0 ? n_pinned : 1;
    {
        const char *e = getenv("MODELBLASTER_POOL_THREADS");
        if (e && *e) {
            long v = strtol(e, NULL, 10);
            /* The helpers go on the listed CPUs, so a different count has no
             * CPUs to run on (or leaves some unused). Guessing which is the
             * failure this override used to hide; refuse instead. */
            if (n_pinned > 0 ? v != n_pinned : v < 1) {
                fprintf(stderr, "modelblaster harness: MODELBLASTER_POOL_THREADS="
                                "%s disagrees with MODELBLASTER_CPU (%d cpus); "
                                "refusing to guess\n", e, n_pinned);
                return 3;
            }
            threads = v;
        }
    }
    if (threads > 1) {
        pool = (void *)(n_pinned > 0
            ? modelblaster_pool_create_on_harts((int)threads, cpus)
            : modelblaster_pool_create((int)threads));
        if (pool == NULL) {
            /* Loud, and fatal. A NULL pool is the wrappers' "run serially"
             * signal, so a silent fallback would file a serial measurement
             * under a multi-core topo tag -- the exact shape of bookkeeping
             * fiction this project exists to keep out of the cost database. */
            fprintf(stderr, "modelblaster harness: pool_create(%ld) FAILED; "
                            "refusing to record a serial run as multi-core\n",
                    threads);
            return 3;
        }
        printf("modelblaster harness: pool of %ld thread(s)\n", threads);
    } else {
        printf("modelblaster harness: no pool (1 core)\n");
    }
#else
    (void)n_pinned;
#endif

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
        run_model(model_test_input, model_output, pool);
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

#ifdef MODELBLASTER_USE_POOL
    if (pool)
        modelblaster_pool_destroy((modelblaster_pool_t)pool);
#endif
    return 0;
}
