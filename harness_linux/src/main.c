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
#include <errno.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "model.h"
#include "test_io.h"
#include "mb_ai_task.h"

#ifdef MODELBLASTER_USE_POOL
#include "modelblaster_pool.h"
#endif

static model_output_t model_output[MODEL_OUTPUT_SIZE];

/* MODELBLASTER_CPU pins the process. Accepts one id ("3"), a list ("0,1,2,3")
 * or an inclusive range ("0-3").
 *
 * WHY A SET AND NOT AN ID. It was one id, which is right for a single-core
 * profile and makes a multi-core one impossible: the pool's helper threads
 * inherit the creating process's affinity mask, so a mask of {0} would have
 * pinned every worker onto hart 0 and the "4-core" run would have been four
 * threads time-slicing one core. That does not fail -- it produces a number,
 * and the number looks like poor scaling.
 *
 * Returns how many cpus are in the mask, so the pool can be sized from the
 * same source of truth rather than from a second environment variable that
 * can disagree with it.
 */
/* Upper bound on cpus in MODELBLASTER_CPU; the pool's own limit is 16. */
#define MB_MAX_HARTS 64

static MB_THREAD_MOVER int pin_from_env(int *harts, int max_harts)
{
    const char *s = getenv("MODELBLASTER_CPU");
    if (!s || !*s)
        return 0;

    cpu_set_t set;
    CPU_ZERO(&set);
    int n = 0;
    const char *p = s;
    while (*p) {
        char *end = NULL;
        long lo = strtol(p, &end, 10);
        if (end == p) {
            fprintf(stderr, "MODELBLASTER_CPU=%s is not a cpu list; "
                            "not pinning\n", s);
            return 0;
        }
        long hi = lo;
        p = end;
        if (*p == '-') {
            p++;
            hi = strtol(p, &end, 10);
            if (end == p) {
                fprintf(stderr, "MODELBLASTER_CPU=%s: range with no end; "
                                "not pinning\n", s);
                return 0;
            }
            p = end;
        }
        if (lo < 0 || hi < lo) {
            fprintf(stderr, "MODELBLASTER_CPU=%s: bad range; not pinning\n", s);
            return 0;
        }
        for (long c = lo; c <= hi; c++) {
            if (c >= CPU_SETSIZE) continue;
            if (!CPU_ISSET((int)c, &set)) {
                CPU_SET((int)c, &set);
                if (n < max_harts) harts[n] = (int)c;   /* request order, no dupes */
                n++;
            }
        }
        while (*p == ',' || *p == ' ') p++;
    }
    if (n == 0)
        return 0;
    /* K3 only: cpus 8-15 are A100 cores and an ordinary thread is not allowed
     * on them. The thread must be labelled an AI task BEFORE it is pinned
     * there (see mb_ai_task.h). Failing to label is fatal: carrying on would
     * measure whatever core the kernel did give us and file it under the
     * A100's topo tag. */
    for (int i = 0; i < n && i < max_harts; i++) {
        if (mb_ai_task_wanted(harts[i])) {
            if (mb_ai_task_label_self() != 0) {
                fprintf(stderr, "modelblaster harness: cannot label this thread an "
                                "AI task via %s: %s\n", MB_AI_TASK_FILE, strerror(errno));
                exit(4);
            }
            printf("modelblaster harness: thread labelled AI task (A100 cores)\n");
            break;
        }
    }
    if (sched_setaffinity(0, sizeof set, &set) != 0) {
        perror("sched_setaffinity");
        return 0;
    }
    sched_yield();
    printf("modelblaster harness: pinned to cpu %s (%d core(s), running on %d)\n",
           s, n, sched_getcpu());
    return n;
}

/* MB_THREAD_MOVER: main is on the stack while pin_from_env moves the thread
 * onto the A100s (mb_ai_task.h). */
MB_THREAD_MOVER int main(void)
{
    int harts[MB_MAX_HARTS];
    int n_pinned = pin_from_env(harts, MB_MAX_HARTS);

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
            if (v > 0) threads = v;
        }
    }
    if (threads > 1) {
        if (threads == n_pinned && n_pinned <= MB_MAX_HARTS) {
            /* Helpers go on the cpus that were asked for, in order, and the
             * master takes the first. modelblaster_pool_create(n) alone pins
             * helper w to cpu w whatever MODELBLASTER_CPU says, and leaves the
             * master free to sit on a helper's cpu: two workers on one core,
             * time doubles, and the run still verifies. That is how
             * MB_CORES=0,1,2,3 gave 41,993 on one run and 82,624 on the next,
             * and why MB_CORES=4,5,6,7 ran helpers on cpus 1-3. It also cannot
             * work at all for the A100 set (cpus 8-15). */
            cpu_set_t first;
            CPU_ZERO(&first);
            CPU_SET(harts[0], &first);
            if (sched_setaffinity(0, sizeof first, &first) != 0)
                perror("sched_setaffinity (master)");
            /* Evidence, not intent: the master also runs every layer the pool does not
             * split, so say where it actually is now that it is narrowed to harts[0]. */
            printf("modelblaster harness: master pinned to cpu %d (running on %d)\n",
                   harts[0], sched_getcpu());
            pool = (void *)modelblaster_pool_create_on_harts((int)threads, harts);
        } else {
            /* MODELBLASTER_POOL_THREADS disagrees with the mask: keep the old
             * 0..n-1 layout, but say so rather than file it under a topo tag
             * that implies otherwise. */
            fprintf(stderr, "modelblaster harness: pool width %ld != %d pinned cpus; "
                            "using legacy cpu 0..%ld mapping\n", threads, n_pinned, threads - 1);
            pool = (void *)modelblaster_pool_create((int)threads);
        }
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
        if (threads == n_pinned && n_pinned <= MB_MAX_HARTS) {
            /* Print where the workers went, so a wrong placement can never again
             * hide behind a plausible-looking number. */
            printf("modelblaster harness: pool cpus:");
            for (int i = 0; i < n_pinned; i++) printf(" %d", harts[i]);
            printf(" (master on %d)\n", harts[0]);
        }
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

#ifdef MODELBLASTER_USE_POOL
    /* Placement audit (opt-in, MODELBLASTER_POOL_TRACE=1): arm the pool's slice trace so
     * every slice of every split layer records the cpu it really ran on (sched_getcpu).
     * The pool cpus printed above say where workers were asked to run; this says where
     * the work ran. Off by default: no allocation, no extra reads. */
    struct modelblaster_pool_shard *mb_audit = NULL;
    unsigned mb_audit_cap = 0;
    {
        const char *e = getenv("MODELBLASTER_POOL_TRACE");
        if (pool && e && *e && *e != '0') {
            mb_audit_cap = 1u << 20;
            mb_audit = calloc(mb_audit_cap, sizeof *mb_audit);
            if (mb_audit == NULL) {
                fprintf(stderr, "modelblaster harness: pool trace buffer allocation failed\n");
                return 3;
            }
            modelblaster_pool_trace_arm((modelblaster_pool_t)pool, mb_audit, mb_audit_cap);
        }
    }
#endif

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
    if (pool && threads == n_pinned)
        printf("modelblaster harness: master finished on cpu %d\n", sched_getcpu());
    if (mb_audit) {
        /* One line per worker: the cpus its slices ran on and how many. Worker w was
         * placed on harts[w] (master = worker 0 on harts[0]). FAIL if any slice ran
         * elsewhere or if the buffer filled up. A worker with no slice is reported
         * (idle_workers), not failed: a model may have nothing the pool splits. */
        unsigned n = modelblaster_pool_trace_count((modelblaster_pool_t)pool);
        int ok = n < mb_audit_cap && threads == n_pinned;
        int idle = 0;
        printf("=== MODELBLASTER_POOL_AUDIT_BEGIN === slices=%u cap=%u workers=%ld\n",
               n, mb_audit_cap, threads);
        for (long w = 0; w < threads; w++) {
            unsigned cnt = 0, off = 0;
            int seen[MB_MAX_HARTS + 64];
            int nseen = 0;
            for (unsigned i = 0; i < n; i++) {
                if (mb_audit[i].wid != (int)w) continue;
                cnt++;
                int h = mb_audit[i].hart, dup = 0;
                if (threads == n_pinned && h != harts[w]) off++;
                for (int k = 0; k < nseen; k++) dup |= seen[k] == h;
                if (!dup && nseen < (int)(sizeof seen / sizeof seen[0])) seen[nseen++] = h;
            }
            printf("worker %ld expected_cpu %d slices %u off_cpu %u cpus", w,
                   threads == n_pinned ? harts[w] : -1, cnt, off);
            for (int k = 0; k < nseen; k++) printf(" %d", seen[k]);
            printf("\n");
            if (off != 0) ok = 0;
            if (cnt == 0) idle++;
        }
        printf("=== MODELBLASTER_POOL_AUDIT_END === %s idle_workers=%d\n", ok ? "PASS" : "FAIL", idle);
        modelblaster_pool_trace_arm((modelblaster_pool_t)pool, NULL, 0);
        free(mb_audit);
    }
    if (pool)
        modelblaster_pool_destroy((modelblaster_pool_t)pool);
#endif
    return 0;
}
