"""Emit a multi-network main.c that walks an XPU-RT dispatch table.

Companion to ``ingest_xpurt_schedule`` (which produces the C dispatch
table). Together they close the round trip:

    XPU-RT schedule.json
        -> ingest_xpurt_schedule  -> <name>_dispatch_table.{h,c}
        -> generate_xpurt_main    -> <name>_main.c
        -> west build (harness_xpurt)
        -> spike

The harness can link multiple HW backends per model (e.g. scalar + rvv);
the walker dispatches each entry to whichever backend's per-model table
matches the entry's ``impl``. Symbol disambiguation is handled at
compile time by ``backend_rename.py`` — each model's externally-visible
symbols (kernels, dispatch table, profile API) get suffixed with
``_<bs>`` per backend.

The generated main:

    1. Creates one persistent modelblaster_pool for every distinct composite
       hart set selected by the schedule. Singleton dispatches need no pool.
    2. Resets every involved (model, backend) profile counter.
    3. Spawns one worker per (core_kind, hart) pair the schedule
       actually uses, pinned to that hart. Each worker walks the table
       in start-time order taking only entries that match BOTH its kind
       and its hart. Entries with hart == -1 (unbound) are claimed by
       the kind's designated worker -- the one sitting on the kind's
       lowest-numbered hart -- so nothing is ever dropped.
    4. Inside the dispatch branch, selects the right per-backend table
       by ``impl`` and invokes ``MODEL_<UMID>_DISPATCH_FNS_<BS>``.

       ``impl`` is the entry's KERNEL IMPLEMENTATION, which the ingest
       defaults to ``core_kind``. They differ only when the schedule was
       solved with ``scheduler.enable_impls`` and chose per-dispatch
       implementations -- "this GEMM on the MAC unit, the next on the
       vector unit", on the same core. ``hardware_target`` cannot say
       that: it names WHERE, and ``impl`` names WITH WHAT. Selecting on
       ``impl`` rather than ``core_kind`` is what makes a heterogeneous
       schedule mean something at run time instead of being a claim the
       binary quietly ignores. Dispatch-local state binds the exact composite
       pool for that entry.
    5. Tracks per-network wall cycles inline (no model-side setter
       call — keeps wall-cycle bookkeeping backend-agnostic).
    6. Captures instance 0 only after every dispatch in that DAG has finished,
       compares it with the baked one-invocation golden, and prints the
       standard output and per-backend profile records. Capturing instance 0
       matters for recurrent models whose later periodic outputs carry state.
"""

from __future__ import annotations

import argparse
import json
import os
import re

# Match the ingest module's heuristic for splitting "<network><instance>".
# Falls back to the trailing-digits regex when no known set is provided.
# When known networks are available, longest-prefix match wins — this
# disambiguates model names ending in digits (e.g. "yolov8_nano_640"
# → ("yolov8_nano_64", 0), not ("yolov8_nano_", 640)).
_INSTANCE_RE = re.compile(r"^(?P<base>.+?)(?P<idx>\d+)$")


# Platform prologue. The walker body is identical on both platforms -- it was
# already written against pthreads with pthread_attr_setaffinity_np -- so the
# only Zephyr coupling is the timer, the semaphores, the abort path, and two
# CONFIG_ constants. Defining the Zephyr names in terms of POSIX ones keeps a
# single body rather than forking ~900 lines of generator.
_PROLOGUE_ZEPHYR = """#include <zephyr/kernel.h>      /* k_cycle_get_64, k_sem */
#include <zephyr/sys/reboot.h>

#define XPURT_TICKS_PER_SEC  CONFIG_SYS_CLOCK_HW_CYCLES_PER_SEC
#define XPURT_BOARD_TARGET   CONFIG_BOARD_TARGET"""

_PROLOGUE_LINUX = """#define MODELBLASTER_PLATFORM_LINUX 1
#include "mb_posix_compat.h"   /* k_cycle_get_64, k_sem, sys_reboot on POSIX */
#include "mb_placement.h"      /* mb_place_current_thread: K3 A100-aware pinning */

#define XPURT_TICKS_PER_SEC  MB_POSIX_TICKS_PER_SEC
#define XPURT_BOARD_TARGET   "linux/riscv64"
"""

_PROLOGUES = {"zephyr": _PROLOGUE_ZEPHYR, "linux": _PROLOGUE_LINUX}

# Emitted BEFORE any #include. glibc gates cpu_set_t / CPU_ZERO / CPU_SET and
# pthread_attr_setaffinity_np behind _GNU_SOURCE, and <sched.h> latches the
# decision the first time it is included -- so this cannot live in the
# platform prologue, which lands after the include block. Zephyr gets an
# empty string: its affinity API is Kconfig-gated, not feature-test-gated,
# and defining _GNU_SOURCE there would be a no-op at best.
_TOP_LINUX = "#define _GNU_SOURCE 1  /* cpu_set_t, pthread_attr_setaffinity_np */\n"
_TOPS = {"zephyr": "", "linux": _TOP_LINUX}


def _warmup_blocks(upper: str, warm_branches: list[str],
                   warm_restore: list[str]) -> tuple[str, str, str, str]:
    """C fragments for the optional pre-t0 warm-up pass (Linux only).

    WHY: the first execution of a dispatch on a core pays for page faults on
    its weights, cold L1/L2/icache and, for a composite entry, waking the pool
    helpers. Measured on the K3 baseline run: dronet instance 0 dispatches 3
    and 8 took 3.8 / 7.4 ms against ~0.5 / 1 ms warm, and mlp_control missed 4
    deadlines, all because the first instance of each network ran cold INSIDE
    the timed window. The schedule's predicted durations are warm numbers.

    HOW:
      * It runs on the same worker (hence the same hart / cache / pool) that
        will run the entry, because caches and pool helpers are per core --
        warming from main would warm the wrong core.
      * Strictly serialised in table order through a ticket (g_warm_cursor).
        Entry ids are a topological order, so every dispatch reads inputs its
        producers already wrote (no garbage activations, no data race between
        workers on the shared per-model buffers).
      * One run per distinct (core_kind, hart, network, dispatch_id, impl,
        hart set): later instances of a network hit the same code and weights.
      * Two pthread barriers (workers + main): barrier 1 = everybody warmed;
        main then restores state and captures run_t0; barrier 2 releases the
        workers, which read run_t0 only after it. So t0 is taken after warm-up
        and the warm-up never appears in the trace (it bypasses the trace,
        hart accounting, wall counters and completion sems).

    STATE RESTORED before t0: every per-model intermediate buffer is zeroed
    (model_<m>_zero_buffers, from buffers.c: they are .bss, so zero is exactly
    their initial state -- this is what resets stateful networks such as the
    lstm h/c pair, which live in those buffers), the per-network output
    buffers are zeroed, and the profile counters are reset. The golden INPUTS
    need no re-staging: dispatch state points at the const model_<m>_test_input
    in rodata, which no kernel writes. Completion sems, wall_start_*,
    completed_dispatches_* are never touched by the warm-up.
    """
    branches = "\n".join(warm_branches)
    restore = "\n".join(warm_restore)
    defs = r'''
/* ---- warm-up pass (--warmup): see _warmup_blocks in generate_xpurt_main.py */
static pthread_barrier_t g_warm_b1, g_warm_b2;
static int g_warm_order[@UPPER@_N_ENTRIES];
static int g_n_warm;
static int g_warm_cursor;

static void xpurt_warm_entry(const xpurt_sched_entry_t *e_)
{
@BRANCHES@
    printf("xpurt: WARN warmup: unknown network %s in entry %d\n",
           e_->network, e_->entry_id);
}

/* The worker-claim predicate, same as the walker's. */
static int warm_claims(const struct xpurt_worker_arg *wa,
                       const xpurt_sched_entry_t *e_)
{
    if (strcmp(e_->core_kind, wa->kind) != 0) return 0;
    if (e_->hart >= 0) return e_->hart == wa->hart;
    return wa->claims_unbound;
}

static int warm_same_entry(const xpurt_sched_entry_t *a,
                           const xpurt_sched_entry_t *b)
{
    if (a->hart != b->hart || a->dispatch_id != b->dispatch_id ||
        a->n_harts != b->n_harts) return 0;
    for (int i = 0; i < a->n_harts; i++)
        if (a->harts[i] != b->harts[i]) return 0;
    return strcmp(a->core_kind, b->core_kind) == 0 &&
           strcmp(a->impl, b->impl) == 0 &&
           strcmp(a->network, b->network) == 0;
}
'''.replace("@UPPER@", upper).replace("@BRANCHES@", branches)

    worker = r'''
    /* Warm-up: runs after this worker placed itself (above) and before the
     * timed run. Serialised in table order via g_warm_cursor. */
    for (int k_ = 0; k_ < g_n_warm; k_++) {
        const xpurt_sched_entry_t *we_ = &@UPPER@_TABLE[g_warm_order[k_]];
        if (!warm_claims(wa, we_)) continue;
        while (__atomic_load_n(&g_warm_cursor, __ATOMIC_ACQUIRE) != k_)
            usleep(20);
        xpurt_warm_entry(we_);
        __atomic_store_n(&g_warm_cursor, k_ + 1, __ATOMIC_RELEASE);
    }
    pthread_barrier_wait(&g_warm_b1);   /* everyone warmed */
    pthread_barrier_wait(&g_warm_b2);   /* main restored state + set run_t0 */
'''.replace("@UPPER@", upper)

    pre = r'''
    /* Warm-up plan: the distinct entries, in table order. Built before any
     * worker exists so the list is immutable while they read it. */
    g_n_warm = 0;
    for (int i = 0; i < @UPPER@_N_ENTRIES; i++) {
        const xpurt_sched_entry_t *e = &@UPPER@_TABLE[i];
        if (e->dispatch_id < 0) continue;   /* zero-cost op: no kernel */
        int dup = 0;
        for (int j = 0; j < g_n_warm && !dup; j++)
            dup = warm_same_entry(&@UPPER@_TABLE[g_warm_order[j]], e);
        if (!dup) g_warm_order[g_n_warm++] = i;
    }
    if (pthread_barrier_init(&g_warm_b1, NULL, (unsigned)g_n_workers + 1) != 0 ||
        pthread_barrier_init(&g_warm_b2, NULL, (unsigned)g_n_workers + 1) != 0) {
        printf("FATAL: warm-up barrier init failed\n");
        sys_reboot(SYS_REBOOT_COLD);
        return -1;
    }
    uint64_t warm_t0 = (uint64_t)k_cycle_get_64();
'''.replace("@UPPER@", upper)

    post = r'''
    {
        pthread_barrier_wait(&g_warm_b1);   /* all workers warmed */
        uint64_t warm_t1 = (uint64_t)k_cycle_get_64();
@RESTORE@
        __asm__ volatile("fence rw, rw" ::: "memory");
        printf("xpurt: warmup done %d entries in %.3f ms\n", g_n_warm,
               (double)(warm_t1 - warm_t0) * 1000.0 / (double)XPURT_TICKS_PER_SEC);
        run_t0 = (uint64_t)k_cycle_get_64();   /* the REAL t=0 */
        pthread_barrier_wait(&g_warm_b2);
    }
'''.replace("@RESTORE@", restore)
    return defs, worker, pre, post


def _v2_blocks(upper: str, dispatch_branch: str, hyper_ms: float | None,
               rt_prio_default: int) -> tuple[str, str, str, str, str]:
    """C fragments for the v2 runtime (--walker skip, Linux only).

    Three changes against the strict walker, each with a run-time switch so it
    can be turned off on the same binary:

    1. NO HEAD-OF-LINE BLOCKING (XPURT_SKIP_WINDOW; 0 = whole table, the
       default; 1 = strict table order). The strict walker runs a worker's
       entries in table order and sleeps on the head entry's dependency
       semaphores; one late producer on another core then holds up every later
       entry of that worker, including entries of other networks that are
       ready. Here a worker keeps the list of its own entries, and at every
       step runs the FIRST entry in table order that is (a) eligible -- all
       data dependencies done and the previous instance of its network
       complete -- and (b) past its start gate. If none is runnable it sleeps
       until the earliest gate among its eligible entries, or until a producer
       posts its wake semaphore, whichever is first.

       What still orders execution:
         * data dependencies (deps[]), checked against per-entry done stamps;
         * instance order: an entry of instance k of a network waits for every
           entry of instance k-1 (the instances share the model's buffers; the
           strict walker relied on table order for this and had no check);
         * mutual exclusion on harts: every entry takes the locks of its hart
           set in numeric order, as before;
         * start gates: lower bounds, as before.
       time_dep_entry_id is NOT a dependency here. XPU-RT emits it as "the
       previous dispatch on the same hardware target", i.e. the same worker's
       previous table entry; honouring it would forbid exactly the skip this
       walker exists for. With XPURT_SKIP_WINDOW=1 it is implied by table
       order, as in the strict walker.

       DEADLOCK FREEDOM. Let G be the graph whose edges are the data
       dependencies plus an edge from every entry of (network, k-1) to every
       entry of (network, k). Data dependencies are intra-instance and acyclic
       (ingest topologically sorts them and refuses a cycle); instance edges
       only go from k-1 to k; so G is a DAG.

       Claim: while some entry of the cycle is unfinished, some entry gets
       run after a finite time. Finitely many entries then gives termination.

       Whole-table window (the default). Let U be the unfinished entries and
       m a G-minimal element of U. Every G-predecessor of m is done, so m is
       eligible, and it stays eligible: done stamps and instance counters only
       grow within a cycle. Let w be m's owner; m is in w's scan range. w is
       either executing an entry or in the scan/wait loop.
         * Executing: it blocks only on hart locks. Locks are taken in numeric
           order and held only around a kernel call that does not wait on
           anything, so the holder of the highest lock anyone waits for is
           running; lock waits end, the entry completes: progress.
         * Scanning: if the scan finds a runnable entry, w runs it: progress.
           If not, and the scan saw an eligible entry, w waits with a timeout
           no later than the earliest eligible gate; after it, a rescan finds
           that entry (or another) past its gate and runs it: progress.
         * The wait without a timeout is taken only when the scan saw NO
           eligible entry, i.e. m became eligible after the scan read its
           state. The completion that made it eligible posts w's wake
           semaphore AFTER publishing its done stamp (a data producer posts
           the owner of each consumer in fanout[]; the entry that completes an
           instance posts every worker), unless w itself ran it, in which case
           w is not waiting. w drains the semaphore BEFORE scanning, so that
           token is still there: the wait returns and the rescan sees m.
       Table order is not used, so a schedule whose instances overlap in
       table order is still walked to completion.

       Bounded window (XPURT_SKIP_WINDOW = W >= 1; W = 1 is strict order). m
       must be inside its owner's window, which holds only if table order is
       a topological order of G: then take m = the unfinished entry with the
       smallest table index. All its G-predecessors have smaller indices and
       are done, and no earlier entry of its owner is unfinished, so m is the
       owner's window head; the argument above applies. Data dependencies
       point backward in table order by construction (ingest asserts it);
       main() checks the instance edges at start-up and refuses to run a
       bounded window on a table that violates them.

    2. SLEEPING START GATES (XPURT_GATE=sleep|spin, default sleep;
       XPURT_GATE_SPIN_US, default 50). The strict walker's gate is
       `while (now < target) sched_yield();`, which keeps a pinned core at
       100 % for the whole wait. Here the wait is a timed semaphore wait until
       (gate - spin margin), then a short spin to the gate. `spin` restores
       the yield loop.

    3. REAL-TIME PRIORITY (XPURT_RT_PRIO, default = --rt-priority, 0 = keep
       the inherited policy). main() asks for SCHED_FIFO at that priority
       before the pools and workers exist, so they inherit it. If the kernel
       refuses (EPERM without CAP_SYS_NICE / RLIMIT_RTPRIO) the run continues
       under the inherited policy and prints that it did.

    Also: XPURT_CYCLES=<k> replays the table k times when --hyperperiod-ms was
    given (gate = t0 + cycle*H + start; all workers meet at a barrier between
    cycles), and XPURT_TRACE_OUT=<path> gets one {start, end, cpu} record per
    dispatch per cycle (raw, 24 bytes each after a 32-byte header).
    """
    N = f"{upper}_N_ENTRIES"
    T = f"{upper}_TABLE"
    give = ("            for (int _f = 0; _f < e_->n_fanout; _f++) {\n"
            "                k_sem_give(&completion_sems[i_]);\n"
            "            }\n")
    give0 = ("                for (int _f = 0; _f < e_->n_fanout; _f++) {\n"
             "                    k_sem_give(&completion_sems[i_]);\n"
             "                }\n")
    br = dispatch_branch
    if br:
        # v2 signals completion through done stamps + wake semaphores, not
        # through the per-entry completion semaphores.
        assert br.count(give) >= 1 and br.count(give0) >= 1, "branch text drifted"
        br = br.replace(give, "").replace(give0, "")
        t0 = "            uint64_t t_disp0 = (uint64_t)k_cycle_get_64();\n"
        t1 = "            uint64_t t_disp1 = (uint64_t)k_cycle_get_64();\n"
        assert br.count(t0) >= 1 and br.count(t0) == br.count(t1), "branch text drifted"
        br = br.replace(t0, t0 +
            f"            g_v2_trace[(size_t)cyc_ * {N} + i_].start = t_disp0 - run_t0;\n")
        br = br.replace(t1, t1 +
            f"            g_v2_trace[(size_t)cyc_ * {N} + i_].end = t_disp1 - run_t0;\n"
            f"            g_v2_trace[(size_t)cyc_ * {N} + i_].cpu = sched_getcpu();\n")
    hyper = "0.0" if hyper_ms is None else repr(float(hyper_ms))

    defs = r'''
/* ---- v2 runtime (--walker skip): see _v2_blocks in generate_xpurt_main.py */
#include <semaphore.h>
#include <time.h>
#define XPURT_V2_HYPER_MS @HYPER@
#define XPURT_V2_RT_PRIO_DEFAULT @RTPRIO@
struct xpurt_v2_slot { uint64_t start; uint64_t end; int32_t cpu; int32_t pad; };
static struct xpurt_v2_slot *g_v2_trace;      /* [cycle][entry]; 0xff.. = not run */
static int g_cycles = 1;
static uint64_t g_cycle_ticks;
static pthread_barrier_t g_cyc_barrier;
static int g_skip_window;                     /* 0 = whole table, 1 = strict */
static int g_gate_sleep = 1;                  /* 0 = yield-spin gates */
static uint64_t g_gate_spin_ticks;
static int g_rt_prio = XPURT_V2_RT_PRIO_DEFAULT;
static int g_rt_active;                       /* 1 if SCHED_FIFO was granted */
static uint32_t g_done[@N@];                  /* cycle+1 of the last completion */
static uint64_t g_gate_ticks[@N@];            /* start gate, ticks from cycle start */
static int g_owner[@N@];                      /* worker index that claims the entry */
static int g_group[@N@];                      /* (network, instance) group */
static int g_prev_group[@N@];                 /* group of instance-1, or -1 */
static int g_group_size[@N@];
static uint64_t g_group_done[@N@];            /* completions, all cycles */
static int g_n_groups;
static sem_t g_wake[XPURT_MAX_WORKERS];
static uint64_t g_v2_skips[XPURT_MAX_WORKERS];   /* entries run ahead of the head */
static uint64_t g_v2_sleeps[XPURT_MAX_WORKERS];  /* timed/untimed sleeps taken */

static inline int xpurt_v2_eligible(int j, uint32_t stamp)
{
    const xpurt_sched_entry_t *e = &@T@[j];
    for (int d = 0; d < e->n_deps; d++) {
        if (__atomic_load_n(&g_done[e->deps[d]], __ATOMIC_ACQUIRE) != stamp)
            return 0;
    }
    int pg = g_prev_group[j];
    if (pg >= 0 &&
        __atomic_load_n(&g_group_done[pg], __ATOMIC_ACQUIRE) !=
            (uint64_t)stamp * (uint64_t)g_group_size[pg])
        return 0;
    return 1;
}

static inline void xpurt_v2_complete(int i, uint32_t stamp, int me)
{
    const xpurt_sched_entry_t *e = &@T@[i];
    __atomic_store_n(&g_done[i], stamp, __ATOMIC_RELEASE);
    int g = g_group[i];
    uint64_t c = __atomic_add_fetch(&g_group_done[g], 1, __ATOMIC_ACQ_REL);
    for (int f = 0; f < e->n_fanout; f++) {
        int w = g_owner[e->fanout[f]];
        if (w != me) sem_post(&g_wake[w]);
    }
    if (c == (uint64_t)stamp * (uint64_t)g_group_size[g]) {
        /* instance complete: the next instance of this network may start */
        for (int w = 0; w < g_n_workers; w++)
            if (w != me) sem_post(&g_wake[w]);
    }
}

static long xpurt_v2_env_long(const char *name, long dflt)
{
    const char *v = getenv(name);
    return (v && *v) ? strtol(v, NULL, 10) : dflt;
}
'''.replace("@HYPER@", hyper).replace("@RTPRIO@", str(int(rt_prio_default))) \
   .replace("@N@", N).replace("@T@", T)

    loop = r'''    int *own_ = malloc(sizeof(int) * (size_t)@N@);
    int n_own_ = 0;
    if (!own_) { printf("FATAL: v2 own-list alloc\n"); sys_reboot(SYS_REBOOT_COLD); }
    for (int i_ = 0; i_ < @N@; i_++)
        if (g_owner[i_] == my_acc_idx) own_[n_own_++] = i_;
    (void)my_claims_unbound;
    for (int cyc_ = 0; cyc_ < g_cycles; cyc_++) {
        const uint64_t cyc_base_ = run_t0 + (uint64_t)cyc_ * g_cycle_ticks;
        const uint32_t stamp_ = (uint32_t)cyc_ + 1u;
        int head_ = 0, left_ = n_own_;
        while (left_ > 0) {
            while (head_ < n_own_ && g_done[own_[head_]] == stamp_) head_++;
            /* Drain wake tokens BEFORE scanning: a completion published after
             * this point leaves a token that ends the wait below. */
            while (sem_trywait(&g_wake[my_acc_idx]) == 0) { }
            uint64_t t_iter_start = (uint64_t)k_cycle_get_64();
            int lim_ = n_own_;
            if (g_skip_window > 0 && head_ + g_skip_window < n_own_)
                lim_ = head_ + g_skip_window;
            int pick_ = -1, pick_pos_ = -1;
            uint64_t next_gate_ = UINT64_MAX;
            for (int k_ = head_; k_ < lim_; k_++) {
                int j_ = own_[k_];
                if (g_done[j_] == stamp_) continue;
                if (!xpurt_v2_eligible(j_, stamp_)) continue;
                uint64_t gate_ = cyc_base_ + g_gate_ticks[j_];
                if (gate_ <= t_iter_start) { pick_ = j_; pick_pos_ = k_; break; }
                if (gate_ < next_gate_) next_gate_ = gate_;
            }
            if (pick_ < 0) {
                if (next_gate_ == UINT64_MAX) {
                    /* nothing eligible: sleep until a producer wakes us */
                    while (sem_wait(&g_wake[my_acc_idx]) != 0 && errno == EINTR) { }
                    g_v2_sleeps[my_acc_idx]++;
                    g_hart_acc[my_acc_idx].dep_wait +=
                        (uint64_t)k_cycle_get_64() - t_iter_start;
                } else if (!g_gate_sleep) {
                    k_yield();
                    g_hart_acc[my_acc_idx].target_gate_spin +=
                        (uint64_t)k_cycle_get_64() - t_iter_start;
                } else if (next_gate_ - t_iter_start <= g_gate_spin_ticks) {
                    /* final approach: spin to the gate */
                    while ((uint64_t)k_cycle_get_64() < next_gate_) { }
                    g_hart_acc[my_acc_idx].target_gate_spin +=
                        (uint64_t)k_cycle_get_64() - t_iter_start;
                } else {
                    /* sleep until (gate - spin margin) or a wake token */
                    uint64_t rel_ticks_ = next_gate_ - t_iter_start - g_gate_spin_ticks;
                    struct timespec ts_;
                    clock_gettime(CLOCK_MONOTONIC, &ts_);
                    uint64_t ns_ = (uint64_t)ts_.tv_nsec +
                        rel_ticks_ * 1000000000ull / (uint64_t)XPURT_TICKS_PER_SEC;
                    ts_.tv_sec += (time_t)(ns_ / 1000000000ull);
                    ts_.tv_nsec = (long)(ns_ % 1000000000ull);
                    while (sem_clockwait(&g_wake[my_acc_idx], CLOCK_MONOTONIC, &ts_) != 0 &&
                           errno == EINTR) { }
                    g_v2_sleeps[my_acc_idx]++;
                    g_hart_acc[my_acc_idx].hart_idle +=
                        (uint64_t)k_cycle_get_64() - t_iter_start;
                }
                continue;
            }
            const int i_ = pick_;
            const xpurt_sched_entry_t *e_ = &@T@[i_];
            if (pick_pos_ != head_) g_v2_skips[my_acc_idx]++;
            do {
@BRANCH@
                printf("xpurt: WARN unknown network %s in entry %d\n",
                       e_->network, e_->entry_id);
                g_hart_acc[my_acc_idx].entries_done++;
            } while (0);
            xpurt_v2_complete(i_, stamp_, my_acc_idx);
            left_--;
        }
        pthread_barrier_wait(&g_cyc_barrier);
    }
    free(own_);
    (void)prev_iter_end;'''.replace("@N@", N).replace("@T@", T).replace("@BRANCH@", br)

    main_early = r'''    /* v2: real-time priority, asked for before the pools and workers exist so
     * that every runtime thread inherits it. A refusal is not fatal. */
    g_rt_prio = (int)xpurt_v2_env_long("XPURT_RT_PRIO", XPURT_V2_RT_PRIO_DEFAULT);
    if (g_rt_prio > 0) {
        struct sched_param sp_ = { .sched_priority = g_rt_prio };
        if (sched_setscheduler(0, SCHED_FIFO, &sp_) == 0) {
            g_rt_active = 1;
            printf("xpurt: v2 rt_priority=SCHED_FIFO %d\n", g_rt_prio);
        } else {
            printf("xpurt: v2 rt_priority %d refused (errno=%d); continuing "
                   "under the inherited policy\n", g_rt_prio, errno);
        }
    } else {
        printf("xpurt: v2 rt_priority off (inherited policy kept)\n");
    }
'''

    main_pre = r'''    /* ---- v2 runtime set-up ------------------------------------------- */
    {
        g_cycles = (int)xpurt_v2_env_long("XPURT_CYCLES", 1);
        if (g_cycles < 1) g_cycles = 1;
        if (XPURT_V2_HYPER_MS <= 0.0 && g_cycles > 1) {
            printf("FATAL: XPURT_CYCLES=%d needs --hyperperiod-ms at codegen\n", g_cycles);
            sys_reboot(SYS_REBOOT_COLD);
            return -1;
        }
        g_cycle_ticks = (uint64_t)(XPURT_V2_HYPER_MS * (double)XPURT_TICKS_PER_SEC / 1000.0 + 0.5);
        g_skip_window = (int)xpurt_v2_env_long("XPURT_SKIP_WINDOW", 0);
        if (g_skip_window < 0) g_skip_window = 0;
        {
            const char *g_ = getenv("XPURT_GATE");
            g_gate_sleep = !(g_ && strcmp(g_, "spin") == 0);
        }
        g_gate_spin_ticks = (uint64_t)xpurt_v2_env_long("XPURT_GATE_SPIN_US", 50) *
                            (uint64_t)XPURT_TICKS_PER_SEC / 1000000ull;
        g_n_groups = 0;
        for (int i = 0; i < @N@; i++) {
            const xpurt_sched_entry_t *e = &@T@[i];
            g_gate_ticks[i] = (uint64_t)((double)e->start_time_ms *
                                         (double)XPURT_TICKS_PER_SEC / 1000.0);
            g_owner[i] = -1;
            for (int w = 0; w < g_n_workers; w++) {
                if (strcmp(e->core_kind, g_workers[w].kind) != 0) continue;
                if (e->hart >= 0 ? (e->hart == g_workers[w].hart)
                                 : g_workers[w].claims_unbound) { g_owner[i] = w; break; }
            }
            int g = -1;
            for (int k = 0; k < i && g < 0; k++) {
                if (@T@[k].instance == e->instance &&
                    strcmp(@T@[k].network, e->network) == 0) g = g_group[k];
            }
            if (g < 0) g = g_n_groups++;
            g_group[i] = g;
            g_group_size[g]++;
        }
        for (int i = 0; i < @N@; i++) {
            const xpurt_sched_entry_t *e = &@T@[i];
            g_prev_group[i] = -1;
            if (e->instance == 0) continue;
            for (int k = 0; k < @N@; k++) {
                if (@T@[k].instance == e->instance - 1 &&
                    strcmp(@T@[k].network, e->network) == 0) {
                    g_prev_group[i] = g_group[k];
                    break;
                }
            }
        }
        /* A bounded window (1 = strict order) is only deadlock-free if table
         * order is a topological order of the instance edges too; data deps
         * are already backward (ingest). Refuse rather than hang. */
        if (g_skip_window > 0) {
            for (int i = 0; i < @N@; i++) {
                if (g_prev_group[i] < 0) continue;
                for (int k = i + 1; k < @N@; k++) {
                    if (g_group[k] == g_prev_group[i]) {
                        printf("FATAL: bounded XPURT_SKIP_WINDOW but entry %d (%s#%d) precedes "
                               "entry %d of the previous instance in table order\n",
                               i, @T@[i].network, @T@[i].instance, k);
                        sys_reboot(SYS_REBOOT_COLD);
                        return -1;
                    }
                }
            }
        }
        for (int w = 0; w < g_n_workers; w++) sem_init(&g_wake[w], 0, 0);
        g_v2_trace = malloc((size_t)g_cycles * @N@ * sizeof *g_v2_trace);
        if (!g_v2_trace) { printf("FATAL: v2 trace alloc\n"); return -1; }
        memset(g_v2_trace, 0xff, (size_t)g_cycles * @N@ * sizeof *g_v2_trace);
        pthread_barrier_init(&g_cyc_barrier, NULL, (unsigned)g_n_workers);
        printf("xpurt: v2 walker cycles=%d hyper_ms=%.6f entries=%d workers=%d "
               "skip_window=%d gate=%s spin_us=%ld rt=%d\n",
               g_cycles, (double)XPURT_V2_HYPER_MS, @N@, g_n_workers,
               g_skip_window, g_gate_sleep ? "sleep" : "spin",
               xpurt_v2_env_long("XPURT_GATE_SPIN_US", 50), g_rt_active);
    }
'''.replace("@N@", N).replace("@T@", T)

    main_post = r'''    {
        uint64_t t_end_ = (uint64_t)k_cycle_get_64() - run_t0;
        printf("xpurt: v2 run_ticks=%llu ticks_per_sec=%llu\n",
               (unsigned long long)t_end_, (unsigned long long)XPURT_TICKS_PER_SEC);
        for (int w = 0; w < g_n_workers; w++)
            printf("xpurt: v2 worker[%d] skips=%llu sleeps=%llu\n", w,
                   (unsigned long long)g_v2_skips[w], (unsigned long long)g_v2_sleeps[w]);
        const char *p_ = getenv("XPURT_TRACE_OUT");
        if (p_ && *p_) {
            FILE *f_ = fopen(p_, "wb");
            if (f_) {
                uint64_t hdr_[4] = { 0x4b33525431ULL, (uint64_t)g_cycles,
                                     (uint64_t)@N@, g_cycle_ticks };
                fwrite(hdr_, sizeof hdr_, 1, f_);
                fwrite(g_v2_trace, sizeof *g_v2_trace, (size_t)g_cycles * @N@, f_);
                fclose(f_);
                printf("xpurt: v2 trace -> %s\n", p_);
            } else {
                printf("xpurt: v2 cannot write %s\n", p_);
            }
        }
    }
'''.replace("@N@", N)
    return defs, loop, main_early, main_pre, main_post


def _static_blocks(upper: str, dispatch_branch: str, program: dict,
                   networks: list[str], core_kinds: list[str],
                   hyper_ms: float | None,
                   rt_prio_default: int,
                   static_env: dict | None = None) -> tuple[str, str, str, str, str]:
    """C fragments for the static walker (--walker static; Linux and Zephyr
    through runtime/modelblaster_pool/xs_os.h; `static_env` bakes run-time
    switches in for Zephyr, which has no environment).

    The schedule is compiled into one const program per worker by
    pipeline/xpurt_static_program.py (read its module docstring for S1-S6:
    per-worker lists, edge classification and reduction, spin/sleep per edge,
    gates on roots only, static gangs, run-ahead alternatives). This function
    emits those programs and the walker that executes them.

    SHARED STATE. Each worker owns one cache-line-aligned block: a progress
    counter (epoch * list length + length of the completed prefix; monotone),
    an event word and a waiter count (an eventcount on a futex). Nothing else
    is written by one worker and read by another; accounting, trace and done
    flags are per-worker and merged by main() after the join. (The
    per-network verification counters of the generated dispatch branches are
    the one exception: a network whose dispatches span workers bumps one
    atomic per dispatch.)

    WAITING. A consumer that needs entry j of worker p waits until p's counter
    reaches j's position: a bounded spin when the schedule expects a short
    wait (per edge, decided at generation time), then a futex wait on p's
    event word. p issues a wake only when its waiter count is non-zero.

    ORDER AND SAFETY OF UNGATED INTERIOR ENTRIES (S4). With every worker
    executing its list in order, the start of an item is
        max(gate if it is a root, end of the previous item of the worker,
            end of each dependency),
    and its end is start + duration: a system of max and + only. Every start
    and end is therefore a non-decreasing function of every duration and
    gate; removing interior gates or shortening any kernel can only move
    starts earlier, never later. A gang is one such node spanning several
    lists (slices sit at fixed list positions), so this holds with static
    gangs as well. It does NOT hold with run-ahead (XPURT_SKIP=1): a worker
    that fills a wait with a later item can be busy when its head becomes
    ready, which delays the head by at most that item's duration. Nor does it
    hold for the pool fallback, where hart locks are granted in arrival order.

    DEADLOCK FREEDOM. Let m be the unfinished item with the smallest table
    index (slices take the index of their entry). Every dependency of m, and
    every earlier item of m's worker, has a smaller index and is finished, so
    m is its worker's head, and each producer worker's prefix counter covers
    the dependency (all earlier items of that producer are finished too). So
    every wait of m is already satisfied, or is satisfied by a completion
    whose wake cannot be lost (the waiter count is raised before the final
    check, the producer bumps the event word before reading the count). For a
    gang, m is the master or a slice of the gang with the smallest index: the
    master opens the gang without waiting for helpers, each helper reaches its
    slice because everything before it is finished, serves and returns at
    close. Run-ahead only ever finishes items early and never makes a worker
    wait on anything but its head (or a timeout), so the argument stands with
    it on. Cross-epoch edges point from epoch E-1 to E and epochs are
    finished in order by every worker, so the same argument applies with
    (epoch, index) ordered lexicographically.

    TARGETED WAKES AND EVENT-DRIVEN RUN-AHEAD (S7a/S7b, 2026-11-26). A
    waiter registers the progress value it needs in the producer's block
    (need[self], then bit self of wmask) and sleeps on its OWN futex word; the
    producer wakes only registered consumers whose need its publish meets.
    The register-then-recheck and publish-then-read-mask pairs are seq_cst,
    so either the waiter sees the progress or the producer sees the
    registration: no wake is lost, and the deadlock-freedom argument above
    holds with "wake cannot be lost" read for this protocol. A worker whose
    head is blocked and that has alternatives registers on the head's
    blocking producer (so the head's wake is never missed) and on the
    producers of the alternatives' first unmet cross-worker dependencies, and
    re-scans when any of them advances instead of every XPURT_STALL_US. Every
    sleep also carries a 2 ms safety timeout; one that ends with the
    registered need already met, and met more than 200 us earlier (the
    producer's last publish is older), is counted (ev_timeouts, a lost-wake
    suspect). XPURT_WAKE=all restores the broadcast eventcount for dependency
    waits (S7b registrations are still served by the targeted path), and
    XPURT_RUNAHEAD_WAIT=poll the periodic re-scan.

    Run-time switches: XPURT_CYCLES, XPURT_SKIP (1/0 run-ahead),
    XPURT_STALL_US (re-scan period while blocked with alternatives, 250),
    XPURT_GATE_ALL (1 = gate every item at its scheduled start),
    XPURT_GATE (sleep|spin), XPURT_GATE_SPIN_US (50), XPURT_WAIT
    (auto|spin|sleep: per-edge choice, or force), XPURT_SPIN_US (bounded spin
    of a SPIN edge, 100), XPURT_RT_PRIO.
    """
    N = f"{upper}_N_ENTRIES"
    T = f"{upper}_TABLE"
    workers = program["workers"]
    use_gang = bool(program["gang"])
    needs_locks = bool(program["needs_locks"])
    net_idx = {n: i for i, n in enumerate(networks)}
    kind_idx = {k: i for i, k in enumerate(core_kinds)}

    # ---- branch text: index compares instead of strcmp, per-worker state ----
    br = dispatch_branch
    give = ("            for (int _f = 0; _f < e_->n_fanout; _f++) {\n"
            "                k_sem_give(&completion_sems[i_]);\n"
            "            }\n")
    give0 = ("                for (int _f = 0; _f < e_->n_fanout; _f++) {\n"
             "                    k_sem_give(&completion_sems[i_]);\n"
             "                }\n")
    assert br.count(give) >= 1 and br.count(give0) >= 1, "branch text drifted"
    br = br.replace(give, "").replace(give0, "")
    for net in networks:
        old = (f"        if (e_->network[0] == '{net[0]}' && "
               f'strcmp(e_->network, "{net}") == 0) {{\n')
        assert br.count(old) == 1, f"branch head for {net} drifted"
        br = br.replace(old, f"        if (it_->net == {net_idx[net]}) {{   /* {net} */\n")
    for kind in core_kinds:
        old = f'(strcmp(e_->impl, "{kind}") == 0)'
        assert br.count(old) >= 1, "impl select drifted"
        br = br.replace(old, f"(it_->impl == {kind_idx[kind]})")
    assert br.count("pool_for_entry(e_)") >= 1
    br = br.replace("pool_for_entry(e_)", "xs_pool_")
    if not needs_locks:
        assert br.count("            lock_entry_harts(e_);\n") >= 1
        br = br.replace("            lock_entry_harts(e_);\n", "")
        br = br.replace("            unlock_entry_harts(e_);\n", "")
    br = br.replace("g_hart_acc[my_acc_idx]", "acc_")
    br = br.replace("xpurt_trace[i_]", "xs_last_[pos_]")
    t0 = "            uint64_t t_disp0 = (uint64_t)k_cycle_get_64();\n"
    t1 = "            uint64_t t_disp1 = (uint64_t)k_cycle_get_64();\n"
    assert br.count(t0) >= 1 and br.count(t0) == br.count(t1), "branch text drifted"
    br = br.replace(t0, t0 + "            tr_[pos_].start = t_disp0 - run_t0;\n")
    br = br.replace(t1, t1 + "            tr_[pos_].end = t_disp1 - run_t0;\n"
                             "            tr_[pos_].cpu = xs_os_cpu();\n")
    hyper = "0.0" if hyper_ms is None else repr(float(hyper_ms))

    # ---- const programs -----------------------------------------------------
    tabs: list[str] = []
    wrows: list[str] = []
    max_len = 1
    for w, wk in enumerate(workers):
        items = wk["items"]
        max_len = max(max_len, len(items))
        rows = []
        for p, it in enumerate(items):
            def deps(name, lst, p=p):
                if not lst:
                    return "NULL"
                arr = ", ".join(
                    "{ %d, %d, %d, %d }" % (d["worker"], d["eoff"],
                                            d.get("spin", 0) if name == "w" else d.get("own", 0), d["pos"])
                    for d in lst)
                tabs.append(f"static const xs_dep_t xs_{name}_{w}_{p}[] = {{ {arr} }};")
                return f"xs_{name}_{w}_{p}"
            wref = deps("w", it["wait"])
            fref = deps("f", it["full"])
            aref = "NULL"
            if it["alts"]:
                tabs.append(f"static const uint16_t xs_a_{w}_{p}[] = {{ "
                            + ", ".join(str(a) for a in it["alts"]) + " };")
                aref = f"xs_a_{w}_{p}"
            rows.append("    { %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %s, %s, %s }," % (
                it["entry"], net_idx[it["network"]], kind_idx[it["impl"]], it["kind"],
                it["gated"], it["slice"], it["gang"], len(it["wait"]), len(it["full"]),
                len(it["alts"]), wref, fref, aref))
        tabs.append(f"static const xs_item_t xs_items_{w}[] = {{\n" + "\n".join(rows) + "\n};")
        wrows.append('    { "%s", %d, %d, xs_items_%d },' % (wk["kind"], wk["hart"], len(items), w))
    # A (doorbell): per producer, the workers that can ever wait on it (a wait
    # edge, or a cross-worker `full` dependency checked by run-ahead).
    waiters: list[set] = [set() for _ in workers]
    for w, wk in enumerate(workers):
        for it in wk["items"]:
            for d in it["wait"]:
                if d["worker"] != w:
                    waiters[d["worker"]].add(w)
            for d in it["full"]:
                if not d.get("own", 0) and d["worker"] != w:
                    waiters[d["worker"]].add(w)
    wlrows = []
    for p_, ws in enumerate(waiters):
        if ws:
            tabs.append(f"static const uint8_t xs_wl_{p_}[] = {{ " + ", ".join(str(x) for x in sorted(ws)) + " };")
            wlrows.append("    { %d, xs_wl_%d }," % (len(ws), p_))
        else:
            wlrows.append("    { 0, NULL },")
    grows = []
    for g in program["gangs"]:
        grows.append("    { %d, { %s } }," % (len(g["harts"]), ", ".join(str(h) for h in g["harts"])))
    if not grows:
        grows.append("    { 0, { 0 } },")
    st = program["stats"]
    baked = ""
    if static_env:
        kv = ", ".join('{ "%s", "%s" }' % (k, v) for k, v in sorted(static_env.items()))
        baked = ("#ifndef MODELBLASTER_PLATFORM_LINUX\n#define XS_BAKED_ENV_N %d\n"
                 "static const char *const XS_BAKED_ENV[%d][2] = { %s };\n#endif\n"
                 % (len(static_env), len(static_env), kv))

    defs = (r'''
/* ---- static walker (--walker static): see _static_blocks in
 * generate_xpurt_main.py and pipeline/xpurt_static_program.py.
 * @STATS@ */
#include <limits.h>
#include <stdatomic.h>
#include <time.h>
@BAKEDENV@#include "xs_os.h"            /* OS layer: Linux / Zephyr (runtime/modelblaster_pool) */
#define XS_HYPER_MS @HYPER@
#define XS_RT_PRIO_DEFAULT @RTPRIO@
#define XS_N_WORKERS @NW@
#define XS_N_GANGS @NG@
#define XS_USE_GANG @USEGANG@
#define XS_MAX_LEN @MAXLEN@
typedef struct { int16_t worker; uint8_t eoff; uint8_t flag; uint32_t pos; } xs_dep_t;
typedef struct {
    int32_t entry; int16_t net; int16_t impl;
    uint8_t kind;      /* 0 normal, 1 gang master, 2 gang slice */
    uint8_t gated;     /* released by time (root) */
    uint8_t slice; int8_t gang;
    uint16_t n_wait, n_full, n_alt;
    const xs_dep_t *wait;     /* reduced cross-worker edges; flag = spin first */
    const xs_dep_t *full;     /* every direct dependency; flag = same worker */
    const uint16_t *alts;     /* later positions independent of this item */
} xs_item_t;
typedef struct { const char *kind; int hart; int n; const xs_item_t *items; } xs_worker_t;
typedef struct { int n; int harts[XPURT_MAX_POOL_WIDTH]; } xs_gang_t;
@TABS@
static const xs_worker_t XS_WORKERS[XS_N_WORKERS] = {
@WROWS@
};
typedef struct { int n; const uint8_t *w; } xs_wl_t;
static const xs_wl_t XS_WAITERS[XS_N_WORKERS] = {
@WLROWS@
};
static const xs_gang_t XS_GANGS[] = {
@GROWS@
};
static modelblaster_pool_t g_xs_gang_pool[XS_N_GANGS + 1];

struct xs_slot { uint64_t start; uint64_t end; int32_t cpu; int32_t pad; };
/* One block per worker, on its own cache lines. `prog` is written by the
 * owner only; `evt`/`waiters` implement the wake. */
struct xs_wstate {
    _Alignas(64) _Atomic uint64_t prog;     /* epoch * n + completed prefix */
    _Atomic uint32_t evt;                   /* futex word */
    _Atomic uint32_t waiters;
    _Alignas(64) struct xs_slot *trace;     /* [epoch][pos], owner only */
    struct xpurt_trace_slot *last;          /* [pos], owner only */
    uint64_t skips, sleeps, spins_ok, futex_waits, wakes_sent, stall_scans, ev_timeouts;
    struct xpurt_hart_acc acc;
    /* S7a targeted wakes (2026-11-26): consumers register here the progress
     * they need (need[c], then bit c of wmask); the owner wakes only those
     * whose need its publish meets, each on the consumer's own word. */
    _Alignas(64) _Atomic uint64_t wmask;
    _Atomic uint64_t need[XS_N_WORKERS];
    _Atomic uint64_t pub_t;                  /* tick of the last publish (lost-wake check) */
    _Alignas(64) _Atomic uint32_t wword;     /* this worker's own futex word */
    /* A doorbell (2026-11-26): dneed[p] = progress this worker needs from
     * producer p (0 = none), written by this worker, claimed (CAS to 0) by p,
     * which then rings this worker's doorbell: a semaphore or a futex count. */
    _Alignas(64) _Atomic uint64_t dneed[XS_N_WORKERS];
    xs_os_db_t db;
};
static struct xs_wstate g_xs[XS_N_WORKERS];
static int g_xs_of_worker[XPURT_MAX_WORKERS];   /* g_workers index -> program worker */
static int g_cycles = 1;
static uint64_t g_cycle_ticks;
static int g_xs_skip = 1, g_xs_gate_all, g_xs_gate_sleep = 1, g_xs_wait_mode;  /* 0 auto 1 spin 2 sleep */
static int g_xs_wake_all, g_xs_ra_poll;  /* XPURT_WAKE=all, XPURT_RUNAHEAD_WAIT=poll: the pre-S7 behaviour */
static int g_xs_db;                      /* XPURT_WAKE=doorbell_sem (1) | doorbell_futex (2); 0 = S7a mask */
_Static_assert(XS_N_WORKERS <= 64, "S7a wake mask is 64 bits");
static uint64_t g_xs_gate_spin_ticks, g_xs_spin_ticks, g_xs_stall_ticks;
static int g_rt_prio = XS_RT_PRIO_DEFAULT, g_rt_active;
static uint64_t g_xs_gate_ticks[@N@];

static long xs_env_long(const char *name, long dflt)
{
    const char *v = xs_os_getenv(name);
    return (v && *v) ? strtol(v, NULL, 10) : dflt;
}

static inline int xs_dep_ok(const xs_dep_t *d, uint64_t epoch)
{
    if (epoch < d->eoff) return 1;      /* producer in the epoch before the first */
    uint64_t need = (epoch - d->eoff) * (uint64_t)XS_WORKERS[d->worker].n + d->pos + 1u;
    return atomic_load_explicit(&g_xs[d->worker].prog, memory_order_acquire) >= need;
}

/* Publish the owner's progress and wake waiters, if any. */
static inline void xs_publish(struct xs_wstate *me, uint64_t prog)
{
    atomic_store_explicit(&me->pub_t, (uint64_t)k_cycle_get_64(), memory_order_relaxed);
    atomic_store(&me->prog, prog);                /* seq_cst: pairs with xs_reg */
    if (g_xs_db) {                                /* A: scan this producer's possible waiters */
        const int self_ = (int)(me - g_xs);
        const xs_wl_t *wl_ = &XS_WAITERS[self_];
        for (int k_ = 0; k_ < wl_->n; k_++) {
            struct xs_wstate *c_ = &g_xs[wl_->w[k_]];
            uint64_t v_ = atomic_load(&c_->dneed[self_]);
            if (v_ && v_ <= prog && atomic_compare_exchange_strong(&c_->dneed[self_], &v_, 0)) {
                xs_os_db_give(&c_->db, g_xs_db);
                me->wakes_sent++;
            }
        }
        return;
    }
    if (g_xs_wake_all) {                          /* pre-S7: broadcast eventcount */
        atomic_fetch_add(&me->evt, 1);
        if (atomic_load(&me->waiters) != 0) {
            xs_os_evt_wake(&me->evt, 1);
            me->wakes_sent++;
        }
        /* fall through: S7b run-ahead registrations are still served */
    }
    uint64_t m = atomic_load(&me->wmask);         /* seq_cst */
    while (m) {
        int c = __builtin_ctzll(m);
        uint64_t bit = 1ull << c;
        m &= m - 1;
        if (atomic_load(&me->need[c]) > prog) continue;
        if (atomic_fetch_and(&me->wmask, ~bit) & bit) {
            atomic_fetch_add(&g_xs[c].wword, 1);
            xs_os_evt_wake(&g_xs[c].wword, 0);
            me->wakes_sent++;
        }
    }
}

/* S7a: register consumer `self` on producer `prod` for progress `need`. The
 * need is stored before the mask bit (the producer reads need only for a set
 * bit); register (seq_cst RMW) then re-check the counter, against the
 * producer's publish (seq_cst store) then mask load: one of the two sees the
 * other, so a wake cannot be lost. */
static inline void xs_reg(int self, int prod, uint64_t need)
{
    atomic_store(&g_xs[prod].need[self], need);
    atomic_fetch_or(&g_xs[prod].wmask, 1ull << self);
}
static inline void xs_unreg(int self, int prod)
{
    atomic_fetch_and(&g_xs[prod].wmask, ~(1ull << self));
}
static inline uint64_t xs_need_of(const xs_dep_t *d, uint64_t epoch)
{
    return (epoch - d->eoff) * (uint64_t)XS_WORKERS[d->worker].n + d->pos + 1u;
}
#define XS_SAFETY_TICKS ((uint64_t)XPURT_TICKS_PER_SEC / 500u)   /* 2 ms: a lost wake shows as ev_timeouts */
/* A safety timeout is a lost-wake suspect only if the producer's last publish
 * (which is at or after the one that met the need) is older than 200 us: a
 * publish racing the timeout itself is not a lost wake. */
static inline int xs_stale(int prod)
{
    return (uint64_t)k_cycle_get_64() - atomic_load_explicit(&g_xs[prod].pub_t, memory_order_relaxed)
           > (uint64_t)XPURT_TICKS_PER_SEC / 5000u;
}

static inline int xs_db_take(struct xs_wstate *me, uint64_t left_ticks)
{
    return xs_os_db_take(&me->db, g_xs_db, left_ticks);
}
/* A: register on every producer in `pm` for needv[p], re-check, take the
 * doorbell unless something is already met (or left_ticks runs out), then
 * cancel every registration; a cancel that finds its slot claimed owes one
 * token, taken here so the count stays exact. Returns 1 if woken by a token. */
static int xs_db_wait(struct xs_wstate *me, uint64_t pm, const uint64_t *needv, uint64_t left_ticks)
{
    for (uint64_t m_ = pm; m_; m_ &= m_ - 1) { int p_ = __builtin_ctzll(m_); atomic_store(&me->dneed[p_], needv[p_]); }
    int ready_ = 0;
    for (uint64_t m_ = pm; m_ && !ready_; m_ &= m_ - 1) {
        int p_ = __builtin_ctzll(m_);
        if (atomic_load(&g_xs[p_].prog) >= needv[p_]) ready_ = 1;
    }
    int got_ = 0;
    if (!ready_) { got_ = xs_db_take(me, left_ticks); me->futex_waits++; }
    int owed_ = 0;
    for (uint64_t m_ = pm; m_; m_ &= m_ - 1) {
        int p_ = __builtin_ctzll(m_);
        if (atomic_exchange(&me->dneed[p_], 0) == 0) owed_++;
    }
    for (owed_ -= got_; owed_ > 0; owed_--) (void)xs_db_take(me, 0);
    return got_;
}

/* Wait for one edge. spin_ticks: bounded spin first. timeout_ticks: 0 = none.
 * Returns 1 when the dependency is met, 0 on timeout. */
static int xs_wait_dep(struct xs_wstate *me, const xs_dep_t *d, uint64_t epoch,
                       uint64_t spin_ticks, uint64_t timeout_ticks)
{
    struct xs_wstate *p = &g_xs[d->worker];
    uint64_t t0 = (uint64_t)k_cycle_get_64();
    if (spin_ticks) {
        do {
            if (xs_dep_ok(d, epoch)) { me->spins_ok++; return 1; }
        } while ((uint64_t)k_cycle_get_64() - t0 < spin_ticks);
    }
    if (g_xs_db) {                            /* A: doorbell */
        uint64_t needv[XS_N_WORKERS];
        needv[d->worker] = xs_need_of(d, epoch);
        for (;;) {
            if (xs_dep_ok(d, epoch)) return 1;
            uint64_t el = (uint64_t)k_cycle_get_64() - t0;
            if (timeout_ticks && el >= timeout_ticks) return 0;
            uint64_t left = timeout_ticks ? timeout_ticks - el : XS_SAFETY_TICKS;
            if (left > XS_SAFETY_TICKS) left = XS_SAFETY_TICKS;
            int got_ = xs_db_wait(me, 1ull << d->worker, needv, left);
            if (xs_dep_ok(d, epoch)) {
                if (!got_ && xs_stale(d->worker)) me->ev_timeouts++;
                return 1;
            }
        }
    }
    if (!g_xs_wake_all) {                     /* S7a: own word, registered need */
        const int self = (int)(me - g_xs);
        const uint64_t need = xs_need_of(d, epoch);
        for (;;) {
            uint32_t s = atomic_load(&me->wword);
            xs_reg(self, d->worker, need);
            if (xs_dep_ok(d, epoch)) { xs_unreg(self, d->worker); return 1; }
            uint64_t el = (uint64_t)k_cycle_get_64() - t0;
            if (timeout_ticks && el >= timeout_ticks) { xs_unreg(self, d->worker); return 0; }
            uint64_t left = timeout_ticks ? timeout_ticks - el : XS_SAFETY_TICKS;
            if (left > XS_SAFETY_TICKS) left = XS_SAFETY_TICKS;
            int r_ = xs_os_evt_wait(&me->wword, s, left);
            xs_unreg(self, d->worker);
            me->futex_waits++;
            if (xs_dep_ok(d, epoch)) {
                /* woke by the safety timeout although the need is met: a lost-wake suspect */
                if (r_ < 0 && xs_stale(d->worker)) me->ev_timeouts++;
                return 1;
            }
            if (timeout_ticks && (uint64_t)k_cycle_get_64() - t0 >= timeout_ticks) return 0;
        }
    }
    for (;;) {
        uint32_t s = atomic_load(&p->evt);
        atomic_fetch_add(&p->waiters, 1);
        if (xs_dep_ok(d, epoch)) { atomic_fetch_sub(&p->waiters, 1); return 1; }
        uint64_t left_ = 0;
        if (timeout_ticks) {
            uint64_t el = (uint64_t)k_cycle_get_64() - t0;
            if (el >= timeout_ticks) { atomic_fetch_sub(&p->waiters, 1); return 0; }
            left_ = timeout_ticks - el;
        }
        (void)xs_os_evt_wait(&p->evt, s, left_);
        atomic_fetch_sub(&p->waiters, 1);
        me->futex_waits++;
        if (xs_dep_ok(d, epoch)) return 1;
        if (timeout_ticks && (uint64_t)k_cycle_get_64() - t0 >= timeout_ticks) return 0;
    }
}

/* Sleep until `target` ticks minus the spin margin (xs_os_sleep_until:
 * clock_nanosleep TIMER_ABSTIME on Linux, k_sleep(K_TIMEOUT_ABS_TICKS) on
 * Zephyr), then spin to `target` on k_cycle_get_64. */
static void xs_sleep_until(uint64_t target, uint64_t *slept_ticks, uint64_t *spun_ticks)
{
    uint64_t now = (uint64_t)k_cycle_get_64();
    if (g_xs_gate_sleep) {
        if (now + g_xs_gate_spin_ticks < target) {
            xs_os_sleep_until(target - g_xs_gate_spin_ticks);
            uint64_t t1 = (uint64_t)k_cycle_get_64();
            *slept_ticks += t1 - now;
            now = t1;
        }
        while ((uint64_t)k_cycle_get_64() < target) { }
    } else {
        while ((uint64_t)k_cycle_get_64() < target) k_yield();
    }
    *spun_ticks += (uint64_t)k_cycle_get_64() - now;
}

/* S7b: the head is blocked and no alternative is ready. Register on the
 * producer of the head's blocking edge and on the producer of the first unmet
 * cross-worker dependency of every alternative in the window (minimum need per
 * producer), then sleep on this worker's own word until one of them reaches
 * its need, or until the head's gate (minus the spin margin). Returns at once
 * if something became ready meanwhile. The caller re-scans. */
static void xs_event_wait(struct xs_wstate *me, const xs_worker_t *pw, const xs_item_t *it,
                          const xs_dep_t *blk, uint64_t gate, uint64_t epoch,
                          const uint8_t *done, uint64_t cyc_base, struct xpurt_hart_acc *acc)
{
    const int self = (int)(me - g_xs);
    uint64_t t0 = (uint64_t)k_cycle_get_64();
    if (!blk && gate) {                         /* gate only, and close: sleep+spin to it */
        if (gate <= t0 + g_xs_gate_spin_ticks) {
            uint64_t sl_ = 0, sp_ = 0;
            xs_sleep_until(gate, &sl_, &sp_);
            acc->hart_idle += sl_; acc->target_gate_spin += sp_;
            return;
        }
    }
    if (blk) {                                  /* bounded spin on a SPIN head edge first */
        uint64_t spin_ = g_xs_wait_mode == 1 ? g_xs_spin_ticks :
                         g_xs_wait_mode == 2 ? 0 : (blk->flag ? g_xs_spin_ticks : 0);
        while (spin_ && (uint64_t)k_cycle_get_64() - t0 < spin_) {
            if (xs_dep_ok(blk, epoch)) { me->spins_ok++; acc->dep_wait += (uint64_t)k_cycle_get_64() - t0; return; }
        }
    }
    uint64_t needv[XS_N_WORKERS];
    uint64_t pm = 0;                            /* producers to register on */
#define XS_WATCH(d_) do { const xs_dep_t *w_ = (d_); \
        if (epoch >= w_->eoff) { uint64_t n_ = xs_need_of(w_, epoch); uint64_t b_ = 1ull << w_->worker; \
            if (!(pm & b_) || n_ < needv[w_->worker]) needv[w_->worker] = n_; pm |= b_; } } while (0)
    if (blk) XS_WATCH(blk);
    uint64_t tnow = (uint64_t)k_cycle_get_64();
    for (int a_ = 0; a_ < it->n_alt; a_++) {
        int q_ = it->alts[a_];
        if (done[q_]) continue;
        const xs_item_t *c_ = &pw->items[q_];
        if (g_xs_gate_all && cyc_base + g_xs_gate_ticks[c_->entry] > tnow) continue;
        const xs_dep_t *miss_ = NULL; int own_ = 0;
        for (int d_ = 0; d_ < c_->n_full; d_++) {
            const xs_dep_t *fd_ = &c_->full[d_];
            if (fd_->flag) { if (!fd_->eoff && !done[fd_->pos]) { own_ = 1; break; } continue; }
            if (!xs_dep_ok(fd_, epoch)) { miss_ = fd_; break; }
        }
        if (own_) continue;                     /* waits on this worker itself: no event while asleep */
        if (!miss_) return;                     /* became ready: re-scan now */
        XS_WATCH(miss_);
    }
#undef XS_WATCH
    if (g_xs_db) {                              /* A: doorbell */
        uint64_t left = XS_SAFETY_TICKS; int timed_gate_ = 0;
        uint64_t tn = (uint64_t)k_cycle_get_64();
        if (!blk && gate) {
            uint64_t g2 = gate - g_xs_gate_spin_ticks;
            left = g2 > tn ? g2 - tn : 0; timed_gate_ = 1;
        }
        if (left > XS_SAFETY_TICKS) { left = XS_SAFETY_TICKS; timed_gate_ = 0; }
        if (left) {
            int got_ = xs_db_wait(me, pm, needv, left);
            if (!got_ && !timed_gate_) {
                for (uint64_t m_ = pm; m_; m_ &= m_ - 1) {
                    int p_ = __builtin_ctzll(m_);
                    if (atomic_load(&g_xs[p_].prog) >= needv[p_] && xs_stale(p_)) { me->ev_timeouts++; break; }
                }
            }
        }
        uint64_t el = (uint64_t)k_cycle_get_64() - t0;
        if (blk) acc->dep_wait += el; else acc->hart_idle += el;
        me->sleeps++;
        (void)self;
        return;
    }
    uint32_t s = atomic_load(&me->wword);
    for (uint64_t m_ = pm; m_; m_ &= m_ - 1) { int p_ = __builtin_ctzll(m_); xs_reg(self, p_, needv[p_]); }
    int ready_ = 0;
    for (uint64_t m_ = pm; m_ && !ready_; m_ &= m_ - 1) {
        int p_ = __builtin_ctzll(m_);
        if (atomic_load_explicit(&g_xs[p_].prog, memory_order_seq_cst) >= needv[p_]) ready_ = 1;
    }
    if (!ready_) {
        uint64_t left = XS_SAFETY_TICKS; int timed_gate_ = 0;
        uint64_t tn = (uint64_t)k_cycle_get_64();
        if (!blk && gate) {
            uint64_t g2 = gate - g_xs_gate_spin_ticks;
            left = g2 > tn ? g2 - tn : 0; timed_gate_ = 1;
        }
        if (left > XS_SAFETY_TICKS) { left = XS_SAFETY_TICKS; timed_gate_ = 0; }
        if (left) {
            int r_ = xs_os_evt_wait(&me->wword, s, left);
            me->futex_waits++;
            if (r_ < 0 && !timed_gate_) {
                /* safety timeout with a registered need already met: lost-wake suspect */
                for (uint64_t m_ = pm; m_; m_ &= m_ - 1) {
                    int p_ = __builtin_ctzll(m_);
                    if (atomic_load(&g_xs[p_].prog) >= needv[p_] && xs_stale(p_)) { me->ev_timeouts++; break; }
                }
            }
        }
    }
    for (uint64_t m_ = pm; m_; m_ &= m_ - 1) xs_unreg(self, __builtin_ctzll(m_));
    uint64_t el = (uint64_t)k_cycle_get_64() - t0;
    if (blk) acc->dep_wait += el; else acc->hart_idle += el;
    me->sleeps++;
}

static int xs_init_locks_only(void)
{
    for (int h = 0; h < XPURT_MAX_HARTS; h++) {
        if (pthread_mutex_init(&g_hart_locks[h], NULL) != 0) return -1;
    }
    g_n_entry_pools = 0;
    return 0;
}
''').replace("@HYPER@", hyper).replace("@RTPRIO@", str(int(rt_prio_default))) \
        .replace("@BAKEDENV@", baked).replace("@NW@", str(len(workers))).replace("@NG@", str(len(program["gangs"]))) \
        .replace("@USEGANG@", "1" if use_gang else "0").replace("@MAXLEN@", str(max_len)) \
        .replace("@TABS@", "\n".join(tabs)).replace("@WROWS@", "\n".join(wrows)).replace("@WLROWS@", "\n".join(wlrows)) \
        .replace("@GROWS@", "\n".join(grows)).replace("@N@", N) \
        .replace("@STATS@", "edges total %d, elided same-worker %d, removed by reduction %d, "
                 "remaining cross-worker %d (+%d epoch)" % (
                     st["edges_total"], st["elided_same_worker"], st["removed_by_reduction"],
                     st["remaining_cross_worker"], st["epoch_cross_worker"]))

    loop = r'''    (void)my_claims_unbound; (void)my_kind; (void)prev_iter_end;
#ifndef MODELBLASTER_PLATFORM_LINUX
    if (g_rt_prio > 0) (void)xs_os_set_rt_prio(g_rt_prio);   /* Zephyr: per worker */
#endif
    const int xw_ = g_xs_of_worker[my_acc_idx];
    const xs_worker_t *const pw_ = &XS_WORKERS[xw_];
    struct xs_wstate *const me_ = &g_xs[xw_];
    const int n_ = pw_->n;
    struct xpurt_hart_acc acc_ = { 0 };
    struct xpurt_trace_slot *const xs_last_ = me_->last;
    uint8_t *done_ = calloc((size_t)n_ + 1, 1);
    if (!done_) { printf("FATAL: static walker alloc\n"); sys_reboot(SYS_REBOOT_COLD); }
    for (int cyc_ = 0; cyc_ < g_cycles; cyc_++) {
        const uint64_t epoch_ = (uint64_t)cyc_;
        const uint64_t cyc_base_ = run_t0 + epoch_ * g_cycle_ticks;
        struct xs_slot *const tr_ = me_->trace + (size_t)cyc_ * (size_t)n_;
        memset(done_, 0, (size_t)n_);
        int head_ = 0;
        while (head_ < n_) {
            int pos_ = head_;
            const xs_item_t *it_ = &pw_->items[pos_];
            uint64_t t_iter_ = (uint64_t)k_cycle_get_64();
            if (done_[pos_]) {            /* ran ahead earlier: fold into the prefix */
                head_++;
                while (head_ < n_ && done_[head_]) head_++;
                xs_publish(me_, epoch_ * (uint64_t)n_ + (uint64_t)head_);
                continue;
            }
            if (it_->kind == 2) {         /* gang slice: serve the master's forks */
                tr_[pos_].start = t_iter_ - run_t0;
                modelblaster_pool_gang_serve(g_xs_gang_pool[it_->gang], it_->slice,
                    epoch_ * (uint64_t)@N@ + (uint64_t)it_->entry + 1u, 1);
                tr_[pos_].end = (uint64_t)k_cycle_get_64() - run_t0;
                tr_[pos_].cpu = xs_os_cpu();
                acc_.kernel += tr_[pos_].end - tr_[pos_].start;
                head_++;
                xs_publish(me_, epoch_ * (uint64_t)n_ + (uint64_t)head_);
                continue;
            }
            /* ---- is the head ready? ---------------------------------- */
            const xs_dep_t *blk_ = NULL;
            for (int d_ = 0; d_ < it_->n_wait; d_++) {
                if (!xs_dep_ok(&it_->wait[d_], epoch_)) { blk_ = &it_->wait[d_]; break; }
            }
            uint64_t gate_ = 0;
            if (!blk_ && (it_->gated || g_xs_gate_all)) {
                gate_ = cyc_base_ + g_xs_gate_ticks[it_->entry];
                if (gate_ <= t_iter_) gate_ = 0;
            }
            if (blk_ || gate_) {
                /* ---- blocked: run an independent ready item instead ---- */
                int alt_ = -1;
                if (g_xs_skip && it_->kind == 0) {
                    for (int a_ = 0; a_ < it_->n_alt && alt_ < 0; a_++) {
                        int q_ = it_->alts[a_];
                        if (done_[q_]) continue;
                        const xs_item_t *c_ = &pw_->items[q_];
                        if (g_xs_gate_all && cyc_base_ + g_xs_gate_ticks[c_->entry] > t_iter_) continue;
                        int ok_ = 1;
                        for (int d_ = 0; d_ < c_->n_full && ok_; d_++) {
                            const xs_dep_t *fd_ = &c_->full[d_];
                            if (fd_->flag) ok_ = fd_->eoff ? 1 : done_[fd_->pos];
                            else ok_ = xs_dep_ok(fd_, epoch_);
                        }
                        if (ok_) alt_ = q_;
                    }
                }
                if (alt_ < 0) {
                    const int can_alt_ = g_xs_skip && it_->kind == 0 && it_->n_alt;
                    uint64_t stall_ = (can_alt_ && g_xs_ra_poll) ? g_xs_stall_ticks : 0;
                    if (can_alt_ && !g_xs_ra_poll) {
                        /* S7b: event-driven run-ahead */
                        xs_event_wait(me_, pw_, it_, blk_, gate_, epoch_, done_, cyc_base_, &acc_);
                    } else if (blk_) {
                        uint64_t spin_ = g_xs_wait_mode == 1 ? g_xs_spin_ticks :
                                         g_xs_wait_mode == 2 ? 0 :
                                         (blk_->flag ? g_xs_spin_ticks : 0);
                        (void)xs_wait_dep(me_, blk_, epoch_, spin_, stall_);
                        acc_.dep_wait += (uint64_t)k_cycle_get_64() - t_iter_;
                    } else {
                        uint64_t tgt_ = gate_;
                        if (stall_ && t_iter_ + stall_ < tgt_) tgt_ = t_iter_ + stall_;
                        uint64_t sl_ = 0, sp_ = 0;
                        xs_sleep_until(tgt_, &sl_, &sp_);
                        acc_.hart_idle += sl_;
                        acc_.target_gate_spin += sp_;
                        me_->sleeps += (sl_ != 0);
                    }
                    if (g_xs_ra_poll ? !stall_ : !can_alt_) continue;
                    /* stalled with alternatives: one unbounded scan */
                    me_->stall_scans++;
                    for (int q_ = head_ + 1; q_ < n_ && alt_ < 0; q_++) {
                        const xs_item_t *c_ = &pw_->items[q_];
                        if (done_[q_] || c_->kind != 0 || c_->gated) continue;
                        if (g_xs_gate_all &&
                            cyc_base_ + g_xs_gate_ticks[c_->entry] > (uint64_t)k_cycle_get_64()) continue;
                        int ok_ = 1;
                        for (int d_ = 0; d_ < c_->n_full && ok_; d_++) {
                            const xs_dep_t *fd_ = &c_->full[d_];
                            if (fd_->flag) ok_ = fd_->eoff ? 1 : done_[fd_->pos];
                            else ok_ = xs_dep_ok(fd_, epoch_);
                        }
                        if (ok_) alt_ = q_;
                    }
                    if (alt_ < 0) continue;
                }
                pos_ = alt_;
                it_ = &pw_->items[pos_];
                me_->skips++;
            }
            /* ---- run item pos_ ----------------------------------------- */
            const int i_ = it_->entry;
            const xpurt_sched_entry_t *e_ = &@T@[i_];
            modelblaster_pool_t xs_pool_ = NULL;
            if (it_->kind == 1) {
                xs_pool_ = g_xs_gang_pool[it_->gang];
                modelblaster_pool_gang_open(xs_pool_, epoch_ * (uint64_t)@N@ + (uint64_t)i_ + 1u);
            } else if (e_->n_harts > 1) {
                xs_pool_ = pool_for_entry(e_);      /* pool fallback */
            }
            do {
@BRANCH@
                printf("xpurt: WARN unknown network %s in entry %d\n",
                       e_->network, e_->entry_id);
                acc_.entries_done++;
            } while (0);
            if (it_->kind == 1)
                modelblaster_pool_gang_close(xs_pool_, epoch_ * (uint64_t)@N@ + (uint64_t)i_ + 1u);
            done_[pos_] = 1;
            if (pos_ == head_) {
                head_++;
                while (head_ < n_ && done_[head_]) head_++;
                xs_publish(me_, epoch_ * (uint64_t)n_ + (uint64_t)head_);
            }
        }
    }
    free(done_);
    acc_.wall_total = (uint64_t)k_cycle_get_64() - worker_t0;
    me_->acc = acc_;'''.replace("@N@", N).replace("@T@", T).replace("@BRANCH@", br)

    main_early = r'''    /* static walker: real-time priority before pools and workers exist, so
     * every runtime thread inherits it (Linux). A refusal is not fatal. On
     * Zephyr threads do not inherit it: each worker sets a cooperative
     * priority itself at start, and only when XPURT_RT_PRIO is baked in
     * (default off there). */
#ifdef MODELBLASTER_PLATFORM_LINUX
    g_rt_prio = (int)xs_env_long("XPURT_RT_PRIO", XS_RT_PRIO_DEFAULT);
    if (g_rt_prio > 0) {
        int rc_ = xs_os_set_rt_prio(g_rt_prio);
        if (rc_ == 0) {
            g_rt_active = 1;
            printf("xpurt: static rt_priority=SCHED_FIFO %d\n", g_rt_prio);
        } else {
            printf("xpurt: static rt_priority %d refused (errno=%d); continuing "
                   "under the inherited policy\n", g_rt_prio, rc_);
        }
    } else {
        printf("xpurt: static rt_priority off (inherited policy kept)\n");
    }
#else
    g_rt_prio = (int)xs_env_long("XPURT_RT_PRIO", 0);
    printf("xpurt: static rt_priority %s\n", g_rt_prio > 0 ? "cooperative (per worker)" : "off");
#endif
'''

    main_pre = r'''    /* ---- static walker set-up ------------------------------------------ */
    {
        g_cycles = (int)xs_env_long("XPURT_CYCLES", 1);
        if (g_cycles < 1) g_cycles = 1;
        if (XS_HYPER_MS <= 0.0 && g_cycles > 1) {
            printf("FATAL: XPURT_CYCLES=%d needs --hyperperiod-ms at codegen\n", g_cycles);
            sys_reboot(SYS_REBOOT_COLD);
            return -1;
        }
        g_cycle_ticks = (uint64_t)(XS_HYPER_MS * (double)XPURT_TICKS_PER_SEC / 1000.0 + 0.5);
        g_xs_skip = (int)xs_env_long("XPURT_SKIP", 1) != 0;
        g_xs_gate_all = (int)xs_env_long("XPURT_GATE_ALL", 0) != 0;
        {
            const char *g_ = xs_os_getenv("XPURT_GATE");
            g_xs_gate_sleep = !(g_ && strcmp(g_, "spin") == 0);
            const char *w_ = xs_os_getenv("XPURT_WAIT");
            g_xs_wait_mode = (w_ && strcmp(w_, "spin") == 0) ? 1 : (w_ && strcmp(w_, "sleep") == 0) ? 2 : 0;
            const char *k_ = xs_os_getenv("XPURT_WAKE");
            g_xs_wake_all = (k_ && strcmp(k_, "all") == 0);
            g_xs_db = (k_ && strcmp(k_, "doorbell_sem") == 0) ? 1 :
                      (k_ && strcmp(k_, "doorbell_futex") == 0) ? 2 : 0;
#ifndef MODELBLASTER_PLATFORM_LINUX
            /* Zephyr: no futex, so the default is the k_sem doorbell (an
             * event word would be spun on). Linux keeps the S7a mask +
             * own futex word, measured cheapest (xpurt_fix Phase R). */
            if (!k_ || !*k_) g_xs_db = 1;
#endif
            for (int x_ = 0; x_ < XS_N_WORKERS; x_++) xs_os_db_init(&g_xs[x_].db);
            const char *r_ = xs_os_getenv("XPURT_RUNAHEAD_WAIT");
            g_xs_ra_poll = (r_ && strcmp(r_, "poll") == 0);
        }
        const uint64_t tpus_ = (uint64_t)XPURT_TICKS_PER_SEC / 1000000ull;
        g_xs_gate_spin_ticks = (uint64_t)xs_env_long("XPURT_GATE_SPIN_US", 50) * tpus_;
        g_xs_spin_ticks = (uint64_t)xs_env_long("XPURT_SPIN_US", 100) * tpus_;
        g_xs_stall_ticks = (uint64_t)xs_env_long("XPURT_STALL_US", 250) * tpus_;
        for (int i = 0; i < @N@; i++)
            g_xs_gate_ticks[i] = (uint64_t)((double)@T@[i].start_time_ms *
                                            (double)XPURT_TICKS_PER_SEC / 1000.0);
        /* Map the program's workers onto the discovered ones; a worker that is
         * only ever a gang helper has no table entry of its own and is added. */
        for (int w = 0; w < XPURT_MAX_WORKERS; w++) g_xs_of_worker[w] = -1;
        for (int x = 0; x < XS_N_WORKERS; x++) {
            int found = -1;
            for (int w = 0; w < g_n_workers; w++) {
                if (g_workers[w].hart == XS_WORKERS[x].hart &&
                    strcmp(g_workers[w].kind, XS_WORKERS[x].kind) == 0) { found = w; break; }
            }
            if (found < 0) {
                if (g_n_workers >= XPURT_MAX_WORKERS) {
                    printf("FATAL: XPURT_MAX_WORKERS exhausted by gang helpers\n");
                    sys_reboot(SYS_REBOOT_COLD);
                    return -1;
                }
                found = g_n_workers++;
                int k_ = 0;
                while (k_ < (int)(sizeof kinds / sizeof kinds[0]) &&
                       strcmp(kinds[k_], XS_WORKERS[x].kind) != 0) k_++;
                g_workers[found].kind = XS_WORKERS[x].kind;
                g_workers[found].hart = XS_WORKERS[x].hart;
                g_workers[found].kind_idx = k_;
                g_workers[found].worker_idx = found;
                g_workers[found].claims_unbound = 0;
                g_workers[found].observed_cpu = -1;
            }
            g_xs_of_worker[found] = x;
            g_xs[x].trace = malloc((size_t)g_cycles * (size_t)(XS_WORKERS[x].n + 1) * sizeof(struct xs_slot));
            g_xs[x].last = calloc((size_t)XS_WORKERS[x].n + 1, sizeof(struct xpurt_trace_slot));
            if (!g_xs[x].trace || !g_xs[x].last) { printf("FATAL: static trace alloc\n"); return -1; }
            memset(g_xs[x].trace, 0xff, (size_t)g_cycles * (size_t)(XS_WORKERS[x].n + 1) * sizeof(struct xs_slot));
        }
        for (int w = 0; w < g_n_workers; w++) {
            if (g_xs_of_worker[w] < 0) {
                printf("FATAL: worker %s/%d has no static program\n", g_workers[w].kind, g_workers[w].hart);
                sys_reboot(SYS_REBOOT_COLD);
                return -1;
            }
        }
        for (int g = 0; g < XS_N_GANGS; g++) {
            g_xs_gang_pool[g] = modelblaster_pool_create_gang(XS_GANGS[g].n);
            if (!g_xs_gang_pool[g]) { printf("FATAL: gang %d create\n", g); return -1; }
        }
        printf("xpurt: static walker cycles=%d hyper_ms=%.6f entries=%d workers=%d gangs=%d "
               "use_gang=%d skip=%d gate_all=%d gate=%s wait=%d rt=%d wake=%s runahead_wait=%s\n",
               g_cycles, (double)XS_HYPER_MS, @N@, XS_N_WORKERS, XS_N_GANGS, XS_USE_GANG,
               g_xs_skip, g_xs_gate_all, g_xs_gate_sleep ? "sleep" : "spin", g_xs_wait_mode, g_rt_active,
               g_xs_wake_all ? "all" : g_xs_db == 1 ? "doorbell_sem" : g_xs_db == 2 ? "doorbell_futex" : "targeted",
               g_xs_ra_poll ? "poll" : "event");
        printf("xpurt: static edges @STATS@\n");
    }
'''.replace("@N@", N).replace("@T@", T).replace(
        "@STATS@", "total=%d elided_same_worker=%d removed_by_reduction=%d remaining_cross_worker=%d "
        "epoch_cross_worker=%d spin=%d sleep=%d" % (
            st["edges_total"], st["elided_same_worker"], st["removed_by_reduction"],
            st["remaining_cross_worker"], st["epoch_cross_worker"], st["spin_edges"], st["sleep_edges"]))

    main_post = r'''    {
        uint64_t t_end_ = (uint64_t)k_cycle_get_64() - run_t0;
        printf("xpurt: static run_ticks=%llu ticks_per_sec=%llu\n",
               (unsigned long long)t_end_, (unsigned long long)XPURT_TICKS_PER_SEC);
        for (int w = 0; w < g_n_workers; w++) {
            const struct xs_wstate *s_ = &g_xs[g_xs_of_worker[w]];
            g_hart_acc[w] = s_->acc;
            printf("xpurt: static worker[%d] %s/%d items=%d skips=%llu sleeps=%llu spins_ok=%llu "
                   "futex_waits=%llu wakes_sent=%llu stall_scans=%llu ev_timeouts=%llu\n", w, g_workers[w].kind,
                   g_workers[w].hart, XS_WORKERS[g_xs_of_worker[w]].n,
                   (unsigned long long)s_->skips, (unsigned long long)s_->sleeps,
                   (unsigned long long)s_->spins_ok, (unsigned long long)s_->futex_waits,
                   (unsigned long long)s_->wakes_sent, (unsigned long long)s_->stall_scans,
                   (unsigned long long)s_->ev_timeouts);
        }
#ifdef MODELBLASTER_XPURT_TRACE
        for (int x = 0; x < XS_N_WORKERS; x++)
            for (int p = 0; p < XS_WORKERS[x].n; p++)
                if (XS_WORKERS[x].items[p].kind != 2)
                    xpurt_trace[XS_WORKERS[x].items[p].entry] = g_xs[x].last[p];
#endif
#ifdef MODELBLASTER_PLATFORM_LINUX   /* trace files: hosted only (Zephyr prints the trace block) */
        const char *p_ = getenv("XPURT_TRACE_OUT");
        if (p_ && *p_) {
            FILE *f_ = fopen(p_, "wb");
            struct xs_slot *row_ = malloc((size_t)@N@ * sizeof *row_);
            if (f_ && row_) {
                uint64_t hdr_[4] = { 0x4b33525431ULL, (uint64_t)g_cycles,
                                     (uint64_t)@N@, g_cycle_ticks };
                fwrite(hdr_, sizeof hdr_, 1, f_);
                for (int c = 0; c < g_cycles; c++) {
                    memset(row_, 0xff, (size_t)@N@ * sizeof *row_);
                    for (int x = 0; x < XS_N_WORKERS; x++)
                        for (int p = 0; p < XS_WORKERS[x].n; p++)
                            if (XS_WORKERS[x].items[p].kind != 2)
                                row_[XS_WORKERS[x].items[p].entry] =
                                    g_xs[x].trace[(size_t)c * (size_t)XS_WORKERS[x].n + p];
                    fwrite(row_, sizeof *row_, (size_t)@N@, f_);
                }
                fclose(f_);
                printf("xpurt: static trace -> %s\n", p_);
            } else {
                printf("xpurt: static cannot write %s\n", p_);
            }
            free(row_);
        }
        /* slice records (helper side of gangs), for the gang overhead numbers */
        const char *q_ = getenv("XPURT_SLICE_TRACE_OUT");
        if (q_ && *q_) {
            FILE *f_ = fopen(q_, "w");
            if (f_) {
                fprintf(f_, "cycle,worker_hart,entry,slice,start_ticks,end_ticks\n");
                for (int x = 0; x < XS_N_WORKERS; x++)
                    for (int p = 0; p < XS_WORKERS[x].n; p++)
                        if (XS_WORKERS[x].items[p].kind == 2)
                            for (int c = 0; c < g_cycles; c++) {
                                const struct xs_slot *s_ = &g_xs[x].trace[(size_t)c * (size_t)XS_WORKERS[x].n + p];
                                fprintf(f_, "%d,%d,%d,%d,%llu,%llu\n", c, XS_WORKERS[x].hart,
                                        XS_WORKERS[x].items[p].entry, XS_WORKERS[x].items[p].slice,
                                        (unsigned long long)s_->start, (unsigned long long)s_->end);
                            }
                fclose(f_);
            }
        }
#endif
    }
'''.replace("@N@", N)
    return defs, loop, main_early, main_pre, main_post


def _split_job_name(job: str, known: set[str] | None = None) -> tuple[str, int]:
    if known:
        for base in sorted(known, key=len, reverse=True):
            if job == base:
                return base, 0
            if job.startswith(base):
                rest = job[len(base):]
                if rest.isdigit():
                    return base, int(rest)
    m = _INSTANCE_RE.match(job)
    if not m:
        return job, 0
    return m.group("base"), int(m.group("idx"))


HEADER = "/* @generated by modelblaster/pipeline/generate_xpurt_main.py — do not edit. */"


def _c_ident(name: str) -> str:
    return name.replace(".", "_").replace("-", "_")


def _emit(networks: list[str], schedule_name: str,
          dispatch_table_header: str,
          core_kinds: list[str],
          backends: list[str],
          pool_sizes: list[int],
          n_instances: dict[str, int],
          platform: str = "zephyr",
          warmup: bool = False,
          walker: str = "strict",
          hyper_ms: float | None = None,
          rt_prio_default: int = 20,
          program: dict | None = None,
          static_env: dict | None = None) -> str:
    """Render xpurt_main.c source.

    `networks`     is the ordered list of distinct network names the
                   schedule references.
    `core_kinds`   is the ordered list of distinct core_kind values.
                   NOTE this no longer fixes the thread count: the walker
                   discovers the (kind, hart) worker set at RUN time by
                   scanning the dispatch table, so `core_kinds` only fixes
                   the kind -> backend pairing and which kinds are
                   executable at all.
    `backends`     is the ordered list of HW backends compiled into the
                   binary (== the per-model OBJECT-lib suffixes). Usually
                   identical to `core_kinds`, but separated to allow
                   binaries with backend pools larger than the schedule
                   used (e.g. a future fallback path).
    `pool_sizes`   is the legacy per-kind default used only when a schedule
                   does not declare a composite machine. Composite entries
                   create one persistent pool for their exact ordered hart
                   set. Per-hart locks prevent scheduler workers and pool
                   helpers from oversubscribing the same physical hart.
    `warmup`       Linux only. Before the timed t0, run each distinct
                   (network, dispatch, impl, hart set) the schedule uses ONCE,
                   untimed, on the worker/pool that will run it in the timed
                   run (see _warmup_blocks), then restore every piece of
                   state the timed run depends on.
    `n_instances`  per-network instance count (max instance index + 1).
                   Sized arrays for wall-cycle bookkeeping so periodic
                   instances of the same network don't clobber each
                   other's start/elapsed scalars."""

    if walker not in ("strict", "skip", "static"):
        raise SystemExit(f"--walker must be strict, skip or static, got {walker!r}")
    if walker == "static" and program is None:
        raise SystemExit("--walker static needs --program-json (from "
                         "ingest_xpurt_schedule --static-program-json)")
    if walker == "skip" and platform != "linux":
        raise SystemExit("--walker skip is Linux-only (futex, timed semaphore "
                         "waits, SCHED_FIFO, clock_nanosleep); on Zephyr use "
                         "--walker static (OS layer xs_os.h) or the strict "
                         "default")
    if warmup and platform != "linux":
        raise SystemExit("--warmup is Linux-only (needs pthread barriers and a "
                         "hosted OS); the Zephyr walker has no warm-up pass")
    warm_branches: list[str] = []
    warm_restore: list[str] = []
    inc_lines: list[str] = []
    output_buf_decls: list[str] = []
    forward_decls: list[str] = []
    reset_calls: list[str] = []
    branch_lines: list[str] = []
    print_blocks: list[str] = []
    wall_decls: list[str] = []

    for net in networks:
        mid = _c_ident(net)
        umid = mid.upper()
        n_inst = max(1, n_instances.get(net, 1))
        inc_lines.append(f'#include "{net}_model.h"')
        inc_lines.append(f'#include "{net}_test_io.h"')

        output_buf_decls.append(
            f"static model_{mid}_output_t out_{mid}[MODEL_{umid}_OUTPUT_SIZE];"
        )

        # Per-instance wall-cycle bookkeeping. Periodic networks issue
        # multiple instances (`dronet0`, `dronet1`, ...) — sharing one
        # scalar would mean each instance overwrites the previous.
        wall_decls.append(
            f"static uint64_t wall_start_{mid}[{n_inst}] = {{ 0 }};")
        wall_decls.append(
            f"static uint64_t wall_cycles_{mid}[{n_inst}] = {{ 0 }};")
        wall_decls.append(
            f"static int completed_dispatches_{mid}[{n_inst}] = {{ 0 }};")
        wall_decls.append(
            f"static float verify_abs_{mid} = 0.0f, verify_rel_{mid} = 0.0f;")
        wall_decls.append(f"static int verify_ready_{mid} = 0;")

        # For each (model, backend) pair: forward-declare the per-backend
        # dispatch table + the renamed profile API. The harness CMakeLists
        # builds a separate OBJECT lib per pair so these live in distinct
        # TUs and link without collision.
        for bs in backends:
            BS = bs.upper()
            forward_decls.append(
                f"extern const model_{mid}_dispatch_fn "
                f"MODEL_{umid}_DISPATCH_FNS_{BS}[MODEL_{umid}_OP_COUNT];"
            )
            forward_decls.append(
                f"void model_{mid}_reset_profile_{bs}(void);")
            forward_decls.append(
                f"const model_{mid}_op_record_t *"
                f"model_{mid}_profile_records_{bs}(int *count);")
            reset_calls.append(f"    model_{mid}_reset_profile_{bs}();")

        # Build the per-backend selection inside the dispatch branch.
        # The selection axis is the entry's `impl` -- the kernel
        # implementation for THIS dispatch -- which the ingest defaults to
        # core_kind, so a schedule that never chose per-dispatch impls
        # selects exactly as it always did. We dispatch with the
        # dispatch-local state so `.pool` follows the exact composite target
        # selected for this entry, not merely its kind or implementation.
        #
        # Cross-tile memory coherence: Gemmini's RoCC DMA reads/writes
        # memory through an independent port at the SoC coherence point,
        # NOT through the CPU store buffer or L1 dcache. On
        # GemminiAndOPUShuttleConfig, the CPU on tile 0 (which dispatches
        # to Gemmini) might still have writes sitting in its store buffer
        # when Gemmini's mvin reads, and Gemmini's mvout writes may not
        # be visible to subsequent CPU loads on tile 1 (Saturn). The
        # `fence rw,rw` before+after each dispatch fixes both. Diagnosed
        # in merlin/third_party/iree_bar/.../embedded_elf_loader.c
        # (2026-05-14) and ported here for the multi-backend xpurt
        # dispatch loop. Without these fences, hetero output diverges
        # from golden (linf ~52 on dronet).
        # Dispatch branch: strcmp against the entry's `impl` (which the
        # ingest set from the schedule, defaulting to the registry kind),
        # but invoke through the BACKEND-suffixed dispatch fn symbol (which
        # is how each backend's per-model OBJECT lib is named). The two
        # can legitimately differ (e.g. kind="gemmini" → backend tag
        # "gemmini_q31") as long as they're paired by index in the
        # callsite's --core-kinds / --backends args. Before 2026-06-03
        # this loop strcmped against `bs` (the backend tag), so any
        # mismatch silently dropped every dispatch into the else clause
        # — verify then showed all-zero outputs (static init) and the
        # bug looked like FPGA corruption. Pool selection is independent of
        # this branch and comes from the entry's complete hart set.
        if len(core_kinds) != len(backends):
            raise SystemExit(
                f"_emit: core_kinds and backends must have equal length "
                f"(got {core_kinds} vs {backends})"
            )
        bs_select_lines: list[str] = []
        for i, (kind, bs) in enumerate(zip(core_kinds, backends)):
            BS = bs.upper()
            kw = "if" if i == 0 else "else if"
            bs_select_lines.append(
                f'            {kw} (strcmp(e_->impl, "{kind}") == 0) {{\n'
                f"                __asm__ volatile(\"fence rw,rw\" ::: \"memory\");\n"
                f"                MODEL_{umid}_DISPATCH_FNS_{BS}[e_->dispatch_id](&dispatch_state_{mid});\n"
                f"                __asm__ volatile(\"fence rw,rw\" ::: \"memory\");\n"
                f"            }}"
            )
        bs_select_lines.append(
            "            else {\n"
            f'                printf("xpurt: FATAL entry %d of {net} asks for impl \'%s\', '
            f'which this binary was not built with (has: {"/".join(core_kinds)}). '
            f'Either the schedule chose a per-dispatch implementation the build '
            f'lacks, or the kind→backend pairing is wrong at codegen; see '
            f'pipeline/generate_xpurt_main.py\\n", e_->entry_id, e_->impl);\n'
            f"                sys_reboot(SYS_REBOOT_COLD);\n"
            "            }"
        )
        bs_select = "\n".join(bs_select_lines)
        if warmup:
            warm_branches.append(
                f'    if (strcmp(e_->network, "{net}") == 0) {{\n'
                f"        model_{mid}_state_t dispatch_state_{mid} = {{\n"
                f"            .input = model_{mid}_test_input,\n"
                f"            .output = out_{mid},\n"
                f"            .pool = (void *)pool_for_entry(e_),\n"
                f"        }};\n"
                f"        lock_entry_harts(e_);\n"
                f"{bs_select}\n"
                f"        unlock_entry_harts(e_);\n"
                f"        return;\n"
                f"    }}")
            forward_decls.append(f"void model_{mid}_zero_buffers(void);")
            warm_restore.append(
                f"        model_{mid}_zero_buffers();\n"
                f"        memset(out_{mid}, 0, sizeof out_{mid});")

        branch_lines.append(
            f'        if (e_->network[0] == \'{net[0]}\' && '
            f'strcmp(e_->network, "{net}") == 0) {{\n'
            f"            /* Zero-cost IR ops (view, chunk2_c1) get\n"
            f"             * dispatch_id=-1 from ingest — they were\n"
            f"             * filtered out of the dispatch fn table by\n"
            f"             * generate_skeleton.py. Skip the kernel call\n"
            f"             * but still post the completion sem so any\n"
            f"             * dependents downstream unblock. */\n"
            f"            if (e_->dispatch_id < 0) {{\n"
            f"                /* Phase G2d producer-side fanout: give this\n"
            f"                 * entry's sem once per downstream consumer\n"
            f"                 * (data dep + time dep, dedup'd at ingest).\n"
            f"                 * Consumers take once with no re-give. Leaf\n"
            f"                 * entries (n_fanout=0) skip the give entirely. */\n"
            f"                for (int _f = 0; _f < e_->n_fanout; _f++) {{\n"
            f"                    k_sem_give(&completion_sems[i_]);\n"
            f"                }}\n"
            f"                prev_iter_end = (uint64_t)k_cycle_get_64();\n"
            f"                g_hart_acc[my_acc_idx].entries_done++;\n"
            f"                continue;\n"
            f"            }}\n"
            f"            if (e_->dispatch_id == 0) {{\n"
            + "\n".join(
                f"                model_{mid}_reset_profile_{bs}();"
                for bs in backends
            ) + "\n"
            f"                wall_start_{mid}[e_->instance] = (uint64_t)k_cycle_get_64();\n"
            f"            }}\n"
            f"            model_{mid}_state_t dispatch_state_{mid} = {{\n"
            f"                .input = model_{mid}_test_input,\n"
            f"                .output = out_{mid},\n"
            f"                .pool = (void *)pool_for_entry(e_),\n"
            f"            }};\n"
            f"            lock_entry_harts(e_);\n"
            f"            uint64_t t_disp0 = (uint64_t)k_cycle_get_64();\n"
            f"#ifdef MODELBLASTER_XPURT_TRACE\n"
            f"            xpurt_trace[i_].start_cycles = t_disp0 - run_t0;\n"
            f"#endif\n"
            f"#ifndef MODELBLASTER_PLATFORM_LINUX\n"
            f"            /* Re-arm mstatus.VS right before the kernel\n"
            f"             * dispatch. The worker's once-at-entry VS=Initial\n"
            f"             * write gets clobbered by Zephyr's V state\n"
            f"             * save/restore on context switches triggered by\n"
            f"             * k_sem_take in dep_wait — so by the time we\n"
            f"             * reach the kernel, VS may be Off again. csrs\n"
            f"             * to MSTATUS.VS is a no-op on harts whose VS is\n"
            f"             * hardwired-zero (no misa.V), so this is safe to\n"
            f"             * issue unconditionally.\n"
            f"             *\n"
            f"             * MACHINE MODE ONLY -- csrs mstatus traps in U-mode.\n"
            f"             * A hosted OS saves and restores vector state across\n"
            f"             * context switches itself, so there is nothing to\n"
            f"             * re-arm; this whole hazard is Zephyr-specific. */\n"
            f"            asm volatile(\"csrs mstatus, %0\" : : \"r\"((unsigned long)(1UL << 9)));\n"
            f"#endif\n"
            f"{bs_select}\n"
            f"            uint64_t t_disp1 = (uint64_t)k_cycle_get_64();\n"
            f"            unlock_entry_harts(e_);\n"
            f"            g_hart_acc[my_acc_idx].kernel += (t_disp1 - t_disp0);\n"
            f"#ifdef MODELBLASTER_XPURT_TRACE\n"
            f"            xpurt_trace[i_].end_cycles = t_disp1 - run_t0;\n"
            f"            xpurt_trace[i_].worker_kind_idx = my_kind_idx;\n"
            f"            xpurt_trace[i_].worker_hart = my_hart;\n"
            f"#endif\n"
            f"#ifdef MODELBLASTER_XPURT_STREAM\n"
            f"            /* One JSON line per dispatch END, for the host-side\n"
            f"             * feedback tailer. Same numbers the trace block\n"
            f"             * dumps at exit -- the difference is WHEN: the trace\n"
            f"             * is only readable once the run is over, so nothing\n"
            f"             * can respond to a deadline miss while the run is\n"
            f"             * still going. Cycles are rdtime TICKS at 24 MHz on\n"
            f"             * this board, not core cycles.\n"
            f"             *\n"
            f"             * ONE write() PER LINE, not printf. Several workers\n"
            f"             * reach this concurrently, and interleaved printf\n"
            f"             * calls tear a line into fragments that are not\n"
            f"             * JSON. Formatting into a stack buffer and issuing a\n"
            f"             * single write of well under PIPE_BUF keeps each\n"
            f"             * line whole. */\n"
            f"            {{\n"
            f"                char sbuf_[320];\n"
            f"                int sn_ = snprintf(sbuf_, sizeof sbuf_,\n"
            f"                    \"{{\\\"entry_id\\\":%d,\\\"network\\\":\\\"%s\\\",\\\"instance\\\":%d,\"\n"
            f"                    \"\\\"dispatch_id\\\":%d,\\\"impl\\\":\\\"%s\\\",\\\"hart\\\":%d,\"\n"
            f"                    \"\\\"predicted_start_ms\\\":%.6f,\\\"predicted_duration_ms\\\":%.6f,\"\n"
            f"                    \"\\\"start_ticks\\\":%llu,\\\"end_ticks\\\":%llu}}\\n\",\n"
            f"                    e_->entry_id, e_->network, e_->instance,\n"
            f"                    e_->dispatch_id, e_->impl, my_hart,\n"
            f"                    (double)e_->start_time_ms, (double)e_->duration_ms,\n"
            f"                    (unsigned long long)(t_disp0 - run_t0),\n"
            f"                    (unsigned long long)(t_disp1 - run_t0));\n"
            f"                if (sn_ > 0) {{ ssize_t w_ = write(1, sbuf_, (size_t)sn_); (void)w_; }}\n"
            f"            }}\n"
            f"#endif\n"
            f"            /* A DAG may have several output leaves. Numeric\n"
            f"             * verification and wall completion belong to the\n"
            f"             * LAST completed dispatch, not the numerically last\n"
            f"             * dispatch id (DroNet's two heads finish in either\n"
            f"             * order). */\n"
            f"            int _model_done = __atomic_add_fetch(\n"
            f"                &completed_dispatches_{mid}[e_->instance], 1,\n"
            f"                __ATOMIC_ACQ_REL);\n"
            f"            if (_model_done == MODEL_{umid}_OP_COUNT) {{\n"
            f"                wall_cycles_{mid}[e_->instance] =\n"
            f"                    (uint64_t)k_cycle_get_64() - wall_start_{mid}[e_->instance];\n"
            f"                /* Capture instance 0 before a stateful model's\n"
            f"                 * later invocations advance its state. The baked\n"
            f"                 * golden is a one-invocation golden. */\n"
            f"                if (e_->instance == 0) {{\n"
            f"                    __asm__ volatile(\"fence rw, rw\" ::: \"memory\");\n"
            f"                    float _mae = 0.0f, _mre = 0.0f;\n"
            f"                    for (int _v = 0; _v < MODEL_{umid}_TEST_OUTPUT_LEN; _v++) {{\n"
            f"                        float a = (float)out_{mid}[_v];\n"
            f"                        float g = (float)model_{mid}_test_golden[_v];\n"
            f"                        float ae = a > g ? a - g : g - a;\n"
            f"                        float ag = g > 0.0f ? g : -g;\n"
            f"                        float re = ae / (ag > 1e-12f ? ag : 1e-12f);\n"
            f"                        if (ae > _mae) _mae = ae;\n"
            f"                        if (re > _mre) _mre = re;\n"
            f"                    }}\n"
            f"                    verify_abs_{mid} = _mae;\n"
            f"                    verify_rel_{mid} = _mre;\n"
            f"                    verify_ready_{mid} = 1;\n"
            f"                }}\n"
            f"            }}\n"
            f"            /* Phase G2d: producer-side fanout. Give this\n"
            f"             * entry's sem once per downstream consumer; the\n"
            f"             * consumer-side take loop does NOT re-give (saves\n"
            f"             * one sem op per dep edge). Leaf entries skip the\n"
            f"             * give entirely. */\n"
            f"            for (int _f = 0; _f < e_->n_fanout; _f++) {{\n"
            f"                k_sem_give(&completion_sems[i_]);\n"
            f"            }}\n"
            f"            uint64_t t_iter_end = (uint64_t)k_cycle_get_64();\n"
            f"            g_hart_acc[my_acc_idx].sync_overhead += (t_iter_end - t_disp1);\n"
            f"            prev_iter_end = t_iter_end;\n"
            f"            g_hart_acc[my_acc_idx].entries_done++;\n"
            f"            continue;\n"
            f"        }}"
        )

        # Per-backend profile record dump. Each backend's TU has its own
        # records array; entries that ran on backend X land in X's array.
        # We tag rows with the backend so the host can split per-op cost
        # by backend.
        per_bs_profile_blocks: list[str] = []
        for bs in backends:
            per_bs_profile_blocks.append(f"""\
        {{
            int n_records = 0;
            const model_{mid}_op_record_t *records =
                model_{mid}_profile_records_{bs}(&n_records);
            for (int i = 0; i < n_records; i++) {{
                printf("{bs},%d,%s,%s,%s,%lu\\n",
                       records[i].dispatch_id,
                       records[i].name, records[i].op, records[i].shape,
                       records[i].cycles);
            }}
        }}""")
        per_bs_profile = "\n".join(per_bs_profile_blocks)

        # Per-instance wall-cycles dump under a NEW marker so streamed
        # runners (firesim_runner, spike_runner) that count
        # `=== MODELBLASTER_WALL_CYCLES [<net>] === <int>` for end-of-run
        # detection still see exactly one bare WALL_CYCLES per network
        # below. The per-instance variant uses
        # `=== MODELBLASTER_WALL_CYCLES_INST [<net>#<i>] ===` and is captured
        # for analysis by the host parser.
        # Output dump strategy: HTIF UART throughput is ~6 KB/sec on
        # FireSim Saturn. Dumping every output value as ASCII works for
        # small models (dronet: 2 values, mlp_control: 4) but is the
        # workflow bottleneck for yolov8-scale outputs (75600 values
        # blow well past the FIRESIM_TIMEOUT). For large outputs we
        # emit a *sampled* dump — first 8, last 8, plus a checksum/L1
        # summary for spot-check parity — and rely on per-model unit
        # tests (or a separate dedicated verify run) for full
        # bit-exactness. The marker tags stay the same so existing
        # parsers continue to find OUTPUT_BEGIN/END.
        print_blocks.append(f"""\
    /* Instance 0 is comparable to the baked one-invocation golden even when
     * this model is stateful and later invocations intentionally evolve it. */
    printf("=== MODELBLASTER_VERIFY [{net}] === "
           "max_abs_err=%.9g max_rel_err=%.9g n=%d instance=0 ready=%d\\n",
           (double)verify_abs_{mid}, (double)verify_rel_{mid},
           MODEL_{umid}_TEST_OUTPUT_LEN, verify_ready_{mid});
    printf("=== MODELBLASTER_OUTPUT_BEGIN [{net}] ===\\n");
    {{
        const int _osz = MODEL_{umid}_OUTPUT_SIZE;
        const int _SAMPLE_THRESHOLD = 64;
        if (_osz <= _SAMPLE_THRESHOLD) {{
            for (int i = 0; i < _osz; i++)
                printf("%.9g\\n", (double)out_{mid}[i]);
        }} else {{
            /* Sampled dump + summary. */
            double _sum = 0.0, _abs_sum = 0.0;
            double _omax = -1.0/0.0, _omin = 1.0/0.0;
            for (int i = 0; i < _osz; i++) {{
                double v = (double)out_{mid}[i];
                _sum += v;
                _abs_sum += (v < 0) ? -v : v;
                if (v > _omax) _omax = v;
                if (v < _omin) _omin = v;
            }}
            printf("OUTPUT_SAMPLED size=%d head=8 tail=8\\n", _osz);
            for (int i = 0; i < 8; i++)
                printf("  [%d] %.9g\\n", i, (double)out_{mid}[i]);
            for (int i = _osz - 8; i < _osz; i++)
                printf("  [%d] %.9g\\n", i, (double)out_{mid}[i]);
            printf("OUTPUT_SUMMARY sum=%.6g abs_sum=%.6g min=%.6g max=%.6g\\n",
                   _sum, _abs_sum, _omin, _omax);
        }}
    }}
    printf("=== MODELBLASTER_OUTPUT_END [{net}] ===\\n");
    printf("=== MODELBLASTER_PROFILE_BEGIN [{net}] ===\\n");
    printf("backend,dispatch_id,name,op,shape,cycles\\n");
{per_bs_profile}
    printf("=== MODELBLASTER_PROFILE_END [{net}] ===\\n");
    {{
        unsigned long _max_wc = 0;
        for (int _i = 0; _i < {n_inst}; _i++) {{
            unsigned long _wc = (unsigned long)wall_cycles_{mid}[_i];
            printf("=== MODELBLASTER_WALL_CYCLES_INST [{net}#%d] === %lu\\n",
                   _i, _wc);
            if (_wc > _max_wc) _max_wc = _wc;
        }}
        /* Worst-case per-instance wall cycles — meaningful for periodic
         * tasks (highest observed latency) and identical to the single
         * value for non-periodic networks. */
        printf("=== MODELBLASTER_WALL_CYCLES [{net}] === %lu\\n", _max_wc);
    }}""")

    upper = schedule_name.replace(".", "_").replace("-", "_").upper()

    # Build the per-(network, instance, dispatch_id) dispatch branch.
    dispatch_branch = "\n".join(branch_lines)

    # One Zephyr worker thread per distinct core_kind. Each pins itself
    # to a hart (the entry's .hart) and pulls only entries whose
    # core_kind matches.
    n_kinds = len(core_kinds)
    kind_strs = ", ".join(f'"{k}"' for k in core_kinds)
    if len(pool_sizes) != n_kinds:
        raise SystemExit(
            f"pool_sizes length ({len(pool_sizes)}) must match core_kinds "
            f"length ({n_kinds})")
    pool_size_strs = ", ".join(str(s) for s in pool_sizes)

    # Forward decls go right after the model headers — they reference
    # types from those headers (model_<mid>_dispatch_fn etc.).
    forward_decl_block = "\n".join(forward_decls)
    try:
        platform_prologue = _PROLOGUES[platform]
        platform_top = _TOPS[platform]
    except KeyError:
        raise SystemExit(f"--platform must be one of {sorted(_PROLOGUES)}, "
                         f"got {platform!r}")

    warm_defs, warm_worker, warm_main_pre, warm_main_post = _warmup_blocks(
        upper, warm_branches, warm_restore + reset_calls) if warmup else ("", "", "", "")

    # The strict table-order walker loop (the only walker until 2026-11; see
    # _v2_blocks for the skipping walker). Kept verbatim so the default output
    # is byte-identical to the pre-v2 generator.
    walker_loop = f"""\
    for (int i_ = 0; i_ < {upper}_N_ENTRIES; i_++) {{
        const xpurt_sched_entry_t *e_ = &{upper}_TABLE[i_];
        if (strcmp(e_->core_kind, my_kind) != 0) continue;
        if (e_->hart >= 0) {{
            if (e_->hart != my_hart) continue;
        }} else if (!my_claims_unbound) {{
            continue;
        }}

        uint64_t t_iter_start = (uint64_t)k_cycle_get_64();
        if (g_hart_acc[my_acc_idx].entries_done > 0) {{
            g_hart_acc[my_acc_idx].hart_idle +=
                (t_iter_start - prev_iter_end);
        }}

        /* Wait for all data deps + the time_dep edge to complete.
         * Phase G2d: producer gives its sem once per UNIQUE consumer
         * (dedup'd at ingest), consumer takes once per UNIQUE dep
         * here. No re-give — the counts balance. The time_dep edge
         * dedup against deps[] preserves the invariant when one
         * producer is both a data and a time dep of the same
         * consumer (without dedup, consumer takes twice from one
         * producer give → deadlock). */
        for (int d = 0; d < e_->n_deps; d++) {{
            k_sem_take(&completion_sems[e_->deps[d]], K_FOREVER);
        }}
        if (e_->time_dep_entry_id >= 0) {{
            int _td_is_dup = 0;
            for (int _d = 0; _d < e_->n_deps; _d++) {{
                if (e_->deps[_d] == e_->time_dep_entry_id) {{
                    _td_is_dup = 1;
                    break;
                }}
            }}
            if (!_td_is_dup) {{
                k_sem_take(&completion_sems[e_->time_dep_entry_id], K_FOREVER);
            }}
        }}
        uint64_t t_deps_done = (uint64_t)k_cycle_get_64();
        g_hart_acc[my_acc_idx].dep_wait += (t_deps_done - t_iter_start);

        /* Every dispatch has an absolute schedule-issued earliest start.
         * Gating only dispatch_id==0 is insufficient for a DAG with multiple
         * roots: another root can otherwise run before its periodic release,
         * then post dependencies early. Dependencies and this gate are both
         * lower bounds; actual resource drift may still start an entry later. */
        uint64_t target_start = run_t0 +
            (uint64_t)((double)e_->start_time_ms *
                       (double)XPURT_CYCLES_PER_MS);
        uint64_t t_gate0 = (uint64_t)k_cycle_get_64();
        while ((uint64_t)k_cycle_get_64() < target_start) {{
            k_yield();
        }}
        g_hart_acc[my_acc_idx].target_gate_spin +=
            ((uint64_t)k_cycle_get_64() - t_gate0);

{dispatch_branch}
        printf("xpurt: WARN unknown network %s in entry %d\\n",
               e_->network, e_->entry_id);
        /* Unknown network — give the sem anyway so we don't deadlock.
         * Fan out per the producer-side invariant so any downstream
         * consumers actually unblock. */
        for (int _f = 0; _f < e_->n_fanout; _f++) {{
            k_sem_give(&completion_sems[i_]);
        }}
        prev_iter_end = (uint64_t)k_cycle_get_64();
        g_hart_acc[my_acc_idx].entries_done++;
    }}"""
    v2_defs = v2_main_early = v2_main_pre = v2_main_post = ""
    if walker == "skip":
        v2_defs, walker_loop, v2_main_early, v2_main_pre, v2_main_post = \
            _v2_blocks(upper, dispatch_branch, hyper_ms, rt_prio_default)
    elif walker == "static":
        v2_defs, walker_loop, v2_main_early, v2_main_pre, v2_main_post = \
            _static_blocks(upper, dispatch_branch, program, networks,
                           core_kinds, hyper_ms, rt_prio_default, static_env)

    src = f"""{HEADER}
{platform_top}/*
 * Schedule-driven multi-network entry point with per-(core_kind, hart)
 * worker threads and HETEROGENEOUS per-backend dispatch tables. One
 * worker is spawned for each (core_kind, hart) pair the dispatch table
 * uses, pinned to that hart, so the schedule's per-core placement is what
 * actually executes. Each worker:
 *   1. Walks the XPU-RT-emitted dispatch table in start_time order,
 *      taking entries whose .core_kind AND .hart match its own.
 *   2. Before invoking, waits on k_sems posted by the producers of its
 *      .deps[] (intra-job data deps) and .time_dep_entry_id (cross-job
 *      ordering edges).
 *   3. Selects the per-(model, backend) dispatch table by .impl -- the
 *      IMPLEMENTATION the schedule chose for this dispatch, which
 *      defaults to .core_kind -- and invokes the matching kernel.
 *      Selecting on .impl rather than .core_kind is what lets one core
 *      run a MAC-unit GEMM and then a vector one. backend_rename.py at compile
 *      time has suffixed every model's externally-visible symbol with
 *      _<bs>, so multiple backends per model link cleanly.
 *   4. After invoking, gives the entry's k_sem so consumers unblock.
 *
 * Deadlock freedom is unchanged by the extra threads. ingest asserts that
 * every dep and time_dep edge points STRICTLY BACKWARD in entry_id order,
 * and every worker walks 0..N-1 monotonically. Take the blocked worker
 * with the smallest current index i: it waits on some j < i, whose owner
 * has current index <= j < i, so that owner is either running or is a
 * blocked worker with a smaller index -- contradicting minimality. The
 * argument never mentions how entries are partitioned across threads, so
 * it holds for one worker per (kind, hart) exactly as it did for one
 * worker per kind.
 *
 * Workers are pinned to harts via pthread_attr_setaffinity_np on Zephyr; on
 * Linux each worker places ITSELF with mb_place_current_thread() as its first
 * action (a K3 A100 thread must register its own TID before it may pin, and
 * attr affinity runs before any TID exists). Composite
 * entries reserve every hart in their machine combination under ordered
 * per-hart locks and select a persistent modelblaster_pool pinned to that
 * exact set. Scheduler workers on helper harts sleep on the locks while the
 * shard is active, so inter-op and intra-op parallelism never oversubscribe.
 *
 * Outputs follow the same multi-line marker protocol as the
 * straight-line multi_main, so spike_runner verifies them against
 * PyTorch goldens unchanged. The PROFILE block now has a leading
 * `backend` column so per-op cycles can be split per backend.
 */

#define MODELBLASTER_DISABLE_UNMANGLED

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include "modelblaster_pool.h"
{platform_prologue}

{chr(10).join(inc_lines)}
#include "{dispatch_table_header}"

/* Note (2026-05-28): we previously tried several memory-fence and
 * cache-invalidation strategies to fix `dronet_hetero` verify-fail
 * (max_abs_err=52 vs golden). All failed identically:
 *
 *   - fence rw,rw before+after each dispatch:    output unchanged
 *   - L1 dcache eviction via 128KB scratch read: output unchanged
 *   - Zicbom cbo.inval:                          illegal-instr trap
 *     (mcause=2, mtval=0x7a00f) -- bitstream
 *     was not built with Zicbom.
 *
 * Persistent identical failure across these very different remedies
 * indicates the bug is NOT cross-tile cache staleness. The real
 * cause is cross-backend per-op numerical drift: gemmini and
 * rvv_opu each match PyTorch golden bit-exactly END-TO-END, but
 * their intermediate per-op outputs differ slightly due to
 * backend-specific rounding/quantization details (each backend's
 * full chain happens to cancel its own drift into a golden-matching
 * final answer; mixing chains doesn't preserve that property).
 *
 * Documented in notes/known_issues.md. Fix path is cross-backend
 * numerical alignment, not memory mgmt. fence rw,rw is kept on the
 * dispatch boundaries below as good hygiene (it WAS load-bearing
 * for Gemmini DMA in merlin's loader; harmless and ~zero cost). */

/* Per-(model, backend) externs — the harness compiles a separate
 * OBJECT lib per pair, with externally-visible symbols suffixed _<bs>
 * via -D renames (see modelblaster/pipeline/backend_rename.py). */
{forward_decl_block}

{chr(10).join(output_buf_decls)}

/* Per-entry completion sem: posted once when the dispatch finishes;
 * dependents take it before invoking. Initialized as 0/1 (binary, but
 * grow the limit so multiple consumers can each take their own
 * "completed" reading without blocking each other). */
static struct k_sem completion_sems[{upper}_N_ENTRIES];

/* Phase G1 — per-hart runtime attribution counters, accumulated in
 * mtime ticks (1 us each on chipyard FireSim). Read at end-of-run and
 * dumped under MODELBLASTER_HART_ACC for host-side decomposition of
 * the predicted-vs-measured wall gap. Indexed by my_kind_idx. */
#ifndef XPURT_MAX_KINDS
#define XPURT_MAX_KINDS 8
#endif
/* Upper bound on scheduler-worker threads == distinct (core_kind, hart)
 * pairs the dispatch table uses. 32 covers every board we target (the K1
 * has 8 harts); main() aborts loudly rather than truncating if a schedule
 * ever exceeds it. */
#ifndef XPURT_MAX_WORKERS
#define XPURT_MAX_WORKERS 32
#endif
struct xpurt_hart_acc {{
    uint64_t kernel;            /* mtime delta around bs_select dispatch */
    uint64_t dep_wait;          /* mtime inside k_sem_take loop */
    uint64_t sync_overhead;     /* mtime between kernel-end and end of iter */
    uint64_t target_gate_spin;  /* mtime inside schedule-start gate */
    uint64_t hart_idle;         /* mtime between iters (gap to next match) */
    uint64_t gemmini_cfg_emit;  /* reserved for G2b caching wrappers */
    uint64_t entries_done;      /* count of dispatched entries */
    uint64_t wall_total;        /* worker_t0 → worker_exit delta */
}};
/* Indexed by WORKER index, not kind index — since 2026-08-27 there is one
 * worker per (core_kind, hart), so a per-kind array would have several
 * threads racing on the same counters and would report a kind's busy time
 * as if one hart had done it all. */
static struct xpurt_hart_acc g_hart_acc[XPURT_MAX_WORKERS];

/* Per-instance wall-clock state. Each periodic instance writes its own
 * slot indexed by entry->instance, so back-to-back instances of one
 * network don't clobber each other's start/elapsed scalars. */
{chr(10).join(wall_decls)}

/* mtime baseline for the run, captured immediately before workers spawn.
 * Used by the worker to (a) honor every entry's absolute schedule-issued
 * start time and (b) tag actual_start/end_cycles in the trace. */
static uint64_t run_t0;
#define XPURT_CYCLES_PER_MS \\
    ((uint64_t)(XPURT_TICKS_PER_SEC / 1000))

#ifdef MODELBLASTER_XPURT_TRACE
/* Per-entry execution trace. Each slot is still touched by exactly one
 * worker, and that is now enforced rather than asserted: the claim
 * predicate in xpurt_worker() partitions the table by (core_kind, hart),
 * a partition with no overlap by construction, and main() audits at
 * startup that every entry is claimed by EXACTLY ONE worker before any
 * thread is spawned (see the claim-coverage audit). `worker_hart` records
 * which hart actually executed the slot, so an entry that ran somewhere
 * other than where the schedule placed it (column `hart`) is visible in
 * the CSV instead of being invisible behind a shared kind index.
 * Pre-initialized to -1/-1 so an entry NO worker ran is distinguishable
 * from one that ran on kind 0 / hart 0 at cycle 0.
 * Cycles are mtime-based (k_cycle_get_64) and offset against `run_t0`,
 * captured immediately before the first worker starts. The host parser
 * reconstructs (start_ms, end_ms) by dividing by the configured
 * clock_mhz. */
struct xpurt_trace_slot {{
    uint64_t start_cycles;
    uint64_t end_cycles;
    int      worker_kind_idx;   /* index into kinds[] */
    int      worker_hart;       /* hart the executing worker was pinned to */
}};
static struct xpurt_trace_slot xpurt_trace[{upper}_N_ENTRIES];
#endif

struct xpurt_worker_arg {{
    const char *kind;
    int hart;             /* hart this worker is pinned to; -1 = unpinned */
    int kind_idx;         /* index into kinds[] / pools[] */
    int worker_idx;       /* index into g_hart_acc[] / tids[] */
    int claims_unbound;   /* 1 => also claims this kind's hart<0 entries */
    int observed_cpu;     /* sched_getcpu() at worker entry; -1 if unknown */
}};

/* The worker set: one entry per (core_kind, hart) pair the dispatch table
 * actually uses. Built in main() by scanning the table, BEFORE the pools
 * are created (the pool sizing depends on how many workers a kind got). */
static struct xpurt_worker_arg g_workers[XPURT_MAX_WORKERS];
static int g_n_workers;

/* A schedule may reserve one hart or an aligned 2/4/...-hart block for one
 * dispatch. Keep one persistent pool per distinct composite block and one
 * mutex per physical hart. The dispatch-owning worker is harts[0]; helpers
 * are pinned to the remaining entries by modelblaster_pool_create_on_harts.
 * Every master acquires hart locks in numeric order, so overlapping blocks
 * serialize without a lock-order cycle even when actual durations drift. */
#ifndef XPURT_MAX_HARTS
#define XPURT_MAX_HARTS 256
#endif
#ifndef XPURT_MAX_POOL_WIDTH
#define XPURT_MAX_POOL_WIDTH 16
#endif
#ifndef XPURT_MAX_POOLS
#define XPURT_MAX_POOLS 32
#endif
struct xpurt_pool_desc {{
    int n_harts;
    int harts[XPURT_MAX_POOL_WIDTH];
    modelblaster_pool_t pool;
}};
static pthread_mutex_t g_hart_locks[XPURT_MAX_HARTS];
static struct xpurt_pool_desc g_entry_pools[XPURT_MAX_POOLS];
static int g_n_entry_pools;

static int same_harts(const struct xpurt_pool_desc *p,
                      const xpurt_sched_entry_t *e)
{{
    if (p->n_harts != e->n_harts) return 0;
    for (int i = 0; i < e->n_harts; i++) {{
        if (p->harts[i] != e->harts[i]) return 0;
    }}
    return 1;
}}

static modelblaster_pool_t pool_for_entry(const xpurt_sched_entry_t *e)
{{
    if (e->n_harts <= 1) return NULL;
    for (int i = 0; i < g_n_entry_pools; i++) {{
        if (same_harts(&g_entry_pools[i], e)) return g_entry_pools[i].pool;
    }}
    return NULL;  /* init_entry_runtime audits this before workers spawn */
}}

static int init_entry_runtime(void)
{{
    for (int h = 0; h < XPURT_MAX_HARTS; h++) {{
        if (pthread_mutex_init(&g_hart_locks[h], NULL) != 0) return -1;
    }}
    g_n_entry_pools = 0;
    for (int i = 0; i < {upper}_N_ENTRIES; i++) {{
        const xpurt_sched_entry_t *e = &{upper}_TABLE[i];
        if (e->n_harts < 1 || e->n_harts > XPURT_MAX_POOL_WIDTH) return -2;
        for (int j = 0; j < e->n_harts; j++) {{
            if (e->harts[j] < 0 || e->harts[j] >= XPURT_MAX_HARTS) return -3;
        }}
        if (e->n_harts == 1 || pool_for_entry(e) != NULL) continue;
        if (g_n_entry_pools >= XPURT_MAX_POOLS) return -4;
        struct xpurt_pool_desc *p = &g_entry_pools[g_n_entry_pools];
        p->n_harts = e->n_harts;
        for (int j = 0; j < e->n_harts; j++) p->harts[j] = e->harts[j];
        p->pool = modelblaster_pool_create_on_harts(p->n_harts, p->harts);
        if (p->pool == NULL) return -5;
        printf("modelblaster_pool[block=%d]: width=%d harts=",\
               g_n_entry_pools, p->n_harts);
        for (int j = 0; j < p->n_harts; j++)
            printf("%s%d", j ? "+" : "", p->harts[j]);
        printf("\\n");
        g_n_entry_pools++;
    }}
    return 0;
}}

static void lock_entry_harts(const xpurt_sched_entry_t *e)
{{
    int ordered[XPURT_MAX_POOL_WIDTH];
    for (int i = 0; i < e->n_harts; i++) ordered[i] = e->harts[i];
    for (int i = 1; i < e->n_harts; i++) {{
        int v = ordered[i], j = i - 1;
        while (j >= 0 && ordered[j] > v) {{ ordered[j + 1] = ordered[j]; j--; }}
        ordered[j + 1] = v;
    }}
    for (int i = 0; i < e->n_harts; i++)
        pthread_mutex_lock(&g_hart_locks[ordered[i]]);
}}

static void unlock_entry_harts(const xpurt_sched_entry_t *e)
{{
    int ordered[XPURT_MAX_POOL_WIDTH];
    for (int i = 0; i < e->n_harts; i++) ordered[i] = e->harts[i];
    for (int i = 1; i < e->n_harts; i++) {{
        int v = ordered[i], j = i - 1;
        while (j >= 0 && ordered[j] > v) {{ ordered[j + 1] = ordered[j]; j--; }}
        ordered[j + 1] = v;
    }}
    for (int i = e->n_harts - 1; i >= 0; i--)
        pthread_mutex_unlock(&g_hart_locks[ordered[i]]);
}}

{v2_defs}\
{warm_defs}
static void *xpurt_worker(void *arg)
{{
    struct xpurt_worker_arg *wa = (struct xpurt_worker_arg *)arg;
    const char *my_kind = wa->kind;
    int my_kind_idx = wa->kind_idx;
    int my_hart = wa->hart;
    int my_acc_idx = wa->worker_idx;
    int my_claims_unbound = wa->claims_unbound;
#ifdef MODELBLASTER_PLATFORM_LINUX
    /* FIRST ACTION: place this thread on its hart. On a K3 an A100 CPU is
     * reachable only after the thread writes its own TID to
     * /proc/set_ai_thread, so this cannot be done by the creator through
     * pthread_attr_setaffinity_np (no TID exists yet). mb_place_current_thread
     * registers if needed, pins, and verifies sched_getcpu(). Failure is fatal:
     * an unplaced worker makes every per-hart number in this trace a fiction.
     * Unbound workers (hart < 0) are deliberately left floating. */
    if (my_hart >= 0 && mb_place_current_thread(my_hart) != 0) {{
        printf("FATAL: mb_place_current_thread worker=%d kind=%s hart=%d -- "
               "refusing to run unpinned, every per-hart number in the trace "
               "would be fiction\\n", my_acc_idx, my_kind, my_hart);
        sys_reboot(SYS_REBOOT_COLD);
        return NULL;
    }}
    /* Record where the scheduler ACTUALLY put us; main() prints observed_cpu
     * next to the requested hart after the join. */
    wa->observed_cpu = sched_getcpu();
#else
    wa->observed_cpu = -1;
#endif
    /* Phase G1 attribution: track per-iter timing so we can decompose
     * wall_total into kernel + dep_wait + sync_overhead +
     * target_gate_spin + hart_idle. prev_iter_end is updated INSIDE
     * each per-net branch just before `continue` so hart_idle on the
     * next iter is correctly (this_iter_start - last_iter_end). */

    /* Per-hart V-extension enable. Zephyr's HAS_V() macro reads
     * misa from the PRIMARY hart only — on our 2-tile
     * FireSimGemminiAndOPUShuttleConfig that's hart 0 (Rocket +
     * Gemmini, misa.V=0) so Zephyr leaves mstatus.VS = Off
     * EVERYWHERE, including on hart 1 (Shuttle + Saturn-OPU) which
     * does have V (misa.V=1, verified by examples/opu_probe). If we
     * don't flip VS to Initial here, the first vsetvli in any RVV
     * kernel dispatched to this hart traps with illegal-instruction.
     *
     * Behavior: read THIS hart's misa, and if the V bit is set,
     * raise mstatus.VS to Initial (writes to mstatus.VS are silently
     * ignored on harts whose misa.V=0, so this is a no-op on
     * non-V harts and safe to run unconditionally per worker). */
#ifndef MODELBLASTER_PLATFORM_LINUX
    /* MACHINE MODE ONLY. Both `csrr misa` and `csrs mstatus` are privileged;
     * under a hosted OS the process runs in U-mode and the first of them traps
     * with SIGILL before a single dispatch executes. On Linux there is also
     * nothing to do: the kernel owns per-thread vector state and enables it
     * lazily on first use, which is exactly what this block hand-rolls for
     * Zephyr. Measured on the SpaceMiT K1: without this guard the harness dies
     * immediately, `dmesg` showing cause=2 badaddr=0x301027f3, which decodes to
     * `csrr a5, misa`. */
    {{
        unsigned long _misa;
        asm volatile("csrr %0, misa" : "=r"(_misa));
        if (_misa & (1UL << 21)) {{  /* 'V' - 'A' = 21 */
            unsigned long _vs_init = (1UL << 9);  /* MSTATUS_VS = Initial */
            asm volatile("csrs mstatus, %0" : : "r"(_vs_init));
        }}
    }}
#endif

{warm_worker}
    uint64_t worker_t0 = (uint64_t)k_cycle_get_64();
    uint64_t prev_iter_end = worker_t0;
    /* Take entries that match `my_kind` AND `my_hart`, in start_time
     * order.
     *
     * Until 2026-08-27 the predicate was kind-only, so ONE thread ran
     * every entry of a kind serially no matter how many distinct cores
     * the scheduler had placed them on. Measured on the K1 with a
     * 3-model schedule: zero overlapping dispatch pairs in 1617 entries,
     * the single `rvv` worker 97.9% busy across a 1017.8 ms run, and 119
     * of 123 instances missing their deadline even though per-dispatch
     * durations matched the cost model (median actual/predicted 0.93).
     * The schedule's core assignment was simply not honored at run time.
     *
     * UNBOUND ENTRIES: `hart == -1` means ingest could not resolve the
     * schedule's machine slot to a physical hart (a registry core with an
     * empty `harts` list). Those entries are NOT dropped -- exactly one
     * worker per kind carries `claims_unbound`, the one on the kind's
     * lowest-numbered hart, and it runs them interleaved with its own.
     * If a kind has ONLY unbound entries it gets a single unpinned worker
     * that carries the flag. Either way the (kind, hart) predicate stays
     * a partition of the table: every entry is claimed by exactly one
     * worker, which main() audits before spawning anything. */
{walker_loop}
    g_hart_acc[my_acc_idx].wall_total =
        (uint64_t)k_cycle_get_64() - worker_t0;
    return NULL;
}}

int main(void)
{{
    printf("xpurt-runner: schedule={schedule_name} entries=%d kinds=%d on %s\\n",
           {upper}_N_ENTRIES, {n_kinds}, XPURT_BOARD_TARGET);
#ifdef MODELBLASTER_PLATFORM_LINUX
    struct sched_param observed_sched_param = {{ 0 }};
    int observed_sched_policy = sched_getscheduler(0);
    if (observed_sched_policy < 0 ||
        sched_getparam(0, &observed_sched_param) != 0) {{
        printf("FATAL: cannot read process scheduler policy errno=%d\\n", errno);
        return -1;
    }}
    const char *observed_sched_name =
        observed_sched_policy == SCHED_FIFO ? "SCHED_FIFO" :
        observed_sched_policy == SCHED_RR ? "SCHED_RR" :
        observed_sched_policy == SCHED_OTHER ? "SCHED_OTHER" : "OTHER";
    printf("xpurt: observed_sched_policy=%s priority=%d\\n",
           observed_sched_name, observed_sched_param.sched_priority);
#endif

{v2_main_early}\
    static const char *kinds[] = {{ {kind_strs} }};

    /* ---- Worker-set discovery -----------------------------------------
     * One worker per (core_kind, hart) pair the DISPATCH TABLE actually
     * uses -- discovered here rather than passed in at codegen, because
     * the hart column is resolved by ingest against the core registry and
     * the generator never sees the registry. A kind the schedule does not
     * use gets zero workers (e.g. rvv_c1 in the 2-model B1 schedule).
     *
     * Unbound entries (hart == -1): the kind's lowest-hart worker is
     * marked `claims_unbound` and runs them interleaved with its own. If
     * a kind has ONLY unbound entries it gets one unpinned worker
     * carrying the flag. Never dropped -- a dropped entry surfaces as a
     * zeroed trace row and reads exactly like a crash. */
    g_n_workers = 0;
    for (int k = 0; k < {n_kinds}; k++) {{
        int kind_first = g_n_workers;
        int has_unbound = 0;
        for (int i = 0; i < {upper}_N_ENTRIES; i++) {{
            const xpurt_sched_entry_t *e = &{upper}_TABLE[i];
            if (strcmp(e->core_kind, kinds[k]) != 0) continue;
            if (e->hart < 0) {{ has_unbound = 1; continue; }}
            int seen = 0;
            for (int w = kind_first; w < g_n_workers; w++) {{
                if (g_workers[w].hart == e->hart) {{ seen = 1; break; }}
            }}
            if (seen) continue;
            if (g_n_workers >= XPURT_MAX_WORKERS) {{
                printf("FATAL: schedule uses more than %d (core_kind,hart) "
                       "pairs; raise XPURT_MAX_WORKERS\\n", XPURT_MAX_WORKERS);
                sys_reboot(SYS_REBOOT_COLD);
                return -1;
            }}
            g_workers[g_n_workers].kind = kinds[k];
            g_workers[g_n_workers].hart = e->hart;
            g_workers[g_n_workers].kind_idx = k;
            g_workers[g_n_workers].worker_idx = g_n_workers;
            g_workers[g_n_workers].claims_unbound = 0;
            g_workers[g_n_workers].observed_cpu = -1;
            g_n_workers++;
        }}
        if (has_unbound) {{
            if (g_n_workers > kind_first) {{
                int lo = kind_first;
                for (int w = kind_first + 1; w < g_n_workers; w++) {{
                    if (g_workers[w].hart < g_workers[lo].hart) lo = w;
                }}
                g_workers[lo].claims_unbound = 1;
            }} else {{
                if (g_n_workers >= XPURT_MAX_WORKERS) {{
                    printf("FATAL: XPURT_MAX_WORKERS exhausted on unbound "
                           "kind %s\\n", kinds[k]);
                    sys_reboot(SYS_REBOOT_COLD);
                    return -1;
                }}
                g_workers[g_n_workers].kind = kinds[k];
                g_workers[g_n_workers].hart = -1;   /* unpinned */
                g_workers[g_n_workers].kind_idx = k;
                g_workers[g_n_workers].worker_idx = g_n_workers;
                g_workers[g_n_workers].claims_unbound = 1;
                g_workers[g_n_workers].observed_cpu = -1;
                g_n_workers++;
            }}
        }}
    }}

    /* ---- Claim-coverage audit -----------------------------------------
     * Every table entry must be claimed by EXACTLY ONE worker. Zero
     * claimants means the entry silently never runs -- that is how the
     * kind-vs-backend strcmp mismatch of 2026-06-03 presented, as all-zero
     * outputs that read like FPGA corruption. More than one means two
     * threads write the same trace slot and run the same kernel twice.
     * Both are fatal and both are cheap to rule out right here, before a
     * single thread exists. This is also what keeps the trace-slot
     * "exactly one writer" invariant a fact rather than a hope. */
    for (int i = 0; i < {upper}_N_ENTRIES; i++) {{
        const xpurt_sched_entry_t *e = &{upper}_TABLE[i];
        int claimants = 0;
        for (int w = 0; w < g_n_workers; w++) {{
            if (strcmp(e->core_kind, g_workers[w].kind) != 0) continue;
            if (e->hart >= 0) {{
                if (e->hart != g_workers[w].hart) continue;
            }} else if (!g_workers[w].claims_unbound) {{
                continue;
            }}
            claimants++;
        }}
        if (claimants != 1) {{
            printf("FATAL: entry %d (%s dispatch %d, core_kind=%s hart=%d) "
                   "has %d claimant workers, expected 1\\n",
                   e->entry_id, e->network, e->dispatch_id,
                   e->core_kind, e->hart, claimants);
            sys_reboot(SYS_REBOOT_COLD);
            return -1;
        }}
    }}

    /* Build one pool for each distinct composite target. This preserves both
     * the exact width and exact harts selected by XPU-RT. Singleton entries
     * deliberately receive NULL and execute on their owning worker. */
#ifdef MODELBLASTER_PLATFORM_LINUX
    /* Pin main to the first non-AI CPU in its allowed mask BEFORE the pools
     * exist: pool helpers inherit main's mask, so they start X100-masked and
     * then place themselves. (A no-op when no AI CPUs are configured, i.e. on
     * a K1.) main must never itself register as an AI thread -- registration
     * is one-way and main would lose the X100 cores. */
    if (mb_pin_self_to_first_non_ai_cpu() != 0) {{
        printf("FATAL: cannot pin main thread off the AI CPUs\\n");
        sys_reboot(SYS_REBOOT_COLD);
        return -1;
    }}
#endif
    int entry_runtime_rc = init_entry_runtime();
    if (entry_runtime_rc != 0) {{
        printf("FATAL: composite-target runtime init failed rc=%d\\n",
               entry_runtime_rc);
        sys_reboot(SYS_REBOOT_COLD);
        return -1;
    }}

    /* Dispatch-local state selects pool_for_entry(e), so the same model may
     * alternate singleton, two-hart, and four-hart implementations. */

{chr(10).join(reset_calls)}

    /* Init completion sems. Limit needs to be >= max fanout across
     * all entries (producer gives n_fanout times before any consumer
     * has taken). For our schedules max fanout is observed at <16; a
     * 32 cap gives headroom while keeping the sem accounting tight
     * (Phase G2d). The old limit=64 + take/re-give pattern is gone. */
    for (int i = 0; i < {upper}_N_ENTRIES; i++) {{
        k_sem_init(&completion_sems[i], 0, 32);
    }}

    /* One worker per (core_kind, hart) pair, discovered above. Each pins
     * itself to its own hart, so the schedule's placement decision is the
     * one that executes. The previous version spawned one worker per KIND
     * and pinned it to the first matching entry's hart, which quietly
     * serialized every dispatch of a kind onto a single core.
     *
     * Zephyr quirk: pthread_attr_init() casts the public pthread_attr_t*
     * to the internal `struct posix_thread_attr*` and zeroes it; with
     * CONFIG_POSIX_THREADS_AFFINITY the internal struct extends past the
     * public 16 bytes (cpu_affinity sits at offset 16). Without padding,
     * pthread_attr_setaffinity_np(&attrs[k+1]) silently corrupts the
     * adjacent attrs[k+1] neighbour. Pad each slot to 64 bytes so the
     * cast stays in its lane. */
    static pthread_t tids[XPURT_MAX_WORKERS];
    union xpurt_attr_slot {{ pthread_attr_t a; char _pad[64]; }};
    static union xpurt_attr_slot attrs[XPURT_MAX_WORKERS];

#ifdef MODELBLASTER_XPURT_TRACE
    /* -1/-1 rather than the static 0/0, so a row NO worker executed is
     * distinguishable from one executed by kind 0 on hart 0. */
    for (int i = 0; i < {upper}_N_ENTRIES; i++) {{
        xpurt_trace[i].worker_kind_idx = -1;
        xpurt_trace[i].worker_hart = -1;
    }}
#endif

{v2_main_pre}\
{warm_main_pre}
    /* Run baseline for both the trace and the periodic-start-time gate.
     * Captured just before workers spawn so it's the moment "t=0" of the
     * schedule maps to. With --warmup it is captured AGAIN, after the
     * warm-up barrier, and that later value is the one workers read. */
    run_t0 = (uint64_t)k_cycle_get_64();

    for (int w = 0; w < g_n_workers; w++) {{
        pthread_attr_init(&attrs[w].a);
        /* Affinity. The Linux arm used to be missing: the whole block sat
         * under `#ifdef CONFIG_POSIX_THREADS_AFFINITY`, a Zephyr Kconfig
         * symbol that the harness_xpurt_linux CMake never defines -- so on
         * the K1 every worker floated across all 8 harts and the
         * "pinned_hart=N" line printed at exit was a claim about an
         * attribute nobody had set. It is now done BY the worker
         * (mb_place_current_thread, first thing in xpurt_worker) because a
         * K3 A100 thread must register its own TID before it may pin; attr
         * affinity at create time cannot do that. So this block is
         * Zephyr-only. */
#if defined(CONFIG_POSIX_THREADS_AFFINITY) && !defined(MODELBLASTER_PLATFORM_LINUX)
        if (g_workers[w].hart >= 0) {{
            cpu_set_t cs;
            CPU_ZERO(&cs);
            CPU_SET(g_workers[w].hart, &cs);
            int arc = pthread_attr_setaffinity_np(&attrs[w].a, sizeof(cs), &cs);
            if (arc != 0) {{
                printf("FATAL: pthread_attr_setaffinity_np worker=%d kind=%s "
                       "hart=%d rc=%d -- refusing to run unpinned, every "
                       "per-hart number in the trace would be fiction\\n",
                       w, g_workers[w].kind, g_workers[w].hart, arc);
                sys_reboot(SYS_REBOOT_COLD);
                return -1;
            }}
        }}
#endif
        int rc = pthread_create(&tids[w], &attrs[w].a, xpurt_worker,
                                &g_workers[w]);
        if (rc != 0) {{
            printf("FATAL: pthread_create worker=%d kind=%s hart=%d rc=%d\\n",
                   w, g_workers[w].kind, g_workers[w].hart, rc);
            sys_reboot(SYS_REBOOT_COLD);
            return -1;
        }}
        /* Per-worker spawn diagnostic deferred to after pthread_join.
         * Inline printf here costs ~tens-of-ms over FireSim HTIF UART
         * and starves any worker pinned to the same hart as main.
         * The (kind, hart) info is already in g_workers[w]; print at the
         * end so it doesn't perturb the schedule's actual_start cycles. */
    }}

{warm_main_post}
    /* Wait for every worker to drain. */
    for (int w = 0; w < g_n_workers; w++) {{
        pthread_join(tids[w], NULL);
        pthread_attr_destroy(&attrs[w].a);
    }}

{v2_main_post}\
    /* Now safe to flush the spawn diagnostics — workers are done, so
     * nothing pinned to hart 0 is competing with main for UART time.
     * `observed_cpu` is sched_getcpu() sampled inside the worker: it must
     * equal `hart` for a pinned worker, and a mismatch means the pinning
     * did not take even though the call returned 0. */
    for (int w = 0; w < g_n_workers; w++) {{
        printf("xpurt: worker[%d] kind=%s pinned_hart=%d observed_cpu=%d "
               "claims_unbound=%d entries_done=%llu\\n",
               w, g_workers[w].kind, g_workers[w].hart,
               g_workers[w].observed_cpu, g_workers[w].claims_unbound,
               (unsigned long long)g_hart_acc[w].entries_done);
    }}

    /* Phase G1 — per-WORKER runtime attribution. All values are mtime
     * ticks (1 us each on chipyard FireSim, 1/24 us on the K1 whose
     * rdtime runs at 24 MHz). Sum of categories should equal wall_total
     * within ~2%; residual is unattributed overhead (k_cycle_get_64
     * itself, branch mispredict, etc.).
     *
     * The `kind_idx` and `kind` columns are kept for
     * scripts/parse_runtime_breakdown.py, which keys on them; `worker_idx`
     * and `hart` are new and are what actually identify a row now that a
     * kind can have several workers. */
    printf("=== MODELBLASTER_HART_ACC_BEGIN ===\\n");
    printf("worker_idx,kind_idx,kind,hart,kernel_us,dep_wait_us,"
           "sync_overhead_us,target_gate_spin_us,hart_idle_us,"
           "gemmini_cfg_emit_us,entries_done,wall_total_us\\n");
    for (int w = 0; w < g_n_workers; w++) {{
        printf("%d,%d,%s,%d,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\\n",
               w, g_workers[w].kind_idx, g_workers[w].kind, g_workers[w].hart,
               (unsigned long long)g_hart_acc[w].kernel,
               (unsigned long long)g_hart_acc[w].dep_wait,
               (unsigned long long)g_hart_acc[w].sync_overhead,
               (unsigned long long)g_hart_acc[w].target_gate_spin,
               (unsigned long long)g_hart_acc[w].hart_idle,
               (unsigned long long)g_hart_acc[w].gemmini_cfg_emit,
               (unsigned long long)g_hart_acc[w].entries_done,
               (unsigned long long)g_hart_acc[w].wall_total);
    }}
    printf("=== MODELBLASTER_HART_ACC_END ===\\n");

#ifdef MODELBLASTER_XPURT_TRACE
    /* Trace dump — one CSV row per scheduled entry, with the actual
     * mtime-cycle window the worker ran, plus the kind-index AND the
     * hart of the worker that owned it (`worker_hart` vs the schedule's
     * `hart` column: they should be equal for every executed row). The host parser cross-references this
     * with the dispatch table for op/network/predicted-time fields. */
    printf("=== MODELBLASTER_XPURT_TRACE_BEGIN ===\\n");
    printf("entry_id,network,instance,dispatch_id,op,name,core_kind,hart,"
           "predicted_start_ms,predicted_duration_ms,worker_kind_idx,"
           "worker_hart,actual_start_cycles,actual_end_cycles\\n");
    for (int i = 0; i < {upper}_N_ENTRIES; i++) {{
        const xpurt_sched_entry_t *e = &{upper}_TABLE[i];
        printf("%d,%s,%d,%d,%s,%s,%s,%d,%.6f,%.6f,%d,%d,%llu,%llu\\n",
               e->entry_id, e->network, e->instance, e->dispatch_id,
               e->op, e->name, e->core_kind, e->hart,
               (double)e->start_time_ms, (double)e->duration_ms,
               xpurt_trace[i].worker_kind_idx,
               xpurt_trace[i].worker_hart,
               (unsigned long long)xpurt_trace[i].start_cycles,
               (unsigned long long)xpurt_trace[i].end_cycles);
    }}
    printf("=== MODELBLASTER_XPURT_TRACE_END ===\\n");
#endif

{chr(10).join(print_blocks)}

    for (int p = 0; p < g_n_entry_pools; p++) {{
        modelblaster_pool_destroy(g_entry_pools[p].pool);
    }}
    for (int h = 0; h < XPURT_MAX_HARTS; h++) {{
        pthread_mutex_destroy(&g_hart_locks[h]);
    }}
#ifndef MODELBLASTER_PLATFORM_LINUX
    /* Zephyr: a bare-metal run ends by rebooting the board, so the harness
     * driver sees a clean restart rather than a hung shell. This is the
     * SUCCESS path -- the error paths above reach sys_reboot too.
     *
     * On a hosted OS it must not be a reboot and, more importantly, must not
     * share an exit status with the failure paths: the POSIX shim maps
     * sys_reboot to _exit(1), which is right for an error and wrong here.
     * Falling through to `return 0` is what lets a caller distinguish "the
     * schedule ran" from "the schedule died". */
    sys_reboot(SYS_REBOOT_COLD);
#endif
    return 0;
}}
"""
    if walker == "static":
        # Gangs replace the pools: no pool threads are created. The worker's
        # own wall_total/accounting come from its private block.
        a = "    int entry_runtime_rc = init_entry_runtime();\n"
        assert src.count(a) == 1
        src = src.replace(a, "    int entry_runtime_rc = XS_USE_GANG ? xs_init_locks_only() "
                             ": init_entry_runtime();\n")
        b = ("    g_hart_acc[my_acc_idx].wall_total =\n"
             "        (uint64_t)k_cycle_get_64() - worker_t0;\n")
        assert src.count(b) == 1
        src = src.replace(b, "")
    return src


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", required=True,
                    help="path to the schedule.json (XPU-RT format)")
    ap.add_argument("--out", required=True,
                    help="path to write xpurt_main.c")
    ap.add_argument("--name", required=True,
                    help="schedule name — must match the --name passed to "
                         "ingest_xpurt_schedule (drives the table symbol prefix)")
    ap.add_argument("--dispatch-table-header", required=True,
                    help="basename of the .h emitted by ingest_xpurt_schedule "
                         "(included from xpurt_main.c)")
    ap.add_argument("--platform", choices=("zephyr", "linux"),
                    default="zephyr",
                    help="zephyr: k_cycle_get_64/k_sem/sys_reboot as-is. "
                         "linux: rdtime + POSIX semaphores, for a hosted "
                         "riscv64 board such as the SpaceMiT K1.")
    ap.add_argument("--warmup", action="store_true",
                    help="Linux only: run every distinct (network, dispatch, "
                         "impl, hart set) once, untimed, before t0, then "
                         "restore state. Removes first-instance cold-start "
                         "from the timed window. Off by default (K1); "
                         "run_xpurt_k1.sh turns it on for k3.")
    ap.add_argument("--core-kinds", default="rvv,scalar",
                    help="comma list of distinct core_kind values the schedule "
                         "uses. One worker thread is spawned per kind. "
                         "(default: rvv,scalar)")
    ap.add_argument("--backends", default=None,
                    help="comma list of HW backends the harness was compiled "
                         "for (== per-model OBJECT-lib suffixes). The walker "
                         "forward-declares MODEL_<UMID>_DISPATCH_FNS_<BS> for "
                         "each backend and dispatches by core_kind. "
                         "Defaults to --core-kinds.")
    ap.add_argument("--networks", default=None,
                    help="comma-separated network names in the schedule. "
                         "Supplies the known set that makes the "
                         "<network><instance> split unambiguous when a network "
                         "name ENDS IN A DIGIT (yolov8_nano_64x96), which the "
                         "trailing-digit regex otherwise reads as "
                         "'yolov8_nano_64x' + instance 96. The caller almost "
                         "always knows these -- run_xpurt_k1.sh has them as "
                         "--models -- and passing them removes the guess.")
    ap.add_argument("--registry", default=None,
                    help="path to the cores/*.json registry that drove the "
                         "schedule. Used to derive each kind's pool size as "
                         "(harts_of_kind - 1). If both --registry and "
                         "--pool-sizes are absent, all kinds get pool=0 "
                         "(NULL) and intra-op parallel_<op> calls run "
                         "synchronously on the scheduler worker.")
    ap.add_argument("--pool-sizes", default=None,
                    help='explicit per-kind pool helper-thread count, '
                         'e.g. "rvv:0,scalar:0". Overrides --registry. '
                         "Use 0 for kinds with one hart (NULL pool).")
    ap.add_argument("--program-json", default=None,
                    help="--walker static: the per-worker static program "
                         "written by ingest_xpurt_schedule "
                         "--static-program-json.")
    ap.add_argument("--static-env", default=None,
                    help="--walker static on Zephyr: run-time switches baked "
                         "into the binary, K=V[,K=V...] (e.g. "
                         "XPURT_CYCLES=10,XPURT_WAKE=doorbell_sem).")
    ap.add_argument("--walker", choices=("strict", "skip", "static"), default="strict",
                    help="strict (default): each worker runs its entries in "
                         "table order and blocks on the head entry's "
                         "dependencies (output unchanged). skip (Linux only): "
                         "the v2 runtime -- a worker whose head entry is "
                         "waiting on a dependency runs a later entry that is "
                         "ready, start gates sleep instead of spinning, and "
                         "the workers take a real-time priority when allowed. "
                         "See _v2_blocks for the behaviour, the run-time "
                         "switches (XPURT_SKIP_WINDOW, XPURT_GATE, "
                         "XPURT_RT_PRIO, XPURT_CYCLES) and the deadlock-"
                         "freedom argument.")
    ap.add_argument("--hyperperiod-ms", type=float, default=None,
                    help="--walker skip only: the schedule covers one "
                         "hyperperiod of this length and XPURT_CYCLES=<k> "
                         "replays it k times (gate = t0 + cycle*H + start). "
                         "Without it the table is walked once.")
    ap.add_argument("--rt-priority", type=int, default=20,
                    help="--walker skip only: default SCHED_FIFO priority the "
                         "runtime asks for (XPURT_RT_PRIO overrides at run "
                         "time; 0 keeps the inherited policy). If the kernel "
                         "refuses, the run continues under the inherited "
                         "policy and says so.")
    args = ap.parse_args()

    with open(args.schedule) as f:
        sched = json.load(f)
    networks: list[str] = []
    seen: set[str] = set()
    n_instances: dict[str, int] = {}
    # Multi-network bridge stamps an explicit instances list in
    # provenance — use it when present (it's the only way to be
    # unambiguous about model names ending in digits, e.g. "yolov8_nano_64"
    # which the trailing-digits regex parses as "yolov8_nano_" + idx=64).
    # An explicitly supplied network list is the most reliable source and
    # takes precedence over both the provenance block and the regex: the
    # caller knows what it built.
    cli_known = {n.strip() for n in (args.networks or "").split(",") if n.strip()}
    prov_instances = sched.get("_provenance", {}).get("instances")
    if cli_known:
        for d in sched["dispatches"].values():
            net, inst = _split_job_name(d["job_name"], cli_known)
            if net not in cli_known:
                raise SystemExit(
                    f"job_name {d['job_name']!r} does not start with any of "
                    f"--networks {sorted(cli_known)}. Refusing to guess: a "
                    f"wrong split emits an #include for a model that does not "
                    f"exist, or worse, one that does.")
            if net not in seen:
                seen.add(net)
                networks.append(net)
            n_instances[net] = max(n_instances.get(net, 0), inst + 1)
    elif prov_instances:
        known = {ins["network"] for ins in prov_instances}
        for ins in prov_instances:
            net = ins["network"]
            inst = ins["instance"]
            if net not in seen:
                seen.add(net)
                networks.append(net)
            n_instances[net] = max(n_instances.get(net, 0), inst + 1)
    else:
        known = None
        for d in sched["dispatches"].values():
            net, inst = _split_job_name(d["job_name"], known)
            if net not in seen:
                seen.add(net)
                networks.append(net)
            n_instances[net] = max(n_instances.get(net, 0), inst + 1)

    core_kinds = [k.strip() for k in args.core_kinds.split(",") if k.strip()]
    if not core_kinds:
        raise SystemExit("--core-kinds must be a non-empty comma list")
    if args.backends is None:
        backends = list(core_kinds)
    else:
        backends = [b.strip() for b in args.backends.split(",") if b.strip()]
        if not backends:
            raise SystemExit("--backends must be a non-empty comma list")

    # Guard against the silent-dispatch-drop bug (2026-06-03):
    # the schedule's dispatch table carries .core_kind = registry-kind
    # (e.g. "gemmini"). The generated dispatch branch must strcmp the
    # entry's .core_kind against the SAME string — so kind→backend
    # mapping is now done by zip(core_kinds, backends) inside _emit
    # (strcmp uses kind, dispatch fn symbol uses backend). Asserting
    # length parity here so the index-aligned mapping is well-defined.
    if len(core_kinds) != len(backends):
        raise SystemExit(
            f"--core-kinds and --backends must have the same length "
            f"(got core_kinds={core_kinds}, backends={backends}). The "
            f"dispatch branch pairs kind[i] -> backend[i] by index."
        )

    # Pool size resolution:
    #   1. explicit --pool-sizes wins
    #   2. else --registry: harts_of_kind - 1 per kind
    #   3. else default 0 per kind (NULL pool, no intra-op fanout)
    pool_sizes_map: dict[str, int] = {}
    if args.pool_sizes:
        for kv in args.pool_sizes.split(","):
            k, _, v = kv.strip().partition(":")
            if not k or not v:
                raise SystemExit(f"--pool-sizes: bad entry '{kv}'")
            pool_sizes_map[k] = int(v)
    elif args.registry:
        with open(args.registry) as f:
            reg = json.load(f)
        harts_per_kind: dict[str, set[int]] = {}
        for c in reg.get("cores", []):
            kind = c.get("kind")
            harts = c.get("harts", []) or []
            if kind is None:
                continue
            harts_per_kind.setdefault(kind, set()).update(harts)
        for k in core_kinds:
            n_harts = len(harts_per_kind.get(k, set()))
            # modelblaster_pool_create(N) makes a pool of N threads INCLUDING
            # the caller, so use n_harts directly. 1-hart kinds still get
            # 0 (NULL pool, serial fallback) since create(1) just allocates
            # a degenerate pool with no helpers.
            pool_sizes_map[k] = n_harts if n_harts >= 2 else 0
    pool_sizes = [pool_sizes_map.get(k, 0) for k in core_kinds]

    src = _emit(networks, args.name, args.dispatch_table_header,
                core_kinds, backends, pool_sizes, n_instances,
                platform=args.platform, warmup=args.warmup,
                walker=args.walker, hyper_ms=args.hyperperiod_ms,
                rt_prio_default=args.rt_priority,
                program=(json.load(open(args.program_json))
                         if args.program_json else None),
                static_env=(dict(kv.split("=", 1) for kv in args.static_env.split(","))
                            if args.static_env else None))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(src)
    print(f"wrote {args.out}  (platform: {args.platform} "
          f"networks: {networks} "
          f"instances: {n_instances} kinds: {core_kinds} "
          f"backends: {backends} pool_sizes: {pool_sizes} "
          f"warmup: {args.warmup}"
          + (f" walker: {args.walker} hyper_ms: {args.hyperperiod_ms}"
             if args.walker != "strict" else "") + ")")


if __name__ == "__main__":
    main()
