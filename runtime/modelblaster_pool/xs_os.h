/* xs_os.h -- the OS-facing operations of the static XPU-RT walker
 * (generate_xpurt_main --walker static) and of the pool's static gangs.
 *
 * One header, two implementations: hosted Linux (MODELBLASTER_PLATFORM_LINUX)
 * and Zephyr. Everything the walker does to the OS goes through here:
 *
 *   xs_os_evt_wait(w, expect, timeout)  sleep while *w == expect (event word)
 *   xs_os_evt_wake(w, all)              wake sleepers on an event word
 *   xs_os_db_*                          the per-worker doorbell (counting)
 *   xs_os_sleep_until(target)           coarse sleep to an absolute tick
 *                                       (the caller spins the last stretch)
 *   xs_os_set_rt_prio(prio)             real-time priority for this process/thread
 *   xs_os_cpu()                         current CPU
 *   xs_os_getenv(name)                  run-time switches
 *
 * Ticks are k_cycle_get_64() ticks (XPURT_TICKS_PER_SEC) on both platforms.
 * Publishing progress and the register/re-check handshake use C11 seq_cst
 * atomics directly in the walker; they need no OS call.
 *
 * Linux: futex event words, sem_t or futex-count doorbells, clock_nanosleep
 * TIMER_ABSTIME, SCHED_FIFO. Zephyr: no futex without CONFIG_USERSPACE, so an
 * event word is waited on by spinning (one walker thread per hart, nothing
 * else to run there); the doorbell is a k_sem; sleep_until is
 * k_sleep(K_TIMEOUT_ABS_TICKS(...)); priority is a cooperative priority
 * (K_PRIO_COOP), which keeps other threads off the hart but does NOT mask
 * interrupts -- the dispatch guard (arch_irq_lock for single-hart dispatches,
 * commit d4fe81e) stays as it is.
 *
 * Added 2026-11-26 for the static walker.
 */
#ifndef XS_OS_H
#define XS_OS_H

#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(MODELBLASTER_PLATFORM_LINUX) || defined(MB_PLATFORM_POSIX)

#ifndef XPURT_TICKS_PER_SEC
#ifdef MB_POSIX_TICKS_PER_SEC
#define XPURT_TICKS_PER_SEC MB_POSIX_TICKS_PER_SEC   /* mb_posix_compat.h: rdtime ticks */
#else
#define XPURT_TICKS_PER_SEC 1000000000ull            /* mb_posix.h: k_cycle_get_64 is ns */
#endif
#endif
#include <errno.h>
#include <limits.h>
#include <linux/futex.h>
#include <sched.h>
#include <semaphore.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define XS_OS_HAS_FUTEX 1
#define XS_OS_NAME "linux"

static inline struct timespec *xs_os_rel_ts(struct timespec *ts, uint64_t ticks)
{
    uint64_t ns = ticks * 1000000000ull / (uint64_t)XPURT_TICKS_PER_SEC;
    ts->tv_sec = (time_t)(ns / 1000000000ull);
    ts->tv_nsec = (long)(ns % 1000000000ull);
    return ts;
}
/* Returns 0 when woken or the word changed, -1 on timeout. */
static inline int xs_os_evt_wait(_Atomic uint32_t *w, uint32_t expect, uint64_t timeout_ticks)
{
    struct timespec ts, *tp = timeout_ticks ? xs_os_rel_ts(&ts, timeout_ticks) : NULL;
    long r = syscall(SYS_futex, w, FUTEX_WAIT_PRIVATE, expect, tp, NULL, 0);
    return (r != 0 && errno == ETIMEDOUT) ? -1 : 0;
}
static inline void xs_os_evt_wake(_Atomic uint32_t *w, int all)
{
    syscall(SYS_futex, w, FUTEX_WAKE_PRIVATE, all ? INT_MAX : 1, NULL, NULL, 0);
}

/* Doorbell: mode 1 = sem_t, mode 2 = futex count. */
typedef struct { sem_t sem; _Atomic uint32_t cnt; } xs_os_db_t;
static inline void xs_os_db_init(xs_os_db_t *d) { sem_init(&d->sem, 0, 0); atomic_store(&d->cnt, 0); }
static inline void xs_os_db_give(xs_os_db_t *d, int mode)
{
    if (mode == 1) { sem_post(&d->sem); return; }
    atomic_fetch_add(&d->cnt, 1);
    xs_os_evt_wake(&d->cnt, 0);
}
/* left_ticks 0 = block. Returns 1 if a token was taken, 0 on timeout. */
static inline int xs_os_db_take(xs_os_db_t *d, int mode, uint64_t left_ticks)
{
    if (mode == 1) {
        if (!left_ticks) { while (sem_wait(&d->sem) != 0) { } return 1; }
        struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
        uint64_t ns = (uint64_t)ts.tv_nsec + left_ticks * 1000000000ull / (uint64_t)XPURT_TICKS_PER_SEC;
        ts.tv_sec += (time_t)(ns / 1000000000ull); ts.tv_nsec = (long)(ns % 1000000000ull);
#if defined(_GNU_SOURCE)
        for (;;) {
            if (sem_clockwait(&d->sem, CLOCK_MONOTONIC, &ts) == 0) return 1;
            if (errno == ETIMEDOUT) return 0;
        }
#else   /* no sem_clockwait without _GNU_SOURCE: CLOCK_REALTIME deadline */
        struct timespec rt; clock_gettime(CLOCK_REALTIME, &rt);
        uint64_t ns2 = (uint64_t)rt.tv_nsec + left_ticks * 1000000000ull / (uint64_t)XPURT_TICKS_PER_SEC;
        rt.tv_sec += (time_t)(ns2 / 1000000000ull); rt.tv_nsec = (long)(ns2 % 1000000000ull);
        (void)ts;
        for (;;) {
            if (sem_timedwait(&d->sem, &rt) == 0) return 1;
            if (errno == ETIMEDOUT) return 0;
        }
#endif
    }
    uint64_t t0 = (uint64_t)k_cycle_get_64();
    for (;;) {
        uint32_t c = atomic_load(&d->cnt);
        while (c) { if (atomic_compare_exchange_weak(&d->cnt, &c, c - 1)) return 1; }
        uint64_t left = 0;
        if (left_ticks) {
            uint64_t el = (uint64_t)k_cycle_get_64() - t0;
            if (el >= left_ticks) return 0;
            left = left_ticks - el;
        }
        (void)xs_os_evt_wait(&d->cnt, 0, left);
    }
}

/* Coarse absolute sleep: the CLOCK_MONOTONIC deadline comes from a fresh
 * (rdtime, clock) pair, so slew only acts over this one sleep. */
static inline void xs_os_sleep_until(uint64_t target_ticks)
{
    uint64_t now = (uint64_t)k_cycle_get_64();
    if (target_ticks <= now) return;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    uint64_t ns = (uint64_t)ts.tv_nsec + (target_ticks - now) * 1000000000ull / (uint64_t)XPURT_TICKS_PER_SEC;
    ts.tv_sec += (time_t)(ns / 1000000000ull);
    ts.tv_nsec = (long)(ns % 1000000000ull);
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL) == EINTR) { }
}
/* 0 on success, else errno. Applies to the calling thread; threads created
 * afterwards inherit it. */
static inline int xs_os_set_rt_prio(int prio)
{
    struct sched_param sp = { .sched_priority = prio };
    return sched_setscheduler(0, SCHED_FIFO, &sp) == 0 ? 0 : errno;
}
static inline int xs_os_cpu(void) { unsigned c = 0; return syscall(SYS_getcpu, &c, NULL, NULL) == 0 ? (int)c : -1; }
static inline const char *xs_os_getenv(const char *name) { return getenv(name); }

#else /* ---------------------------------------------------------- Zephyr */

#include <zephyr/kernel.h>

#ifndef XPURT_TICKS_PER_SEC
#define XPURT_TICKS_PER_SEC CONFIG_SYS_CLOCK_HW_CYCLES_PER_SEC
#endif
#define XS_OS_NAME "zephyr"

static inline int xs_os_evt_wait(_Atomic uint32_t *w, uint32_t expect, uint64_t timeout_ticks)
{
    uint64_t t0 = k_cycle_get_64();
    while (atomic_load(w) == expect) {
        if (timeout_ticks && k_cycle_get_64() - t0 >= timeout_ticks) return -1;
    }
    return 0;
}
static inline void xs_os_evt_wake(_Atomic uint32_t *w, int all) { (void)w; (void)all; }

typedef struct { struct k_sem sem; _Atomic uint32_t cnt; } xs_os_db_t;
static inline void xs_os_db_init(xs_os_db_t *d) { k_sem_init(&d->sem, 0, K_SEM_MAX_LIMIT); atomic_store(&d->cnt, 0); }
static inline void xs_os_db_give(xs_os_db_t *d, int mode) { (void)mode; k_sem_give(&d->sem); }
static inline int xs_os_db_take(xs_os_db_t *d, int mode, uint64_t left_ticks)
{
    (void)mode;
    return k_sem_take(&d->sem, left_ticks ? K_CYC(left_ticks) : K_FOREVER) == 0;
}
static inline void xs_os_sleep_until(uint64_t target_ticks)
{
    if (target_ticks <= k_cycle_get_64()) return;
    k_sleep(K_TIMEOUT_ABS_TICKS(k_cyc_to_ticks_floor64(target_ticks)));
}
/* A cooperative priority: no other thread preempts the walker on its hart.
 * prio > 0 maps to K_PRIO_COOP(CONFIG_NUM_COOP_PRIORITIES - 1 - min(prio, ..)). */
static inline int xs_os_set_rt_prio(int prio)
{
    int lvl = prio > CONFIG_NUM_COOP_PRIORITIES - 1 ? CONFIG_NUM_COOP_PRIORITIES - 1 : prio;
    k_thread_priority_set(k_current_get(), K_PRIO_COOP(CONFIG_NUM_COOP_PRIORITIES - 1 - lvl));
    return 0;
}
static inline int xs_os_cpu(void) { return (int)arch_curr_cpu()->id; }
/* No environment on Zephyr: switches are baked in at code generation
 * (generate_xpurt_main --static-env K=V,...), served from XS_BAKED_ENV. */
#ifndef XS_BAKED_ENV_N
#define XS_BAKED_ENV_N 0
static const char *const XS_BAKED_ENV[1][2] = { { NULL, NULL } };
#endif
static inline const char *xs_os_getenv(const char *name)
{
    for (int i = 0; i < XS_BAKED_ENV_N; i++)
        if (XS_BAKED_ENV[i][0] && __builtin_strcmp(XS_BAKED_ENV[i][0], name) == 0) return XS_BAKED_ENV[i][1];
    return NULL;
}

#endif
#endif /* XS_OS_H */
