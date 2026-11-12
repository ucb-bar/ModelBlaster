/* Zephyr primitives, on a hosted POSIX riscv64 target.
 *
 * ModelBlaster's schedule-driven runtime -- modelblaster_pool and the generated
 * xpurt_main walker -- was written against pthreads from the start
 * (pthread_create, pthread_attr_setaffinity_np), so the Zephyr coupling is
 * narrow: a 64-bit cycle counter, counting semaphores, and an abort path. This
 * header supplies those three on Linux so one body serves both platforms
 * instead of forking the walker generator and the pool.
 *
 * Include this INSTEAD OF <zephyr/kernel.h> when MODELBLASTER_PLATFORM_LINUX is
 * defined.
 *
 * The timer is `rdtime`, not `rdcycle`. That is not a preference:
 *
 *   - rdcycle traps with SIGILL from userspace on the SpaceMiT K1's 6.6 kernel.
 *   - rdtime is unprivileged, ticks at a constant 24 MHz irrespective of DVFS,
 *     and is consistent across harts. rdcycle is none of those things -- a
 *     thread migrated mid-measurement would read another hart's counter.
 *
 * generate_skeleton.py makes the same choice for --platform linux, and the tick
 * rate here MUST agree with the one it emits (24 MHz) or per-op cycles and
 * whole-run wall time describe different clocks.
 */

#ifndef MB_POSIX_COMPAT_H_
#define MB_POSIX_COMPAT_H_

#include <errno.h>
#include <sched.h>
#include <semaphore.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

/* ---- 64-bit tick counter ------------------------------------------------ */

static inline uint64_t mb_posix_rdtime(void)
{
#if defined(__riscv)
	unsigned long t;
	__asm__ volatile("rdtime %0" : "=r"(t));
	return (uint64_t)t;
#else
	/* Host builds (unit tests, x86 dev boxes) have no rdtime. Fall back to
	 * a monotonic clock so the code compiles and runs, and scale it to the
	 * same nominal tick rate so arithmetic downstream is unchanged. */
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
	return (uint64_t)ts.tv_sec * MB_POSIX_TICKS_PER_SEC +
	       (uint64_t)ts.tv_nsec * MB_POSIX_TICKS_PER_SEC / 1000000000ull;
#endif
}

/* 24 MHz rdtime on the SpaceMiT K1 (measured; also
 * /proc/device-tree/cpus/timebase-frequency). Override for another board. */
#ifndef MB_POSIX_TICKS_PER_SEC
#define MB_POSIX_TICKS_PER_SEC 24000000ull
#endif

#if !defined(__riscv)
#include <time.h>
#endif

#define k_cycle_get_64() mb_posix_rdtime()

/* ---- counting semaphores ----------------------------------------------- */

/* The walker and the pool only ever init / give / take-forever, so POSIX
 * sem_t covers the whole surface. `limit` is accepted and ignored: POSIX has no
 * ceiling, and neither call site relies on one. */
struct k_sem {
	sem_t s;
};

#define K_FOREVER 0
#define K_NO_WAIT (-1)

static inline void k_sem_init(struct k_sem *p, unsigned int initial,
			      unsigned int limit)
{
	(void)limit;
	sem_init(&p->s, 0, initial);
}

static inline void k_sem_give(struct k_sem *p)
{
	sem_post(&p->s);
}

static inline int k_sem_take(struct k_sem *p, int timeout)
{
	if (timeout == K_NO_WAIT)
		return sem_trywait(&p->s) == 0 ? 0 : -EBUSY;
	/* Restart on EINTR. Returning early would let a signal be read as a
	 * satisfied dependency, which silently corrupts the dispatch order. */
	while (sem_wait(&p->s) != 0) {
		if (errno != EINTR)
			return -errno;
	}
	return 0;
}

/* ---- yield -------------------------------------------------------------- */

/* The walker calls this in its spin-wait when a periodic instance is gated
 * until its release time. sched_yield is the direct POSIX equivalent: give up
 * the rest of the timeslice without sleeping, so a short gate does not pay a
 * timer-granularity penalty. */
static inline void k_yield(void)
{
	sched_yield();
}

/* ---- schedule-start gate (opt-in: MB_XPURT_GATE_SLEEP=1) ---------------- */

/* The walker waits for an entry's absolute start time by spinning on k_yield.
 * A worker whose entries are far apart therefore never sleeps, and its core is
 * also a helper core of the split (pool) groups that contain it: on the K3
 * (2026-11-03, 50 Hz shard run under ftrace) the worker on cpu 2 ran 16 entries,
 * spun 288 ms and slept 0.3 ms, while the pool helpers pinned to cpu 2 waited
 * 16-20 ms for that core -- stalling whole 8-core layers by milliseconds.
 * With MB_XPURT_GATE_SLEEP set (and not "0") the wait sleeps until
 * MB_POSIX_GATE_SPIN_US before the target, then yields as before; unset keeps
 * the original spin exactly. Read once per process. */
#ifndef MB_POSIX_GATE_SPIN_US
#define MB_POSIX_GATE_SPIN_US 200ull
#endif
#define MB_HAVE_GATE_WAIT 1
static inline void mb_posix_gate_wait(uint64_t target)
{
	static int mode = -1;
	const uint64_t spin = MB_POSIX_GATE_SPIN_US * MB_POSIX_TICKS_PER_SEC / 1000000ull;
	if (mode < 0) {
		const char *e = getenv("MB_XPURT_GATE_SLEEP");
		mode = (e && e[0] && !(e[0] == '0' && e[1] == 0)) ? 1 : 0;
	}
	for (;;) {
		uint64_t now = mb_posix_rdtime();
		if (now >= target)
			return;
		if (mode && target - now > spin) {
			uint64_t ns = (target - now - spin) * 1000000000ull / MB_POSIX_TICKS_PER_SEC;
			struct timespec ts = { (time_t)(ns / 1000000000ull), (long)(ns % 1000000000ull) };
			nanosleep(&ts, NULL);
		} else {
			sched_yield();
		}
	}
}

/* ---- abort ------------------------------------------------------------- */

/* sys_reboot has no meaning on a hosted OS; the runner checks the exit status.
 * stdout is flushed first because the marker protocol the host parser reads is
 * line-buffered and would otherwise lose the failure message. */
#define SYS_REBOOT_COLD 1
#define SYS_REBOOT_WARM 0
#define sys_reboot(kind)                                                       \
	do {                                                                   \
		(void)(kind);                                                  \
		fflush(stdout);                                                \
		_exit(1);                                                      \
	} while (0)

#endif /* MB_POSIX_COMPAT_H_ */
