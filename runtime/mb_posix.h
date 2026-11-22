/*
 * Copyright (c) 2026 Dima Nikiforov <vnikiforov@berkeley.edu>
 * SPDX-License-Identifier: Apache-2.0
 *
 * mb_posix.h -- hosted-POSIX stand-ins for the few Zephyr kernel APIs the
 * ModelBlaster runtime uses (modelblaster_pool and the generated XPU-RT
 * schedule walker), so the same C sources build unchanged for a Linux host
 * (x86-64 for development, riscv64 boards for measurement).
 *
 * Both of those sources already do their threading through POSIX pthreads.
 * Zephyr shows up only in:
 *
 *   k_sem_{init,take,give}   cross-hart rendezvous   -> POSIX sem_t
 *   k_cycle_get_64           timestamps              -> CLOCK_MONOTONIC_RAW, ns
 *   k_yield                  spin back-off           -> sched_yield
 *   k_thread_cpu_pin         pin the master thread   -> pthread_setaffinity_np
 *   k_sched_{lock,unlock}    no-preempt dispatch     -> no-op (see below)
 *   sys_reboot               end of firmware run     -> exit
 *
 * Select it by defining MB_PLATFORM_POSIX before including any runtime
 * header (the walker and pool #include this file instead of <zephyr/...>).
 *
 * Timebase: on POSIX, k_cycle_get_64() returns nanoseconds and
 * CONFIG_SYS_CLOCK_HW_CYCLES_PER_SEC is 1e9, so any code that converts
 * cycles to time through that macro stays correct. Raw cycle counters are
 * deliberately not used: rdcycle is usually not readable from user mode on
 * riscv64 Linux, and the time CSR rate differs per board (24 MHz on the
 * SpacemiT K1); the vDSO clock already reads that CSR without a syscall.
 */
#ifndef MB_POSIX_H_
#define MB_POSIX_H_

#if defined(__linux__) && !defined(_GNU_SOURCE)
#error "mb_posix.h needs _GNU_SOURCE on Linux (pthread_setaffinity_np, CPU_SET); \
define it before the first #include, e.g. -D_GNU_SOURCE"
#endif

/* Linux is the target (host x86-64/aarch64 and boards such as the SpacemiT
 * K1). Other POSIX hosts (macOS) are supported only so the walker can be
 * built and unit-tested there: no thread affinity, and semaphores built on a
 * mutex and condition variable because macOS has no unnamed sem_t. */
#if defined(__linux__)
#define MB_POSIX_LINUX 1
#endif

#include <errno.h>
#include <pthread.h>
#include <sched.h>
#ifdef MB_POSIX_LINUX
#include <semaphore.h>
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* ---- Kconfig values the runtime reads ------------------------------- */
#ifndef CONFIG_SYS_CLOCK_HW_CYCLES_PER_SEC
#define CONFIG_SYS_CLOCK_HW_CYCLES_PER_SEC 1000000000ULL
#endif
#ifndef CONFIG_BOARD_TARGET
#if defined(__riscv)
#define CONFIG_BOARD_TARGET "posix/riscv64"
#elif defined(__x86_64__)
#define CONFIG_BOARD_TARGET "posix/x86_64"
#else
#define CONFIG_BOARD_TARGET "posix"
#endif
#endif
/* Pinning is always available on Linux; the pool keys its affinity code
 * off this Zephyr symbol. */
#if defined(MB_POSIX_LINUX) && !defined(CONFIG_POSIX_THREADS_AFFINITY)
#define CONFIG_POSIX_THREADS_AFFINITY 1
#endif

/* ---- timeouts ------------------------------------------------------- */
typedef struct {
	int64_t ns; /* < 0: forever, 0: no wait */
} k_timeout_t;
#define K_FOREVER ((k_timeout_t){ .ns = -1 })
#define K_NO_WAIT ((k_timeout_t){ .ns = 0 })

/* ---- semaphores ----------------------------------------------------- */
/* Zephyr's k_sem saturates at `limit`; sem_t does not. The runtime only
 * uses limit-1 semaphores as one-shot start/done signals between a fixed
 * producer and consumer, where a give is never issued twice without an
 * intervening take, so the check below only guards against misuse. */
#ifdef MB_POSIX_LINUX
struct k_sem {
	sem_t s;
	unsigned int limit;
};

static inline int k_sem_init(struct k_sem *sem, unsigned int initial,
			     unsigned int limit)
{
	sem->limit = limit;
	return sem_init(&sem->s, 0, initial) == 0 ? 0 : -errno;
}

static inline int k_sem_take(struct k_sem *sem, k_timeout_t t)
{
	if (t.ns == 0) {
		return sem_trywait(&sem->s) == 0 ? 0 : -EBUSY;
	}
	if (t.ns > 0) {
		struct timespec dl;
		clock_gettime(CLOCK_REALTIME, &dl);
		dl.tv_sec += t.ns / 1000000000LL;
		dl.tv_nsec += t.ns % 1000000000LL;
		if (dl.tv_nsec >= 1000000000L) {
			dl.tv_sec++;
			dl.tv_nsec -= 1000000000L;
		}
		while (sem_timedwait(&sem->s, &dl) != 0) {
			if (errno == ETIMEDOUT) {
				return -EAGAIN;
			}
			if (errno != EINTR) {
				return -errno;
			}
		}
		return 0;
	}
	while (sem_wait(&sem->s) != 0) {
		if (errno != EINTR) {
			return -errno;
		}
	}
	return 0;
}

static inline void k_sem_give(struct k_sem *sem)
{
	int v = 0;
	if (sem_getvalue(&sem->s, &v) == 0 && v >= (int)sem->limit) {
		return; /* saturate, as Zephyr does */
	}
	sem_post(&sem->s);
}
#else /* portable fallback: mutex + condition variable */
struct k_sem {
	pthread_mutex_t m;
	pthread_cond_t c;
	unsigned int count;
	unsigned int limit;
};

static inline int k_sem_init(struct k_sem *sem, unsigned int initial,
			     unsigned int limit)
{
	sem->limit = limit;
	sem->count = initial;
	if (pthread_mutex_init(&sem->m, NULL) != 0) {
		return -EINVAL;
	}
	return pthread_cond_init(&sem->c, NULL) == 0 ? 0 : -EINVAL;
}

static inline int k_sem_take(struct k_sem *sem, k_timeout_t t)
{
	int rc = 0;
	pthread_mutex_lock(&sem->m);
	if (t.ns == 0) {
		if (sem->count == 0) {
			rc = -EBUSY;
		}
	} else if (t.ns > 0) {
		struct timespec dl;
		clock_gettime(CLOCK_REALTIME, &dl);
		dl.tv_sec += t.ns / 1000000000LL;
		dl.tv_nsec += t.ns % 1000000000LL;
		if (dl.tv_nsec >= 1000000000L) {
			dl.tv_sec++;
			dl.tv_nsec -= 1000000000L;
		}
		while (sem->count == 0 && rc == 0) {
			if (pthread_cond_timedwait(&sem->c, &sem->m, &dl) != 0) {
				rc = -EAGAIN;
			}
		}
	} else {
		while (sem->count == 0) {
			pthread_cond_wait(&sem->c, &sem->m);
		}
	}
	if (rc == 0) {
		sem->count--;
	}
	pthread_mutex_unlock(&sem->m);
	return rc;
}

static inline void k_sem_give(struct k_sem *sem)
{
	pthread_mutex_lock(&sem->m);
	if (sem->count < sem->limit) { /* saturate, as Zephyr does */
		sem->count++;
		pthread_cond_signal(&sem->c);
	}
	pthread_mutex_unlock(&sem->m);
}
#endif /* MB_POSIX_LINUX */

/* ---- time ----------------------------------------------------------- */
static inline uint64_t k_cycle_get_64(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static inline uint32_t k_cycle_get_32(void)
{
	return (uint32_t)k_cycle_get_64();
}

static inline int64_t k_uptime_get(void)
{
	return (int64_t)(k_cycle_get_64() / 1000000ULL);
}

static inline uint32_t sys_clock_hw_cycles_per_sec(void)
{
	return (uint32_t)CONFIG_SYS_CLOCK_HW_CYCLES_PER_SEC;
}

static inline void k_busy_wait(uint32_t usec)
{
	uint64_t end = k_cycle_get_64() + (uint64_t)usec * 1000ULL;
	while (k_cycle_get_64() < end) {
	}
}

static inline int32_t k_msleep(int32_t ms)
{
	struct timespec ts = { ms / 1000, (long)(ms % 1000) * 1000000L };
	while (nanosleep(&ts, &ts) != 0 && errno == EINTR) {
	}
	return 0;
}

/* ---- threads -------------------------------------------------------- */
typedef pthread_t k_tid_t;

static inline k_tid_t k_current_get(void)
{
	return pthread_self();
}

static inline void k_yield(void)
{
	sched_yield();
}

/* Zephyr pins a thread to one CPU; on Linux that is a one-CPU affinity
 * mask. Returns 0 or a negative errno, as Zephyr does. */
static inline int k_thread_cpu_pin(k_tid_t thread, int cpu)
{
#ifdef MB_POSIX_LINUX
	cpu_set_t cs;
	CPU_ZERO(&cs);
	CPU_SET(cpu, &cs);
	int r = pthread_setaffinity_np(thread, sizeof(cs), &cs);
	return r == 0 ? 0 : -r;
#else
	(void)thread; /* no affinity API on this host; ordering is unaffected */
	(void)cpu;
	return 0;
#endif
}

/* On Zephyr these stop the scheduler preempting the current thread while it
 * dispatches. A hosted process cannot do that; the walker's correctness
 * does not depend on it (its ordering is semaphore-driven), only its
 * timing jitter does, which is why measurement runs pin one worker per
 * hart and keep the board otherwise idle. */
static inline void k_sched_lock(void)
{
}

static inline void k_sched_unlock(void)
{
}

/* ---- console / exit ------------------------------------------------- */
#define printk printf

#define SYS_REBOOT_WARM 0
#define SYS_REBOOT_COLD 1
static inline void sys_reboot(int type)
{
	(void)type;
	fflush(stdout);
	exit(0);
}

#endif /* MB_POSIX_H_ */
