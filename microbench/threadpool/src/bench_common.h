/*
 * Copyright (c) 2026 Dima Nikiforov <vnikiforov@berkeley.edu>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Shared helpers for the cross-hart synchronization microbenchmarks under
 * agents/microbench/threadpool/. The three harnesses (pthreadpool,
 * raw POSIX pthreads, native k_thread+k_sem) all share:
 *   - rdcycle() for per-call cycle deltas (matches the existing
 *     harness's profiling primitive at the 1 GHz target frequency)
 *   - bench_workload_t — the trivial fanned-out body each harness
 *     dispatches across N harts. Tunable in size to expose the cliff
 *     where dispatch overhead dwarfs the work
 *   - results_emit_csv() — prints a single CSV block bracketed by
 *     THREADPOOL_BENCH_{BEGIN,END} markers so the host-side parser can
 *     pick it up regardless of which simulator produced it
 *
 * Per the HTIF UART lesson (FireSim's HTIF is extremely slow; printing
 * inside a timed region starves any worker pinned to hart 0), all
 * printf calls are deferred until after every measurement loop.
 */

#ifndef AGENTS_MICROBENCH_THREADPOOL_BENCH_COMMON_H
#define AGENTS_MICROBENCH_THREADPOOL_BENCH_COMMON_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* RV64 cycle counter at the core's clock (1 GHz target on FireSim
 * quad-rocket-saturn). Inlined so the read happens at the call site
 * with no extra branch / spill.
 *
 * IMPORTANT: rdcycle reads a per-hart counter, not a global one. Each
 * hart's `cycle` CSR counts only the cycles where THIS hart was
 * retiring instructions, so two rdcycle reads on different harts are
 * NOT directly comparable — a worker hart that's been mostly idle
 * since boot can have a cycle value that's millions of cycles below
 * the master's even though they're sampling "at the same wall time".
 * Use `bench_rdmtime` for cross-hart timestamps.
 */
static inline uint64_t bench_rdcycle(void)
{
	uint64_t v;
	__asm__ volatile("rdcycle %0" : "=r"(v));
	return v;
}

/* Global mtime read — coherent across harts. mtime ticks at 1 MHz on
 * the FireSim quad-rocket-saturn build (CONFIG_SYS_CLOCK_HW_CYCLES_PER_SEC),
 * so 1 mtime tick = 1000 target-clock cycles at 1 GHz. The breakdown
 * fields (wake_to_first_worker, etc.) report mtime ticks; convert by
 * multiplying by 1000 to get target cycles, or treat the column as
 * "microseconds at 1 GHz target".
 *
 * On spike, sys_clock_cycle_get_64 returns the same global tick
 * source (spike's clint), so cross-hart values are still comparable;
 * only the rate differs (spike runs at host speed). */
#include <zephyr/sys_clock.h>
static inline uint64_t bench_rdmtime(void)
{
	return sys_clock_cycle_get_64();
}

/* The trivial per-element body each harness dispatches. Tiny on
 * purpose: the point of the benchmark is to characterize *dispatch
 * overhead*, not steady-state throughput. The compiler can't elide it
 * because `out` is declared volatile in the worker (see the harnesses).
 *
 * range = number of elements; each worker processes a contiguous
 * sub-slice. The body is float multiply so the FPU is exercised
 * (matches the kernels that pthreadpool dispatches on in production).
 */
typedef struct {
	const float *in;
	volatile float *out;
	size_t range;
} bench_workload_t;

static inline void bench_apply(const bench_workload_t *wl,
			       size_t start, size_t end)
{
	for (size_t i = start; i < end; i++) {
		wl->out[i] = wl->in[i] * 2.0f;
	}
}

/* CSV row for one (harness, N, range) tuple. We collect these into a
 * fixed-size array on the stack and print the whole batch at the end.
 * Per-iteration cycles are summarized by min/median/max — the spread
 * matters when (e.g.) the first call after warmup includes a JIT-style
 * bring-up cost that isn't representative of steady state.
 *
 * UNITS:
 *   per_call_{min,med,max} — target-clock cycles (master rdcycle delta).
 *   wake_to_first_worker
 *   wake_to_all_finished
 *   finish_to_observed   — mtime ticks (global counter; on FireSim
 *                           quad-rocket-saturn 1 tick = 1 µs target =
 *                           1000 target cycles). Cross-hart-correct
 *                           because mtime is global; rdcycle wouldn't
 *                           be (per-hart counter).
 */
typedef struct {
	const char *harness;	/* "pthreadpool", "pthreads_raw", "k_thread" */
	const char *variant;	/* "default", "spin", "yield" — free-form tag */
	int n_workers;
	int range;
	int reps;
	uint64_t per_call_min;	    /* target cycles */
	uint64_t per_call_med;	    /* target cycles */
	uint64_t per_call_max;	    /* target cycles */
	uint64_t wake_to_first_worker; /* mtime ticks */
	uint64_t wake_to_all_finished; /* mtime ticks */
	uint64_t finish_to_observed;   /* mtime ticks */
} bench_row_t;

/* Median of an in-place buffer. Quick + dirty insertion-sort (reps is
 * O(1k)). Sorts the buffer as a side-effect; caller must accept that. */
static inline uint64_t bench_median_inplace(uint64_t *buf, size_t n)
{
	for (size_t i = 1; i < n; i++) {
		uint64_t v = buf[i];
		size_t j = i;
		while (j > 0 && buf[j - 1] > v) {
			buf[j] = buf[j - 1];
			j--;
		}
		buf[j] = v;
	}
	return n ? buf[n / 2] : 0;
}

static inline uint64_t bench_min(const uint64_t *buf, size_t n)
{
	uint64_t v = buf[0];
	for (size_t i = 1; i < n; i++) {
		if (buf[i] < v) {
			v = buf[i];
		}
	}
	return v;
}

static inline uint64_t bench_max(const uint64_t *buf, size_t n)
{
	uint64_t v = buf[0];
	for (size_t i = 1; i < n; i++) {
		if (buf[i] > v) {
			v = buf[i];
		}
	}
	return v;
}

/* Emit one CSV block. Markers are intentionally distinct from
 * AGENTS_OUTPUT_BEGIN/END so the existing runner doesn't try to parse
 * floats out of them.
 *
 * Schema:
 *   harness,variant,n_workers,range,reps,
 *   per_call_min,per_call_med,per_call_max,
 *   wake_to_first_worker,wake_to_all_finished,finish_to_observed
 */
static inline void bench_emit_csv(const bench_row_t *rows, size_t n)
{
	printf("=== THREADPOOL_BENCH_BEGIN ===\n");
	printf("harness,variant,n_workers,range,reps,"
	       "per_call_min,per_call_med,per_call_max,"
	       "wake_to_first_worker,wake_to_all_finished,"
	       "finish_to_observed\n");
	for (size_t i = 0; i < n; i++) {
		printf("%s,%s,%d,%d,%d,%llu,%llu,%llu,%llu,%llu,%llu\n",
		       rows[i].harness, rows[i].variant,
		       rows[i].n_workers, rows[i].range, rows[i].reps,
		       (unsigned long long)rows[i].per_call_min,
		       (unsigned long long)rows[i].per_call_med,
		       (unsigned long long)rows[i].per_call_max,
		       (unsigned long long)rows[i].wake_to_first_worker,
		       (unsigned long long)rows[i].wake_to_all_finished,
		       (unsigned long long)rows[i].finish_to_observed);
	}
	printf("=== THREADPOOL_BENCH_END ===\n");
}

/* Sweep configuration shared by all three harnesses. Kept identical
 * across the harnesses so the rows of the resulting CSV are directly
 * comparable. */
#define BENCH_RANGE_COUNT 3
static const int BENCH_RANGES[BENCH_RANGE_COUNT] = { 32, 256, 4096 };

#define BENCH_NWORKERS_COUNT 3
static const int BENCH_NWORKERS[BENCH_NWORKERS_COUNT] = { 1, 2, 4 };

/* Rep counts. Trade-off: more reps = tighter median + lower noise on
 * the breakdown averages; fewer reps = much shorter FireSim runs.
 * pthreadpool's dispatch path costs ~13M target cycles per rep, so
 * 256 reps × 9 sweeps would be ~30B cycles (~500s of FireSim host
 * time at our ~60 MHz host clock). 24 reps keeps a single full sweep
 * under ~3B target cycles ≈ ~1 minute of host wallclock per
 * pthreadpool variant, which fits the lab cadence. The cheap
 * harnesses (k_thread, pthreads_raw) finish in seconds either way. */
#define BENCH_WARMUP_REPS 8
#define BENCH_TIMED_REPS  24

/* Static-sized I/O buffers — sized for the largest range we sweep.
 * Both buffers are 4-byte aligned (float). Declared in main TUs because
 * each harness owns its own copies (no shared linker scope across
 * separate elf builds). */
#define BENCH_MAX_RANGE 4096

#endif /* AGENTS_MICROBENCH_THREADPOOL_BENCH_COMMON_H */
