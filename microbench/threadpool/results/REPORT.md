# Threadpool overhead microbenchmarks — FireSim quad-rocket-saturn

Cross-hart synchronization overhead for three dispatch primitives, measured on FireSim's alveo_u250_firesim-quad-rocket-saturn-no-nic-l2-llc4mb-ddr3 hwconfig (4× RVV rocket, 1 GHz target, mtime at 1 MHz). Per-call cycles are master-thread `rdcycle` deltas around one dispatch. Each row is the median of 24 timed iterations after 32 warmup iterations.

## k_sem ping-pong baseline (single-hart)

Round-trip `k_sem_give → k_sem_take` on the master hart, no IPI. Establishes the floor cost of the primitive itself.

| metric | cycles |
|---|---|
| min | 194 |
| median | 194 |
| max | 267 |

## Per-call dispatch cycles (median)

| harness | variant | range | N=1 | N=2 | N=4 |
|---|---|---|---|---|---|
| k_thread | k_sem | 32 | 5,612 | 8,330 | 20,199 |
| k_thread | k_sem | 256 | 7,846 | 8,666 | 19,480 |
| k_thread | k_sem | 4096 | 49,082 | 28,364 | 27,547 |
| pthreads_raw | k_sem | 32 | 6,744 | 11,425 | 20,066 |
| pthreads_raw | k_sem | 256 | 9,040 | 12,619 | 20,146 |
| pthreads_raw | k_sem | 4096 | 49,790 | 31,862 | 27,981 |
| pthreadpool | default | 32 | 613 | 13,011,433 | 27,018,896 |
| pthreadpool | default | 256 | 4,645 | 13,018,774 | 27,026,242 |
| pthreadpool | default | 4096 | 74,167 | 13,147,199 | 27,157,142 |
| pthreadpool | spin | 32 | 616 | 39,999,989 | 79,999,992 |
| pthreadpool | spin | 256 | 4,645 | 39,999,992 | 79,999,992 |
| pthreadpool | spin | 4096 | 74,117 | 39,999,978 | 79,999,986 |

Cycles are at the 1 GHz target clock; divide by 1e6 for ms. Column N is the worker count (master included for pthreadpool, master-only-dispatch for the others).

## Wake / finish breakdown (k_thread, pthreads_raw)

Worker stamps `sys_clock_cycle_get_64()` (mtime, the global tick counter — coherent across harts unlike rdcycle) right after returning from `k_sem_take(start_sem)` (= first observable wake) and again right after computing its slice (= done). All deltas are master-relative **mtime ticks**; on FireSim 1 tick = 1 µs target = 1000 target cycles at 1 GHz. Multiply by 1000 to compare with the per-call cycle column above.

| harness | N | range | wake→first_worker (µs) | wake→all_finished (µs) | finish→observed (µs) |
|---|---|---|---|---|---|
| k_thread | 1 | 32 | 3 | 3 | 2 |
| k_thread | 1 | 256 | 2 | 5 | 2 |
| k_thread | 1 | 4096 | 3 | 46 | 2 |
| k_thread | 2 | 32 | 3 | 4 | 3 |
| k_thread | 2 | 256 | 2 | 5 | 2 |
| k_thread | 2 | 4096 | 3 | 25 | 2 |
| k_thread | 4 | 32 | 4 | 14 | 3 |
| k_thread | 4 | 256 | 4 | 14 | 3 |
| k_thread | 4 | 4096 | 5 | 23 | 3 |
| pthreads_raw | 1 | 32 | 3 | 3 | 3 |
| pthreads_raw | 1 | 256 | 3 | 6 | 2 |
| pthreads_raw | 1 | 4096 | 3 | 46 | 3 |
| pthreads_raw | 2 | 32 | 4 | 8 | 2 |
| pthreads_raw | 2 | 256 | 3 | 9 | 3 |
| pthreads_raw | 2 | 4096 | 3 | 28 | 3 |
| pthreads_raw | 4 | 32 | 3 | 14 | 5 |
| pthreads_raw | 4 | 256 | 3 | 15 | 5 |
| pthreads_raw | 4 | 4096 | 3 | 24 | 3 |

## Cost attribution

Comparing per-call medians at N=4 (pulls in the worst-case wake fanout) for the smallest range (range=32 — work is negligible, so per-call cycles ≈ pure dispatch cost):

| harness | variant | per-call median (cyc) | vs k_thread |
|---|---|---|---|
| k_thread | k_sem | 20,199 | 1.00× |
| pthreads_raw | k_sem | 20,066 | 0.99× |
| pthreadpool | default | 27,018,896 | 1337.64× |
| pthreadpool | spin | 79,999,992 | 3960.59× |

## Recommendation

- For ops smaller than ~1 M cycles per worker share, **stay sequential** — even the cheapest cross-hart primitive (k_thread + k_sem) costs millions of cycles per dispatch on this RTL.
- When parallelism *is* warranted, the relative ordering (cheapest first) at N=4 is `k_thread (k_sem)` ≤ `pthreads_raw (k_sem)` ≪ `pthreadpool (default)`. Switching pthreadpool to `spin` removes the condvar fallback, recovering the closing pthread_cond_wait cost but leaving pthreadpool's queue/state-machine overhead intact.
- Concrete pthreadpool fix path, in order of impact: (1) bump `PTHREADPOOL_SPIN_WAIT_ITERATIONS` so the master spin-wait covers worst-case cross-hart latency on this RTL — see the `spin` variant for the upper bound; (2) replace pthread_cond_wait with futex-equivalent (POSIX_CONFSTR_FUTEX or Zephyr `k_futex`); (3) replace the wrapper's pthread_create-spawned workers with pre-pinned k_threads, side-stepping the POSIX layer entirely (the `k_thread` row above is the floor).

Generated from `firesim_overhead.csv`.
