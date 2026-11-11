/*
 * mb_placement -- put the CURRENT thread on a specific CPU, including the
 * SpaceMiT K3's A100 cluster, which plain sched_setaffinity cannot reach.
 *
 * K3 facts (measured on the board, not read from a datasheet):
 *
 *   - 16 CPUs, one shared memory. X100 = Linux CPUs 0-7, A100 = CPUs 8-15.
 *   - A process starts with mask 0-7.
 *   - A thread may run on an A100 CPU only after writing its Linux TID
 *     (gettid, NOT the pid) followed by "\n" to /proc/set_ai_thread. Before
 *     that, sched_setaffinity({8}) fails with EINVAL. After it, the thread's
 *     mask becomes 8-15 at once and sched_setaffinity({9}) works.
 *   - Registration is ONE-WAY: a registered thread can never pin to CPU 0-7
 *     again (EINVAL). Other threads of the process are unaffected.
 *
 * Consequences that shape this API:
 *
 *   - Placement must be done BY the thread being placed, after it exists and
 *     has a TID. pthread_attr_setaffinity_np at create time cannot work for
 *     A100: there is no TID yet to register.
 *   - The process main thread must never be registered unless it is itself
 *     meant to be an A100 thread, or it can no longer share work with X100.
 *
 * Which CPUs are "AI" CPUs is a COMPILE-TIME property of the target,
 * -DMODELBLASTER_AI_CPUS="8-15" (or "8,9,10"). Empty / undefined means no AI
 * CPUs: mb_place_current_thread() is then a plain pin + verify, which is
 * exactly the K1 behaviour. Nothing is detected at run time on purpose; a
 * board that silently looked like a K1 would file X100 numbers under an A100
 * tag.
 *
 * Compiled to stubs (everything "succeeds", nothing is touched) when
 * MODELBLASTER_PLATFORM_LINUX is not defined, so Zephyr builds are unchanged.
 */

#ifndef MODELBLASTER_RUNTIME_MB_PLACEMENT_H
#define MODELBLASTER_RUNTIME_MB_PLACEMENT_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(MODELBLASTER_PLATFORM_LINUX)

/* Nonzero iff `cpu` is in the compile-time AI-CPU set. */
int mb_cpu_is_ai(int cpu);

/* Register (if `cpu` is an AI CPU), pin the calling thread to exactly `cpu`,
 * and verify sched_getcpu() == cpu (one sched_yield + recheck allowed).
 * Returns 0 on success; on failure returns nonzero after printing ONE
 * diagnostic line to stderr naming tid, cpu, the failing step and errno.
 * The proc file is /proc/set_ai_thread unless MODELBLASTER_AI_THREAD_PATH is
 * set (host tests). */
int mb_place_current_thread(int cpu);

/* Pin the calling thread to the lowest-numbered CPU in its CURRENT allowed
 * mask that is not an AI CPU. Intended for the process main thread, so that
 * threads it creates (which inherit its mask) start X100-masked and then
 * place themselves. A no-op returning 0 when no AI CPUs are configured, so
 * K1 behaviour is unchanged. Nonzero (with a stderr diagnostic) on failure,
 * including "every allowed CPU is an AI CPU". */
int mb_pin_self_to_first_non_ai_cpu(void);

#else /* !MODELBLASTER_PLATFORM_LINUX: Zephyr / bare metal */

static inline int mb_cpu_is_ai(int cpu) { (void)cpu; return 0; }
static inline int mb_place_current_thread(int cpu) { (void)cpu; return 0; }
static inline int mb_pin_self_to_first_non_ai_cpu(void) { return 0; }

#endif

#ifdef __cplusplus
}
#endif

#endif /* MODELBLASTER_RUNTIME_MB_PLACEMENT_H */
