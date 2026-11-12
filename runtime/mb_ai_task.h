/* SPDX-License-Identifier: Apache-2.0
 *
 * SpacemiT K3: move the calling thread onto the A100 ("AI") cores.
 *
 * WHY THIS EXISTS. The K3 has two core types behind one Linux scheduler
 * (K3 paper, section 4.1): cpus 0-7 are X100, cpus 8-15 are A100. By default
 * every thread is bound to the X100 set and is never migrated to an A100 core,
 * even when the X100s are saturated. A thread reaches the A100 cores only if
 * the application explicitly labels it an AI task, and once labelled it stays
 * on the A100 set for its lifetime. SpacemiT's Bianbu kernel exposes that
 * through a write-only /proc file:
 *
 *     echo $$ > /proc/set_ai_thread        # label this shell
 *
 * Nothing about that file is documented in the paper beyond the policy, so
 * everything here is written to FAIL LOUDLY: a thread that could not be
 * labelled must not go on to be measured as if it had been.
 *
 * It is per THREAD, so a pool's helper threads are labelled by the helpers
 * themselves (see modelblaster_pool.c); whether the label is inherited across
 * pthread_create is exactly what scripts/a100_bringup.sh tests.
 */
#ifndef MB_AI_TASK_H
#define MB_AI_TASK_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <unistd.h>

#define MB_AI_TASK_FILE "/proc/set_ai_thread"

/* First A100 cpu id. Cpus below it are X100. Measured on our board: lscpu
 * shows 16 cpus, model name "Spacemit(R) X100" for 0-7 and "A100" for 8-15. */
#ifndef MB_A100_FIRST_CPU
#define MB_A100_FIRST_CPU 8
#endif

/* A function that is running while its thread moves must not use the vector
 * unit.
 *
 * Labelling (below) changes the thread's allowed cpus from 0-7 to 8-15, so the
 * kernel moves it onto an A100 inside the write(); a later sched_setaffinity
 * can move it again. X100 vectors are 256-bit (vlenb 32), A100 ones 1024-bit
 * (vlenb 128). GCC reads vlenb once, in a function's prologue, to size the
 * vector part of its stack frame, and assumes it never changes: a function
 * that is on the stack when the thread moves goes on addressing that frame
 * with the new vlenb, and its locals land in the wrong place. That is what
 * zeroed main's return address in the TASK-006 bench
 * (k3-work/research/k3-vlen-migration-hazard.md).
 *
 * Mark every such function MB_THREAD_MOVER: the label call, its caller, and
 * the thread's entry function. In a file compiled with the vector extension it
 * compiles the function without it, so the frame cannot depend on vlenb;
 * otherwise it expands to nothing. The harness, walker and pool sources are
 * compiled without the vector extension (only kernels.c gets -march=...v), so
 * there it is a guard against that ever changing. Single-file benches that
 * #include kernels compile main with it: mark main. */
#ifndef MB_THREAD_MOVER
#if defined(__riscv_vector) && defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 14
#define MB_THREAD_MOVER __attribute__((target("arch=rv64gc")))
#else
#define MB_THREAD_MOVER
#endif
#endif

/* Should a thread that will run on `cpu` be labelled?
 *   MODELBLASTER_AI_TASK unset -> yes iff cpu is an A100 core (auto)
 *   MODELBLASTER_AI_TASK=1     -> always (to test the label on an X100 core)
 *   MODELBLASTER_AI_TASK=0     -> never
 */
static inline int mb_ai_task_wanted(int cpu)
{
	const char *e = getenv("MODELBLASTER_AI_TASK");
	if (e && e[0] == '0') {
		return 0;
	}
	if (e && e[0] == '1') {
		return 1;
	}
	return cpu >= MB_A100_FIRST_CPU;
}

/* Label the calling thread, which moves it onto the A100s. 0 on success, -1
 * with errno set on failure. */
static inline MB_THREAD_MOVER int mb_ai_task_label_self(void)
{
	char buf[32];
	int n = snprintf(buf, sizeof buf, "%ld\n", (long)syscall(SYS_gettid));
	int fd = open(MB_AI_TASK_FILE, O_WRONLY | O_CLOEXEC);
	if (fd < 0) {
		return -1;
	}
	ssize_t w = write(fd, buf, (size_t)n);
	int saved = errno;
	close(fd);
	if (w != (ssize_t)n) {
		errno = (w < 0) ? saved : EIO;
		return -1;
	}
	return 0;
}

#endif /* MB_AI_TASK_H */
