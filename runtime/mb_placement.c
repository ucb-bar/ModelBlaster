/*
 * mb_placement implementation -- see mb_placement.h for the K3 facts this is
 * built around.
 */

#if defined(MODELBLASTER_PLATFORM_LINUX)

#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif

#include "mb_placement.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifndef MODELBLASTER_AI_CPUS
#define MODELBLASTER_AI_CPUS ""
#endif

#define MB_AI_PROC_DEFAULT "/proc/set_ai_thread"

static cpu_set_t g_ai_set;
static pthread_once_t g_ai_once = PTHREAD_ONCE_INIT;

/* Parse "8-15" / "8,9,10" / "0,2-3". Anything malformed is fatal: a typo in
 * the AI set would silently route A100 threads through the plain-pin path
 * (EINVAL at best) or register X100 threads (one-way!) at worst. */
static void parse_ai_set(void)
{
	CPU_ZERO(&g_ai_set);
	const char *s = MODELBLASTER_AI_CPUS;
	const char *p = s;
	while (*p) {
		char *end = NULL;
		long lo = strtol(p, &end, 10);
		long hi = lo;
		if (end == p || lo < 0) goto bad;
		p = end;
		if (*p == '-') {
			p++;
			hi = strtol(p, &end, 10);
			if (end == p || hi < lo) goto bad;
			p = end;
		}
		for (long c = lo; c <= hi; c++) {
			if (c >= CPU_SETSIZE) goto bad;
			CPU_SET((int)c, &g_ai_set);
		}
		if (*p == ',') p++;
		else if (*p) goto bad;
	}
	return;
bad:
	fprintf(stderr, "FATAL: mb_placement: bad MODELBLASTER_AI_CPUS=\"%s\"\n", s);
	abort();
}

int mb_cpu_is_ai(int cpu)
{
	pthread_once(&g_ai_once, parse_ai_set);
	return cpu >= 0 && cpu < CPU_SETSIZE && CPU_ISSET(cpu, &g_ai_set);
}

static int ai_any(void)
{
	pthread_once(&g_ai_once, parse_ai_set);
	return CPU_COUNT(&g_ai_set) > 0;
}

static void diag(long tid, int cpu, const char *step, int err)
{
	fprintf(stderr, "mb_placement: FAILED tid=%ld cpu=%d step=%s errno=%d (%s)\n",
		tid, cpu, step, err, err ? strerror(err) : "none");
}

int mb_place_current_thread(int cpu)
{
	long tid = (long)syscall(SYS_gettid);
	if (cpu < 0 || cpu >= CPU_SETSIZE) {
		diag(tid, cpu, "cpu-range", EINVAL);
		return 1;
	}

	if (mb_cpu_is_ai(cpu)) {
		const char *path = getenv("MODELBLASTER_AI_THREAD_PATH");
		if (path == NULL || *path == '\0') path = MB_AI_PROC_DEFAULT;
		char buf[32];
		int n = snprintf(buf, sizeof buf, "%ld\n", tid);
		int fd = open(path, O_WRONLY);
		if (fd < 0) {
			diag(tid, cpu, "open-ai-thread-file", errno);
			return 2;
		}
		/* One write: the kernel handler parses one line per write. */
		ssize_t w = write(fd, buf, (size_t)n);
		int werr = errno;
		close(fd);
		if (w != n) {
			diag(tid, cpu, "write-ai-thread-file", w < 0 ? werr : EIO);
			return 3;
		}
	}

	cpu_set_t cs;
	CPU_ZERO(&cs);
	CPU_SET(cpu, &cs);
	if (sched_setaffinity(0, sizeof cs, &cs) != 0) {
		diag(tid, cpu, "sched_setaffinity", errno);
		return 4;
	}

	int got = sched_getcpu();
	if (got != cpu) {
		sched_yield();
		got = sched_getcpu();
	}
	if (got != cpu) {
		char step[48];
		snprintf(step, sizeof step, "verify-sched_getcpu(observed=%d)", got);
		diag(tid, cpu, step, got < 0 ? errno : 0);
		return 5;
	}
	return 0;
}

int mb_pin_self_to_first_non_ai_cpu(void)
{
	if (!ai_any()) return 0;
	long tid = (long)syscall(SYS_gettid);
	cpu_set_t allowed;
	if (sched_getaffinity(0, sizeof allowed, &allowed) != 0) {
		diag(tid, -1, "sched_getaffinity", errno);
		return 1;
	}
	for (int c = 0; c < CPU_SETSIZE; c++) {
		if (!CPU_ISSET(c, &allowed) || mb_cpu_is_ai(c)) continue;
		cpu_set_t cs;
		CPU_ZERO(&cs);
		CPU_SET(c, &cs);
		if (sched_setaffinity(0, sizeof cs, &cs) != 0) {
			diag(tid, c, "sched_setaffinity(main)", errno);
			return 2;
		}
		return 0;
	}
	diag(tid, -1, "no-non-ai-cpu-in-allowed-mask", 0);
	return 3;
}

#endif /* MODELBLASTER_PLATFORM_LINUX */
