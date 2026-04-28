/*
 * Smoke test for the vendored Phase-A POSIX affinity patch.
 *
 * Spawns NUM_THREADS pthreads, each pinned via
 * pthread_attr_setaffinity_np() to a distinct hart, and reports the
 * hart each thread actually runs on (via arch_curr_cpu()->id). On a
 * working patch every thread's reported hart matches its pinned one.
 *
 * Run on spike with -p4:
 *   spike -p4 build/affinity_smoketest/zephyr/zephyr.elf
 */

#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>

#include <zephyr/arch/cpu.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/reboot.h>

#define NUM_THREADS CONFIG_MP_MAX_NUM_CPUS

struct worker_arg {
	int worker_id;
	int pinned_hart;
	int observed_hart;
};

static struct worker_arg args[NUM_THREADS];
static struct k_mutex print_mutex;

static void *worker(void *p)
{
	struct worker_arg *a = p;
	a->observed_hart = arch_curr_cpu()->id;

	k_mutex_lock(&print_mutex, K_FOREVER);
	printf("worker %d pinned=%d observed=%d %s\n", a->worker_id,
	       a->pinned_hart, a->observed_hart,
	       a->observed_hart == a->pinned_hart ? "OK" : "MISMATCH");
	k_mutex_unlock(&print_mutex);
	return NULL;
}

int main(void)
{
	printf("affinity smoketest: %d threads, %d harts\n", NUM_THREADS,
	       (int)CONFIG_MP_MAX_NUM_CPUS);
	k_mutex_init(&print_mutex);

	pthread_t tids[NUM_THREADS];
	pthread_attr_t attrs[NUM_THREADS];

	for (int i = 0; i < NUM_THREADS; i++) {
		args[i].worker_id = i;
		args[i].pinned_hart = i;
		args[i].observed_hart = -1;

		int rc = pthread_attr_init(&attrs[i]);
		if (rc != 0) {
			printf("pthread_attr_init[%d] failed: %d\n", i, rc);
			sys_reboot(SYS_REBOOT_COLD);
		}

		cpu_set_t cs;
		CPU_ZERO(&cs);
		CPU_SET(i, &cs);
		rc = pthread_attr_setaffinity_np(&attrs[i], sizeof(cs), &cs);
		if (rc != 0) {
			printf("pthread_attr_setaffinity_np[%d] failed: %d\n", i, rc);
			sys_reboot(SYS_REBOOT_COLD);
		}

		rc = pthread_create(&tids[i], &attrs[i], worker, &args[i]);
		if (rc != 0) {
			printf("pthread_create[%d] failed: %d\n", i, rc);
			sys_reboot(SYS_REBOOT_COLD);
		}
	}

	int ok = 1;
	for (int i = 0; i < NUM_THREADS; i++) {
		pthread_join(tids[i], NULL);
		pthread_attr_destroy(&attrs[i]);
		if (args[i].observed_hart != args[i].pinned_hart) {
			ok = 0;
		}
	}

	printf("affinity smoketest: %s\n", ok ? "PASS" : "FAIL");
	sys_reboot(SYS_REBOOT_COLD);
	return 0;
}
