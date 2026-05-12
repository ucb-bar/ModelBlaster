/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * micro-ROS baseline runtime: fixed-HW per-network ROS nodes.
 *
 *   net_a (hart NET_A_HART): rclc executor + rmw session 0
 *                            runs MODEL_<NET_A_NAME>_DISPATCH_FNS_<NET_A_BACKEND>
 *                            sequentially through the full graph.
 *                            timer @ NET_A_PERIOD_MS (0 = one-shot).
 *
 *   net_b (hart NET_B_HART): rclc executor + rmw session 1
 *                            runs MODEL_<NET_B_NAME>_DISPATCH_FNS_<NET_B_BACKEND>.
 *                            timer @ NET_B_PERIOD_MS.
 *
 *   net_c (hart NET_C_HART): [optional, MICROROS_3NET] rclc executor +
 *                            rmw session 2. Periodic like net_a. Shares
 *                            a hart with net_a or net_b. timer @
 *                            NET_C_PERIOD_MS.
 *
 *   broker (hart MICROROS_BROKER_HART): in-target XRCE-DDS broker —
 *                            single thread serving all sessions.
 *
 * Run window: net_a (periodic) keeps firing on its timer until net_b
 * (one-shot, the "long-running" network) completes its first iteration.
 * On net_b done both threads exit, main prints a CSV trace block + the
 * AGENTS_WALL_CYCLES markers, then reboots. Mirrors the periodic-dronet-
 * during-yolov8-window pattern that the xpurt schedule expresses, so the
 * two harnesses' traces are directly comparable.
 *
 * Each dispatch within a network call is wrapped with k_cycle_get_64()
 * reads; tuples are appended to ros_trace[] and emitted as a single CSV
 * block once net_b completes — same trace-CSV shape as harness_xpurt's
 * AGENTS_XPURT_TRACE block (entry_id, network, instance, dispatch_id,
 * op, name, kind, hart, start_cycles, end_cycles), with op+name left
 * blank (the host plotter resolves them by dispatch_id from the model's
 * graph.json). Cycle values are wall-relative to a single global t0
 * captured immediately before broker_start_pinned().
 */

#include <zephyr/kernel.h>
#include <string.h>
#include <zephyr/sys/printk.h>
#include <zephyr/sys/reboot.h>
#include <zephyr/sys/atomic.h>
#include <zephyr/fatal.h>
#include <stdint.h>
#include <stdio.h>

#include <rcl/rcl.h>
#include <rcl/init_options.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <rmw_microros/rmw_microros.h>
#include <rmw_microros/custom_transport.h>
#include <std_msgs/msg/int32.h>

#include <microros_local/transport_loopback.h>
#include <microros_local/broker.h>

#include "net_a_includes.h"
#include "net_b_includes.h"
#ifdef MICROROS_3NET
#include "net_c_includes.h"
#endif

#define _STR_(x) #x
#define STR(x) _STR_(x)

/* Default cap on periodic-net iterations before we force-exit. With a
 * 50 ms dronet timer and a typical yolov8 cost of ~120 ms, the periodic
 * net normally fires 3-5 times during the run window. Cap at 30 iters
 * (~1.5 s simulated) to bound wall-clock time on placements where
 * yolov8 hangs/faults — keeps the trace block reachable before OOM. */
#ifndef NET_A_MAX_ITERS
#define NET_A_MAX_ITERS 30
#endif

/* Bigger trace ring so 30-iter dronet (30 ops * 30 iters = 900 entries) +
 * yolov8 (~150 entries) fit without dropping records. */
#ifndef ROS_TRACE_MAX
#define ROS_TRACE_MAX 4096
#endif

#define _DISPATCH_FNS(name_up, bs_up) MODEL_##name_up##_DISPATCH_FNS_##bs_up
#define DISPATCH_FNS(name_up, bs_up) _DISPATCH_FNS(name_up, bs_up)

/* Each generated model.c has a `static int n_` profile counter that
 * `dispatch_<model>_<idx>` increments AND uses to index into a static
 * `records_[OP_COUNT]` array.  `run_model_<name>()` resets `n_` to 0 at
 * each iteration, but our harness calls DISPATCH_FNS[i] directly so n_
 * just keeps growing across iterations — writing far past the end of
 * records_ into adjacent BSS.  Manually invoke the model's reset hook
 * (backend-suffixed by backend_rename.py at compile time of model.c)
 * at the top of each run_graph_X so writes stay bounded. */
#define _RESET_PROFILE(name_c, bs) model_##name_c##_reset_profile_##bs
#define RESET_PROFILE(name_c, bs) _RESET_PROFILE(name_c, bs)
void RESET_PROFILE(NET_A_NAME_C, NET_A_BACKEND)(void);
void RESET_PROFILE(NET_B_NAME_C, NET_B_BACKEND)(void);
#ifdef MICROROS_3NET
void RESET_PROFILE(NET_C_NAME_C, NET_C_BACKEND)(void);
#endif

#define _OP_COUNT(name_up) MODEL_##name_up##_OP_COUNT
#define OP_COUNT(name_up) _OP_COUNT(name_up)

#define _OUTPUT_SIZE(name_up) MODEL_##name_up##_OUTPUT_SIZE
#define OUTPUT_SIZE(name_up) _OUTPUT_SIZE(name_up)

#define _STATE_TYPE(name) model_##name##_state_t
#define STATE_TYPE(name) _STATE_TYPE(name)

#define _OUTPUT_ELEM_TYPE(name) model_##name##_output_t
#define OUTPUT_ELEM_TYPE(name) _OUTPUT_ELEM_TYPE(name)

#define _DISPATCH_FN_TYPE(name) model_##name##_dispatch_fn
#define DISPATCH_FN_TYPE(name) _DISPATCH_FN_TYPE(name)

#define _TEST_INPUT(name) model_##name##_test_input
#define TEST_INPUT(name) _TEST_INPUT(name)

extern const DISPATCH_FN_TYPE(NET_A_NAME_C)
    DISPATCH_FNS(NET_A_NAME_UP, NET_A_BACKEND_UP)[OP_COUNT(NET_A_NAME_UP)];
extern const DISPATCH_FN_TYPE(NET_B_NAME_C)
    DISPATCH_FNS(NET_B_NAME_UP, NET_B_BACKEND_UP)[OP_COUNT(NET_B_NAME_UP)];
#ifdef MICROROS_3NET
extern const DISPATCH_FN_TYPE(NET_C_NAME_C)
    DISPATCH_FNS(NET_C_NAME_UP, NET_C_BACKEND_UP)[OP_COUNT(NET_C_NAME_UP)];
#endif

static OUTPUT_ELEM_TYPE(NET_A_NAME_C) out_a[OUTPUT_SIZE(NET_A_NAME_UP)];
static OUTPUT_ELEM_TYPE(NET_B_NAME_C) out_b[OUTPUT_SIZE(NET_B_NAME_UP)];
#ifdef MICROROS_3NET
static OUTPUT_ELEM_TYPE(NET_C_NAME_C) out_c[OUTPUT_SIZE(NET_C_NAME_UP)];
#endif

static STATE_TYPE(NET_A_NAME_C) s_a;
static STATE_TYPE(NET_B_NAME_C) s_b;
#ifdef MICROROS_3NET
static STATE_TYPE(NET_C_NAME_C) s_c;
#endif

#define RCCHECK(rc) do { \
    rcl_ret_t _rc = (rc); \
    if (_rc != RCL_RET_OK) { \
        printk("rcl error %d at %s:%d\n", (int)_rc, __FILE__, __LINE__); \
        return; \
    } \
} while (0)

/* ----- per-node thread context ------------------------------------- */

struct net_ctx {
    int                    session_idx;
    int                    cpu;
    const char            *node_name;
    const char            *done_topic;
    const char            *kind_str;
    rcl_publisher_t        pub_done;
    std_msgs__msg__Int32   done_msg;
    void                 (*run_graph)(void);
    uint64_t               period_ms;
    uint64_t               wall_cycles;
    uint64_t               iter;
};

static struct net_ctx ctx_a, ctx_b;
#ifdef MICROROS_3NET
static struct net_ctx ctx_c;
#endif

/* 1 MB per executor — micro-ROS init goes deep + dronet's im2col_buf is
 * ~295 KB and yolov8's conv scratch needs similar headroom. */
#define EXECUTOR_STACK_SIZE 1048576
K_THREAD_STACK_DEFINE(stack_a, EXECUTOR_STACK_SIZE);
K_THREAD_STACK_DEFINE(stack_b, EXECUTOR_STACK_SIZE);
#ifdef MICROROS_3NET
K_THREAD_STACK_DEFINE(stack_c, EXECUTOR_STACK_SIZE);
#endif
static struct k_thread thread_a, thread_b;
#ifdef MICROROS_3NET
static struct k_thread thread_c;
#endif

/* ----- trace storage ------------------------------------------------- */

#ifndef ROS_TRACE_MAX
#define ROS_TRACE_MAX 1024
#endif

#define ROS_TRACE_MAGIC 0xDEADBEEFCAFEBABEULL
struct ros_trace_slot {
    uint64_t    magic;          /* set to ROS_TRACE_MAGIC by trace_record */
    const char *net;
    int         instance;
    int         dispatch_id;
    int         hart;
    const char *kind;
    uint64_t    start_cycles;
    uint64_t    end_cycles;
};

/* In the 3-network build, ~13 KB at the start of ros_trace[] gets
 * clobbered by writes that pattern-match mlp_control's `records_[]`
 * profile table (dispatch_id + name + op + shape + cycles per entry).
 * Source confirmed via ROS_TRACE_MAGIC sentinel — 231/1524 slots lose
 * their magic.  nm shows mlp's records_ at a different BSS address
 * (0x80950ea0) than ros_trace (0x80951db8) so it's not a direct
 * overlap; the mechanism is some indirection we haven't pinned down.
 * Adding BSS guards/absorbers didn't help (linker placed them
 * elsewhere); heap allocation just exposed the same overflow hitting
 * other critical data instead.  For now: rely on the magic + pointer
 * sanity checks in emit_trace_block() to skip corrupted slots, and
 * use the 2-net configuration where the issue does not manifest
 * (mlp_control absent) as the canonical concurrent-execution proof. */
static struct ros_trace_slot ros_trace[ROS_TRACE_MAX];
static atomic_t              ros_trace_count = ATOMIC_INIT(0);
static uint64_t              global_t0       = 0;

static atomic_t net_b_done = ATOMIC_INIT(0);
static atomic_t net_b_ready = ATOMIC_INIT(0);
#ifdef MICROROS_3NET
static atomic_t net_a_ready = ATOMIC_INIT(0);
#endif

/* Per-dispatch progress marker for post-mortem inspection in the fatal
 * handler. Updated AT THE TOP of each model dispatch (before the call
 * into the dispatch table) and AGAIN AFTER the dispatch returns. The
 * handler dumps this so we can identify the faulting dispatch by id
 * without printk on the hot path (which contends with the broker). */
static volatile int   last_dispatch_id_a = -1;
static volatile int   last_dispatch_id_a_post = -1;
static volatile int   last_dispatch_id_b = -1;
static volatile int   last_dispatch_id_b_post = -1;
static volatile void *last_dispatch_input_b  = NULL;
static volatile void *last_dispatch_output_b = NULL;
static volatile void *last_dispatch_pool_b   = NULL;

static inline void trace_record(const char *net, int inst, int did, int hart,
                                const char *kind, uint64_t s, uint64_t e) {
    int idx = atomic_inc(&ros_trace_count);
    if (idx < ROS_TRACE_MAX) {
        ros_trace[idx].magic         = ROS_TRACE_MAGIC;
        ros_trace[idx].net           = net;
        ros_trace[idx].instance      = inst;
        ros_trace[idx].dispatch_id   = did;
        ros_trace[idx].hart          = hart;
        ros_trace[idx].kind          = kind;
        ros_trace[idx].start_cycles  = s;
        ros_trace[idx].end_cycles    = e;
    }
}

static void run_graph_a(void) {
    int iter = (int)ctx_a.iter;
    RESET_PROFILE(NET_A_NAME_C, NET_A_BACKEND)();
    uint64_t t_run = (uint64_t)k_cycle_get_64();
    /* irq_lock is only needed when another thread (e.g. mlp_control)
     * shares this hart and can preempt mid-dispatch — the V/FPU context
     * save/restore is broken on Saturn and corrupts state. When yolov8
     * is alone on hart 0 (Config B: mlp on hart 1), the lock is
     * unnecessary AND it appears to block the other hart's executor
     * (see plots/microros_3net_configB_*.png — hart 1 idle while
     * yolov8 runs). MICROROS_NO_LOCK_A=1 disables it for cross-config
     * comparison. */
#ifndef MICROROS_NO_LOCK_A
    unsigned int irq_key = irq_lock();
#endif
    for (size_t i = 0; i < OP_COUNT(NET_A_NAME_UP); i++) {
        last_dispatch_id_a = (int)i;
        uint64_t s = (uint64_t)k_cycle_get_64() - global_t0;
        DISPATCH_FNS(NET_A_NAME_UP, NET_A_BACKEND_UP)[i](&s_a);
        uint64_t e = (uint64_t)k_cycle_get_64() - global_t0;
        last_dispatch_id_a_post = (int)i;
        trace_record(NET_A_NAME, iter, (int)i, NET_A_HART, ctx_a.kind_str, s, e);
    }
#ifndef MICROROS_NO_LOCK_A
    irq_unlock(irq_key);
#endif
    ctx_a.wall_cycles = (uint64_t)k_cycle_get_64() - t_run;
}

#ifdef MICROROS_3NET
static void run_graph_c(void) {
    int iter = (int)ctx_c.iter;
    RESET_PROFILE(NET_C_NAME_C, NET_C_BACKEND)();
    uint64_t t_run = (uint64_t)k_cycle_get_64();
    /* Match run_graph_b: mask only MSIE (M-mode IPI). Leaving MTIE
     * (system tick) enabled keeps Zephyr's SMP timer subsystem alive on
     * hart 1 while the dispatch loop runs — the full irq_lock() this
     * function used to call shut off the tick for ~200 µs per fire and
     * collided with hart 0's own irq_lock during yolov8, producing the
     * ~400 ms hart-1 stall seen in FUSE_BC 3-net traces. */
    unsigned long mask_bits = (1UL << 3); /* MSIE */
    unsigned long saved_mie;
    __asm__ volatile("csrrc %0, mie, %1"
                     : "=r"(saved_mie)
                     : "r"(mask_bits)
                     : "memory");
    for (size_t i = 0; i < OP_COUNT(NET_C_NAME_UP); i++) {
        uint64_t s = (uint64_t)k_cycle_get_64() - global_t0;
        DISPATCH_FNS(NET_C_NAME_UP, NET_C_BACKEND_UP)[i](&s_c);
        uint64_t e = (uint64_t)k_cycle_get_64() - global_t0;
        trace_record(NET_C_NAME, iter, (int)i, NET_C_HART, ctx_c.kind_str, s, e);
    }
    if (saved_mie & mask_bits) {
        __asm__ volatile("csrs mie, %0" :: "r"(saved_mie & mask_bits)
                         : "memory");
    }
    ctx_c.wall_cycles = (uint64_t)k_cycle_get_64() - t_run;
}
#endif

static void run_graph_b(void) {
    int iter = (int)ctx_b.iter;
    RESET_PROFILE(NET_B_NAME_C, NET_B_BACKEND)();
    uint64_t t_run = (uint64_t)k_cycle_get_64();
    /* Debug: bisect which interrupt source on hart 1 is corrupting
     * V/scalar state mid-V-kernel. We already proved it's
     * interrupt-driven, not scheduler-driven (k_sched_lock didn't
     * suppress it; irq_lock did). Now mask only specific mie bits to
     * find the offending source.
     *
     *   MIE_MSIE = 1<<3  (M-mode software interrupt = IPI)
     *   MIE_MTIE = 1<<7  (M-mode timer interrupt = system tick)
     *   MIE_MEIE = 1<<11 (M-mode external interrupt)
     *
     * Pick exactly one MICROROS_MASK_* knob at build time:
     *   MASK_ALL    — same as old irq_lock (sanity baseline)
     *   MASK_TIMER  — mask only MTIE; if fault suppressed, system tick
     *                 ISR is the culprit
     *   MASK_IPI    — mask only MSIE; if fault suppressed, IPI handler
     *                 is the culprit
     *   MASK_EXT    — mask only MEIE
     * Default (no define): MASK_TIMER. Try IPI next if timer doesn't
     * suppress. */
#if defined(MICROROS_MASK_ALL)
    unsigned int irq_key = irq_lock();
#else
    /* Build the bitmask additively so multiple knobs can be combined,
     * e.g. -DMICROROS_MASK_TIMER=1 -DMICROROS_MASK_IPI=1 to mask both
     * timer (MTIE) and software/IPI (MSIE). MASK_TIMER is the default
     * if no knob is set so we never accidentally degrade to "no
     * masking at all" silently. */
    unsigned long mask_bits = 0;
#  if defined(MICROROS_MASK_TIMER)
    mask_bits |= (1UL << 7);   /* MTIE */
#  endif
#  if defined(MICROROS_MASK_IPI)
    mask_bits |= (1UL << 3);   /* MSIE */
#  endif
#  if defined(MICROROS_MASK_EXT)
    mask_bits |= (1UL << 11);  /* MEIE */
#  endif
#  if !defined(MICROROS_MASK_TIMER) && !defined(MICROROS_MASK_IPI) && !defined(MICROROS_MASK_EXT)
    /* Default: MSIE (M-mode software interrupt = IPI) — bisected as
     * the minimum sufficient mask to suppress the yolov8 vtype-desync
     * fault on hart 1.  See agents/notes/zephyr_rvv_fix_summary.md
     * "ISR bisection (2026-05-08, fifth pass)".  MASK_TIMER alone and
     * MASK_EXT alone both fail to suppress; only MSIE-or-broader masks
     * produce a clean PASSED run with the irq_lock baseline's
     * wall_cycles. */
    mask_bits |= (1UL << 3);   /* default: MSIE */
#  endif
    unsigned long saved_mie;
    __asm__ volatile("csrrc %0, mie, %1"
                     : "=r"(saved_mie)
                     : "r"(mask_bits)
                     : "memory");
#endif
    for (size_t i = 0; i < OP_COUNT(NET_B_NAME_UP); i++) {
        /* Snapshot the state struct fields right before invoking the
         * dispatch. The fatal handler dumps these so we can spot
         * input/output/pool corruption between dispatches. */
        last_dispatch_id_b      = (int)i;
        last_dispatch_input_b   = (void *)s_b.input;
        last_dispatch_output_b  = (void *)s_b.output;
        last_dispatch_pool_b    = (void *)s_b.pool;
        uint64_t s = (uint64_t)k_cycle_get_64() - global_t0;
        DISPATCH_FNS(NET_B_NAME_UP, NET_B_BACKEND_UP)[i](&s_b);
        uint64_t e = (uint64_t)k_cycle_get_64() - global_t0;
        last_dispatch_id_b_post = (int)i;
        trace_record(NET_B_NAME, iter, (int)i, NET_B_HART, ctx_b.kind_str, s, e);
    }
#if defined(MICROROS_MASK_ALL)
    irq_unlock(irq_key);
#else
    /* Restore only bits we cleared (handles case where mie already had
     * them clear before we entered). */
    if (saved_mie & mask_bits) {
        __asm__ volatile("csrs mie, %0" :: "r"(saved_mie & mask_bits)
                         : "memory");
    }
#endif
    ctx_b.wall_cycles = (uint64_t)k_cycle_get_64() - t_run;
}

/* Skip the rcl_publish() call when either NO_BROKER (no broker → publish
 * has no peer) or NO_PUBLISH (broker still up, but we want to test the
 * pure executor-spin / kernel-dispatch path with zero RMW write
 * activity) is set. */
#if defined(MICROROS_NO_BROKER) || defined(MICROROS_NO_PUBLISH)
#  define MICROROS_SKIP_PUBLISH 1
#endif

static void timer_cb_a(rcl_timer_t *t, int64_t last_call_time) {
    ARG_UNUSED(t); ARG_UNUSED(last_call_time);
    if (ctx_a.run_graph) ctx_a.run_graph();
    ctx_a.done_msg.data = (int32_t)(ctx_a.wall_cycles / 1000);
#ifndef MICROROS_SKIP_PUBLISH
    (void)rcl_publish(&ctx_a.pub_done, &ctx_a.done_msg, NULL);
#endif
    ctx_a.iter++;
}
static void timer_cb_b(rcl_timer_t *t, int64_t last_call_time) {
    ARG_UNUSED(t); ARG_UNUSED(last_call_time);
    if (ctx_b.run_graph) ctx_b.run_graph();
    ctx_b.done_msg.data = (int32_t)(ctx_b.wall_cycles / 1000);
#ifndef MICROROS_SKIP_PUBLISH
    (void)rcl_publish(&ctx_b.pub_done, &ctx_b.done_msg, NULL);
#endif
    ctx_b.iter++;
}
#ifdef MICROROS_3NET
static void timer_cb_c(rcl_timer_t *t, int64_t last_call_time) {
    ARG_UNUSED(t); ARG_UNUSED(last_call_time);
    if (ctx_c.run_graph) ctx_c.run_graph();
    ctx_c.done_msg.data = (int32_t)(ctx_c.wall_cycles / 1000);
#ifndef MICROROS_SKIP_PUBLISH
    (void)rcl_publish(&ctx_c.pub_done, &ctx_c.done_msg, NULL);
#endif
    ctx_c.iter++;
}

#ifdef MICROROS_2EXEC_FUSE_BC
/* Fused callback: one rcl_timer fires, then we run dronet + 2× mlp
 * (1:2 ratio approximating the natural period ratio). Tests whether
 * the multi-timer setup itself is what triggers the hart-1 stall. */
static void timer_cb_bc_fused(rcl_timer_t *t, int64_t last_call_time) {
    ARG_UNUSED(t); ARG_UNUSED(last_call_time);
    if (ctx_b.run_graph) ctx_b.run_graph();
    ctx_b.iter++;
#ifndef MICROROS_FUSE_BC_NO_C
    if (ctx_c.run_graph) ctx_c.run_graph();
    ctx_c.iter++;
    if (ctx_c.run_graph) ctx_c.run_graph();
    ctx_c.iter++;
#endif
}
#endif
#endif

static void node_thread_fn(void *arg1, void *arg2, void *arg3) {
    ARG_UNUSED(arg2); ARG_UNUSED(arg3);
    struct net_ctx *ctx = (struct net_ctx *)arg1;

    printk("[%s] starting on hart %d, session_idx=%d\n",
           ctx->node_name, arch_proc_id(), ctx->session_idx);

    /* Serialize session init: net_b goes first, then net_a, then net_c.
     * Simultaneous CREATE handshakes saturate the broker queue and the
     * later one faults on a corrupted response. */
    if (ctx == &ctx_a) {
        while (!atomic_get(&net_b_ready)) {
            k_msleep(1);
        }
    }
#ifdef MICROROS_3NET
    if (ctx == &ctx_c) {
        while (!atomic_get(&net_a_ready)) {
            k_msleep(1);
        }
    }
#endif

    rcl_allocator_t allocator = rcl_get_default_allocator();
    rcl_init_options_t init_options = rcl_get_zero_initialized_init_options();
    if (rcl_init_options_init(&init_options, allocator) != RCL_RET_OK) {
        printk("[%s] init_options_init failed\n", ctx->node_name);
        return;
    }
    rmw_init_options_t *rmw_options =
        rcl_init_options_get_rmw_init_options(&init_options);
    if (!rmw_options) return;
    if (rmw_uros_options_set_custom_transport(false,
                          (void *)(intptr_t)ctx->session_idx,
                          loopback_open, loopback_close,
                          loopback_write, loopback_read,
                          rmw_options) != RMW_RET_OK) {
        printk("[%s] custom_transport set failed\n", ctx->node_name);
        return;
    }

    rclc_support_t support;
    if (rclc_support_init_with_options(&support, 0, NULL, &init_options,
                                       &allocator) != RCL_RET_OK) {
        printk("[%s] support_init failed\n", ctx->node_name);
        return;
    }

    rcl_node_t node;
    if (rclc_node_init_default(&node, ctx->node_name, "", &support)
        != RCL_RET_OK) {
        printk("[%s] node_init failed\n", ctx->node_name);
        return;
    }

#ifndef MICROROS_SKIP_PUBLISH
    /* Mode D (NO_BROKER) and Mode E (NO_PUBLISH) both skip publisher_init:
     * Mode D has no broker to handshake with anyway; Mode E keeps the
     * broker but wants the pure executor-spin/kernel path with zero
     * RMW writes. rcl_publish in timer_cb_a/b is also #ifdef'd out via
     * MICROROS_SKIP_PUBLISH. */
    const rosidl_message_type_support_t *int32_ts =
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32);
    if (rclc_publisher_init_default(&ctx->pub_done, &node, int32_ts,
                                    ctx->done_topic) != RCL_RET_OK) {
        printk("[%s] pub_init failed\n", ctx->node_name);
        return;
    }
    std_msgs__msg__Int32__init(&ctx->done_msg);
#endif

    rcl_timer_t timer;
    bool one_shot = (ctx->period_ms == 0);
    uint64_t period_ns = one_shot ? 1u : RCL_MS_TO_NS(ctx->period_ms);
    rcl_timer_callback_t _tcb = (ctx == &ctx_a) ? timer_cb_a : timer_cb_b;
#ifdef MICROROS_3NET
    if (ctx == &ctx_c) _tcb = timer_cb_c;
#endif
    rclc_timer_init_default2(&timer, &support, period_ns, _tcb, true);

    rclc_executor_t executor;
    rclc_executor_init(&executor, &support.context, 1, &allocator);
    rclc_executor_add_timer(&executor, &timer);

    if (ctx == &ctx_b) {
        atomic_set(&net_b_ready, 1);
    }
#ifdef MICROROS_3NET
    if (ctx == &ctx_a) {
        atomic_set(&net_a_ready, 1);
    }
#endif

    if (one_shot) {
        while (ctx->iter == 0) {
            rclc_executor_spin_some(&executor, RCL_MS_TO_NS(50));
            k_msleep(1);
        }
        /* Signal the periodic threads (dronet, mlp) to stop — the
         * one-shot network is what defines the "run window". */
        atomic_set(&net_b_done, 1);
    } else {
        const uint64_t periodic_max_iters = NET_A_MAX_ITERS;
        while (!atomic_get(&net_b_done) && ctx->iter < periodic_max_iters) {
            rclc_executor_spin_some(&executor, RCL_MS_TO_NS(10));
            k_msleep(1);
        }
        if (ctx->iter >= periodic_max_iters) {
            printk("[%s] periodic-iter cap %llu hit before net_b done — "
                   "force-exiting\n", ctx->node_name,
                   (unsigned long long)periodic_max_iters);
            atomic_set(&net_b_done, 1);
        }
    }

    printk("[%s] done — total iters=%llu, last wall_cycles=%llu\n",
           ctx->node_name, (unsigned long long)ctx->iter,
           (unsigned long long)ctx->wall_cycles);

#ifndef MICROROS_SKIP_PUBLISH
    (void)rcl_publisher_fini(&ctx->pub_done, &node);
#endif
    /* rcl_node_fini / rclc_support_fini both send DELETE messages to
     * the broker.  In NO_PUBLISH mode the broker is supposed to be idle
     * after init, but these fini messages still flood the loopback
     * queue (we've seen "loopback_write[s0]: queue full, dropped" on
     * yolov8 exit when the broker is starved by a higher-prio executor
     * on the same hart).  We sys_reboot at the end of main anyway, so
     * skip the protocol-level teardown and just exit. */
#ifndef MICROROS_NO_PUBLISH
    (void)rcl_node_fini(&node);
    (void)rclc_support_fini(&support);
#endif
}

#ifndef MICROROS_BROKER_HART
#define MICROROS_BROKER_HART (CONFIG_MP_MAX_NUM_CPUS - 1)
#endif

#ifdef MICROROS_SINGLE_EXECUTOR
/* Diagnostic mode: collapse both networks onto a single thread + single
 * rclc_executor + single rcl_node + two timers, all on one hart. Tests
 * whether the cross-thread/cross-hart RMW transport is the wedge cause.
 * Hart is NET_B_HART (where the rvv-on-h1 wedge has been observed). */
static void single_executor_thread_fn(void *a, void *b, void *c) {
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);

    printk("[single_exec] starting on hart %d (broker hart=%d)\n",
           arch_proc_id(), MICROROS_BROKER_HART);

    rcl_allocator_t allocator = rcl_get_default_allocator();
    rcl_init_options_t init_options = rcl_get_zero_initialized_init_options();
    if (rcl_init_options_init(&init_options, allocator) != RCL_RET_OK) {
        printk("[single_exec] init_options_init failed\n"); return;
    }
    rmw_init_options_t *rmw_options =
        rcl_init_options_get_rmw_init_options(&init_options);
    if (!rmw_options) return;
    /* One session id — covers both publishers, since the broker routes
     * topic-by-name within a session. */
    if (rmw_uros_options_set_custom_transport(false, (void *)(intptr_t)0,
            loopback_open, loopback_close,
            loopback_write, loopback_read,
            rmw_options) != RMW_RET_OK) {
        printk("[single_exec] custom_transport set failed\n"); return;
    }
    rclc_support_t support;
    if (rclc_support_init_with_options(&support, 0, NULL, &init_options,
                                       &allocator) != RCL_RET_OK) {
        printk("[single_exec] support_init failed\n"); return;
    }
    rcl_node_t node;
    if (rclc_node_init_default(&node, "microros_single_exec", "", &support)
        != RCL_RET_OK) {
        printk("[single_exec] node_init failed\n"); return;
    }
    const rosidl_message_type_support_t *int32_ts =
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32);
    if (rclc_publisher_init_default(&ctx_a.pub_done, &node, int32_ts,
                                    ctx_a.done_topic) != RCL_RET_OK) {
        printk("[single_exec] pub_a_init failed\n"); return;
    }
    if (rclc_publisher_init_default(&ctx_b.pub_done, &node, int32_ts,
                                    ctx_b.done_topic) != RCL_RET_OK) {
        printk("[single_exec] pub_b_init failed\n"); return;
    }
    std_msgs__msg__Int32__init(&ctx_a.done_msg);
    std_msgs__msg__Int32__init(&ctx_b.done_msg);

    rcl_timer_t timer_a, timer_b;
    bool a_one_shot = (ctx_a.period_ms == 0);
    bool b_one_shot = (ctx_b.period_ms == 0);
    rclc_timer_init_default2(&timer_a, &support,
        a_one_shot ? 1u : RCL_MS_TO_NS(ctx_a.period_ms),
        timer_cb_a, true);
    rclc_timer_init_default2(&timer_b, &support,
        b_one_shot ? 1u : RCL_MS_TO_NS(ctx_b.period_ms),
        timer_cb_b, true);

    rclc_executor_t executor;
    rclc_executor_init(&executor, &support.context, 2, &allocator);
    rclc_executor_add_timer(&executor, &timer_a);
    rclc_executor_add_timer(&executor, &timer_b);

    /* Run until ctx_b finishes its (one-shot or periodic-cap) duty. */
    const uint64_t b_max = b_one_shot ? 1 : (uint64_t)NET_A_MAX_ITERS;
    while (ctx_b.iter < b_max) {
        rclc_executor_spin_some(&executor, RCL_MS_TO_NS(50));
        k_msleep(1);
    }
    atomic_set(&net_b_done, 1);

    printk("[single_exec] done — a.iter=%llu (wall=%llu), "
           "b.iter=%llu (wall=%llu)\n",
           (unsigned long long)ctx_a.iter,
           (unsigned long long)ctx_a.wall_cycles,
           (unsigned long long)ctx_b.iter,
           (unsigned long long)ctx_b.wall_cycles);
}
#endif /* MICROROS_SINGLE_EXECUTOR */

#if defined(MICROROS_3NET) && defined(MICROROS_2EXEC_BC)
/* "2-executor" mode for the 3-net build: net_a (yolov8) keeps its own
 * thread/executor on hart 0; net_b and net_c collapse onto ONE Zephyr
 * thread + ONE rclc_executor + TWO timers, pinned to hart 1.  This
 * tests whether the hart-1 idle gap in plain Config B comes from
 * having two separate rclc_executors competing on the same hart.
 *
 * Topology:
 *   hart 0: tid_a → executor_a → 1 timer  (yolov8)
 *   hart 1: tid_bc → executor_bc → 2 timers (dronet + mlp)  + broker
 *
 * One rcl_node "combined_bc" carries both timers; one rmw session
 * (idx 1) handles the broker handshake for both. */
static void bc_executor_thread_fn(void *a, void *b, void *c) {
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);
    uint64_t t0 = (uint64_t)k_cycle_get_64();
    printk("[bc_exec] starting on hart %d (broker hart=%d) cyc=%llu\n",
           arch_proc_id(), MICROROS_BROKER_HART, (unsigned long long)t0);

    rcl_allocator_t allocator = rcl_get_default_allocator();
    rcl_init_options_t init_options = rcl_get_zero_initialized_init_options();
    if (rcl_init_options_init(&init_options, allocator) != RCL_RET_OK) {
        printk("[bc_exec] init_options_init failed\n"); return;
    }
    printk("[bc_exec] T1 after init_options_init cyc=%llu\n",
           (unsigned long long)k_cycle_get_64());
    rmw_init_options_t *rmw_options =
        rcl_init_options_get_rmw_init_options(&init_options);
    if (!rmw_options) return;
    /* Single rmw session — idx 1 — covers both nets.  net_a uses idx 0. */
    if (rmw_uros_options_set_custom_transport(false, (void *)(intptr_t)1,
            loopback_open, loopback_close,
            loopback_write, loopback_read,
            rmw_options) != RMW_RET_OK) {
        printk("[bc_exec] custom_transport set failed\n"); return;
    }
    printk("[bc_exec] T2 after custom_transport cyc=%llu\n",
           (unsigned long long)k_cycle_get_64());
    rclc_support_t support;
    if (rclc_support_init_with_options(&support, 0, NULL, &init_options,
                                       &allocator) != RCL_RET_OK) {
        printk("[bc_exec] support_init failed\n"); return;
    }
    printk("[bc_exec] T3 after rclc_support_init cyc=%llu\n",
           (unsigned long long)k_cycle_get_64());
    rcl_node_t node;
    if (rclc_node_init_default(&node, "combined_bc", "", &support)
        != RCL_RET_OK) {
        printk("[bc_exec] node_init failed\n"); return;
    }
    printk("[bc_exec] T4 after node_init cyc=%llu\n",
           (unsigned long long)k_cycle_get_64());

    /* Executor declared up-front so both the FUSE_BC and dual-timer
     * paths can fill it in below. */
    rclc_executor_t executor;
#ifdef MICROROS_2EXEC_FUSE_BC
    /* FUSE_BC: only add ONE timer to the executor, and its callback
     * runs run_graph_b once + run_graph_c twice (1:2 ratio). This
     * mimics the 2-net topology (1 executor / 1 timer per hart)
     * while still running 3 networks worth of work — tests whether
     * the hart-1 stall is from "multi-timer-per-executor". */
    rcl_timer_t timer_bc;
    rclc_timer_init_default2(&timer_bc, &support,
        RCL_MS_TO_NS(ctx_b.period_ms), timer_cb_bc_fused, true);
    rclc_executor_init(&executor, &support.context, 1, &allocator);
    rclc_executor_add_timer(&executor, &timer_bc);
    printk("[bc_exec] FUSE_BC: 1 timer, callback runs b+2c\n");
    /* Skip the dual-timer + dual-add path below by short-circuiting. */
    goto bc_ready;
#endif
    rcl_timer_t timer_b, timer_c;
    bool b_one_shot = (ctx_b.period_ms == 0);
    bool c_one_shot = (ctx_c.period_ms == 0);
    /* MICROROS_2EXEC_FIRE_FAST: collapse both periods to 1ns so the
     * timers fire on every spin_some call. Tests whether the apparent
     * Config-B "gap" was just the bc_executor correctly waiting for
     * its first periodic fire — in the 3-thread case dronet inited
     * late so its 20ms period was already overdue and fired in
     * catch-up. When this flag is set, both timers are always due. */
#ifdef MICROROS_2EXEC_FIRE_FAST
    rclc_timer_init_default2(&timer_b, &support, 1u, timer_cb_b, true);
    rclc_timer_init_default2(&timer_c, &support, 1u, timer_cb_c, true);
    printk("[bc_exec] FIRE_FAST: both timer periods=1ns\n");
#else
    rclc_timer_init_default2(&timer_b, &support,
        b_one_shot ? 1u : RCL_MS_TO_NS(ctx_b.period_ms),
        timer_cb_b, true);
    rclc_timer_init_default2(&timer_c, &support,
        c_one_shot ? 1u : RCL_MS_TO_NS(ctx_c.period_ms),
        timer_cb_c, true);
#endif
    printk("[bc_exec] T5 after timer_init x2 cyc=%llu\n",
           (unsigned long long)k_cycle_get_64());

    rclc_executor_init(&executor, &support.context, 2, &allocator);
    rclc_executor_add_timer(&executor, &timer_b);
    rclc_executor_add_timer(&executor, &timer_c);
    printk("[bc_exec] T6 after executor_init+add_timer x2 cyc=%llu\n",
           (unsigned long long)k_cycle_get_64());

#ifdef MICROROS_2EXEC_FUSE_BC
bc_ready:
#endif
    /* Now that this thread has gone through the broker handshake first,
     * tell net_a it can proceed with its own init.  Keeps the
     * three-way serialization invariant the old code relied on. */
    atomic_set(&net_b_ready, 1);
    printk("[bc_exec] T7 set net_b_ready=1 cyc=%llu\n",
           (unsigned long long)k_cycle_get_64());

    /* Drive until net_b hits its periodic cap (it sets net_b_done on
     * exit). net_c piggy-backs on the same loop. */
    const uint64_t b_max = (uint64_t)NET_A_MAX_ITERS;
    int loop_iter = 0;
    while (ctx_b.iter < b_max) {
        rclc_executor_spin_some(&executor, RCL_MS_TO_NS(10));
        k_msleep(1);
        /* Log every 100 outer-loop iterations to see if/when dispatches
         * actually fire on hart 1 while yolov8 is running on hart 0. */
        if ((loop_iter++ & 0xff) == 0) {
            printk("[bc_exec] loop_iter=%d ctx_b.iter=%llu ctx_c.iter=%llu cyc=%llu\n",
                   loop_iter, (unsigned long long)ctx_b.iter,
                   (unsigned long long)ctx_c.iter,
                   (unsigned long long)k_cycle_get_64());
        }
    }
    atomic_set(&net_b_done, 1);

    printk("[bc_exec] done — b.iter=%llu (wall=%llu) "
           "c.iter=%llu (wall=%llu)\n",
           (unsigned long long)ctx_b.iter,
           (unsigned long long)ctx_b.wall_cycles,
           (unsigned long long)ctx_c.iter,
           (unsigned long long)ctx_c.wall_cycles);

    /* Same rationale as in node_thread_fn: skip protocol-level teardown
     * in NO_PUBLISH mode (broker would be flooded with DELETE traffic
     * while the executor on its hart starves it). */
#ifndef MICROROS_NO_PUBLISH
    (void)rcl_node_fini(&node);
    (void)rclc_support_fini(&support);
#endif
}
#endif /* MICROROS_3NET && MICROROS_2EXEC_BC */

#if defined(MICROROS_3NET) && defined(MICROROS_2EXEC_BC) && defined(MICROROS_2EXEC_NORCLC)
/* Strip rclc / rmw entirely from the hart-1 side: bc_norclc just calls
 * run_graph_b and run_graph_c in a tight while-loop, no timers, no
 * executor, no broker handshake. Ratio 1:2 (dronet:mlp) approximates
 * the 20ms:10ms period ratio from the timer-driven version. If hart 1
 * still stalls during yolov8's run with this thread, the issue is
 * outside rclc — somewhere in the Zephyr SMP scheduler. If hart 1 runs
 * concurrently, the gap is squarely in rclc_executor_spin_some's
 * internal wait. */
static void bc_norclc_thread_fn(void *a, void *b, void *c) {
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);
    printk("[bc_norclc] starting on hart %d (no rclc, direct dispatch, 1:2 b:c)\n",
           arch_proc_id());

    /* Let net_a (yolov8) proceed with its broker-handshake init. */
    atomic_set(&net_b_ready, 1);

    /* Wait for yolov8's broker handshake to complete (net_a sets
     * net_a_ready when its rclc init finishes). The broker is on
     * hart 1 with us; if we enter the tight loop now, we'll starve
     * the broker (priority 6 < ours at 1) and yolov8 will hang in
     * rclc_node_init. Yield until handshake is done. */
    while (!atomic_get(&net_a_ready)) {
        k_msleep(1);
    }
    printk("[bc_norclc] net_a_ready seen, entering tight loop hart=%d\n",
           arch_proc_id());

    const uint64_t b_max = (uint64_t)NET_A_MAX_ITERS;
    while (ctx_b.iter < b_max) {
        /* 1 dronet + 2 mlp per outer iteration. ctx_*.run_graph
         * already records each dispatch into ros_trace[]. */
        if (ctx_b.run_graph) ctx_b.run_graph();
        ctx_b.iter++;

        if (ctx_c.run_graph) ctx_c.run_graph();
        ctx_c.iter++;

        if (ctx_c.run_graph) ctx_c.run_graph();
        ctx_c.iter++;
    }
    atomic_set(&net_b_done, 1);

    printk("[bc_norclc] done b.iter=%llu c.iter=%llu\n",
           (unsigned long long)ctx_b.iter,
           (unsigned long long)ctx_c.iter);
}
#endif

#ifdef MICROROS_NO_MICROROS
/* Diagnostic mode: skip micro-ROS entirely. Two threads (one per net,
 * pinned to their respective harts), each driven by Zephyr k_msleep
 * ticks acting as the timer/scheduler. No broker, no transport, no
 * publish, no rclc executor. Tests whether the wedge is anywhere in
 * the micro-ROS stack vs. anything intrinsic to kernel/V-state on
 * hart 1. */
static void no_microros_thread_fn(void *arg1, void *arg2, void *arg3) {
    ARG_UNUSED(arg2); ARG_UNUSED(arg3);
    struct net_ctx *ctx = (struct net_ctx *)arg1;

    printk("[%s] no_microros: starting on hart %d, kind=%s\n",
           ctx->node_name, arch_proc_id(), ctx->kind_str);

    /* Net A waits for net B to finish its first (one-shot) dispatch
     * before exiting; mirrors broker-mode handshake ordering. */
    bool one_shot = (ctx->period_ms == 0);
    if (one_shot) {
        if (ctx->run_graph) ctx->run_graph();
        ctx->iter = 1;
        atomic_set(&net_b_done, 1);
    } else {
        const uint64_t periodic_max_iters = NET_A_MAX_ITERS;
        while (!atomic_get(&net_b_done) && ctx->iter < periodic_max_iters) {
            if (ctx->run_graph) ctx->run_graph();
            ctx->iter++;
            k_msleep((int)ctx->period_ms);
        }
    }

    printk("[%s] no_microros: done iters=%llu wall=%llu\n",
           ctx->node_name, (unsigned long long)ctx->iter,
           (unsigned long long)ctx->wall_cycles);
}
#endif /* MICROROS_NO_MICROROS */

/* Direct HTIF tohost-CSR writer that bypasses Zephyr's UART driver and
 * its console mutex. The default `printk` path blocks on htif_lock,
 * which the broker thread on the other hart holds for each HDLC
 * topic-emit frame; in a fault on hart 1 that yields a truncated dump.
 * Writing tohost directly is racy with broker output (the host UART
 * sees interleaved bytes) but all our characters reach stdout. */
extern volatile uint64_t tohost __attribute__((section(".htif")));

static inline void direct_htif_putc(char c)
{
	while (tohost != 0) {
		__asm__ volatile ("nop");
	}
	tohost = (((uint64_t)1) << 56) | (((uint64_t)1) << 48) | (uint64_t)(unsigned char)c;
}

static void direct_htif_puts(const char *s)
{
	while (*s) {
		direct_htif_putc(*s++);
	}
}

static void direct_htif_put_hex(unsigned long v)
{
	const char *digits = "0123456789abcdef";
	char buf[18];
	buf[0] = '0'; buf[1] = 'x';
	for (int i = 0; i < 16; i++) {
		buf[17 - i] = digits[v & 0xf];
		v >>= 4;
	}
	for (int i = 0; i < 18; i++) {
		direct_htif_putc(buf[i]);
	}
}

static void direct_htif_put_dec(unsigned long v)
{
	char buf[24];
	int len = 0;
	if (v == 0) {
		direct_htif_putc('0');
		return;
	}
	while (v) {
		buf[len++] = '0' + (v % 10);
		v /= 10;
	}
	while (len--) {
		direct_htif_putc(buf[len]);
	}
}

/* Custom fatal-error handler: dumps the full exception state via direct
 * HTIF writes, bypassing printk's lock so the broker's in-flight HDLC
 * frames don't starve the diagnostic. The default Zephyr dump runs
 * first (and truncates), but this re-emit gives us mepc/ra/regs. */
void k_sys_fatal_error_handler(unsigned int reason, const struct arch_esf *esf)
{
	/* Disable interrupts before anything else — an IPI between fault
	 * entry and our CSR reads will overwrite mcause/mtval/mepc. The
	 * v6 dump showed mcause=0x800...3 (M-mode software interrupt =
	 * IPI) instead of the actual load-fault mcause=5. */
	unsigned long mcause, mtval, mepc, mstatus;
	__asm__ volatile("csrci mstatus, 8" ::: "memory");
	__asm__ volatile("csrr %0, mcause"  : "=r"(mcause));
	__asm__ volatile("csrr %0, mtval"   : "=r"(mtval));
	__asm__ volatile("csrr %0, mepc"    : "=r"(mepc));
	__asm__ volatile("csrr %0, mstatus" : "=r"(mstatus));

	broker_quiesce();

	direct_htif_puts("\n=== HARNESS_FATAL_BEGIN hart=");
	direct_htif_put_dec((unsigned long)arch_proc_id());
	direct_htif_puts(" reason=");
	direct_htif_put_dec(reason);
	direct_htif_puts(" ===\n");
	/* Live CSRs (may have been overwritten by an IPI between Zephyr's
	 * fatal printer and our handler — see comment above). */
	direct_htif_puts("live: mcause="); direct_htif_put_hex(mcause);
	direct_htif_puts(" mtval="); direct_htif_put_hex(mtval);
	direct_htif_puts(" mepc="); direct_htif_put_hex(mepc);
	direct_htif_puts(" mstatus="); direct_htif_put_hex(mstatus);
	direct_htif_putc('\n');
	/* esf->mepc / esf->mstatus are saved by isr.S at trap entry — those
	 * reflect the ORIGINAL fault, not whatever IPI fired afterwards. */
	if (esf != NULL) {
		direct_htif_puts("esf:  mepc=");
		direct_htif_put_hex(esf->mepc);
		direct_htif_puts(" mstatus=");
		direct_htif_put_hex(esf->mstatus);
		direct_htif_putc('\n');
	}

	if (esf != NULL) {
		direct_htif_puts("a0="); direct_htif_put_hex(esf->a0);
		direct_htif_puts(" a1="); direct_htif_put_hex(esf->a1);
		direct_htif_puts(" a2="); direct_htif_put_hex(esf->a2);
		direct_htif_puts(" a3="); direct_htif_put_hex(esf->a3);
		direct_htif_putc('\n');
		direct_htif_puts("a4="); direct_htif_put_hex(esf->a4);
		direct_htif_puts(" a5="); direct_htif_put_hex(esf->a5);
		direct_htif_puts(" a6="); direct_htif_put_hex(esf->a6);
		direct_htif_puts(" a7="); direct_htif_put_hex(esf->a7);
		direct_htif_putc('\n');
		direct_htif_puts("t0="); direct_htif_put_hex(esf->t0);
		direct_htif_puts(" t1="); direct_htif_put_hex(esf->t1);
		direct_htif_puts(" t2="); direct_htif_put_hex(esf->t2);
		direct_htif_puts(" t3="); direct_htif_put_hex(esf->t3);
		direct_htif_putc('\n');
		direct_htif_puts("t4="); direct_htif_put_hex(esf->t4);
		direct_htif_puts(" t5="); direct_htif_put_hex(esf->t5);
		direct_htif_puts(" t6="); direct_htif_put_hex(esf->t6);
		direct_htif_puts(" ra="); direct_htif_put_hex(esf->ra);
		direct_htif_putc('\n');
	}

	/* Per-dispatch progress markers — id_pre is updated before the
	 * dispatch call, id_post after it returns. If pre==post the fault
	 * came from after the dispatch (e.g., trace_record path). If
	 * pre>post the fault is INSIDE that dispatch's kernel. */
	direct_htif_puts("net_a: dispatch_pre=");
	direct_htif_put_dec((unsigned long)(int)last_dispatch_id_a);
	direct_htif_puts(" dispatch_post=");
	direct_htif_put_dec((unsigned long)(int)last_dispatch_id_a_post);
	direct_htif_putc('\n');

	direct_htif_puts("net_b: dispatch_pre=");
	direct_htif_put_dec((unsigned long)(int)last_dispatch_id_b);
	direct_htif_puts(" dispatch_post=");
	direct_htif_put_dec((unsigned long)(int)last_dispatch_id_b_post);
	direct_htif_putc('\n');

	direct_htif_puts("net_b: input=");
	direct_htif_put_hex((unsigned long)last_dispatch_input_b);
	direct_htif_puts(" output=");
	direct_htif_put_hex((unsigned long)last_dispatch_output_b);
	direct_htif_puts(" pool=");
	direct_htif_put_hex((unsigned long)last_dispatch_pool_b);
	direct_htif_putc('\n');

	direct_htif_puts("net_b: s_b.input=");
	direct_htif_put_hex((unsigned long)s_b.input);
	direct_htif_puts(" s_b.output=");
	direct_htif_put_hex((unsigned long)s_b.output);
	direct_htif_puts(" s_b.pool=");
	direct_htif_put_hex((unsigned long)s_b.pool);
	direct_htif_putc('\n');

	/* V CSR dump — V state corruption is a known failure mode here. If
	 * vstart != 0 at fault entry, the next vlse/vse will index off-base
	 * (matches the original mtval=0xcc747057 signature).
	 *
	 * In lazy V mode (CONFIG_RISCV_V_DECOUPLED_LAZY=y) MSTATUS.VS gets
	 * cleared on trap entry, so reading any V CSR (incl. vstart) raises
	 * an illegal-instruction fault. Re-enable V access via VS=CLEAN
	 * before the dump — we're past saving the fault context anyway,
	 * so dirtying VS is harmless. Bit 9 = MSTATUS.VS[0]; setting both
	 * VS bits ([10:9]) to 01 = CLEAN. */
	unsigned long vstart, vl, vtype, vcsr, vlenb;
	unsigned long ms_vs_clean = (1UL << 9);
	__asm__ volatile (".option push\n"
	                  ".option arch, +v\n"
	                  "csrs mstatus, %5\n"
	                  "csrr %0, vstart\n"
	                  "csrr %1, vl\n"
	                  "csrr %2, vtype\n"
	                  "csrr %3, vcsr\n"
	                  "csrr %4, vlenb\n"
	                  ".option pop"
	                  : "=r"(vstart), "=r"(vl), "=r"(vtype),
	                    "=r"(vcsr), "=r"(vlenb)
	                  : "r"(ms_vs_clean));
	direct_htif_puts("V: vstart=");
	direct_htif_put_hex(vstart);
	direct_htif_puts(" vl=");
	direct_htif_put_hex(vl);
	direct_htif_puts(" vtype=");
	direct_htif_put_hex(vtype);
	direct_htif_puts(" vcsr=");
	direct_htif_put_hex(vcsr);
	direct_htif_puts(" vlenb=");
	direct_htif_put_hex(vlenb);
	direct_htif_putc('\n');

	/* Stack snapshot — s1 (callee-saved) is reloaded from a stack slot
	 * before the faulting `lb`. Dumping the saved frame may reveal a
	 * stack-spill corruption (somebody else wrote into yolov8's stack). */
	if (esf != NULL) {
		uintptr_t sp = (uintptr_t)esf + sizeof(struct arch_esf);
		direct_htif_puts("sp=");
		direct_htif_put_hex(sp);
		direct_htif_putc('\n');
		const unsigned long *stk = (const unsigned long *)sp;
		for (int row = 0; row < 16; row++) {
			direct_htif_puts("sp+");
			direct_htif_put_hex((unsigned long)(row * 32));
			direct_htif_puts(": ");
			for (int col = 0; col < 4; col++) {
				direct_htif_put_hex(stk[row * 4 + col]);
				direct_htif_putc(' ');
			}
			direct_htif_putc('\n');
		}
	}

	direct_htif_puts("=== HARNESS_FATAL_END ===\n");
	/* Loop forever — firesim_runner's fault detector ends the workload. */
	for (;;) {
		__asm__ volatile ("wfi");
	}
}

/* Reject pointers that aren't in the kernel image's .rodata range. The
 * wild write in the 3-network build leaves slot[1].net at e.g. 0x8b1 —
 * dereferencing that in printk("%s") wedges HTIF and the whole dump
 * stalls. Anything outside [0x80000000, 0x90000000) is by definition not
 * a string literal in our image. */
static inline bool is_rodata_str(const char *p) {
    uintptr_t a = (uintptr_t)p;
    return a >= 0x80000000UL && a < 0x90000000UL;
}

static void emit_trace_block(void) {
    int n = atomic_get(&ros_trace_count);
    if (n > ROS_TRACE_MAX) n = ROS_TRACE_MAX;
    /* Count entries that lack the magic — those were never written by
     * trace_record or were clobbered by some other code writing to the
     * same memory.  Reports separately from skipped-due-to-bad-strings. */
    int no_magic = 0;
    for (int i = 0; i < n; i++) {
        if (ros_trace[i].magic != ROS_TRACE_MAGIC) no_magic++;
    }
    printk("=== TRACE_MAGIC_AUDIT: %d/%d slots missing ROS_TRACE_MAGIC ===\n",
           no_magic, n);
    printk("=== AGENTS_ROS_TRACE_BEGIN ===\n");
    printk("entry_id,network,instance,dispatch_id,op,name,kind,hart,start_cycles,end_cycles\n");
    int skipped = 0;
    for (int i = 0; i < n; i++) {
        struct ros_trace_slot *s = &ros_trace[i];
        if (s->magic != ROS_TRACE_MAGIC ||
            !is_rodata_str(s->net) || !is_rodata_str(s->kind)) {
            skipped++;
            continue;
        }
        printk("%d,%s,%d,%d,,,%s,%d,%llu,%llu\n",
               i, s->net, s->instance, s->dispatch_id,
               s->kind, s->hart,
               (unsigned long long)s->start_cycles,
               (unsigned long long)s->end_cycles);
    }
    printk("=== AGENTS_ROS_TRACE_END (skipped=%d corrupted) ===\n", skipped);
}

int main(void) {
    printk("*** microros_demo boot: " NET_A_NAME " on hart %d ("
           STR(NET_A_BACKEND) "), " NET_B_NAME " on hart %d ("
           STR(NET_B_BACKEND) ")"
#ifdef MICROROS_3NET
           ", " NET_C_NAME " on hart %d (" STR(NET_C_BACKEND) ")"
#endif
           " ***\n",
           NET_A_HART, NET_B_HART
#ifdef MICROROS_3NET
           , NET_C_HART
#endif
           );

    s_a.input  = TEST_INPUT(NET_A_NAME_C);
    s_a.output = out_a;
    s_a.pool   = NULL;
    s_b.input  = TEST_INPUT(NET_B_NAME_C);
    s_b.output = out_b;
    s_b.pool   = NULL;
#ifdef MICROROS_3NET
    s_c.input  = TEST_INPUT(NET_C_NAME_C);
    s_c.output = out_c;
    s_c.pool   = NULL;
#endif

    ctx_a.session_idx = 0;
    ctx_a.cpu         = NET_A_HART;
    ctx_a.node_name   = NET_A_NAME;
    ctx_a.done_topic  = "/" NET_A_NAME "/done";
    ctx_a.kind_str    = STR(NET_A_BACKEND);
    ctx_a.run_graph   = run_graph_a;
    ctx_a.period_ms   = NET_A_PERIOD_MS;

    ctx_b.session_idx = 1;
    ctx_b.cpu         = NET_B_HART;
    ctx_b.node_name   = NET_B_NAME;
    ctx_b.done_topic  = "/" NET_B_NAME "/done";
    ctx_b.kind_str    = STR(NET_B_BACKEND);
    ctx_b.run_graph   = run_graph_b;
    ctx_b.period_ms   = NET_B_PERIOD_MS;

#ifdef MICROROS_3NET
    ctx_c.session_idx = 2;
    ctx_c.cpu         = NET_C_HART;
    ctx_c.node_name   = NET_C_NAME;
    ctx_c.done_topic  = "/" NET_C_NAME "/done";
    ctx_c.kind_str    = STR(NET_C_BACKEND);
    ctx_c.run_graph   = run_graph_c;
    ctx_c.period_ms   = NET_C_PERIOD_MS;
#endif

    global_t0 = (uint64_t)k_cycle_get_64();

#if !defined(MICROROS_NO_BROKER) && !defined(MICROROS_NO_MICROROS)
    broker_start_pinned(MICROROS_BROKER_HART);
#endif

    /* K_FP_REGS opts the thread into FPU/vector register save+restore on
     * context switch. Without it, picolibc's RVV-vectorized memcpy
     * inside the model kernels writes to base + stale-vstart * elem,
     * faulting with mcause=6 misaligned-store as soon as the second
     * thread runs. */
#ifdef MICROROS_SINGLE_EXECUTOR
    k_tid_t tid_se = k_thread_create(
        &thread_a, stack_a, K_THREAD_STACK_SIZEOF(stack_a),
        single_executor_thread_fn, NULL, NULL, NULL,
        K_PRIO_PREEMPT(1), K_FP_REGS, K_FOREVER);
    k_thread_name_set(tid_se, "single_exec");
    /* Pin to NET_B_HART — that's where the rvv-on-h1 wedge has been
     * observed. If the wedge persists with both nets here under one
     * executor, the cause is independent of cross-thread/cross-hart
     * RMW interaction. */
    k_thread_cpu_pin(tid_se, NET_B_HART);
    k_thread_start(tid_se);
    k_thread_join(tid_se, K_FOREVER);
#elif defined(MICROROS_NO_MICROROS)
    k_tid_t tid_a = k_thread_create(
        &thread_a, stack_a, K_THREAD_STACK_SIZEOF(stack_a),
        no_microros_thread_fn, &ctx_a, NULL, NULL,
        K_PRIO_PREEMPT(1), K_FP_REGS, K_FOREVER);
    k_thread_name_set(tid_a, NET_A_NAME);
    k_thread_cpu_pin(tid_a, ctx_a.cpu);

    k_tid_t tid_b = k_thread_create(
        &thread_b, stack_b, K_THREAD_STACK_SIZEOF(stack_b),
        no_microros_thread_fn, &ctx_b, NULL, NULL,
        K_PRIO_PREEMPT(1), K_FP_REGS, K_FOREVER);
    k_thread_name_set(tid_b, NET_B_NAME);
    k_thread_cpu_pin(tid_b, ctx_b.cpu);

#ifdef MICROROS_3NET
    k_tid_t tid_c = k_thread_create(
        &thread_c, stack_c, K_THREAD_STACK_SIZEOF(stack_c),
        no_microros_thread_fn, &ctx_c, NULL, NULL,
        K_PRIO_PREEMPT(1), K_FP_REGS, K_FOREVER);
    k_thread_name_set(tid_c, NET_C_NAME);
    k_thread_cpu_pin(tid_c, ctx_c.cpu);
#endif

    k_thread_start(tid_a);
    k_thread_start(tid_b);
#ifdef MICROROS_3NET
    k_thread_start(tid_c);
#endif
    k_thread_join(tid_a, K_FOREVER);
    k_thread_join(tid_b, K_FOREVER);
#ifdef MICROROS_3NET
    k_thread_join(tid_c, K_FOREVER);
#endif
#elif defined(MICROROS_3NET) && defined(MICROROS_2EXEC_BC)
    /* 2-executor mode: tid_a runs yolov8 on hart 0 (normal path), and
     * a single tid_bc runs both dronet and mlp on hart 1 via one
     * rclc_executor with two timers.  Compared to plain 3-net Config B,
     * this collapses two same-hart Zephyr threads + two separate
     * executors into one of each. */
    k_tid_t tid_a = k_thread_create(
        &thread_a, stack_a, K_THREAD_STACK_SIZEOF(stack_a),
        node_thread_fn, &ctx_a, NULL, NULL,
        K_PRIO_PREEMPT(1), K_FP_REGS, K_FOREVER);
    k_thread_name_set(tid_a, NET_A_NAME);
    k_thread_cpu_pin(tid_a, ctx_a.cpu);

    /* Reuse stack_b for the combined thread. */
    k_tid_t tid_bc = k_thread_create(
        &thread_b, stack_b, K_THREAD_STACK_SIZEOF(stack_b),
#ifdef MICROROS_2EXEC_NORCLC
        bc_norclc_thread_fn,
#else
        bc_executor_thread_fn,
#endif
        NULL, NULL, NULL,
        K_PRIO_PREEMPT(1), K_FP_REGS, K_FOREVER);
    k_thread_name_set(tid_bc,
#ifdef MICROROS_2EXEC_NORCLC
                      "bc_norclc"
#else
                      "bc_exec"
#endif
                      );
    k_thread_cpu_pin(tid_bc, ctx_b.cpu);

    k_thread_start(tid_a);
    k_thread_start(tid_bc);
    k_thread_join(tid_a, K_FOREVER);
    k_thread_join(tid_bc, K_FOREVER);
#else
    k_tid_t tid_a = k_thread_create(
        &thread_a, stack_a, K_THREAD_STACK_SIZEOF(stack_a),
        node_thread_fn, &ctx_a, NULL, NULL,
        K_PRIO_PREEMPT(1), K_FP_REGS, K_FOREVER);
    k_thread_name_set(tid_a, NET_A_NAME);
    k_thread_cpu_pin(tid_a, ctx_a.cpu);

    k_tid_t tid_b = k_thread_create(
        &thread_b, stack_b, K_THREAD_STACK_SIZEOF(stack_b),
        node_thread_fn, &ctx_b, NULL, NULL,
        K_PRIO_PREEMPT(1), K_FP_REGS, K_FOREVER);
    k_thread_name_set(tid_b, NET_B_NAME);
    k_thread_cpu_pin(tid_b, ctx_b.cpu);

#ifdef MICROROS_3NET
    /* MICROROS_NO_FPREGS_C: drop K_FP_REGS from the mlp_control thread.
     * Test for the Config-B hart-1 idle gap — see if 2 K_FP_REGS threads
     * sharing a hart is what blocks scheduling vs 1. We expect mlp's
     * fp32 ops to V-fault without it, but the gap dynamics on hart 1
     * are visible long before any mlp dispatch fires. */
    k_tid_t tid_c = k_thread_create(
        &thread_c, stack_c, K_THREAD_STACK_SIZEOF(stack_c),
        node_thread_fn, &ctx_c, NULL, NULL,
        K_PRIO_PREEMPT(1),
#ifdef MICROROS_NO_FPREGS_C
        0,
#else
        K_FP_REGS,
#endif
        K_FOREVER);
    k_thread_name_set(tid_c, NET_C_NAME);
    k_thread_cpu_pin(tid_c, ctx_c.cpu);
#endif

    k_thread_start(tid_a);
    k_thread_start(tid_b);
#ifdef MICROROS_3NET
    k_thread_start(tid_c);
#endif

    k_thread_join(tid_a, K_FOREVER);
    k_thread_join(tid_b, K_FOREVER);
#ifdef MICROROS_3NET
    k_thread_join(tid_c, K_FOREVER);
#endif
#endif

    printk("*** microros_demo exit ***\n");
#if !defined(MICROROS_NO_BROKER) && !defined(MICROROS_NO_MICROROS)
    /* Silence the broker before the trace dump — otherwise its printks
     * interleave byte-for-byte with the CSV stream over HTIF and the
     * trace comes out garbled. */
    broker_quiesce();
#endif
#ifndef MICROROS_SKIP_TRACE
    emit_trace_block();
#endif
    printk("=== AGENTS_WALL_CYCLES [" NET_A_NAME "] === %llu\n",
           (unsigned long long)ctx_a.wall_cycles);
    printk("=== AGENTS_WALL_CYCLES [" NET_B_NAME "] === %llu\n",
           (unsigned long long)ctx_b.wall_cycles);
#ifdef MICROROS_3NET
    printk("=== AGENTS_WALL_CYCLES [" NET_C_NAME "] === %llu\n",
           (unsigned long long)ctx_c.wall_cycles);
#endif
    sys_reboot(SYS_REBOOT_COLD);
    return 0;
}
