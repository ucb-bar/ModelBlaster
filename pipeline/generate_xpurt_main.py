"""Emit a multi-network main.c that walks an XPU-RT dispatch table.

Companion to ``ingest_xpurt_schedule`` (which produces the C dispatch
table). Together they close the round trip:

    XPU-RT schedule.json
        -> ingest_xpurt_schedule  -> <name>_dispatch_table.{h,c}
        -> generate_xpurt_main    -> <name>_main.c
        -> west build (harness_xpurt)
        -> spike

The harness can link multiple HW backends per model (e.g. scalar + rvv);
the walker dispatches each entry to whichever backend's per-model table
matches the entry's ``impl``. Symbol disambiguation is handled at
compile time by ``backend_rename.py`` — each model's externally-visible
symbols (kernels, dispatch table, profile API) get suffixed with
``_<bs>`` per backend.

The generated main:

    1. Creates one persistent modelblaster_pool for every distinct composite
       hart set selected by the schedule. Singleton dispatches need no pool.
    2. Resets every involved (model, backend) profile counter.
    3. Spawns one worker per (core_kind, hart) pair the schedule
       actually uses, pinned to that hart. Each worker walks the table
       in start-time order taking only entries that match BOTH its kind
       and its hart. Entries with hart == -1 (unbound) are claimed by
       the kind's designated worker -- the one sitting on the kind's
       lowest-numbered hart -- so nothing is ever dropped.
    4. Inside the dispatch branch, selects the right per-backend table
       by ``impl`` and invokes ``MODEL_<UMID>_DISPATCH_FNS_<BS>``.

       ``impl`` is the entry's KERNEL IMPLEMENTATION, which the ingest
       defaults to ``core_kind``. They differ only when the schedule was
       solved with ``scheduler.enable_impls`` and chose per-dispatch
       implementations -- "this GEMM on the MAC unit, the next on the
       vector unit", on the same core. ``hardware_target`` cannot say
       that: it names WHERE, and ``impl`` names WITH WHAT. Selecting on
       ``impl`` rather than ``core_kind`` is what makes a heterogeneous
       schedule mean something at run time instead of being a claim the
       binary quietly ignores. Dispatch-local state binds the exact composite
       pool for that entry.
    5. Tracks per-network wall cycles inline (no model-side setter
       call — keeps wall-cycle bookkeeping backend-agnostic).
    6. Captures instance 0 only after every dispatch in that DAG has finished,
       compares it with the baked one-invocation golden, and prints the
       standard output and per-backend profile records. Capturing instance 0
       matters for recurrent models whose later periodic outputs carry state.
"""

from __future__ import annotations

import argparse
import json
import glob
import os
import re

# Match the ingest module's heuristic for splitting "<network><instance>".
# Falls back to the trailing-digits regex when no known set is provided.
# When known networks are available, longest-prefix match wins — this
# disambiguates model names ending in digits (e.g. "yolov8_nano_640"
# → ("yolov8_nano_64", 0), not ("yolov8_nano_", 640)).
_INSTANCE_RE = re.compile(r"^(?P<base>.+?)(?P<idx>\d+)$")


# Platform prologue. The walker body is identical on both platforms -- it was
# already written against pthreads with pthread_attr_setaffinity_np -- so the
# only Zephyr coupling is the timer, the semaphores, the abort path, and two
# CONFIG_ constants. Defining the Zephyr names in terms of POSIX ones keeps a
# single body rather than forking ~900 lines of generator.
_PROLOGUE_ZEPHYR = """#include <zephyr/kernel.h>      /* k_cycle_get_64, k_sem */
#include <zephyr/sys/reboot.h>

#define XPURT_TICKS_PER_SEC  CONFIG_SYS_CLOCK_HW_CYCLES_PER_SEC
#define XPURT_BOARD_TARGET   CONFIG_BOARD_TARGET"""

_PROLOGUE_LINUX = """#define MODELBLASTER_PLATFORM_LINUX 1
#include "mb_posix_compat.h"   /* k_cycle_get_64, k_sem, sys_reboot on POSIX */

#define XPURT_TICKS_PER_SEC  MB_POSIX_TICKS_PER_SEC
#define XPURT_BOARD_TARGET   "linux/riscv64"
"""

_PROLOGUES = {"zephyr": _PROLOGUE_ZEPHYR, "linux": _PROLOGUE_LINUX}

# Emitted BEFORE any #include. glibc gates cpu_set_t / CPU_ZERO / CPU_SET and
# pthread_attr_setaffinity_np behind _GNU_SOURCE, and <sched.h> latches the
# decision the first time it is included -- so this cannot live in the
# platform prologue, which lands after the include block. Zephyr gets an
# empty string: its affinity API is Kconfig-gated, not feature-test-gated,
# and defining _GNU_SOURCE there would be a no-op at best.
_TOP_LINUX = "#define _GNU_SOURCE 1  /* cpu_set_t, pthread_attr_setaffinity_np */\n"
_TOPS = {"zephyr": "", "linux": _TOP_LINUX}


def _split_job_name(job: str, known: set[str] | None = None) -> tuple[str, int]:
    if known:
        for base in sorted(known, key=len, reverse=True):
            if job == base:
                return base, 0
            if job.startswith(base):
                rest = job[len(base):]
                if rest.isdigit():
                    return base, int(rest)
    m = _INSTANCE_RE.match(job)
    if not m:
        return job, 0
    return m.group("base"), int(m.group("idx"))


HEADER = "/* @generated by modelblaster/pipeline/generate_xpurt_main.py — do not edit. */"


def _c_ident(name: str) -> str:
    return name.replace(".", "_").replace("-", "_")


def _state_input_inits(mid: str, gen_dir: str | None, net: str) -> str:
    """Designated-initialiser lines for a dispatch state's INPUT members.

    A model's input count depends on quant/backend (fused_full: 1 input at
    fp16, 3 at int8), so the single hardcoded `.input =` cannot be assumed.
    """
    members = _model_input_members(gen_dir, net, mid)
    if members == ["input"]:
        return f"                .input = model_{mid}_test_input,\n"
    return "".join(
        f"                .{m} = model_{mid}_test_{m},\n" for m in members)


def _model_input_members(gen_dir: str | None, net: str, mid: str) -> list[str]:
    """Return the state struct's input member names for `net`.

    Single-input models declare one `input` member; multi-input models
    (e.g. fused_full, which takes camera + ToF + flow/IMU tensors) declare
    `input0`, `input1`, ... . This used to be hardcoded to `.input`, which
    compiled for dronet/yolov8/mlp_control but broke on the first
    multi-input model in a scheduled build:

        error: 'model_fused_full_state_t' has no member named 'input'
        error: 'model_fused_full_test_input' undeclared
                (did you mean 'model_fused_full_test_input2'?)

    The per-model header is located by CONTENT rather than by a passed-in
    path: at the time this generator runs, the `<net>_model.h` copies do
    not exist yet (harness_xpurt/CMakeLists.txt stages those into
    <build>/modelblaster_xpurt/ later), but every model's own
    examples/<exdir>/<quant>/generated/<backend>/model.h is already on
    disk. We pick the one declaring `model_<mid>_state_t`.

    Falls back to the single-input form when nothing is found, preserving
    the previous behaviour for every model that has one input.
    """
    cands: list[str] = []
    if gen_dir:
        # Explicit per-model generated dir (net=path from --model-gen-dir).
        # REQUIRED for correctness, not just speed: arity is a property of
        # the (quant, backend) actually being built. fused_full's fp16
        # build has ONE input while its int8 build has THREE, so guessing
        # from a repo-wide glob can silently pick the wrong variant.
        cands.append(os.path.join(gen_dir, "model.h"))
        cands.append(os.path.join(gen_dir, f"{net}_model.h"))
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cands += sorted(glob.glob(os.path.join(
        repo, "examples", "*", "*", "generated", "*", "model.h")))
    needle = f"model_{mid}_state_t"
    for hdr in cands:
        try:
            text = open(hdr).read()
        except OSError:
            continue
        if needle not in text:
            continue
        members = re.findall(r"\*\s*(input\d*)\s*;", text)
        seen: list[str] = []
        for m in members:
            if m not in seen:
                seen.append(m)
        if not seen:
            continue
        if seen == ["input"]:
            return seen
        return sorted((m for m in seen if m != "input"),
                      key=lambda m: int(m[len("input"):] or 0))
    return ["input"]


def _emit(networks: list[str], schedule_name: str,
          dispatch_table_header: str,
          core_kinds: list[str],
          backends: list[str],
          pool_sizes: list[int],
          n_instances: dict[str, int],
          platform: str = "zephyr",
          gen_dir: str | None = None,
          model_gen_dirs: dict[str, str] | None = None) -> str:
    """Render xpurt_main.c source.

    `networks`     is the ordered list of distinct network names the
                   schedule references.
    `core_kinds`   is the ordered list of distinct core_kind values.
                   NOTE this no longer fixes the thread count: the walker
                   discovers the (kind, hart) worker set at RUN time by
                   scanning the dispatch table, so `core_kinds` only fixes
                   the kind -> backend pairing and which kinds are
                   executable at all.
    `backends`     is the ordered list of HW backends compiled into the
                   binary (== the per-model OBJECT-lib suffixes). Usually
                   identical to `core_kinds`, but separated to allow
                   binaries with backend pools larger than the schedule
                   used (e.g. a future fallback path).
    `pool_sizes`   is the legacy per-kind default used only when a schedule
                   does not declare a composite machine. Composite entries
                   create one persistent pool for their exact ordered hart
                   set. Per-hart locks prevent scheduler workers and pool
                   helpers from oversubscribing the same physical hart.
    `n_instances`  per-network instance count (max instance index + 1).
                   Sized arrays for wall-cycle bookkeeping so periodic
                   instances of the same network don't clobber each
                   other's start/elapsed scalars."""

    inc_lines: list[str] = []
    output_buf_decls: list[str] = []
    forward_decls: list[str] = []
    reset_calls: list[str] = []
    branch_lines: list[str] = []
    print_blocks: list[str] = []
    wall_decls: list[str] = []

    for net in networks:
        mid = _c_ident(net)
        umid = mid.upper()
        n_inst = max(1, n_instances.get(net, 1))
        inc_lines.append(f'#include "{net}_model.h"')
        inc_lines.append(f'#include "{net}_test_io.h"')

        output_buf_decls.append(
            f"static model_{mid}_output_t out_{mid}[MODEL_{umid}_OUTPUT_SIZE];"
        )

        # Per-instance wall-cycle bookkeeping. Periodic networks issue
        # multiple instances (`dronet0`, `dronet1`, ...) — sharing one
        # scalar would mean each instance overwrites the previous.
        wall_decls.append(
            f"static uint64_t wall_start_{mid}[{n_inst}] = {{ 0 }};")
        wall_decls.append(
            f"static uint64_t wall_cycles_{mid}[{n_inst}] = {{ 0 }};")
        wall_decls.append(
            f"static int completed_dispatches_{mid}[{n_inst}] = {{ 0 }};")
        wall_decls.append(
            f"static float verify_abs_{mid} = 0.0f, verify_rel_{mid} = 0.0f;")
        wall_decls.append(f"static int verify_ready_{mid} = 0;")

        # For each (model, backend) pair: forward-declare the per-backend
        # dispatch table + the renamed profile API. The harness CMakeLists
        # builds a separate OBJECT lib per pair so these live in distinct
        # TUs and link without collision.
        for bs in backends:
            BS = bs.upper()
            forward_decls.append(
                f"extern const model_{mid}_dispatch_fn "
                f"MODEL_{umid}_DISPATCH_FNS_{BS}[MODEL_{umid}_OP_COUNT];"
            )
            forward_decls.append(
                f"void model_{mid}_reset_profile_{bs}(void);")
            forward_decls.append(
                f"const model_{mid}_op_record_t *"
                f"model_{mid}_profile_records_{bs}(int *count);")
            reset_calls.append(f"    model_{mid}_reset_profile_{bs}();")

        # Build the per-backend selection inside the dispatch branch.
        # The selection axis is the entry's `impl` -- the kernel
        # implementation for THIS dispatch -- which the ingest defaults to
        # core_kind, so a schedule that never chose per-dispatch impls
        # selects exactly as it always did. We dispatch with the
        # dispatch-local state so `.pool` follows the exact composite target
        # selected for this entry, not merely its kind or implementation.
        #
        # Cross-tile memory coherence: Gemmini's RoCC DMA reads/writes
        # memory through an independent port at the SoC coherence point,
        # NOT through the CPU store buffer or L1 dcache. On
        # GemminiAndOPUShuttleConfig, the CPU on tile 0 (which dispatches
        # to Gemmini) might still have writes sitting in its store buffer
        # when Gemmini's mvin reads, and Gemmini's mvout writes may not
        # be visible to subsequent CPU loads on tile 1 (Saturn). The
        # `fence rw,rw` before+after each dispatch fixes both. Diagnosed
        # in merlin/third_party/iree_bar/.../embedded_elf_loader.c
        # (2026-05-14) and ported here for the multi-backend xpurt
        # dispatch loop. Without these fences, hetero output diverges
        # from golden (linf ~52 on dronet).
        # Dispatch branch: strcmp against the entry's `impl` (which the
        # ingest set from the schedule, defaulting to the registry kind),
        # but invoke through the BACKEND-suffixed dispatch fn symbol (which
        # is how each backend's per-model OBJECT lib is named). The two
        # can legitimately differ (e.g. kind="gemmini" → backend tag
        # "gemmini_q31") as long as they're paired by index in the
        # callsite's --core-kinds / --backends args. Before 2026-06-03
        # this loop strcmped against `bs` (the backend tag), so any
        # mismatch silently dropped every dispatch into the else clause
        # — verify then showed all-zero outputs (static init) and the
        # bug looked like FPGA corruption. Pool selection is independent of
        # this branch and comes from the entry's complete hart set.
        if len(core_kinds) != len(backends):
            raise SystemExit(
                f"_emit: core_kinds and backends must have equal length "
                f"(got {core_kinds} vs {backends})"
            )
        bs_select_lines: list[str] = []
        for i, (kind, bs) in enumerate(zip(core_kinds, backends)):
            BS = bs.upper()
            kw = "if" if i == 0 else "else if"
            bs_select_lines.append(
                f'            {kw} (strcmp(e_->impl, "{kind}") == 0) {{\n'
                f"                __asm__ volatile(\"fence rw,rw\" ::: \"memory\");\n"
                f"                MODEL_{umid}_DISPATCH_FNS_{BS}[e_->dispatch_id](&dispatch_state_{mid});\n"
                f"                __asm__ volatile(\"fence rw,rw\" ::: \"memory\");\n"
                f"            }}"
            )
        bs_select_lines.append(
            "            else {\n"
            f'                printf("xpurt: FATAL entry %d of {net} asks for impl \'%s\', '
            f'which this binary was not built with (has: {"/".join(core_kinds)}). '
            f'Either the schedule chose a per-dispatch implementation the build '
            f'lacks, or the kind→backend pairing is wrong at codegen; see '
            f'pipeline/generate_xpurt_main.py\\n", e_->entry_id, e_->impl);\n'
            f"                sys_reboot(SYS_REBOOT_COLD);\n"
            "            }"
        )
        bs_select = "\n".join(bs_select_lines)

        branch_lines.append(
            f'        if (e_->network[0] == \'{net[0]}\' && '
            f'strcmp(e_->network, "{net}") == 0) {{\n'
            f"            /* Zero-cost IR ops (view, chunk2_c1) get\n"
            f"             * dispatch_id=-1 from ingest — they were\n"
            f"             * filtered out of the dispatch fn table by\n"
            f"             * generate_skeleton.py. Skip the kernel call\n"
            f"             * but still post the completion sem so any\n"
            f"             * dependents downstream unblock. */\n"
            f"            if (e_->dispatch_id < 0) {{\n"
            f"                /* Phase G2d producer-side fanout: give this\n"
            f"                 * entry's sem once per downstream consumer\n"
            f"                 * (data dep + time dep, dedup'd at ingest).\n"
            f"                 * Consumers take once with no re-give. Leaf\n"
            f"                 * entries (n_fanout=0) skip the give entirely. */\n"
            f"                for (int _f = 0; _f < e_->n_fanout; _f++) {{\n"
            f"                    k_sem_give(&completion_sems[i_]);\n"
            f"                }}\n"
            f"                prev_iter_end = (uint64_t)k_cycle_get_64();\n"
            f"                g_hart_acc[my_acc_idx].entries_done++;\n"
            f"                continue;\n"
            f"            }}\n"
            f"            if (e_->dispatch_id == 0) {{\n"
            + "\n".join(
                f"                model_{mid}_reset_profile_{bs}();"
                for bs in backends
            ) + "\n"
            f"                wall_start_{mid}[e_->instance] = (uint64_t)k_cycle_get_64();\n"
            f"            }}\n"
            f"            model_{mid}_state_t dispatch_state_{mid} = {{\n"
            + _state_input_inits(
                mid, (model_gen_dirs or {}).get(net, gen_dir), net) +
            f"                .output = out_{mid},\n"
            f"                .pool = (void *)pool_for_entry(e_),\n"
            f"            }};\n"
            f"            lock_entry_harts(e_);\n"
            f"            XPURT_DISPATCH_GUARD_ENTER();\n"
            f"            uint64_t t_disp0 = (uint64_t)k_cycle_get_64();\n"
            f"#ifdef MODELBLASTER_XPURT_TRACE\n"
            f"            xpurt_trace[i_].start_cycles = t_disp0 - run_t0;\n"
            f"#endif\n"
            f"#ifndef MODELBLASTER_PLATFORM_LINUX\n"
            f"            /* Re-arm mstatus.VS right before the kernel\n"
            f"             * dispatch. The worker's once-at-entry VS=Initial\n"
            f"             * write gets clobbered by Zephyr's V state\n"
            f"             * save/restore on context switches triggered by\n"
            f"             * k_sem_take in dep_wait — so by the time we\n"
            f"             * reach the kernel, VS may be Off again. csrs\n"
            f"             * to MSTATUS.VS is a no-op on harts whose VS is\n"
            f"             * hardwired-zero (no misa.V), so this is safe to\n"
            f"             * issue unconditionally.\n"
            f"             *\n"
            f"             * MACHINE MODE ONLY -- csrs mstatus traps in U-mode.\n"
            f"             * A hosted OS saves and restores vector state across\n"
            f"             * context switches itself, so there is nothing to\n"
            f"             * re-arm; this whole hazard is Zephyr-specific. */\n"
            f"            asm volatile(\"csrs mstatus, %0\" : : \"r\"((unsigned long)(1UL << 9)));\n"
            f"#endif\n"
            f"{bs_select}\n"
            f"            uint64_t t_disp1 = (uint64_t)k_cycle_get_64();\n"
            f"            XPURT_DISPATCH_GUARD_EXIT();\n"
            f"            unlock_entry_harts(e_);\n"
            f"            g_hart_acc[my_acc_idx].kernel += (t_disp1 - t_disp0);\n"
            f"#ifdef MODELBLASTER_XPURT_TRACE\n"
            f"            xpurt_trace[i_].end_cycles = t_disp1 - run_t0;\n"
            f"            xpurt_trace[i_].worker_kind_idx = my_kind_idx;\n"
            f"            xpurt_trace[i_].worker_hart = my_hart;\n"
            f"#endif\n"
            f"#ifdef MODELBLASTER_XPURT_STREAM\n"
            f"            /* One JSON line per dispatch END, for the host-side\n"
            f"             * feedback tailer. Same numbers the trace block\n"
            f"             * dumps at exit -- the difference is WHEN: the trace\n"
            f"             * is only readable once the run is over, so nothing\n"
            f"             * can respond to a deadline miss while the run is\n"
            f"             * still going. Cycles are rdtime TICKS at 24 MHz on\n"
            f"             * this board, not core cycles.\n"
            f"             *\n"
            f"             * ONE write() PER LINE, not printf. Several workers\n"
            f"             * reach this concurrently, and interleaved printf\n"
            f"             * calls tear a line into fragments that are not\n"
            f"             * JSON. Formatting into a stack buffer and issuing a\n"
            f"             * single write of well under PIPE_BUF keeps each\n"
            f"             * line whole. */\n"
            f"            {{\n"
            f"                char sbuf_[320];\n"
            f"                int sn_ = snprintf(sbuf_, sizeof sbuf_,\n"
            f"                    \"{{\\\"entry_id\\\":%d,\\\"network\\\":\\\"%s\\\",\\\"instance\\\":%d,\"\n"
            f"                    \"\\\"dispatch_id\\\":%d,\\\"impl\\\":\\\"%s\\\",\\\"hart\\\":%d,\"\n"
            f"                    \"\\\"predicted_start_ms\\\":%.6f,\\\"predicted_duration_ms\\\":%.6f,\"\n"
            f"                    \"\\\"start_ticks\\\":%llu,\\\"end_ticks\\\":%llu}}\\n\",\n"
            f"                    e_->entry_id, e_->network, e_->instance,\n"
            f"                    e_->dispatch_id, e_->impl, my_hart,\n"
            f"                    (double)e_->start_time_ms, (double)e_->duration_ms,\n"
            f"                    (unsigned long long)(t_disp0 - run_t0),\n"
            f"                    (unsigned long long)(t_disp1 - run_t0));\n"
            f"                if (sn_ > 0) {{ ssize_t w_ = write(1, sbuf_, (size_t)sn_); (void)w_; }}\n"
            f"            }}\n"
            f"#endif\n"
            f"            /* A DAG may have several output leaves. Numeric\n"
            f"             * verification and wall completion belong to the\n"
            f"             * LAST completed dispatch, not the numerically last\n"
            f"             * dispatch id (DroNet's two heads finish in either\n"
            f"             * order). */\n"
            f"            int _model_done = __atomic_add_fetch(\n"
            f"                &completed_dispatches_{mid}[e_->instance], 1,\n"
            f"                __ATOMIC_ACQ_REL);\n"
            f"            if (_model_done == MODEL_{umid}_OP_COUNT) {{\n"
            f"                wall_cycles_{mid}[e_->instance] =\n"
            f"                    (uint64_t)k_cycle_get_64() - wall_start_{mid}[e_->instance];\n"
            f"                /* Capture instance 0 before a stateful model's\n"
            f"                 * later invocations advance its state. The baked\n"
            f"                 * golden is a one-invocation golden. */\n"
            f"                if (e_->instance == 0) {{\n"
            f"                    __asm__ volatile(\"fence rw, rw\" ::: \"memory\");\n"
            f"                    float _mae = 0.0f, _mre = 0.0f;\n"
            f"                    for (int _v = 0; _v < MODEL_{umid}_TEST_OUTPUT_LEN; _v++) {{\n"
            f"                        float a = (float)out_{mid}[_v];\n"
            f"                        float g = (float)model_{mid}_test_golden[_v];\n"
            f"                        float ae = a > g ? a - g : g - a;\n"
            f"                        float ag = g > 0.0f ? g : -g;\n"
            f"                        float re = ae / (ag > 1e-12f ? ag : 1e-12f);\n"
            f"                        if (ae > _mae) _mae = ae;\n"
            f"                        if (re > _mre) _mre = re;\n"
            f"                    }}\n"
            f"                    verify_abs_{mid} = _mae;\n"
            f"                    verify_rel_{mid} = _mre;\n"
            f"                    verify_ready_{mid} = 1;\n"
            f"                }}\n"
            f"            }}\n"
            f"            /* Phase G2d: producer-side fanout. Give this\n"
            f"             * entry's sem once per downstream consumer; the\n"
            f"             * consumer-side take loop does NOT re-give (saves\n"
            f"             * one sem op per dep edge). Leaf entries skip the\n"
            f"             * give entirely. */\n"
            f"            for (int _f = 0; _f < e_->n_fanout; _f++) {{\n"
            f"                k_sem_give(&completion_sems[i_]);\n"
            f"            }}\n"
            f"            uint64_t t_iter_end = (uint64_t)k_cycle_get_64();\n"
            f"            g_hart_acc[my_acc_idx].sync_overhead += (t_iter_end - t_disp1);\n"
            f"            prev_iter_end = t_iter_end;\n"
            f"            g_hart_acc[my_acc_idx].entries_done++;\n"
            f"            continue;\n"
            f"        }}"
        )

        # Per-backend profile record dump. Each backend's TU has its own
        # records array; entries that ran on backend X land in X's array.
        # We tag rows with the backend so the host can split per-op cost
        # by backend.
        per_bs_profile_blocks: list[str] = []
        for bs in backends:
            per_bs_profile_blocks.append(f"""\
        {{
            int n_records = 0;
            const model_{mid}_op_record_t *records =
                model_{mid}_profile_records_{bs}(&n_records);
            /* Clamp to the model's own op count. Unbounded, this loop walked
             * off the end of the record array on long ViNT runs and faulted
             * in printf's strnlen() on a garbage name pointer -- AFTER the
             * run, its full XPURT_TRACE and every model's OUTPUT block had
             * been emitted, so it destroyed a completed run's exit status
             * for a dump that XPURT_TRACE already duplicates. */
            if (records == NULL || n_records < 0
                    || n_records > MODEL_{umid}_OP_COUNT) {{
                printf("{bs},PROFILE_RECORDS_INVALID,n=%d,cap=%d\\n",
                       n_records, (int)MODEL_{umid}_OP_COUNT);
                n_records = 0;
            }}
            for (int i = 0; i < n_records; i++) {{
                printf("{bs},%d,%s,%s,%s,%lu\\n",
                       records[i].dispatch_id,
                       records[i].name, records[i].op, records[i].shape,
                       records[i].cycles);
            }}
        }}""")
        per_bs_profile = "\n".join(per_bs_profile_blocks)

        # Per-instance wall-cycles dump under a NEW marker so streamed
        # runners (firesim_runner, spike_runner) that count
        # `=== MODELBLASTER_WALL_CYCLES [<net>] === <int>` for end-of-run
        # detection still see exactly one bare WALL_CYCLES per network
        # below. The per-instance variant uses
        # `=== MODELBLASTER_WALL_CYCLES_INST [<net>#<i>] ===` and is captured
        # for analysis by the host parser.
        # Output dump strategy: HTIF UART throughput is ~6 KB/sec on
        # FireSim Saturn. Dumping every output value as ASCII works for
        # small models (dronet: 2 values, mlp_control: 4) but is the
        # workflow bottleneck for yolov8-scale outputs (75600 values
        # blow well past the FIRESIM_TIMEOUT). For large outputs we
        # emit a *sampled* dump — first 8, last 8, plus a checksum/L1
        # summary for spot-check parity — and rely on per-model unit
        # tests (or a separate dedicated verify run) for full
        # bit-exactness. The marker tags stay the same so existing
        # parsers continue to find OUTPUT_BEGIN/END.
        print_blocks.append(f"""\
    /* Instance 0 is comparable to the baked one-invocation golden even when
     * this model is stateful and later invocations intentionally evolve it. */
    printf("=== MODELBLASTER_VERIFY [{net}] === "
           "max_abs_err=%.9g max_rel_err=%.9g n=%d instance=0 ready=%d\\n",
           (double)verify_abs_{mid}, (double)verify_rel_{mid},
           MODEL_{umid}_TEST_OUTPUT_LEN, verify_ready_{mid});
    printf("=== MODELBLASTER_OUTPUT_BEGIN [{net}] ===\\n");
    {{
        const int _osz = MODEL_{umid}_OUTPUT_SIZE;
        const int _SAMPLE_THRESHOLD = 64;
        if (_osz <= _SAMPLE_THRESHOLD) {{
            for (int i = 0; i < _osz; i++)
                printf("%.9g\\n", (double)out_{mid}[i]);
        }} else {{
            /* Sampled dump + summary. */
            double _sum = 0.0, _abs_sum = 0.0;
            double _omax = -1.0/0.0, _omin = 1.0/0.0;
            for (int i = 0; i < _osz; i++) {{
                double v = (double)out_{mid}[i];
                _sum += v;
                _abs_sum += (v < 0) ? -v : v;
                if (v > _omax) _omax = v;
                if (v < _omin) _omin = v;
            }}
            printf("OUTPUT_SAMPLED size=%d head=8 tail=8\\n", _osz);
            for (int i = 0; i < 8; i++)
                printf("  [%d] %.9g\\n", i, (double)out_{mid}[i]);
            for (int i = _osz - 8; i < _osz; i++)
                printf("  [%d] %.9g\\n", i, (double)out_{mid}[i]);
            printf("OUTPUT_SUMMARY sum=%.6g abs_sum=%.6g min=%.6g max=%.6g\\n",
                   _sum, _abs_sum, _omin, _omax);
        }}
    }}
    printf("=== MODELBLASTER_OUTPUT_END [{net}] ===\\n");
    printf("=== MODELBLASTER_PROFILE_BEGIN [{net}] ===\\n");
    printf("backend,dispatch_id,name,op,shape,cycles\\n");
{per_bs_profile}
    printf("=== MODELBLASTER_PROFILE_END [{net}] ===\\n");
    {{
        unsigned long _max_wc = 0;
        for (int _i = 0; _i < {n_inst}; _i++) {{
            unsigned long _wc = (unsigned long)wall_cycles_{mid}[_i];
            printf("=== MODELBLASTER_WALL_CYCLES_INST [{net}#%d] === %lu\\n",
                   _i, _wc);
            if (_wc > _max_wc) _max_wc = _wc;
        }}
        /* Worst-case per-instance wall cycles — meaningful for periodic
         * tasks (highest observed latency) and identical to the single
         * value for non-periodic networks. */
        printf("=== MODELBLASTER_WALL_CYCLES [{net}] === %lu\\n", _max_wc);
    }}""")

    upper = schedule_name.replace(".", "_").replace("-", "_").upper()

    # Build the per-(network, instance, dispatch_id) dispatch branch.
    dispatch_branch = "\n".join(branch_lines)

    # One Zephyr worker thread per distinct core_kind. Each pins itself
    # to a hart (the entry's .hart) and pulls only entries whose
    # core_kind matches.
    n_kinds = len(core_kinds)
    kind_strs = ", ".join(f'"{k}"' for k in core_kinds)
    if len(pool_sizes) != n_kinds:
        raise SystemExit(
            f"pool_sizes length ({len(pool_sizes)}) must match core_kinds "
            f"length ({n_kinds})")
    pool_size_strs = ", ".join(str(s) for s in pool_sizes)

    # Forward decls go right after the model headers — they reference
    # types from those headers (model_<mid>_dispatch_fn etc.).
    forward_decl_block = "\n".join(forward_decls)
    try:
        platform_prologue = _PROLOGUES[platform]
        platform_top = _TOPS[platform]
    except KeyError:
        raise SystemExit(f"--platform must be one of {sorted(_PROLOGUES)}, "
                         f"got {platform!r}")

    return f"""{HEADER}
{platform_top}/*
 * Schedule-driven multi-network entry point with per-(core_kind, hart)
 * worker threads and HETEROGENEOUS per-backend dispatch tables. One
 * worker is spawned for each (core_kind, hart) pair the dispatch table
 * uses, pinned to that hart, so the schedule's per-core placement is what
 * actually executes. Each worker:
 *   1. Walks the XPU-RT-emitted dispatch table in start_time order,
 *      taking entries whose .core_kind AND .hart match its own.
 *   2. Before invoking, waits on k_sems posted by the producers of its
 *      .deps[] (intra-job data deps) and .time_dep_entry_id (cross-job
 *      ordering edges).
 *   3. Selects the per-(model, backend) dispatch table by .impl -- the
 *      IMPLEMENTATION the schedule chose for this dispatch, which
 *      defaults to .core_kind -- and invokes the matching kernel.
 *      Selecting on .impl rather than .core_kind is what lets one core
 *      run a MAC-unit GEMM and then a vector one. backend_rename.py at compile
 *      time has suffixed every model's externally-visible symbol with
 *      _<bs>, so multiple backends per model link cleanly.
 *   4. After invoking, gives the entry's k_sem so consumers unblock.
 *
 * Deadlock freedom is unchanged by the extra threads. ingest asserts that
 * every dep and time_dep edge points STRICTLY BACKWARD in entry_id order,
 * and every worker walks 0..N-1 monotonically. Take the blocked worker
 * with the smallest current index i: it waits on some j < i, whose owner
 * has current index <= j < i, so that owner is either running or is a
 * blocked worker with a smaller index -- contradicting minimality. The
 * argument never mentions how entries are partitioned across threads, so
 * it holds for one worker per (kind, hart) exactly as it did for one
 * worker per kind.
 *
 * Workers are pinned to harts via pthread_attr_setaffinity_np. Composite
 * entries reserve every hart in their machine combination under ordered
 * per-hart locks and select a persistent modelblaster_pool pinned to that
 * exact set. Scheduler workers on helper harts sleep on the locks while the
 * shard is active, so inter-op and intra-op parallelism never oversubscribe.
 *
 * Outputs follow the same multi-line marker protocol as the
 * straight-line multi_main, so spike_runner verifies them against
 * PyTorch goldens unchanged. The PROFILE block now has a leading
 * `backend` column so per-op cycles can be split per backend.
 */

#define MODELBLASTER_DISABLE_UNMANGLED

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include "modelblaster_pool.h"
{platform_prologue}

{chr(10).join(inc_lines)}
#include "{dispatch_table_header}"

/* Note (2026-05-28): we previously tried several memory-fence and
 * cache-invalidation strategies to fix `dronet_hetero` verify-fail
 * (max_abs_err=52 vs golden). All failed identically:
 *
 *   - fence rw,rw before+after each dispatch:    output unchanged
 *   - L1 dcache eviction via 128KB scratch read: output unchanged
 *   - Zicbom cbo.inval:                          illegal-instr trap
 *     (mcause=2, mtval=0x7a00f) -- bitstream
 *     was not built with Zicbom.
 *
 * Persistent identical failure across these very different remedies
 * indicates the bug is NOT cross-tile cache staleness. The real
 * cause is cross-backend per-op numerical drift: gemmini and
 * rvv_opu each match PyTorch golden bit-exactly END-TO-END, but
 * their intermediate per-op outputs differ slightly due to
 * backend-specific rounding/quantization details (each backend's
 * full chain happens to cancel its own drift into a golden-matching
 * final answer; mixing chains doesn't preserve that property).
 *
 * Documented in notes/known_issues.md. Fix path is cross-backend
 * numerical alignment, not memory mgmt. fence rw,rw is kept on the
 * dispatch boundaries below as good hygiene (it WAS load-bearing
 * for Gemmini DMA in merlin's loader; harmless and ~zero cost). */

/* Per-(model, backend) externs — the harness compiles a separate
 * OBJECT lib per pair, with externally-visible symbols suffixed _<bs>
 * via -D renames (see modelblaster/pipeline/backend_rename.py). */
{forward_decl_block}

{chr(10).join(output_buf_decls)}

/* Per-entry completion sem: posted once when the dispatch finishes;
 * dependents take it before invoking. Initialized as 0/1 (binary, but
 * grow the limit so multiple consumers can each take their own
 * "completed" reading without blocking each other). */

/* XPURT_DISPATCH_GUARD: mask this hart's interrupts around ONE kernel call.
 *
 * On the Saturn RVV configs here, a trap taken while a vector kernel runs can
 * return with exactly one scalar register corrupted -- deterministic,
 * bit-reproducible, invisible on spike (see harness/src/main.c, which masks
 * for the WHOLE inference via MODELBLASTER_MASK_IRQ_DURING_RUN).
 *
 * Symptom here (ViNT, scheduled harness):
 *     mcause: 5, Load access fault
 *     mepc -> z_riscv_vstate_restore_thread  (arch/riscv/core/v.c)
 *     ra   -> kernel_sigmoid_s8_vint_gemmini_q31
 * a corrupted vector-context pointer on the restore path. Probabilistic in
 * run length: the 605-dispatch vint-only schedule and the 1546-dispatch
 * seed2 passed; the ~2000-dispatch / ~5.6 s points faulted.
 *
 * harness/src/main.c warns "a thread pool or several models concurrently
 * must NOT do this" -- that is about holding the lock across BLOCKING work.
 * Here it is held only across one kernel call: every k_sem wait, scheduler
 * decision and bookkeeping happens outside it. The two workers are pinned
 * one-per-hart and the intra-op pool has 0 helpers, so a hart never has a
 * second runnable worker to starve, and irq_lock() masks only this hart.
 *
 * XPURT_DISPATCH_IRQ_GUARD=0 disables it (reproduces the fault). */
#if !defined(XPURT_DISPATCH_IRQ_GUARD)
#define XPURT_DISPATCH_IRQ_GUARD 1
#endif
#if XPURT_DISPATCH_IRQ_GUARD
#define XPURT_DISPATCH_GUARD_ENTER() unsigned int _xpurt_irq_key = irq_lock()
#define XPURT_DISPATCH_GUARD_EXIT()  irq_unlock(_xpurt_irq_key)
#else
#define XPURT_DISPATCH_GUARD_ENTER() do {{ }} while (0)
#define XPURT_DISPATCH_GUARD_EXIT()  do {{ }} while (0)
#endif

static struct k_sem completion_sems[{upper}_N_ENTRIES];

/* Phase G1 — per-hart runtime attribution counters, accumulated in
 * mtime ticks (1 us each on chipyard FireSim). Read at end-of-run and
 * dumped under MODELBLASTER_HART_ACC for host-side decomposition of
 * the predicted-vs-measured wall gap. Indexed by my_kind_idx. */
#ifndef XPURT_MAX_KINDS
#define XPURT_MAX_KINDS 8
#endif
/* Upper bound on scheduler-worker threads == distinct (core_kind, hart)
 * pairs the dispatch table uses. 32 covers every board we target (the K1
 * has 8 harts); main() aborts loudly rather than truncating if a schedule
 * ever exceeds it. */
#ifndef XPURT_MAX_WORKERS
#define XPURT_MAX_WORKERS 32
#endif
struct xpurt_hart_acc {{
    uint64_t kernel;            /* mtime delta around bs_select dispatch */
    uint64_t dep_wait;          /* mtime inside k_sem_take loop */
    uint64_t sync_overhead;     /* mtime between kernel-end and end of iter */
    uint64_t target_gate_spin;  /* mtime inside schedule-start gate */
    uint64_t hart_idle;         /* mtime between iters (gap to next match) */
    uint64_t gemmini_cfg_emit;  /* reserved for G2b caching wrappers */
    uint64_t entries_done;      /* count of dispatched entries */
    uint64_t wall_total;        /* worker_t0 → worker_exit delta */
}};
/* Indexed by WORKER index, not kind index — since 2026-08-27 there is one
 * worker per (core_kind, hart), so a per-kind array would have several
 * threads racing on the same counters and would report a kind's busy time
 * as if one hart had done it all. */
static struct xpurt_hart_acc g_hart_acc[XPURT_MAX_WORKERS];

/* Per-instance wall-clock state. Each periodic instance writes its own
 * slot indexed by entry->instance, so back-to-back instances of one
 * network don't clobber each other's start/elapsed scalars. */
{chr(10).join(wall_decls)}

/* mtime baseline for the run, captured immediately before workers spawn.
 * Used by the worker to (a) honor every entry's absolute schedule-issued
 * start time and (b) tag actual_start/end_cycles in the trace. */
static uint64_t run_t0;
#define XPURT_CYCLES_PER_MS \\
    ((uint64_t)(XPURT_TICKS_PER_SEC / 1000))

#ifdef MODELBLASTER_XPURT_TRACE
/* Per-entry execution trace. Each slot is still touched by exactly one
 * worker, and that is now enforced rather than asserted: the claim
 * predicate in xpurt_worker() partitions the table by (core_kind, hart),
 * a partition with no overlap by construction, and main() audits at
 * startup that every entry is claimed by EXACTLY ONE worker before any
 * thread is spawned (see the claim-coverage audit). `worker_hart` records
 * which hart actually executed the slot, so an entry that ran somewhere
 * other than where the schedule placed it (column `hart`) is visible in
 * the CSV instead of being invisible behind a shared kind index.
 * Pre-initialized to -1/-1 so an entry NO worker ran is distinguishable
 * from one that ran on kind 0 / hart 0 at cycle 0.
 * Cycles are mtime-based (k_cycle_get_64) and offset against `run_t0`,
 * captured immediately before the first worker starts. The host parser
 * reconstructs (start_ms, end_ms) by dividing by the configured
 * clock_mhz. */
struct xpurt_trace_slot {{
    uint64_t start_cycles;
    uint64_t end_cycles;
    int      worker_kind_idx;   /* index into kinds[] */
    int      worker_hart;       /* hart the executing worker was pinned to */
}};
static struct xpurt_trace_slot xpurt_trace[{upper}_N_ENTRIES];
#endif

struct xpurt_worker_arg {{
    const char *kind;
    int hart;             /* hart this worker is pinned to; -1 = unpinned */
    int kind_idx;         /* index into kinds[] / pools[] */
    int worker_idx;       /* index into g_hart_acc[] / tids[] */
    int claims_unbound;   /* 1 => also claims this kind's hart<0 entries */
    int observed_cpu;     /* sched_getcpu() at worker entry; -1 if unknown */
}};

/* The worker set: one entry per (core_kind, hart) pair the dispatch table
 * actually uses. Built in main() by scanning the table, BEFORE the pools
 * are created (the pool sizing depends on how many workers a kind got). */
static struct xpurt_worker_arg g_workers[XPURT_MAX_WORKERS];
static int g_n_workers;

/* A schedule may reserve one hart or an aligned 2/4/...-hart block for one
 * dispatch. Keep one persistent pool per distinct composite block and one
 * mutex per physical hart. The dispatch-owning worker is harts[0]; helpers
 * are pinned to the remaining entries by modelblaster_pool_create_on_harts.
 * Every master acquires hart locks in numeric order, so overlapping blocks
 * serialize without a lock-order cycle even when actual durations drift. */
#ifndef XPURT_MAX_HARTS
#define XPURT_MAX_HARTS 256
#endif
#ifndef XPURT_MAX_POOL_WIDTH
#define XPURT_MAX_POOL_WIDTH 16
#endif
#ifndef XPURT_MAX_POOLS
#define XPURT_MAX_POOLS 32
#endif
struct xpurt_pool_desc {{
    int n_harts;
    int harts[XPURT_MAX_POOL_WIDTH];
    modelblaster_pool_t pool;
}};
static pthread_mutex_t g_hart_locks[XPURT_MAX_HARTS];
static struct xpurt_pool_desc g_entry_pools[XPURT_MAX_POOLS];
static int g_n_entry_pools;

static int same_harts(const struct xpurt_pool_desc *p,
                      const xpurt_sched_entry_t *e)
{{
    if (p->n_harts != e->n_harts) return 0;
    for (int i = 0; i < e->n_harts; i++) {{
        if (p->harts[i] != e->harts[i]) return 0;
    }}
    return 1;
}}

static modelblaster_pool_t pool_for_entry(const xpurt_sched_entry_t *e)
{{
    if (e->n_harts <= 1) return NULL;
    for (int i = 0; i < g_n_entry_pools; i++) {{
        if (same_harts(&g_entry_pools[i], e)) return g_entry_pools[i].pool;
    }}
    return NULL;  /* init_entry_runtime audits this before workers spawn */
}}

static int init_entry_runtime(void)
{{
    for (int h = 0; h < XPURT_MAX_HARTS; h++) {{
        if (pthread_mutex_init(&g_hart_locks[h], NULL) != 0) return -1;
    }}
    g_n_entry_pools = 0;
    for (int i = 0; i < {upper}_N_ENTRIES; i++) {{
        const xpurt_sched_entry_t *e = &{upper}_TABLE[i];
        if (e->n_harts < 1 || e->n_harts > XPURT_MAX_POOL_WIDTH) return -2;
        for (int j = 0; j < e->n_harts; j++) {{
            if (e->harts[j] < 0 || e->harts[j] >= XPURT_MAX_HARTS) return -3;
        }}
        if (e->n_harts == 1 || pool_for_entry(e) != NULL) continue;
        if (g_n_entry_pools >= XPURT_MAX_POOLS) return -4;
        struct xpurt_pool_desc *p = &g_entry_pools[g_n_entry_pools];
        p->n_harts = e->n_harts;
        for (int j = 0; j < e->n_harts; j++) p->harts[j] = e->harts[j];
        p->pool = modelblaster_pool_create_on_harts(p->n_harts, p->harts);
        if (p->pool == NULL) return -5;
        printf("modelblaster_pool[block=%d]: width=%d harts=",\
               g_n_entry_pools, p->n_harts);
        for (int j = 0; j < p->n_harts; j++)
            printf("%s%d", j ? "+" : "", p->harts[j]);
        printf("\\n");
        g_n_entry_pools++;
    }}
    return 0;
}}

static void lock_entry_harts(const xpurt_sched_entry_t *e)
{{
    int ordered[XPURT_MAX_POOL_WIDTH];
    for (int i = 0; i < e->n_harts; i++) ordered[i] = e->harts[i];
    for (int i = 1; i < e->n_harts; i++) {{
        int v = ordered[i], j = i - 1;
        while (j >= 0 && ordered[j] > v) {{ ordered[j + 1] = ordered[j]; j--; }}
        ordered[j + 1] = v;
    }}
    for (int i = 0; i < e->n_harts; i++)
        pthread_mutex_lock(&g_hart_locks[ordered[i]]);
}}

static void unlock_entry_harts(const xpurt_sched_entry_t *e)
{{
    int ordered[XPURT_MAX_POOL_WIDTH];
    for (int i = 0; i < e->n_harts; i++) ordered[i] = e->harts[i];
    for (int i = 1; i < e->n_harts; i++) {{
        int v = ordered[i], j = i - 1;
        while (j >= 0 && ordered[j] > v) {{ ordered[j + 1] = ordered[j]; j--; }}
        ordered[j + 1] = v;
    }}
    for (int i = e->n_harts - 1; i >= 0; i--)
        pthread_mutex_unlock(&g_hart_locks[ordered[i]]);
}}

static void *xpurt_worker(void *arg)
{{
    struct xpurt_worker_arg *wa = (struct xpurt_worker_arg *)arg;
    const char *my_kind = wa->kind;
    int my_kind_idx = wa->kind_idx;
    int my_hart = wa->hart;
    int my_acc_idx = wa->worker_idx;
    int my_claims_unbound = wa->claims_unbound;
#ifdef MODELBLASTER_PLATFORM_LINUX
    /* Record where the scheduler ACTUALLY put us. pthread_attr_setaffinity_np
     * failing (or being compiled out) is silent otherwise, and an unpinned
     * worker makes every per-hart number in this trace a fiction. main()
     * prints observed_cpu next to the requested hart after the join. */
    wa->observed_cpu = sched_getcpu();
#else
    wa->observed_cpu = -1;
#endif
    /* Phase G1 attribution: track per-iter timing so we can decompose
     * wall_total into kernel + dep_wait + sync_overhead +
     * target_gate_spin + hart_idle. prev_iter_end is updated INSIDE
     * each per-net branch just before `continue` so hart_idle on the
     * next iter is correctly (this_iter_start - last_iter_end). */

    /* Per-hart V-extension enable. Zephyr's HAS_V() macro reads
     * misa from the PRIMARY hart only — on our 2-tile
     * FireSimGemminiAndOPUShuttleConfig that's hart 0 (Rocket +
     * Gemmini, misa.V=0) so Zephyr leaves mstatus.VS = Off
     * EVERYWHERE, including on hart 1 (Shuttle + Saturn-OPU) which
     * does have V (misa.V=1, verified by examples/opu_probe). If we
     * don't flip VS to Initial here, the first vsetvli in any RVV
     * kernel dispatched to this hart traps with illegal-instruction.
     *
     * Behavior: read THIS hart's misa, and if the V bit is set,
     * raise mstatus.VS to Initial (writes to mstatus.VS are silently
     * ignored on harts whose misa.V=0, so this is a no-op on
     * non-V harts and safe to run unconditionally per worker). */
#ifndef MODELBLASTER_PLATFORM_LINUX
    /* MACHINE MODE ONLY. Both `csrr misa` and `csrs mstatus` are privileged;
     * under a hosted OS the process runs in U-mode and the first of them traps
     * with SIGILL before a single dispatch executes. On Linux there is also
     * nothing to do: the kernel owns per-thread vector state and enables it
     * lazily on first use, which is exactly what this block hand-rolls for
     * Zephyr. Measured on the SpaceMiT K1: without this guard the harness dies
     * immediately, `dmesg` showing cause=2 badaddr=0x301027f3, which decodes to
     * `csrr a5, misa`. */
    {{
        unsigned long _misa;
        asm volatile("csrr %0, misa" : "=r"(_misa));
        if (_misa & (1UL << 21)) {{  /* 'V' - 'A' = 21 */
            unsigned long _vs_init = (1UL << 9);  /* MSTATUS_VS = Initial */
            asm volatile("csrs mstatus, %0" : : "r"(_vs_init));
        }}
    }}
#endif

    uint64_t worker_t0 = (uint64_t)k_cycle_get_64();
    uint64_t prev_iter_end = worker_t0;
    /* Take entries that match `my_kind` AND `my_hart`, in start_time
     * order.
     *
     * Until 2026-08-27 the predicate was kind-only, so ONE thread ran
     * every entry of a kind serially no matter how many distinct cores
     * the scheduler had placed them on. Measured on the K1 with a
     * 3-model schedule: zero overlapping dispatch pairs in 1617 entries,
     * the single `rvv` worker 97.9% busy across a 1017.8 ms run, and 119
     * of 123 instances missing their deadline even though per-dispatch
     * durations matched the cost model (median actual/predicted 0.93).
     * The schedule's core assignment was simply not honored at run time.
     *
     * UNBOUND ENTRIES: `hart == -1` means ingest could not resolve the
     * schedule's machine slot to a physical hart (a registry core with an
     * empty `harts` list). Those entries are NOT dropped -- exactly one
     * worker per kind carries `claims_unbound`, the one on the kind's
     * lowest-numbered hart, and it runs them interleaved with its own.
     * If a kind has ONLY unbound entries it gets a single unpinned worker
     * that carries the flag. Either way the (kind, hart) predicate stays
     * a partition of the table: every entry is claimed by exactly one
     * worker, which main() audits before spawning anything. */
    for (int i_ = 0; i_ < {upper}_N_ENTRIES; i_++) {{
        const xpurt_sched_entry_t *e_ = &{upper}_TABLE[i_];
        if (strcmp(e_->core_kind, my_kind) != 0) continue;
        if (e_->hart >= 0) {{
            if (e_->hart != my_hart) continue;
        }} else if (!my_claims_unbound) {{
            continue;
        }}

        uint64_t t_iter_start = (uint64_t)k_cycle_get_64();
        if (g_hart_acc[my_acc_idx].entries_done > 0) {{
            g_hart_acc[my_acc_idx].hart_idle +=
                (t_iter_start - prev_iter_end);
        }}

        /* Wait for all data deps + the time_dep edge to complete.
         * Phase G2d: producer gives its sem once per UNIQUE consumer
         * (dedup'd at ingest), consumer takes once per UNIQUE dep
         * here. No re-give — the counts balance. The time_dep edge
         * dedup against deps[] preserves the invariant when one
         * producer is both a data and a time dep of the same
         * consumer (without dedup, consumer takes twice from one
         * producer give → deadlock). */
        for (int d = 0; d < e_->n_deps; d++) {{
            k_sem_take(&completion_sems[e_->deps[d]], K_FOREVER);
        }}
        if (e_->time_dep_entry_id >= 0) {{
            int _td_is_dup = 0;
            for (int _d = 0; _d < e_->n_deps; _d++) {{
                if (e_->deps[_d] == e_->time_dep_entry_id) {{
                    _td_is_dup = 1;
                    break;
                }}
            }}
            if (!_td_is_dup) {{
                k_sem_take(&completion_sems[e_->time_dep_entry_id], K_FOREVER);
            }}
        }}
        uint64_t t_deps_done = (uint64_t)k_cycle_get_64();
        g_hart_acc[my_acc_idx].dep_wait += (t_deps_done - t_iter_start);

        /* Every dispatch has an absolute schedule-issued earliest start.
         * Gating only dispatch_id==0 is insufficient for a DAG with multiple
         * roots: another root can otherwise run before its periodic release,
         * then post dependencies early. Dependencies and this gate are both
         * lower bounds; actual resource drift may still start an entry later. */
        uint64_t target_start = run_t0 +
            (uint64_t)((double)e_->start_time_ms *
                       (double)XPURT_CYCLES_PER_MS);
        uint64_t t_gate0 = (uint64_t)k_cycle_get_64();
        while ((uint64_t)k_cycle_get_64() < target_start) {{
            k_yield();
        }}
        g_hart_acc[my_acc_idx].target_gate_spin +=
            ((uint64_t)k_cycle_get_64() - t_gate0);

{dispatch_branch}
        printf("xpurt: WARN unknown network %s in entry %d\\n",
               e_->network, e_->entry_id);
        /* Unknown network — give the sem anyway so we don't deadlock.
         * Fan out per the producer-side invariant so any downstream
         * consumers actually unblock. */
        for (int _f = 0; _f < e_->n_fanout; _f++) {{
            k_sem_give(&completion_sems[i_]);
        }}
        prev_iter_end = (uint64_t)k_cycle_get_64();
        g_hart_acc[my_acc_idx].entries_done++;
    }}
    g_hart_acc[my_acc_idx].wall_total =
        (uint64_t)k_cycle_get_64() - worker_t0;
    return NULL;
}}

int main(void)
{{
    printf("xpurt-runner: schedule={schedule_name} entries=%d kinds=%d on %s\\n",
           {upper}_N_ENTRIES, {n_kinds}, XPURT_BOARD_TARGET);
#ifdef MODELBLASTER_PLATFORM_LINUX
    struct sched_param observed_sched_param = {{ 0 }};
    int observed_sched_policy = sched_getscheduler(0);
    if (observed_sched_policy < 0 ||
        sched_getparam(0, &observed_sched_param) != 0) {{
        printf("FATAL: cannot read process scheduler policy errno=%d\\n", errno);
        return -1;
    }}
    const char *observed_sched_name =
        observed_sched_policy == SCHED_FIFO ? "SCHED_FIFO" :
        observed_sched_policy == SCHED_RR ? "SCHED_RR" :
        observed_sched_policy == SCHED_OTHER ? "SCHED_OTHER" : "OTHER";
    printf("xpurt: observed_sched_policy=%s priority=%d\\n",
           observed_sched_name, observed_sched_param.sched_priority);
#endif

    static const char *kinds[] = {{ {kind_strs} }};

    /* ---- Worker-set discovery -----------------------------------------
     * One worker per (core_kind, hart) pair the DISPATCH TABLE actually
     * uses -- discovered here rather than passed in at codegen, because
     * the hart column is resolved by ingest against the core registry and
     * the generator never sees the registry. A kind the schedule does not
     * use gets zero workers (e.g. rvv_c1 in the 2-model B1 schedule).
     *
     * Unbound entries (hart == -1): the kind's lowest-hart worker is
     * marked `claims_unbound` and runs them interleaved with its own. If
     * a kind has ONLY unbound entries it gets one unpinned worker
     * carrying the flag. Never dropped -- a dropped entry surfaces as a
     * zeroed trace row and reads exactly like a crash. */
    g_n_workers = 0;
    for (int k = 0; k < {n_kinds}; k++) {{
        int kind_first = g_n_workers;
        int has_unbound = 0;
        for (int i = 0; i < {upper}_N_ENTRIES; i++) {{
            const xpurt_sched_entry_t *e = &{upper}_TABLE[i];
            if (strcmp(e->core_kind, kinds[k]) != 0) continue;
            if (e->hart < 0) {{ has_unbound = 1; continue; }}
            int seen = 0;
            for (int w = kind_first; w < g_n_workers; w++) {{
                if (g_workers[w].hart == e->hart) {{ seen = 1; break; }}
            }}
            if (seen) continue;
            if (g_n_workers >= XPURT_MAX_WORKERS) {{
                printf("FATAL: schedule uses more than %d (core_kind,hart) "
                       "pairs; raise XPURT_MAX_WORKERS\\n", XPURT_MAX_WORKERS);
                sys_reboot(SYS_REBOOT_COLD);
                return -1;
            }}
            g_workers[g_n_workers].kind = kinds[k];
            g_workers[g_n_workers].hart = e->hart;
            g_workers[g_n_workers].kind_idx = k;
            g_workers[g_n_workers].worker_idx = g_n_workers;
            g_workers[g_n_workers].claims_unbound = 0;
            g_workers[g_n_workers].observed_cpu = -1;
            g_n_workers++;
        }}
        if (has_unbound) {{
            if (g_n_workers > kind_first) {{
                int lo = kind_first;
                for (int w = kind_first + 1; w < g_n_workers; w++) {{
                    if (g_workers[w].hart < g_workers[lo].hart) lo = w;
                }}
                g_workers[lo].claims_unbound = 1;
            }} else {{
                if (g_n_workers >= XPURT_MAX_WORKERS) {{
                    printf("FATAL: XPURT_MAX_WORKERS exhausted on unbound "
                           "kind %s\\n", kinds[k]);
                    sys_reboot(SYS_REBOOT_COLD);
                    return -1;
                }}
                g_workers[g_n_workers].kind = kinds[k];
                g_workers[g_n_workers].hart = -1;   /* unpinned */
                g_workers[g_n_workers].kind_idx = k;
                g_workers[g_n_workers].worker_idx = g_n_workers;
                g_workers[g_n_workers].claims_unbound = 1;
                g_workers[g_n_workers].observed_cpu = -1;
                g_n_workers++;
            }}
        }}
    }}

    /* ---- Claim-coverage audit -----------------------------------------
     * Every table entry must be claimed by EXACTLY ONE worker. Zero
     * claimants means the entry silently never runs -- that is how the
     * kind-vs-backend strcmp mismatch of 2026-06-03 presented, as all-zero
     * outputs that read like FPGA corruption. More than one means two
     * threads write the same trace slot and run the same kernel twice.
     * Both are fatal and both are cheap to rule out right here, before a
     * single thread exists. This is also what keeps the trace-slot
     * "exactly one writer" invariant a fact rather than a hope. */
    for (int i = 0; i < {upper}_N_ENTRIES; i++) {{
        const xpurt_sched_entry_t *e = &{upper}_TABLE[i];
        int claimants = 0;
        for (int w = 0; w < g_n_workers; w++) {{
            if (strcmp(e->core_kind, g_workers[w].kind) != 0) continue;
            if (e->hart >= 0) {{
                if (e->hart != g_workers[w].hart) continue;
            }} else if (!g_workers[w].claims_unbound) {{
                continue;
            }}
            claimants++;
        }}
        if (claimants != 1) {{
            printf("FATAL: entry %d (%s dispatch %d, core_kind=%s hart=%d) "
                   "has %d claimant workers, expected 1\\n",
                   e->entry_id, e->network, e->dispatch_id,
                   e->core_kind, e->hart, claimants);
            sys_reboot(SYS_REBOOT_COLD);
            return -1;
        }}
    }}

    /* Build one pool for each distinct composite target. This preserves both
     * the exact width and exact harts selected by XPU-RT. Singleton entries
     * deliberately receive NULL and execute on their owning worker. */
    int entry_runtime_rc = init_entry_runtime();
    if (entry_runtime_rc != 0) {{
        printf("FATAL: composite-target runtime init failed rc=%d\\n",
               entry_runtime_rc);
        sys_reboot(SYS_REBOOT_COLD);
        return -1;
    }}

    /* Dispatch-local state selects pool_for_entry(e), so the same model may
     * alternate singleton, two-hart, and four-hart implementations. */

{chr(10).join(reset_calls)}

    /* Init completion sems. Limit needs to be >= max fanout across
     * all entries (producer gives n_fanout times before any consumer
     * has taken). For our schedules max fanout is observed at <16; a
     * 32 cap gives headroom while keeping the sem accounting tight
     * (Phase G2d). The old limit=64 + take/re-give pattern is gone. */
    for (int i = 0; i < {upper}_N_ENTRIES; i++) {{
        k_sem_init(&completion_sems[i], 0, 32);
    }}

    /* One worker per (core_kind, hart) pair, discovered above. Each pins
     * itself to its own hart, so the schedule's placement decision is the
     * one that executes. The previous version spawned one worker per KIND
     * and pinned it to the first matching entry's hart, which quietly
     * serialized every dispatch of a kind onto a single core.
     *
     * Zephyr quirk: pthread_attr_init() casts the public pthread_attr_t*
     * to the internal `struct posix_thread_attr*` and zeroes it; with
     * CONFIG_POSIX_THREADS_AFFINITY the internal struct extends past the
     * public 16 bytes (cpu_affinity sits at offset 16). Without padding,
     * pthread_attr_setaffinity_np(&attrs[k+1]) silently corrupts the
     * adjacent attrs[k+1] neighbour. Pad each slot to 64 bytes so the
     * cast stays in its lane. */
    static pthread_t tids[XPURT_MAX_WORKERS];
    union xpurt_attr_slot {{ pthread_attr_t a; char _pad[64]; }};
    static union xpurt_attr_slot attrs[XPURT_MAX_WORKERS];

#ifdef MODELBLASTER_XPURT_TRACE
    /* -1/-1 rather than the static 0/0, so a row NO worker executed is
     * distinguishable from one executed by kind 0 on hart 0. */
    for (int i = 0; i < {upper}_N_ENTRIES; i++) {{
        xpurt_trace[i].worker_kind_idx = -1;
        xpurt_trace[i].worker_hart = -1;
    }}
#endif

    /* Run baseline for both the trace and the periodic-start-time gate.
     * Captured just before workers spawn so it's the moment "t=0" of the
     * schedule maps to. */
    run_t0 = (uint64_t)k_cycle_get_64();

    for (int w = 0; w < g_n_workers; w++) {{
        pthread_attr_init(&attrs[w].a);
        /* Affinity. The Linux arm used to be missing: the whole block sat
         * under `#ifdef CONFIG_POSIX_THREADS_AFFINITY`, a Zephyr Kconfig
         * symbol that the harness_xpurt_linux CMake never defines -- so on
         * the K1 every worker floated across all 8 harts and the
         * "pinned_hart=N" line printed at exit was a claim about an
         * attribute nobody had set. glibc needs _GNU_SOURCE (emitted at
         * the top of this file) plus the same call. */
#if defined(CONFIG_POSIX_THREADS_AFFINITY) || defined(MODELBLASTER_PLATFORM_LINUX)
        if (g_workers[w].hart >= 0) {{
            cpu_set_t cs;
            CPU_ZERO(&cs);
            CPU_SET(g_workers[w].hart, &cs);
            int arc = pthread_attr_setaffinity_np(&attrs[w].a, sizeof(cs), &cs);
            if (arc != 0) {{
                printf("FATAL: pthread_attr_setaffinity_np worker=%d kind=%s "
                       "hart=%d rc=%d -- refusing to run unpinned, every "
                       "per-hart number in the trace would be fiction\\n",
                       w, g_workers[w].kind, g_workers[w].hart, arc);
                sys_reboot(SYS_REBOOT_COLD);
                return -1;
            }}
        }}
#endif
        int rc = pthread_create(&tids[w], &attrs[w].a, xpurt_worker,
                                &g_workers[w]);
        if (rc != 0) {{
            printf("FATAL: pthread_create worker=%d kind=%s hart=%d rc=%d\\n",
                   w, g_workers[w].kind, g_workers[w].hart, rc);
            sys_reboot(SYS_REBOOT_COLD);
            return -1;
        }}
        /* Per-worker spawn diagnostic deferred to after pthread_join.
         * Inline printf here costs ~tens-of-ms over FireSim HTIF UART
         * and starves any worker pinned to the same hart as main.
         * The (kind, hart) info is already in g_workers[w]; print at the
         * end so it doesn't perturb the schedule's actual_start cycles. */
    }}

    /* Wait for every worker to drain. */
    for (int w = 0; w < g_n_workers; w++) {{
        pthread_join(tids[w], NULL);
        pthread_attr_destroy(&attrs[w].a);
    }}

    /* Now safe to flush the spawn diagnostics — workers are done, so
     * nothing pinned to hart 0 is competing with main for UART time.
     * `observed_cpu` is sched_getcpu() sampled inside the worker: it must
     * equal `hart` for a pinned worker, and a mismatch means the pinning
     * did not take even though the call returned 0. */
    for (int w = 0; w < g_n_workers; w++) {{
        printf("xpurt: worker[%d] kind=%s pinned_hart=%d observed_cpu=%d "
               "claims_unbound=%d entries_done=%llu\\n",
               w, g_workers[w].kind, g_workers[w].hart,
               g_workers[w].observed_cpu, g_workers[w].claims_unbound,
               (unsigned long long)g_hart_acc[w].entries_done);
    }}

    /* Phase G1 — per-WORKER runtime attribution. All values are mtime
     * ticks (1 us each on chipyard FireSim, 1/24 us on the K1 whose
     * rdtime runs at 24 MHz). Sum of categories should equal wall_total
     * within ~2%; residual is unattributed overhead (k_cycle_get_64
     * itself, branch mispredict, etc.).
     *
     * The `kind_idx` and `kind` columns are kept for
     * scripts/parse_runtime_breakdown.py, which keys on them; `worker_idx`
     * and `hart` are new and are what actually identify a row now that a
     * kind can have several workers. */
    printf("=== MODELBLASTER_HART_ACC_BEGIN ===\\n");
    printf("worker_idx,kind_idx,kind,hart,kernel_us,dep_wait_us,"
           "sync_overhead_us,target_gate_spin_us,hart_idle_us,"
           "gemmini_cfg_emit_us,entries_done,wall_total_us\\n");
    for (int w = 0; w < g_n_workers; w++) {{
        printf("%d,%d,%s,%d,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\\n",
               w, g_workers[w].kind_idx, g_workers[w].kind, g_workers[w].hart,
               (unsigned long long)g_hart_acc[w].kernel,
               (unsigned long long)g_hart_acc[w].dep_wait,
               (unsigned long long)g_hart_acc[w].sync_overhead,
               (unsigned long long)g_hart_acc[w].target_gate_spin,
               (unsigned long long)g_hart_acc[w].hart_idle,
               (unsigned long long)g_hart_acc[w].gemmini_cfg_emit,
               (unsigned long long)g_hart_acc[w].entries_done,
               (unsigned long long)g_hart_acc[w].wall_total);
    }}
    printf("=== MODELBLASTER_HART_ACC_END ===\\n");

#ifdef MODELBLASTER_XPURT_TRACE
    /* Trace dump — one CSV row per scheduled entry, with the actual
     * mtime-cycle window the worker ran, plus the kind-index AND the
     * hart of the worker that owned it (`worker_hart` vs the schedule's
     * `hart` column: they should be equal for every executed row). The host parser cross-references this
     * with the dispatch table for op/network/predicted-time fields. */
    printf("=== MODELBLASTER_XPURT_TRACE_BEGIN ===\\n");
    printf("entry_id,network,instance,dispatch_id,op,name,core_kind,hart,"
           "predicted_start_ms,predicted_duration_ms,worker_kind_idx,"
           "worker_hart,actual_start_cycles,actual_end_cycles\\n");
    for (int i = 0; i < {upper}_N_ENTRIES; i++) {{
        const xpurt_sched_entry_t *e = &{upper}_TABLE[i];
        printf("%d,%s,%d,%d,%s,%s,%s,%d,%.6f,%.6f,%d,%d,%llu,%llu\\n",
               e->entry_id, e->network, e->instance, e->dispatch_id,
               e->op, e->name, e->core_kind, e->hart,
               (double)e->start_time_ms, (double)e->duration_ms,
               xpurt_trace[i].worker_kind_idx,
               xpurt_trace[i].worker_hart,
               (unsigned long long)xpurt_trace[i].start_cycles,
               (unsigned long long)xpurt_trace[i].end_cycles);
    }}
    printf("=== MODELBLASTER_XPURT_TRACE_END ===\\n");
#endif

{chr(10).join(print_blocks)}

    for (int p = 0; p < g_n_entry_pools; p++) {{
        modelblaster_pool_destroy(g_entry_pools[p].pool);
    }}
    for (int h = 0; h < XPURT_MAX_HARTS; h++) {{
        pthread_mutex_destroy(&g_hart_locks[h]);
    }}
#ifndef MODELBLASTER_PLATFORM_LINUX
    /* Zephyr: a bare-metal run ends by rebooting the board, so the harness
     * driver sees a clean restart rather than a hung shell. This is the
     * SUCCESS path -- the error paths above reach sys_reboot too.
     *
     * On a hosted OS it must not be a reboot and, more importantly, must not
     * share an exit status with the failure paths: the POSIX shim maps
     * sys_reboot to _exit(1), which is right for an error and wrong here.
     * Falling through to `return 0` is what lets a caller distinguish "the
     * schedule ran" from "the schedule died". */
    sys_reboot(SYS_REBOOT_COLD);
#endif
    return 0;
}}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", required=True,
                    help="path to the schedule.json (XPU-RT format)")
    ap.add_argument("--out", required=True,
                    help="path to write xpurt_main.c")
    ap.add_argument("--name", required=True,
                    help="schedule name — must match the --name passed to "
                         "ingest_xpurt_schedule (drives the table symbol prefix)")
    ap.add_argument("--dispatch-table-header", required=True,
                    help="basename of the .h emitted by ingest_xpurt_schedule "
                         "(included from xpurt_main.c)")
    ap.add_argument("--platform", choices=("zephyr", "linux"),
                    default="zephyr",
                    help="zephyr: k_cycle_get_64/k_sem/sys_reboot as-is. "
                         "linux: rdtime + POSIX semaphores, for a hosted "
                         "riscv64 board such as the SpaceMiT K1.")
    ap.add_argument("--core-kinds", default="rvv,scalar",
                    help="comma list of distinct core_kind values the schedule "
                         "uses. One worker thread is spawned per kind. "
                         "(default: rvv,scalar)")
    ap.add_argument("--backends", default=None,
                    help="comma list of HW backends the harness was compiled "
                         "for (== per-model OBJECT-lib suffixes). The walker "
                         "forward-declares MODEL_<UMID>_DISPATCH_FNS_<BS> for "
                         "each backend and dispatches by core_kind. "
                         "Defaults to --core-kinds.")
    ap.add_argument("--model-gen-dir", action="append", default=[],
                    metavar="NET=PATH",
                    help="per-network generated dir (repeatable). Used to read "
                         "each model's model.h for input arity -- required "
                         "because a model's input count can differ between "
                         "quants/backends (fused_full: 1 input at fp16, 3 at "
                         "int8). Without it the generator falls back to a "
                         "repo-wide search and then to the single-input form.")
    ap.add_argument("--networks", default=None,
                    help="comma-separated network names in the schedule. "
                         "Supplies the known set that makes the "
                         "<network><instance> split unambiguous when a network "
                         "name ENDS IN A DIGIT (yolov8_nano_64x96), which the "
                         "trailing-digit regex otherwise reads as "
                         "'yolov8_nano_64x' + instance 96. The caller almost "
                         "always knows these -- run_xpurt_k1.sh has them as "
                         "--models -- and passing them removes the guess.")
    ap.add_argument("--registry", default=None,
                    help="path to the cores/*.json registry that drove the "
                         "schedule. Used to derive each kind's pool size as "
                         "(harts_of_kind - 1). If both --registry and "
                         "--pool-sizes are absent, all kinds get pool=0 "
                         "(NULL) and intra-op parallel_<op> calls run "
                         "synchronously on the scheduler worker.")
    ap.add_argument("--pool-sizes", default=None,
                    help='explicit per-kind pool helper-thread count, '
                         'e.g. "rvv:0,scalar:0". Overrides --registry. '
                         "Use 0 for kinds with one hart (NULL pool).")
    args = ap.parse_args()

    with open(args.schedule) as f:
        sched = json.load(f)
    networks: list[str] = []
    seen: set[str] = set()
    n_instances: dict[str, int] = {}
    # Multi-network bridge stamps an explicit instances list in
    # provenance — use it when present (it's the only way to be
    # unambiguous about model names ending in digits, e.g. "yolov8_nano_64"
    # which the trailing-digits regex parses as "yolov8_nano_" + idx=64).
    # An explicitly supplied network list is the most reliable source and
    # takes precedence over both the provenance block and the regex: the
    # caller knows what it built.
    cli_known = {n.strip() for n in (args.networks or "").split(",") if n.strip()}
    prov_instances = sched.get("_provenance", {}).get("instances")
    if cli_known:
        for d in sched["dispatches"].values():
            net, inst = _split_job_name(d["job_name"], cli_known)
            if net not in cli_known:
                raise SystemExit(
                    f"job_name {d['job_name']!r} does not start with any of "
                    f"--networks {sorted(cli_known)}. Refusing to guess: a "
                    f"wrong split emits an #include for a model that does not "
                    f"exist, or worse, one that does.")
            if net not in seen:
                seen.add(net)
                networks.append(net)
            n_instances[net] = max(n_instances.get(net, 0), inst + 1)
    elif prov_instances:
        known = {ins["network"] for ins in prov_instances}
        for ins in prov_instances:
            net = ins["network"]
            inst = ins["instance"]
            if net not in seen:
                seen.add(net)
                networks.append(net)
            n_instances[net] = max(n_instances.get(net, 0), inst + 1)
    else:
        known = None
        for d in sched["dispatches"].values():
            net, inst = _split_job_name(d["job_name"], known)
            if net not in seen:
                seen.add(net)
                networks.append(net)
            n_instances[net] = max(n_instances.get(net, 0), inst + 1)

    core_kinds = [k.strip() for k in args.core_kinds.split(",") if k.strip()]
    if not core_kinds:
        raise SystemExit("--core-kinds must be a non-empty comma list")
    if args.backends is None:
        backends = list(core_kinds)
    else:
        backends = [b.strip() for b in args.backends.split(",") if b.strip()]
        if not backends:
            raise SystemExit("--backends must be a non-empty comma list")

    # Guard against the silent-dispatch-drop bug (2026-06-03):
    # the schedule's dispatch table carries .core_kind = registry-kind
    # (e.g. "gemmini"). The generated dispatch branch must strcmp the
    # entry's .core_kind against the SAME string — so kind→backend
    # mapping is now done by zip(core_kinds, backends) inside _emit
    # (strcmp uses kind, dispatch fn symbol uses backend). Asserting
    # length parity here so the index-aligned mapping is well-defined.
    if len(core_kinds) != len(backends):
        raise SystemExit(
            f"--core-kinds and --backends must have the same length "
            f"(got core_kinds={core_kinds}, backends={backends}). The "
            f"dispatch branch pairs kind[i] -> backend[i] by index."
        )

    # Pool size resolution:
    #   1. explicit --pool-sizes wins
    #   2. else --registry: harts_of_kind - 1 per kind
    #   3. else default 0 per kind (NULL pool, no intra-op fanout)
    pool_sizes_map: dict[str, int] = {}
    if args.pool_sizes:
        for kv in args.pool_sizes.split(","):
            k, _, v = kv.strip().partition(":")
            if not k or not v:
                raise SystemExit(f"--pool-sizes: bad entry '{kv}'")
            pool_sizes_map[k] = int(v)
    elif args.registry:
        with open(args.registry) as f:
            reg = json.load(f)
        harts_per_kind: dict[str, set[int]] = {}
        for c in reg.get("cores", []):
            kind = c.get("kind")
            harts = c.get("harts", []) or []
            if kind is None:
                continue
            harts_per_kind.setdefault(kind, set()).update(harts)
        for k in core_kinds:
            n_harts = len(harts_per_kind.get(k, set()))
            # modelblaster_pool_create(N) makes a pool of N threads INCLUDING
            # the caller, so use n_harts directly. 1-hart kinds still get
            # 0 (NULL pool, serial fallback) since create(1) just allocates
            # a degenerate pool with no helpers.
            pool_sizes_map[k] = n_harts if n_harts >= 2 else 0
    pool_sizes = [pool_sizes_map.get(k, 0) for k in core_kinds]

    src = _emit(networks, args.name, args.dispatch_table_header,
                core_kinds, backends, pool_sizes, n_instances,
                platform=args.platform,
                gen_dir=os.path.dirname(args.out),
                model_gen_dirs=dict(
                    kv.split("=", 1) for kv in args.model_gen_dir if "=" in kv))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(src)
    print(f"wrote {args.out}  (platform: {args.platform} "
          f"networks: {networks} "
          f"instances: {n_instances} kinds: {core_kinds} "
          f"backends: {backends} pool_sizes: {pool_sizes})")


if __name__ == "__main__":
    main()
