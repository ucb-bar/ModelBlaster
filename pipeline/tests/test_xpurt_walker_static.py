"""The v2 walker (--walker skip): order is preserved, nothing deadlocks, skipping happens.

The generated main is compiled for the HOST against stub networks whose kernels
sleep for a random time, so the actual execution order differs from the table
on every run. Each run is checked two ways:

  * inside the stub kernels: when op k of a network starts, every data
    dependency must have run exactly once more than k has (so a dependency that
    has not run yet, and one that belongs to the NEXT instance, are both
    caught), and the previous instance must be complete;
  * from the per-dispatch trace (XPURT_TRACE_OUT): every entry ran in every
    cycle, started after its dependencies and after the whole previous instance
    ended, not before its start gate, and no two entries overlapped on a hart.

A deadlock shows up as the run hitting its timeout.

Cases: random DAG workloads at window 0 (whole table), 1 (strict order) and 3;
a table whose instances are interleaved against table order (window 0 must
finish, a bounded window must refuse to start); a head-of-line case where the
skipping walker runs a ready entry ahead of a blocked head and the strict one
does not; and the default (--walker strict) output of the generator, which must
not mention the v2 runtime at all.

Host-only liberties, none of which touch the walker logic: the two
`fence rw,rw` asm statements are removed from the generated text (RISC-V only),
thread placement is stubbed (no pinning), and XPURT_RT_PRIO=0.
"""

from __future__ import annotations

import os
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from pipeline import generate_xpurt_main_static as gen  # noqa: E402
from pipeline import ingest_xpurt_schedule as ingest  # noqa: E402
from pipeline import xpurt_static_program as sprog  # noqa: E402

TPS = 24_000_000.0
KINDS = ("ka", "kb")
HARTS = {"ka": (0, 1, 2, 3), "kb": (4, 5)}
_CC = shutil.which("gcc") or shutil.which("cc")


class Op:
    def __init__(self, net, did, deps):
        self.net, self.did, self.deps = net, did, deps


def _random_workload(rng, n_nets=3, max_inst=3):
    """nets -> (ops, n_instances); ops form a random DAG (deps on earlier ids)."""
    nets = {}
    for k in range(n_nets):
        name = f"net{chr(ord('a') + k)}"
        n_ops = rng.randint(4, 9)
        ops = []
        for d in range(n_ops):
            deps = sorted(rng.sample(range(d), k=min(d, rng.randint(0, 2)))) if d else []
            ops.append(Op(name, d, deps))
        nets[name] = (ops, rng.randint(1, max_inst))
    return nets


def _plan(rng, nets, composite_p=0.2, period_ms=6.0):
    """A consistent nominal schedule: (key -> dict(start, dur, harts, kind))."""
    free = {h: 0.0 for hs in HARTS.values() for h in hs}
    end = {}
    plan = {}
    order = []
    for name, (ops, n_inst) in nets.items():
        for inst in range(n_inst):
            for op in ops:
                order.append((inst, name, op))
    # instance-major order keeps instance k-1 before instance k in the plan
    order.sort(key=lambda x: (x[0], rng.random()))
    order.sort(key=lambda x: (x[0], x[1], x[2].did))
    last_end = {}
    for inst, name, op in order:
        kind = rng.choice(KINDS)
        hs = HARTS[kind]
        h = rng.choice(hs)
        harts = (h,)
        if rng.random() < composite_p and h + 1 in hs:
            harts = (h, h + 1)
        dur = rng.uniform(0.2, 0.8)
        ready = inst * period_ms
        for d in op.deps:
            ready = max(ready, end[(name, inst, d)])
        if inst:
            ready = max(ready, last_end[(name, inst - 1)])
        start = max([ready] + [free[x] for x in harts])
        for x in harts:
            free[x] = start + dur
        end[(name, inst, op.did)] = start + dur
        last_end[(name, inst)] = max(last_end.get((name, inst), 0.0), start + dur)
        plan[(name, inst, op.did)] = dict(start=start, dur=dur, harts=harts, kind=kind, deps=op.deps)
    return plan, max(end.values())


def _entries(plan, table_key=None):
    keys = sorted(plan, key=table_key or (lambda k: (plan[k]["start"], k)))
    eid = {k: i for i, k in enumerate(keys)}
    out = []
    last_on = {}
    for i, k in enumerate(keys):
        name, inst, did = k
        p = plan[k]
        deps = tuple(sorted(eid[(name, inst, d)] for d in p["deps"]))
        assert all(d < i for d in deps), "table order must keep data deps backward"
        td = last_on.get(p["harts"], -1)
        last_on[p["harts"]] = i
        out.append(ingest.XpurtEntry(
            entry_id=i, schedule_key=f"{name}{inst}_dispatch_{did}", job_name=f"{name}{inst}",
            network=name, instance=inst, dispatch_id=did, op="stub", name=f"op{did}",
            core_name=f"{p['kind']}{p['harts'][0]}", core_kind=p["kind"], impl=p["kind"],
            hart=p["harts"][0], harts=tuple(p["harts"]), start_time_ms=float(p["start"]),
            duration_ms=float(p["dur"]), deps_entry_ids=deps, time_dep_entry_id=td))
    fan = [[] for _ in out]
    for e in out:
        for d in e.deps_entry_ids:
            fan[d].append(e.entry_id)
        if e.time_dep_entry_id >= 0:
            fan[e.time_dep_entry_id].append(e.entry_id)
    for e in out:
        e.fanout_entry_ids = tuple(sorted(set(fan[e.entry_id])))
    return out


def _stub_sources(nets, slow=None):
    """Headers + one C file with stub kernels for every (network, backend).

    slow: {(net, did): extra_us} fixed extra delay, for the head-of-line case."""
    slow = slow or {}
    files = {}
    c = ['#include <stdio.h>', '#include <stdint.h>', '#include <stdlib.h>', '#include <time.h>',
         '#include <pthread.h>', '#include "mb_placement.h"',
         'int mb_cpu_is_ai(int cpu) { (void)cpu; return 0; }',
         'int mb_place_current_thread(int cpu) { (void)cpu; return 0; }',
         'int mb_pin_self_to_first_non_ai_cpu(void) { return 0; }',
         'static __thread uint64_t rs_;',
         'static void stub_delay(long extra_us) {',
         '    if (!rs_) { const char *s = getenv("XSTUB_SEED");',
         '        rs_ = (uint64_t)(s ? atol(s) : 1) * 0x9E3779B97F4A7C15ull ^ (uint64_t)pthread_self(); if (!rs_) rs_ = 1; }',
         '    rs_ ^= rs_ << 13; rs_ ^= rs_ >> 7; rs_ ^= rs_ << 17;',
         '    const char *m = getenv("XSTUB_MAX_US"); long mx = m ? atol(m) : 1500;',
         '    long us = (long)(rs_ % (uint64_t)(mx + 1)) + extra_us;',
         '    struct timespec ts = { us / 1000000, (us % 1000000) * 1000 }; nanosleep(&ts, NULL);',
         '}',
         'int stub_violations;',
         '#include "modelblaster_pool.h"',
         'struct stub_job { int hits; long extra; };',
         'static void stub_slice(void *c, size_t i) { (void)i; struct stub_job *j = c; stub_delay(j->extra); __atomic_add_fetch(&j->hits, 1, __ATOMIC_ACQ_REL); }',
         'static void stub_kernel(void *pool, long extra_us) {',
         '    if (!pool) { stub_delay(extra_us); return; }',
         '    int n = (int)modelblaster_pool_get_threads_count(pool);',
         '    for (int rep = 0; rep < 2; rep++) {       /* two forks per dispatch */',
         '        struct stub_job j = { 0, extra_us };',
         '        modelblaster_pool_parallelize_1d(pool, stub_slice, &j, (size_t)n, 0);',
         '        if (__atomic_load_n(&j.hits, __ATOMIC_ACQUIRE) != n) { __atomic_add_fetch(&stub_violations, 1, __ATOMIC_ACQ_REL); printf("STUB_FORK_VIOLATION hits=%d n=%d\\n", j.hits, n); }',
         '    }',
         '}']
    for name, (ops, _n_inst) in nets.items():
        up = name.upper()
        n = len(ops)
        files[f"{name}_model.h"] = f"""#pragma once
#include <stdint.h>
#define MODEL_{up}_OUTPUT_SIZE 4
#define MODEL_{up}_OP_COUNT {n}
typedef int8_t model_{name}_input_t;
typedef int8_t model_{name}_output_t;
typedef struct {{ const model_{name}_input_t *input; model_{name}_output_t *output; void *pool; }} model_{name}_state_t;
typedef struct {{ int dispatch_id; const char *name; const char *op; const char *shape; unsigned long cycles; }} model_{name}_op_record_t;
typedef void (*model_{name}_dispatch_fn)(model_{name}_state_t *);
"""
        files[f"{name}_test_io.h"] = f"""#pragma once
#include <stdint.h>
#define MODEL_{up}_TEST_OUTPUT_LEN 4
#define MODEL_{up}_TEST_INPUT_LEN 4
extern const int8_t model_{name}_test_input[4];
extern const int8_t model_{name}_test_golden[4];
"""
        c += [f'#include "{name}_model.h"', f'#include "{name}_test_io.h"',
              f'const int8_t model_{name}_test_input[4] = {{1, 2, 3, 4}};',
              f'const int8_t model_{name}_test_golden[4] = {{{n}, 0, 0, 0}};',
              f'static int gen_{name}[{n}];',
              f'void model_{name}_zero_buffers(void) {{ }}']
        for op in ops:
            checks = "".join(
                f'    if (__atomic_load_n(&gen_{name}[{d}], __ATOMIC_ACQUIRE) != g + 1) '
                f'{{ __atomic_add_fetch(&stub_violations, 1, __ATOMIC_ACQ_REL); '
                f'printf("STUB_ORDER_VIOLATION {name} op {op.did} dep {d}\\n"); }}\n' for d in op.deps)
            # every op of the previous instance must have completed (same count as this op has)
            inst_check = "".join(
                f'    if (__atomic_load_n(&gen_{name}[{o.did}], __ATOMIC_ACQUIRE) < g) '
                f'{{ __atomic_add_fetch(&stub_violations, 1, __ATOMIC_ACQ_REL); '
                f'printf("STUB_INSTANCE_VIOLATION {name} op {op.did} vs {o.did}\\n"); }}\n' for o in ops)
            c.append(f'static void k_{name}_{op.did}(model_{name}_state_t *s) {{\n'
                     f'    int g = __atomic_load_n(&gen_{name}[{op.did}], __ATOMIC_ACQUIRE);\n'
                     f'{checks}{inst_check}'
                     f'    stub_kernel(s->pool, {int(slow.get((name, op.did), 0))});\n'
                     f'    if (g == 0) __atomic_add_fetch(&s->output[0], 1, __ATOMIC_ACQ_REL);\n'
                     f'    __atomic_add_fetch(&gen_{name}[{op.did}], 1, __ATOMIC_ACQ_REL);\n}}')
        for bs in KINDS:
            fns = ", ".join(f"k_{name}_{op.did}" for op in ops)
            c += [f'const model_{name}_dispatch_fn MODEL_{up}_DISPATCH_FNS_{bs.upper()}[{n}] = {{ {fns} }};',
                  f'void model_{name}_reset_profile_{bs}(void) {{ }}',
                  f'const model_{name}_op_record_t *model_{name}_profile_records_{bs}(int *count) {{ *count = 0; return NULL; }}']
    c.append('__attribute__((destructor)) static void stub_report(void) '
             '{ printf("STUB_VIOLATIONS %d\\n", stub_violations); fflush(stdout); }')
    files["stubs.c"] = "\n".join(c) + "\n"
    return files


def _build(tmp, nets, entries, hyper_ms, slow=None, walker="skip", gang=True, program_out=None):
    name = "wtest"
    ingest.emit_table(entries, str(tmp / f"{name}.c"), name)
    n_inst = {n: v[1] for n, v in nets.items()}
    program = None
    if walker == "static":
        program = sprog.build(entries, gang=gang, cyclic=True)
        if program_out is not None:
            program_out.update(program)
    src = gen._emit(list(nets), name, f"{name}.h", list(KINDS), list(KINDS), [0, 0], n_inst,
                    platform="linux", warmup=False, walker=walker, hyper_ms=hyper_ms, program=program)
    for fence in ('__asm__ volatile("fence rw,rw" ::: "memory");', '__asm__ volatile("fence rw, rw" ::: "memory");'):
        src = src.replace(fence, ";")
    (tmp / "main.c").write_text(src)
    for fn, body in _stub_sources(nets, slow).items():
        (tmp / fn).write_text(body)
    rt = _ROOT / "runtime"
    cmd = [_CC, "-O1", "-g", "-pthread", "-DMODELBLASTER_XPURT_TRACE=1", "-DMODELBLASTER_PLATFORM_LINUX=1",
           "-I", str(tmp), "-I", str(rt), "-I", str(rt / "modelblaster_pool"),
           str(tmp / "main.c"), str(tmp / f"{name}.c"), str(tmp / "stubs.c"),
           str(rt / "modelblaster_pool" / "modelblaster_pool.c"), "-o", str(tmp / "walker")]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise AssertionError("host build failed:\n" + p.stderr[-4000:])
    return tmp / "walker"


def _run(binary, tmp, cycles, window, seed, timeout=60, extra_env=None):
    trace = tmp / f"trace_w{window}_s{seed}.bin"
    env = dict(os.environ, XPURT_CYCLES=str(cycles), XPURT_SKIP_WINDOW=str(window), XPURT_RT_PRIO="0",
               XPURT_TRACE_OUT=str(trace), XSTUB_SEED=str(seed), XPURT_GATE_SPIN_US="20")
    env.update(extra_env or {})
    p = subprocess.run([str(binary)], capture_output=True, text=True, timeout=timeout, env=env)
    return p, trace


def _check_trace(testcase, trace, entries, cycles, hyper_ms, static=False, gate_all=False, gang=True):
    raw = trace.read_bytes()
    magic, cyc, n, H = struct.unpack_from("<4Q", raw, 0)
    testcase.assertEqual((magic, cyc, n), (0x4B33525431, cycles, len(entries)))
    recs = [struct.unpack_from("<QQii", raw, 32 + 24 * i) for i in range(cyc * n)]
    inst_entries = {}
    for e in entries:
        inst_entries.setdefault((e.network, e.instance), []).append(e.entry_id)
    for c in range(cyc):
        s = [recs[c * n + i][0] for i in range(n)]
        en = [recs[c * n + i][1] for i in range(n)]
        for e in entries:
            i = e.entry_id
            testcase.assertNotEqual(s[i], 2**64 - 1, f"cycle {c} entry {i} never ran")
            gate = c * H + int(e.start_time_ms * TPS / 1000.0)
            if (not static) or gate_all or not e.deps_entry_ids:
                testcase.assertGreaterEqual(s[i], gate, f"cycle {c} entry {i} ran before its gate")
            for d in e.deps_entry_ids:
                testcase.assertGreaterEqual(s[i], en[d], f"cycle {c} entry {i} started before dep {d} ended")
            if e.instance:
                for j in inst_entries[(e.network, e.instance - 1)]:
                    testcase.assertGreaterEqual(s[i], en[j], f"cycle {c} entry {i} overlapped previous instance entry {j}")
        by_hart = {}
        for e in entries:
            # with static gangs a helper joins when its own list reaches the slice, so the
            # master's interval may start while the helper hart still runs its previous item
            for h in (e.harts[:1] if (static and gang) else e.harts):
                by_hart.setdefault(h, []).append((s[e.entry_id], en[e.entry_id], e.entry_id))
        for h, iv in by_hart.items():
            iv.sort()
            for (s0, e0, i0), (s1, _e1, i1) in zip(iv, iv[1:]):
                testcase.assertGreaterEqual(s1, e0, f"cycle {c} hart {h}: entries {i0} and {i1} overlap")
    return recs, n


@unittest.skipUnless(_CC, "no host C compiler")
class WalkerV2(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="xpurt_v2_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _ok(self, p):
        self.assertEqual(p.returncode, 0, p.stdout[-3000:] + p.stderr[-2000:])
        self.assertIn("STUB_VIOLATIONS 0", p.stdout)
        self.assertNotIn("STUB_ORDER_VIOLATION", p.stdout)
        self.assertNotIn("STUB_INSTANCE_VIOLATION", p.stdout)
        self.assertNotIn("STUB_FORK_VIOLATION", p.stdout)
        for m in re.finditer(r"=== MODELBLASTER_VERIFY \[(\w+)\] === max_abs_err=(\S+) .* ready=(\d)", p.stdout):
            self.assertEqual((m.group(2), m.group(3)), ("0", "1"), m.group(0))

    def test_random_workloads_keep_order_and_finish(self):
        """Random DAGs, placements and kernel delays; windows 0, 1 and 3."""
        for wseed in range(4):
            rng = random.Random(1000 + wseed)
            nets = _random_workload(rng)
            plan, span = _plan(rng, nets)
            entries = _entries(plan)
            hyper = span + 1.0
            d = self.tmp / f"w{wseed}"
            d.mkdir()
            binary = _build(d, nets, entries, hyper)
            for window in (0, 1, 3):
                for seed in (1, 2):
                    with self.subTest(workload=wseed, window=window, seed=seed):
                        p, trace = _run(binary, d, cycles=3, window=window, seed=seed)
                        self._ok(p)
                        _check_trace(self, trace, entries, 3, hyper)

    def test_spin_gates_and_single_cycle(self):
        rng = random.Random(7)
        nets = _random_workload(rng, n_nets=2, max_inst=2)
        plan, span = _plan(rng, nets)
        entries = _entries(plan)
        binary = _build(self.tmp, nets, entries, span + 1.0)
        p, trace = _run(binary, self.tmp, cycles=1, window=0, seed=3, extra_env={"XPURT_GATE": "spin"})
        self._ok(p)
        self.assertIn("gate=spin", p.stdout)
        _check_trace(self, trace, entries, 1, span + 1.0)

    def test_table_order_against_instance_order(self):
        """Instance 1 placed BEFORE instance 0 in table order (data deps still backward).
        The whole-table window must finish in the right order; a bounded window must refuse."""
        rng = random.Random(42)
        nets = _random_workload(rng, n_nets=2, max_inst=2)
        for name in nets:
            nets[name] = (nets[name][0], 2)
        plan, span = _plan(rng, nets)
        # reverse instance order in the table; keep all gates at 0 so nothing waits on time
        for k in plan:
            plan[k]["start"] = 0.0
        entries = _entries(plan, table_key=lambda k: (-k[1], k[0], k[2]))
        binary = _build(self.tmp, nets, entries, span + 1.0)
        p, trace = _run(binary, self.tmp, cycles=2, window=0, seed=5)
        self._ok(p)
        _check_trace(self, trace, entries, 2, span + 1.0)
        p, _ = _run(binary, self.tmp, cycles=2, window=1, seed=5)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("FATAL: bounded XPURT_SKIP_WINDOW", p.stdout)

    def test_skips_past_a_blocked_head(self):
        """Worker ka0's first entry waits on a slow producer on kb4; its second entry is independent.
        Skipping: the second entry starts before the slow producer ends. Strict: after."""
        nets = {"neta": ([Op("neta", 0, []), Op("neta", 1, [0])], 1), "netb": ([Op("netb", 0, [])], 1)}
        plan = {("neta", 0, 0): dict(start=0.0, dur=0.1, harts=(4,), kind="kb", deps=[]),
                ("neta", 0, 1): dict(start=0.1, dur=0.1, harts=(0,), kind="ka", deps=[0]),
                ("netb", 0, 0): dict(start=0.2, dur=0.1, harts=(0,), kind="ka", deps=[])}
        entries = _entries(plan)
        self.assertEqual([e.network for e in entries], ["neta", "neta", "netb"])
        binary = _build(self.tmp, nets, entries, 100.0, slow={("neta", 0): 30000})
        env = {"XSTUB_MAX_US": "100"}
        p, trace = _run(binary, self.tmp, cycles=1, window=0, seed=1, extra_env=env)
        self._ok(p)
        recs, _n = _check_trace(self, trace, entries, 1, 100.0)
        self.assertLess(recs[2][0], recs[0][1], "skip walker did not run the ready entry ahead of the blocked head")
        self.assertRegex(p.stdout, r"skips=1")
        p, trace = _run(binary, self.tmp, cycles=1, window=1, seed=1, extra_env=env)
        self._ok(p)
        recs, _n = _check_trace(self, trace, entries, 1, 100.0)
        self.assertGreater(recs[2][0], recs[0][1], "strict window ran out of table order")

    def test_rt_priority_refusal_is_not_fatal(self):
        nets = {"neta": ([Op("neta", 0, [])], 1)}
        plan = {("neta", 0, 0): dict(start=0.0, dur=0.1, harts=(0,), kind="ka", deps=[])}
        entries = _entries(plan)
        binary = _build(self.tmp, nets, entries, 10.0)
        if os.geteuid() == 0:
            self.skipTest("root may take SCHED_FIFO; the refusal path needs an unprivileged user")
        p, _ = _run(binary, self.tmp, cycles=1, window=0, seed=1, extra_env={"XPURT_RT_PRIO": "20"})
        self.assertEqual(p.returncode, 0, p.stdout[-2000:])
        self.assertTrue("refused" in p.stdout or "rt_priority=SCHED_FIFO" in p.stdout, p.stdout[:600])


@unittest.skipUnless(_CC, "no host C compiler")
class StaticWalker(unittest.TestCase):
    """--walker static: per-worker programs, prefix counters, gangs, run-ahead."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="xpurt_static_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    _ok = WalkerV2._ok

    def test_random_workloads_all_switches(self):
        modes = [dict(), dict(XPURT_SKIP="0"), dict(XPURT_GATE_ALL="1"), dict(XPURT_WAIT="spin"),
                 dict(XPURT_WAIT="sleep", XPURT_STALL_US="50"), dict(XPURT_GATE="spin", XPURT_SKIP="0"),
                 dict(XPURT_WAKE="all"), dict(XPURT_RUNAHEAD_WAIT="poll"),
                 dict(XPURT_WAKE="all", XPURT_RUNAHEAD_WAIT="poll", XPURT_WAIT="sleep"),
                 dict(XPURT_WAIT="sleep"), dict(XPURT_WAKE="doorbell_sem"), dict(XPURT_WAKE="doorbell_futex"),
                 dict(XPURT_WAKE="doorbell_sem", XPURT_WAIT="sleep"), dict(XPURT_WAKE="doorbell_futex", XPURT_WAIT="sleep"),
                 dict(XPURT_WAKE="doorbell_futex", XPURT_RUNAHEAD_WAIT="poll", XPURT_WAIT="sleep")]
        for wseed in range(4):
            rng = random.Random(2000 + wseed)
            nets = _random_workload(rng)
            plan, span = _plan(rng, nets, composite_p=0.3)
            entries = _entries(plan)
            hyper = span + 1.0
            for gang in (True, False):
                d = self.tmp / f"w{wseed}_{int(gang)}"
                d.mkdir()
                binary = _build(d, nets, entries, hyper, walker="static", gang=gang)
                for mi, env in enumerate(modes):
                    with self.subTest(workload=wseed, gang=gang, mode=env):
                        p, trace = _run(binary, d, cycles=3, window=0, seed=10 + mi, extra_env=env)
                        self._ok(p)
                        _check_trace(self, trace, entries, 3, hyper, static=True,
                                     gate_all=env.get("XPURT_GATE_ALL") == "1", gang=gang)
                        # S7: no sleep may end on the safety timeout with its need already met (a lost wake)
                        to = [int(x) for x in re.findall(r"ev_timeouts=(\d+)", p.stdout)]
                        self.assertTrue(to, "walker printed no ev_timeouts counters")
                        self.assertEqual(sum(to), 0, "lost-wake suspect: a sleep timed out with its need met")

    def test_overlapping_instances_are_refused_at_build(self):
        rng = random.Random(42)
        nets = _random_workload(rng, n_nets=2, max_inst=2)
        for name in nets:
            nets[name] = (nets[name][0], 2)
        plan, _span = _plan(rng, nets)
        for k in plan:
            plan[k]["start"] = 0.0
        entries = _entries(plan, table_key=lambda k: (-k[1], k[0], k[2]))
        with self.assertRaises(ValueError):
            sprog.build(entries)

    def test_edge_classification(self):
        """a0 -> a1 -> a2 with a1 on another worker, plus a0 -> a2 directly:
        a0->a1 and a1->a2 stay cross-worker, a0->a2 is same-worker (elided)."""
        nets = {"neta": ([Op("neta", 0, []), Op("neta", 1, [0]), Op("neta", 2, [0, 1])], 1)}
        plan = {("neta", 0, 0): dict(start=0.0, dur=0.1, harts=(0,), kind="ka", deps=[]),
                ("neta", 0, 1): dict(start=0.1, dur=0.1, harts=(4,), kind="kb", deps=[0]),
                ("neta", 0, 2): dict(start=0.2, dur=0.1, harts=(0,), kind="ka", deps=[0, 1])}
        st = sprog.build(_entries(plan), cyclic=False)["stats"]
        self.assertEqual((st["edges_data"], st["remaining_cross_worker"]), (3, 2))
        # now put a2 on a third worker: a0->a2 is cross-worker but implied by a0->a1->a2
        plan[("neta", 0, 2)]["harts"] = (1,)
        prog = sprog.build(_entries(plan), cyclic=False)
        st = prog["stats"]
        self.assertEqual((st["removed_by_reduction"], st["remaining_cross_worker"]), (1, 2))
        self.assertEqual(st["cross_worker_data_edges_per_instance"], {"neta0": 3})

    def test_runs_ahead_of_a_blocked_head(self):
        nets = {"neta": ([Op("neta", 0, []), Op("neta", 1, [0])], 1), "netb": ([Op("netb", 0, []), Op("netb", 1, [0])], 1)}
        plan = {("neta", 0, 0): dict(start=0.0, dur=0.1, harts=(4,), kind="kb", deps=[]),
                ("netb", 0, 0): dict(start=0.0, dur=0.1, harts=(0,), kind="ka", deps=[]),
                ("neta", 0, 1): dict(start=0.1, dur=0.1, harts=(0,), kind="ka", deps=[0]),
                ("netb", 0, 1): dict(start=0.2, dur=0.1, harts=(0,), kind="ka", deps=[0])}
        entries = _entries(plan)
        idx = {(e.network, e.dispatch_id): e.entry_id for e in entries}
        binary = _build(self.tmp, nets, entries, 100.0, slow={("neta", 0): 30000}, walker="static")
        env = {"XSTUB_MAX_US": "100"}
        p, trace = _run(binary, self.tmp, cycles=1, window=0, seed=1, extra_env=env)
        self._ok(p)
        recs, _n = _check_trace(self, trace, entries, 1, 100.0, static=True)
        self.assertLess(recs[idx[("netb", 1)]][0], recs[idx[("neta", 0)]][1])
        p, trace = _run(binary, self.tmp, cycles=1, window=0, seed=1, extra_env=dict(env, XPURT_SKIP="0"))
        self._ok(p)
        recs, _n = _check_trace(self, trace, entries, 1, 100.0, static=True)
        self.assertGreater(recs[idx[("netb", 1)]][0], recs[idx[("neta", 0)]][1])

    def test_ungated_interior_entries_never_finish_later(self):
        """S4, checked with fixed kernel durations and static order (no run-ahead): every entry
        ends no later without interior gates than with them (tolerance 2 ms for host noise)."""
        rng = random.Random(77)
        nets = _random_workload(rng, n_nets=3, max_inst=2)
        plan, span = _plan(rng, nets, composite_p=0.3)
        # stretch the plan so that interior gates actually hold entries back
        for k in plan:
            plan[k]["start"] *= 3.0
        entries = _entries(plan)
        hyper = 3.0 * span + 1.0
        slow = {(name, op.did): 300 for name, (ops, _n) in nets.items() for op in ops}
        binary = _build(self.tmp, nets, entries, hyper, slow=slow, walker="static")
        env = {"XSTUB_MAX_US": "0", "XPURT_SKIP": "0"}
        ends = {}
        for label, extra in (("gated", {"XPURT_GATE_ALL": "1"}), ("free", {})):
            p, trace = _run(binary, self.tmp, cycles=2, window=0, seed=1, extra_env=dict(env, **extra))
            self._ok(p)
            recs, n = _check_trace(self, trace, entries, 2, hyper, static=True, gate_all=bool(extra))
            ends[label] = [recs[i][1] for i in range(2 * n)]
        late = [(i, (f - g) / TPS * 1e3) for i, (g, f) in enumerate(zip(ends["gated"], ends["free"])) if f > g + 0.002 * TPS]
        self.assertEqual(late, [], f"entries that ended later without interior gates: {late[:5]}")
        earlier = sum(1 for g, f in zip(ends["gated"], ends["free"]) if f < g - 0.0002 * TPS)
        self.assertGreater(earlier, 0, "the gated plan held nothing back; the test has no teeth")


class DefaultOutputUnchanged(unittest.TestCase):
    def test_strict_walker_has_no_v2_code(self):
        src = gen._emit(["neta"], "s", "s.h", ["ka"], ["ka"], [0], {"neta": 1}, platform="linux")
        self.assertNotIn("v2", src)
        self.assertNotIn("xs_", src)
        self.assertIn("k_sem_take(&completion_sems[e_->deps[d]], K_FOREVER);", src)
        zsrc = gen._emit(["neta"], "s", "s.h", ["ka"], ["ka"], [0], {"neta": 1}, platform="zephyr")
        self.assertNotIn("v2", zsrc)

    def test_skip_walker_is_linux_only(self):
        with self.assertRaises(SystemExit):
            gen._emit(["neta"], "s", "s.h", ["ka"], ["ka"], [0], {"neta": 1}, platform="zephyr", walker="skip")


if __name__ == "__main__":
    unittest.main()
