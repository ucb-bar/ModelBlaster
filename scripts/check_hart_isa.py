#!/usr/bin/env python3
"""Fail if code a hart without a vector unit can run contains RVV instructions.

  check_hart_isa.py ELF DISPATCH_TABLE.c --model NAME [--hart 1] [--objdump OBJDUMP]
                    [--impl-backend IMPL=BACKEND ...]

DISPATCH_TABLE.c is ingest_xpurt_schedule's output. Roots: for every schedule entry
on --hart, the backend kernel it calls, kernel_<op>_<model>_<backend>, where
<backend> is the entry's impl mapped through --impl-backend (defaults below). The check
walks the static call graph from those roots through `jal`/`j`/`call`/`tail` targets in
the linked ELF, and reports every 32-bit instruction that is OP-V (major opcode 0x57) or
a vector load/store (LOAD-FP 0x07 / STORE-FP 0x27 with width field 0, 5, 6 or 7; the
scalar FP loads and stores use 1-4).

Why: gemmini_q31_rvv's kernels.c is compiled with zve64x (other kernels in that TU use
RVV intrinsics) but runs on the Gemmini hart, which has no vector unit. GCC
auto-vectorizes ordinary loops and memcpy there, and the result is an illegal
instruction on the hardware, never seen in spike, which has V. A build recipe for a
heterogeneous SoC runs this on its linked ELF for every hart without V.
"""
import argparse, re, subprocess, sys

IMPL_BACKEND = {"gemmini_q31": "gemmini_q31_rvv", "rvv": "rvv", "scalar": "scalar"}


def is_vector(word: int) -> bool:
    op = word & 0x7F
    if op == 0x57:
        return True
    if op in (0x07, 0x27):
        return ((word >> 12) & 7) in (0, 5, 6, 7)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("elf"); ap.add_argument("table")
    ap.add_argument("--hart", type=int, default=1)
    ap.add_argument("--model", required=True)
    ap.add_argument("--objdump", default="riscv64-zephyr-elf-objdump")
    ap.add_argument("--impl-backend", action="append", default=[], metavar="IMPL=BACKEND",
                    help="schedule impl -> generated backend whose kernels it calls "
                         f"(repeatable; defaults {IMPL_BACKEND})")
    a = ap.parse_args()
    impl_backend = dict(IMPL_BACKEND, **dict(x.split("=", 1) for x in a.impl_backend))
    tab = open(a.table).read()
    roots = set()
    for m in re.finditer(r'\.op = "([a-z0-9_]+)".*?\.impl = "([a-z0-9_]+)", \.hart = (-?\d+)', tab):
        op, impl, hart = m.group(1), m.group(2), int(m.group(3))
        if hart == a.hart:
            roots.add(f"kernel_{op}_{a.model}_{impl_backend.get(impl, impl)}")
    if not roots:
        sys.exit(f"[check_hart_isa] no schedule entries on hart {a.hart} in {a.table}")
    dis = subprocess.run([a.objdump, "-d", a.elf], capture_output=True, text=True, check=True).stdout
    funcs, cur = {}, None
    for line in dis.splitlines():
        m = re.match(r"^([0-9a-f]+) <([^>]+)>:$", line)
        if m:
            cur = m.group(2); funcs.setdefault(cur, []); continue
        if cur is None:
            continue
        m = re.match(r"^\s*([0-9a-f]+):\s+([0-9a-f]{4,8})\s+(\S+)\s*(.*)$", line)
        if m:
            funcs[cur].append((m.group(1), m.group(2), m.group(3), m.group(4)))
    missing = sorted(r for r in roots if r not in funcs)
    if len(missing) == len(roots):
        # Not one root links: the --model or --impl-backend mapping is wrong, and the walk
        # would check nothing and pass.
        sys.exit(f"[check_hart_isa] FAIL: none of the {len(roots)} hart-{a.hart} kernel roots is in "
                 f"the ELF (wrong --model or --impl-backend?): {missing}")
    seen, todo, bad = set(), sorted(r for r in roots if r in funcs), []
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        for addr, raw, mn, args in funcs.get(f, []):
            if len(raw) == 8 and is_vector(int(raw, 16)):
                bad.append((f, addr, mn, args))
            if mn in ("jal", "j", "call", "tail", "c.j", "c.jal"):
                t = re.search(r"<([^>+]+)(\+0x[0-9a-f]+)?>", args)
                if t and t.group(1) != f:
                    todo.append(t.group(1))
    print(f"[check_hart_isa] hart {a.hart}: {len(roots)} kernel roots, {len(seen)} functions reachable"
          + (f"; roots not in the ELF (inlined or unused): {missing}" if missing else ""))
    for f, addr, mn, args in bad[:40]:
        print(f"  VECTOR on hart {a.hart}: {f} @0x{addr}: {mn} {args}")
    if bad:
        print(f"[check_hart_isa] FAIL: {len(bad)} vector instruction(s) reachable on hart {a.hart}")
        sys.exit(1)
    print(f"[check_hart_isa] PASS: no vector instruction reachable on hart {a.hart}")


if __name__ == "__main__":
    main()
