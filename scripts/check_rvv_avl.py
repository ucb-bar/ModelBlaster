#!/usr/bin/env python3
"""Catch a vsetvli handed a POINTER as its AVL operand.

Why this exists
---------------
GCC 14.3.0 miscompiles a vsetvl chained on a previous vsetvl's result. Four
curated kernels contained this, verbatim from kernel_cat2_c1_s8:

    subw    a3,a1,a4          ; a3 = stride - hw = 6
    add     a5,a6,a4          ; a5 = src + hw    <- a POINTER
    vsetvli a3,a3,e8,m2       ; vl = 6, correct
    vle8.v  v2,(a5)
    vsetvli zero,a5,e16,m4    ; AVL = a5 -- THE ADDRESS REGISTER

An address is astronomically larger than VLMAX, so `vl` saturates to VLMAX and
the vl-preserving `vsetvli zero,zero` forms that follow carry it to the store.
The tail then writes VLMAX elements where it owes the remainder.

The reason it is worth a checker rather than a code review: saturating to VLMAX
is CORRECT on every full iteration, so the bug is invisible unless a shape
produces a partial tail. yolov8n's cat_15 (stride 2*3=6 over 66 channels) is a
tail on its first iteration and wrote 58 bytes past the output buffer. The same
kernels had been shipping for as long as they existed on shapes that happened
not to expose it.

It is also worth checking the DISASSEMBLY and not the C: nine curated kernels
use the chained source pattern, and only four are actually miscompiled. Grepping
the source over-reports by more than half.

The heuristic: a register used as the base address of a vector load/store, then
used as the AVL operand of a vsetvli before being redefined. That is never
something a kernel means to do.

Usage:
    check_rvv_avl.py <elf-or-object> [...]      # exit 1 if any found
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

_FUNC = re.compile(r"^[0-9a-f]+ <(?P<name>.+)>:")
_INSN = re.compile(r"^\s*(?P<addr>[0-9a-f]+):\s+[0-9a-f ]+\t(?P<mn>[a-z0-9._]+)\s*(?P<ops>.*)$")
#: `vle8.v v2,(a5)` / `vse8.v v8,(a2)` -- the base register is what matters.
_MEM = re.compile(r"^v(?:le|se)[0-9]+\.v\s+v\d+,\((?P<base>\w+)\)")
_VSET = re.compile(r"^vsetvli\s+(?P<rd>\w+),(?P<avl>\w+),")
#: Anything writing rd kills the association; keep this deliberately small and
#: assume a write when unsure -- a missed report beats a false one here.
_WRITES_RD = re.compile(r"^(?P<mn>add|addi|addw|addiw|sub|subw|mv|li|lui|ld|lw|lh|lb|"
                        r"lbu|lhu|lwu|sll|slli|srl|srli|sra|srai|and|andi|or|ori|"
                        r"xor|xori|mul|mulw|div|divw|rem|remw|c\.\w+)$")


def scan(text: str):
    out = []
    func = "<unknown>"
    base_of = {}          # register -> addr of the load/store that used it
    for line in text.splitlines():
        m = _FUNC.match(line)
        if m:
            func, base_of = m.group("name"), {}
            continue
        m = _INSN.match(line)
        if not m:
            continue
        addr, mn, ops = m.group("addr"), m.group("mn"), m.group("ops")
        full = f"{mn} {ops}"

        mm = _MEM.match(full)
        if mm:
            base_of[mm.group("base")] = addr
            continue

        vs = _VSET.match(full)
        if vs:
            avl = vs.group("avl")
            if avl != "zero" and avl in base_of:
                out.append((func, addr, avl, base_of[avl]))
            continue

        # A branch or call ends the straight-line reasoning.
        if mn.startswith(("b", "j", "call", "tail", "ret")):
            base_of = {}
            continue
        # An instruction that redefines the register breaks the association.
        if _WRITES_RD.match(mn):
            rd = ops.split(",")[0].strip()
            base_of.pop(rd, None)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--objdump", default="riscv64-unknown-linux-gnu-objdump")
    a = ap.parse_args()

    bad = 0
    for f in a.files:
        try:
            text = subprocess.run([a.objdump, "-d", f], capture_output=True,
                                  text=True, check=True, timeout=600).stdout
        except (OSError, subprocess.CalledProcessError) as e:
            print(f"{f}: could not disassemble ({e})", file=sys.stderr)
            return 2
        hits = scan(text)
        if hits:
            bad += len(hits)
            print(f"FAIL {f}: {len(hits)} vsetvli given an address as AVL")
            for func, addr, reg, src in hits:
                print(f"  {func} @ {addr}: AVL={reg}, which is the base of the "
                      f"vector load/store at {src}")
        else:
            print(f"OK   {f}")

    if bad:
        print(f"\n{bad} site(s). vl saturates to VLMAX at each, so the code is "
              f"correct on full iterations and writes VLMAX elements on a "
              f"PARTIAL TAIL -- past the end of the destination.\n"
              f"The fix is in the KERNEL: hand every width the same plain "
              f"element count instead of chaining one vsetvl on another's "
              f"result. See kernels/rvv/rvv_cat2_c1_s8_direct.c.",
              file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
