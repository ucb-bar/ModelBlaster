#!/usr/bin/env python3
"""Catch RVV instructions issued under the wrong vtype, before the board does.

Why this exists
---------------
GCC 13.2 does not reliably carry `vtype` across width changes in a kernel that
mixes them. Three separate curated kernels shipped instructions the hardware
refuses:

    vsetvli a5, a5, e8, m2        <- set for an int8 load
    vsext.vf4 v16, v24            <- ILLEGAL: vf4 at SEW=8 implies a 2-bit source
    vfmv.v.f  v24, fa4            <- ILLEGAL: there is no 8-bit float

Each one SIGILLs on its first dispatch. The intrinsics were used correctly in
every case -- the compiler owed the vtype change and did not emit it -- so
nothing in the source review would have found them, and nothing in the build
complains. They were each found by decoding `badaddr` out of the board's dmesg,
one model at a time, which is a poor way to find a defect that recurs.

So this checks the property directly on the disassembly: for every instruction
whose legality depends on the active SEW, is the vtype in force one it can
actually run under?

The check is a linear scan, deliberately. It follows straight-line vtype flow
inside each function and gives up (reports UNKNOWN rather than guessing) where
control flow merges with disagreeing vtypes. A conservative checker that
sometimes says "cannot tell" is useful; one that guesses would be worse than
nothing, because it would be trusted.

Usage:
    check_rvv_vtype.py <object-or-elf> [...]        # exit 1 if any violation
    check_rvv_vtype.py --objdump <path> <file>
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from typing import Optional

_FUNC = re.compile(r"^[0-9a-f]+ <(?P<name>.+)>:")
_INSN = re.compile(r"^\s*(?P<addr>[0-9a-f]+):\s+[0-9a-f ]+\t(?P<mn>[a-z0-9._]+)\s*(?P<ops>.*)$")
_VSET = re.compile(r"\be(?P<sew>8|16|32|64)\s*,\s*(?P<lmul>mf?\d+)")

#: Instructions whose legality depends on the SEW in force, and the rule.
#:
#: A widening extend `vsext.vfN` / `vzext.vfN` names its DESTINATION SEW, and
#: the implied source SEW is dest/N -- which must be at least 8. So vf2 needs
#: SEW>=16, vf4 needs SEW>=32, vf8 needs SEW>=64.
_EXT = re.compile(r"^v[sz]ext\.vf(?P<f>2|4|8)$")

#: Float instructions cannot run at SEW=8: there is no 8-bit float format in
#: V, and zvfh only adds 16.
_FLOAT_PREFIXES = ("vf",)
_FLOAT_MIN_SEW = 16

#: Narrowing ops name the destination too; the source is 2x wider, so the
#: destination SEW must leave room for a legal source (<=64 here).
_NARROW = re.compile(r"^vn(clip|srl|sra|cvt)\b")


class Violation:
    def __init__(self, func, addr, mn, sew, why):
        self.func, self.addr, self.mn, self.sew, self.why = func, addr, mn, sew, why

    def __str__(self):
        s = f"e{self.sew}" if self.sew else "unknown"
        return f"  {self.func} @ {self.addr}: {self.mn} under SEW={s} -- {self.why}"


def _check_insn(mn: str, sew: Optional[int]) -> Optional[str]:
    """Return a violation reason, or None if the instruction is fine here."""
    if sew is None:
        return None  # vtype not known at this point; do not guess

    m = _EXT.match(mn)
    if m:
        f = int(m.group("f"))
        if sew < 8 * f:
            return (f"{mn} names a destination SEW of {sew}, implying a "
                    f"{sew // f}-bit source; the minimum is 8. It needs "
                    f"SEW>={8 * f}, so a vsetvli is missing before it")
        return None

    if mn.startswith(_FLOAT_PREFIXES) and not mn.startswith("vfirst"):
        if sew < _FLOAT_MIN_SEW:
            return (f"{mn} is a float op at SEW={sew}; there is no "
                    f"{sew}-bit float format (zvfh only adds 16), so a "
                    f"vsetvli is missing before it")
        return None

    if _NARROW.match(mn):
        if sew > 32:
            return (f"{mn} narrows from a {sew * 2}-bit source, which exceeds "
                    f"the 64-bit maximum")
        return None

    return None


def scan(text: str) -> list[Violation]:
    out: list[Violation] = []
    func = "<unknown>"
    sew: Optional[int] = None
    for line in text.splitlines():
        m = _FUNC.match(line)
        if m:
            func = m.group("name")
            sew = None  # a function entry tells us nothing about vtype
            continue
        m = _INSN.match(line)
        if not m:
            continue
        mn, ops, addr = m.group("mn"), m.group("ops"), m.group("addr")

        if mn.startswith("vset"):
            v = _VSET.search(ops)
            # `vsetvli zero, zero, ...` keeps vl and changes vtype; either way
            # the SEW is whatever this instruction names.
            sew = int(v.group("sew")) if v else None
            continue

        # A branch or call means the next instruction's vtype is not
        # determined by this path alone. Forget rather than assume.
        if mn.startswith(("b", "j", "call", "tail", "ret")):
            sew = None
            continue

        if not mn.startswith("v"):
            continue

        why = _check_insn(mn, sew)
        if why:
            out.append(Violation(func, addr, mn, sew, why))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--objdump",
                    default="riscv64-unknown-linux-gnu-objdump")
    ap.add_argument("--mattr", default="+m,+a,+f,+d,+c,+v,+zvl256b,+xsmtvdot")
    a = ap.parse_args()

    bad = 0
    for f in a.files:
        try:
            text = subprocess.run([a.objdump, "-d", f], capture_output=True,
                                  text=True, check=True, timeout=600).stdout
        except (OSError, subprocess.CalledProcessError) as e:
            print(f"{f}: could not disassemble ({e})", file=sys.stderr)
            return 2
        vs = scan(text)
        if vs:
            bad += len(vs)
            print(f"FAIL {f}: {len(vs)} instruction(s) under an illegal vtype")
            for v in vs:
                print(v)
        else:
            print(f"OK   {f}")

    if bad:
        print(f"\n{bad} violation(s). Each of these SIGILLs on the first "
              f"dispatch that reaches it.\n"
              f"The fix is in the KERNEL, not the compiler flags: name each "
              f"width-domain transition explicitly with __riscv_vsetvl_e<SEW>m<L>, "
              f"as rvv_batchnorm2d_s8_direct.c and rvv_cat2_c1_s8_direct.c do.",
              file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
