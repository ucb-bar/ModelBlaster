#!/usr/bin/env python3
"""Refuse a `vsetvl` whose AVL is another `vsetvl`'s result.

WHAT THIS CATCHES, AND WHY THE DISASSEMBLY CHECKER CANNOT
---------------------------------------------------------
`check_rvv_vtype.py` finds instructions the hardware refuses -- a vf4 extend
under SEW=8, say -- by decoding the disassembly. Those SIGILL, loudly, on the
first dispatch. This is the other failure: every instruction is legal, the
binary runs to completion, and `vl` is simply wrong.

Written as

    size_t vl8 = __riscv_vsetvl_e8m1(n_col);
    size_t vl  = __riscv_vsetvl_e32m4(vl8);      <- AVL is a previous vl

GCC 14.3 substitutes an unrelated register for the second AVL. Measured in
kernels/rvv/rvv_avgpool2d_s8_rvv_ow_lanes.c: the e32m4 vsetvl was issued with
the enclosing loop's BOUND as its AVL, `vl` came out 5 where the output row is
11 wide, the `vsetvli zero,zero` forms carried that 5 down to the store, and
six of every eleven outputs were never written. max_abs_err=68 against the
op's own reference, with no crash and no warning.

WHY THE FORM IS TEMPTING AND STILL WRONG. It reads as correct RVV: e8m1 and
e32m4 hold the same number of elements, so `vsetvl_e32m4(vl8)` and
`vsetvl_e32m4(n)` return the same value, and the first says "the same count,
in the other domain" more clearly. It is the compiler that does not honour it.

WHY IT SURVIVED REVIEW FOR SO LONG. The kernels that use it were verified
bit-exact on the board under GCC 13.2, which compiles it correctly. 13.2 has
the OTHER bug -- it reorders a vsetvl across a widening op and the binary
SIGILLs -- which is why 14.3 is now mandatory. So the mandate that fixed a
loud failure introduced a silent one, and the only form correct under both
compilers is: pass the ELEMENT COUNT to every width, every time.

    const size_t n_elem = (size_t)(n - i);
    size_t vl8 = __riscv_vsetvl_e8m1(n_elem);
    size_t vl  = __riscv_vsetvl_e32m4(n_elem);

Source-level and deliberately simple: it does not model scope or control flow.
It records every identifier assigned from a `__riscv_vsetvl_*` call in a file
and flags any later `__riscv_vsetvl_*` whose argument is exactly one of them.
That is the whole shape of the defect, and a broader analysis would find
nothing more while being able to be wrong.

Exit 0 clean, 1 if any kernel chains.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

#: `size_t vl8 = __riscv_vsetvl_e8m1(...)` / `vl = __riscv_vsetvl_e8m2(...)`
_ASSIGN = re.compile(
    r"^\s*(?:const\s+)?(?:size_t\s+)?([A-Za-z_]\w*)\s*=\s*"
    r"__riscv_vsetvl_[a-z0-9]+\s*\(")
#: any call, with its argument captured when the argument is a bare identifier
_CALL = re.compile(r"__riscv_vsetvl_([a-z0-9]+)\s*\(\s*([A-Za-z_]\w*)\s*\)")


def strip_comments(text: str) -> str:
    """Blank out comments, preserving line numbering.

    Necessary, not tidy: every one of these kernels now carries a header
    paragraph that QUOTES the bad form in order to explain it, and a checker
    that flags its own explanation is a checker people turn off.
    """
    out, i, n = [], 0, len(text)
    while i < n:
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join(c if c == "\n" else " " for c in text[i:j]))
            i = j
        elif text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def check_file(path: str) -> list[str]:
    src = strip_comments(open(path, encoding="utf-8").read())
    vl_names = {m.group(1) for line in src.splitlines()
                for m in [_ASSIGN.match(line)] if m}
    bad = []
    for lineno, line in enumerate(src.splitlines(), 1):
        for m in _CALL.finditer(line):
            if m.group(2) in vl_names:
                bad.append(f"{path}:{lineno}: __riscv_vsetvl_{m.group(1)}"
                           f"({m.group(2)}) -- {m.group(2)} is itself a vsetvl "
                           f"result; pass the element count instead")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*",
                    help="C sources to check (default: every curated kernel)")
    a = ap.parse_args()

    files = a.files
    if not files:
        root = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "kernels")
        files = sorted(os.path.join(d, f)
                       for d, _, fs in os.walk(root) for f in fs
                       if f.endswith(".c"))
    if not files:
        print("no files to check", file=sys.stderr)
        return 1

    bad = [v for f in files for v in check_file(f)]
    for v in bad:
        print("CHAINED AVL: " + v)
    print(f"{len(files)} file(s) checked, {len(bad)} chained AVL(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
