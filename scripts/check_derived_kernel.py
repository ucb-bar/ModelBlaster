#!/usr/bin/env python3
"""Re-derive a curated kernel from the one it claims to be derived from.

WHY THIS EXISTS. Five distinct routes to "labelled vector, actually wrong"
have been found in this tree. The most recent one -- cos_s8's lookup table
built in float where its reference computes in double -- passed the
`_ALGORITHM_REQUIRED_SUBSTRINGS` structural gate AND a board numeric check,
because at int8-in/int8-out the two precisions agree on most inputs. Its
bit-exactness was a property of the test data.

The escape from that was to stop generating the kernel and derive it from one
already verified on the board: cos from sin, mul from add. That is a strictly
stronger guarantee -- but only while the derivation actually holds. A later
edit to the base kernel, or a well-meant "optimisation" of the derived one,
silently breaks the relationship that the derived file's header asserts, and
nothing would notice.

So the header's claim is executable. Each entry below names the base file, the
derived file, and the COMPLETE list of substitutions that turns one into the
other. If applying exactly those to the base does not reproduce the derived
file's code section, this fails and says which line diverged.

Only the code section is compared -- from the first `#include` onward. The
headers are prose and are expected to differ; that is where each file explains
why it exists.

Exit 0 all derivations hold, 1 one does not, 2 a file is missing.
"""

from __future__ import annotations

import argparse
import difflib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_KERNELS = os.path.join(os.path.dirname(_HERE), "kernels")

#: (derived, base, [(from, to), ...]). The substitution list must be complete:
#: anything else that differs is a finding, not a detail.
DERIVATIONS = [
    (
        "rvv/rvv_mul_s8_rvv_frm_rmm.c",
        "rvv/rvv_add_s8_rvv_frm_rmm.c",
        [
            ("void kernel_add_s8(", "void kernel_mul_s8("),
            ("vfloat32m4_t vsum = __riscv_vfadd_vv_f32m4(vfa, vfb, vl);",
             "vfloat32m4_t vprod = __riscv_vfmul_vv_f32m4(vfa, vfb, vl);"),
            ("__riscv_vfdiv_vf_f32m4(vsum, scale_out, vl)",
             "__riscv_vfdiv_vf_f32m4(vprod, scale_out, vl)"),
            ("""         * separately rounded. The caller's frm (RNE) is in force here. */""",
             """         * separately rounded. The caller's frm (RNE) is in force here. This
         * is the ONLY paragraph that differs from the add_s8 kernel: vfmul
         * where that one has vfadd. */"""),
        ],
    ),
    (
        "rvv/rvv_cos_s8_rvv_memo_lut_gather.c",
        "rvv/rvv_sin_s8_rvv_memo_lut_gather.c",
        [
            ("void kernel_sin_s8(", "void kernel_cos_s8("),
            ("const double y = sin((double)input[i] * (double)scale_in);",
             "const double y = cos((double)input[i] * (double)scale_in);"),
            ("const double y = sin((double)x * (double)scale_in);",
             "const double y = cos((double)x * (double)scale_in);"),
        ],
    ),
]


def code_section(path: str) -> str:
    """Everything from the first #include. The header above it is prose."""
    text = open(path, encoding="utf-8").read()
    i = text.find("#include")
    if i < 0:
        raise SystemExit(f"{path}: no #include, cannot find the code section")
    return text[i:]


def check_one(derived_rel, base_rel, subs, verbose=False) -> bool:
    derived_p = os.path.join(_KERNELS, derived_rel)
    base_p = os.path.join(_KERNELS, base_rel)
    for p in (derived_p, base_p):
        if not os.path.exists(p):
            print(f"MISSING: {p}", file=sys.stderr)
            raise SystemExit(2)

    got = code_section(base_p)
    for a, b in subs:
        if a not in got:
            print(f"FAIL {derived_rel}: the base no longer contains {a!r} -- "
                  f"{base_rel} was edited and the derivation is stale.")
            return False
        got = got.replace(a, b)

    want = code_section(derived_p)
    if got == want:
        n = len(subs)
        print(f"OK   {derived_rel}\n     = {base_rel} + {n} substitution(s)")
        return True

    print(f"FAIL {derived_rel} is NOT {base_rel} plus the listed "
          f"substitutions.")
    print("     Either the derivation gained an unlisted change (state it "
          "here, or undo it), or the base moved underneath it.")
    for line in difflib.unified_diff(
            got.splitlines(), want.splitlines(),
            fromfile=f"{base_rel} + substitutions", tofile=derived_rel,
            lineterm="", n=1):
        print("     " + line)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="check just the derived kernel whose path contains this")
    a = ap.parse_args()

    todo = [d for d in DERIVATIONS if not a.only or a.only in d[0]]
    if not todo:
        print(f"no derivation matches {a.only!r}", file=sys.stderr)
        return 2
    ok = all([check_one(*d) for d in todo])
    print(f"\n{sum(1 for _ in todo)} derivation(s) checked: "
          f"{'all hold' if ok else 'AT LEAST ONE BROKEN'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
