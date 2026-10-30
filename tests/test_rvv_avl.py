"""Every curated kernel passes the element count to every vsetvl.

WHY THIS IS A TEST AND NOT JUST A SCRIPT. The defect it guards against is
invisible in three of the four places you would look for it: the source reads
as correct RVV, the build is clean, and the binary runs to completion. It shows
up only as wrong numbers, and only under one of the two compilers in use.

kernels/rvv/rvv_avgpool2d_s8_rvv_ow_lanes.c wrote

    size_t vl8 = __riscv_vsetvl_e8m1((size_t)(OW - ow));
    size_t vl  = __riscv_vsetvl_e32m4(vl8);

which says "the same element count, in the 32-bit domain" and is true of the
instruction set. GCC 14.3 issued the second vsetvl with the ENCLOSING LOOP'S
BOUND as its AVL: vl came out 5 where the output row is 11 wide, the
`vsetvli zero,zero` forms carried the 5 down to the store, and six of every
eleven outputs were never written. The board verify measured max_abs_err=68
against the op's own reference -- on a kernel whose header claimed
`accuracy_class: bit_exact`, and which HAD been bit-exact when it was verified
under GCC 13.2.

So the property is not "we checked once". It is a form that must not come
back, in a file nobody is currently looking at, under a compiler upgrade
nobody connects to it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECK = os.path.join(REPO, "scripts", "check_rvv_avl.py")
sys.path.insert(0, os.path.join(REPO, "scripts"))

import importlib.util
_spec = importlib.util.spec_from_file_location("check_rvv_avl", CHECK)
check_rvv_avl = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(check_rvv_avl)


class NoKernelChainsItsAVL(unittest.TestCase):

    def test_every_committed_kernel_is_clean(self):
        r = subprocess.run([sys.executable, CHECK], capture_output=True,
                           text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_the_checker_finds_the_form_it_exists_for(self):
        """The regression, verbatim, as avgpool had it."""
        import tempfile
        src = """
#include <riscv_vector.h>
void k(const signed char *a, signed char *o, int n) {
    for (int i = 0; i < n; ) {
        size_t vl8 = __riscv_vsetvl_e8m1((size_t)(n - i));
        size_t vl  = __riscv_vsetvl_e32m4(vl8);
        (void)vl; i += (int)vl8;
    }
}
"""
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "bad.c")
            open(p, "w").write(src)
            bad = check_rvv_avl.check_file(p)
        self.assertEqual(len(bad), 1, bad)
        self.assertIn("__riscv_vsetvl_e32m4(vl8)", bad[0])

    def test_the_element_count_form_is_accepted(self):
        import tempfile
        src = """
#include <riscv_vector.h>
void k(const signed char *a, signed char *o, int n) {
    for (int i = 0; i < n; ) {
        const size_t n_elem = (size_t)(n - i);
        size_t vl8 = __riscv_vsetvl_e8m1(n_elem);
        size_t vl  = __riscv_vsetvl_e32m4(n_elem);
        (void)vl; i += (int)vl8;
    }
}
"""
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "good.c")
            open(p, "w").write(src)
            self.assertEqual(check_rvv_avl.check_file(p), [])

    def test_a_header_that_quotes_the_bad_form_is_not_flagged(self):
        """Every fixed kernel now carries a paragraph naming the form in
        order to explain it. A checker that flags its own documentation is a
        checker someone switches off."""
        import tempfile
        src = """
/* Do NOT write __riscv_vsetvl_e32m4(vl8) -- see the header. */
#include <riscv_vector.h>
void k(int n) {
    const size_t n_elem = (size_t)n;
    size_t vl8 = __riscv_vsetvl_e8m1(n_elem);  // not __riscv_vsetvl_e32m4(vl8)
    (void)vl8;
}
"""
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "doc.c")
            open(p, "w").write(src)
            self.assertEqual(check_rvv_avl.check_file(p), [])


if __name__ == "__main__":
    unittest.main()
