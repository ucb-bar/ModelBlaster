"""ModelBlaster's side of the compile pool (analysis/compile_pool.md): the shared build cache in kb_export's nvcc
builds (``KERNELBLASTER_BUILD_CACHE``), flag-off byte identity, negative entries, key sensitivity. Stub nvcc, no GPU."""
from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modelblaster.pipeline import build_cache, kb_export  # noqa: E402
from modelblaster.pipeline.tests.test_kb_export_slot_units import _fake_problem  # noqa: E402

STUB = """#!/bin/bash
D="$(cd "$(dirname "$0")" && pwd)"
if [ "$1" = "--version" ]; then cat "$D/version"; exit 0; fi
out=""; src=""; prev=""
for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; case "$a" in *.cu|*.cpp) src="$a";; esac; done
echo "$*" >> "$D/calls.log"
if [ -n "$src" ] && grep -q COMPILE_ERROR "$src"; then echo "$src(3): error: expected a ';'" >&2; exit 1; fi
{ echo stub; cat "$D/version"; echo "$*" | tr ' ' '\\n' | grep '^-[A-Za-z]' ; [ -n "$src" ] && cat "$src"
  for a in "$@"; do case "$a" in *.o) [ "$a" != "$out" ] && [ -f "$a" ] && head -c 100000 "$a";; esac; done; true; } > "$out"
"""
VARS = ("KERNELBLASTER_BUILD_CACHE", "MB_BUILD_CACHE", "KERNELBLASTER_BUILD_CACHE_NEG_TTL", "NVCC_APPEND_FLAGS",
        "KERNELBLASTER_BUILD_CACHE_MAX_GB", "KERNELBLASTER_BUILD_CACHE_STATS_DIR", "MB_BUILD_UNITS")


class BuildCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        self.nvcc = self.bin / "nvcc"
        self.nvcc.write_text(STUB)
        self.nvcc.chmod(self.nvcc.stat().st_mode | stat.S_IXUSR)
        (self.bin / "version").write_text("Cuda compilation tools, release 12.6, V12.6.77\nBuild x\n")
        self.saved = {k: os.environ.pop(k, None) for k in VARS}
        build_cache._TOOL.clear()
        build_cache._CACHE.clear()
        self.prob = _fake_problem(self.tmp / "p") or self.tmp / "p"

    def tearDown(self):
        for k, v in self.saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
        build_cache._CACHE.clear()

    def calls(self):
        p = self.bin / "calls.log"
        return p.read_text().splitlines() if p.exists() else []

    def so(self, work, sources=None, nvcc=None, arch="89"):
        return kb_export.build_so(self.prob, arch=arch, sources=sources, work=self.tmp / work, nvcc=str(nvcc or self.nvcc))

    def test_off_is_the_plain_build(self):
        so, log = self.so("w1")
        self.assertIsNotNone(so)
        n = len(self.calls())
        self.assertEqual(n, 4)                                   # 2 slot units + walker tail + the link
        self.so("w2")
        self.assertEqual(len(self.calls()), 2 * n)               # no cache: everything again
        self.assertFalse(list(self.tmp.rglob("*.meta")) or list(self.tmp.rglob(".stats")))

    def test_off_commands_equal_the_legacy_commands(self):
        """Cache off: the nvcc argv are exactly the ones the build always ran (object paths included)."""
        so, _ = self.so("w1")
        lines = self.calls()
        units = self.tmp / "w1" / "kernels_units" / "units"
        self.assertTrue(any(f"-c {units}/slot_a.cu -o {units}/slot_a.cu.o" in l for l in lines))
        self.assertTrue(any(l.startswith("-shared -arch=sm_89") and str(so) in l for l in lines))

    def test_hit_across_work_dirs_and_misses_on_every_input(self):
        os.environ["KERNELBLASTER_BUILD_CACHE"] = str(self.tmp / "shared")
        so1, _ = self.so("w1")
        n = len(self.calls())
        so2, _ = self.so("w2")
        self.assertEqual(len(self.calls()), n)                    # full hit: not one nvcc call
        self.assertEqual(so1.read_bytes(), so2.read_bytes())
        src = kb_export.seed_sources(self.prob) if hasattr(kb_export, "seed_sources") else None
        if src is None:
            src = {"a": (self.prob / "slots" / "a.cu").read_text(), "b": (self.prob / "slots" / "b.cu").read_text()}
        src2 = dict(src, b=src["b"] + "\n// edit\n")
        self.so("w3", src2)
        self.assertEqual(len(self.calls()) - n, 2)                # the edited unit and the link; a, tail hit
        n2 = len(self.calls())
        self.so("w4", arch="80")                                   # another arch: all units + link rebuilt
        self.assertEqual(len(self.calls()) - n2, 4)
        n3 = len(self.calls())
        (self.bin / "version").write_text("Cuda compilation tools, release 12.8, V12.8.61\nBuild y\n")
        build_cache._TOOL.clear()
        self.so("w5")                                              # another toolkit
        self.assertEqual(len(self.calls()) - n3, 4)
        n4 = len(self.calls())
        os.environ["NVCC_APPEND_FLAGS"] = "-DX=1"                  # compiler environment
        build_cache._TOOL.clear()
        self.so("w6")
        self.assertEqual(len(self.calls()) - n4, 4)

    def test_changed_header_is_a_miss(self):
        os.environ["KERNELBLASTER_BUILD_CACHE"] = str(self.tmp / "shared")
        h = self.tmp / "inc.h"
        h.write_text("#define V 1\n")
        a = self.prob / "slots" / "a.cu"
        a.write_text(a.read_text() + f'\n#include "{h}"\n')
        self.so("w1")
        n = len(self.calls())
        self.so("w2")
        self.assertEqual(len(self.calls()), n)
        st = os.stat(h)
        h.write_text("#define V 2\n")                             # same size
        os.utime(h, ns=(st.st_atime_ns, st.st_mtime_ns))          # same mtime
        self.so("w3")
        self.assertGreater(len(self.calls()), n)

    def test_compile_error_is_cached_and_not_for_other_text(self):
        os.environ["KERNELBLASTER_BUILD_CACHE"] = str(self.tmp / "shared")
        a = self.prob / "slots" / "a.cu"
        a.write_text(a.read_text() + "\nCOMPILE_ERROR\n")
        so, log = self.so("w1")
        self.assertIsNone(so)
        self.assertIn("expected", log)
        n = len(self.calls())
        # the per-unit split fails, the assembled TU fallback fails: both stored
        so, log2 = self.so("w2")
        self.assertIsNone(so)
        self.assertEqual(len(self.calls()), n)                     # served from negative entries
        self.assertIn("expected", log2)
        a.write_text(a.read_text().replace("COMPILE_ERROR", "int fixed;"))
        so, _ = self.so("w3")
        self.assertIsNotNone(so)                                   # a different source is compiled

    def test_neg_ttl_zero_disables_negative_entries(self):
        os.environ["KERNELBLASTER_BUILD_CACHE"] = str(self.tmp / "shared")
        os.environ["KERNELBLASTER_BUILD_CACHE_NEG_TTL"] = "0"
        a = self.prob / "slots" / "a.cu"
        a.write_text(a.read_text() + "\nCOMPILE_ERROR\n")
        self.so("w1")
        n = len(self.calls())
        self.so("w2")
        self.assertGreater(len(self.calls()), n)

    def test_stats_and_eviction(self):
        os.environ["KERNELBLASTER_BUILD_CACHE"] = str(self.tmp / "shared")
        self.so("w1")
        self.so("w2")
        c = build_cache.cache()
        self.assertGreaterEqual(c.st["hits"], 3)
        self.assertGreaterEqual(c.st["misses"], 3)
        c.flush()
        files = list((self.tmp / "shared" / ".stats").glob("*.json"))
        self.assertEqual(len(files), 1)
        d = json.loads(files[0].read_text())
        self.assertEqual(d["hits"], c.st["hits"])
        self.assertEqual(c.evict(max_bytes=1, min_age=0) > 0, True)

    def test_interop_with_kernelblaster_cache_layout(self):
        """Same directory layout / suffix naming as KernelBlaster's BuildCache."""
        os.environ["KERNELBLASTER_BUILD_CACHE"] = str(self.tmp / "shared")
        self.so("w1")
        ents = [p for p in (self.tmp / "shared").rglob("*") if p.is_file() and p.suffix in (".o", ".so")]
        self.assertTrue(ents)
        for p in ents:
            self.assertEqual(p.parent.name, p.name[:2])


if __name__ == "__main__":
    unittest.main()
