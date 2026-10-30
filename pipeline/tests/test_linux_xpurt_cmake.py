"""Regression guards for the hosted XPU-RT harness build graph."""

from __future__ import annotations

from pathlib import Path
import unittest


_ROOT = Path(__file__).resolve().parents[2]


class LinuxXpurtCmakeTests(unittest.TestCase):

    def test_links_weights_for_every_backend_but_buffers_once_per_model(self):
        """Backend-packed weights cannot be shared across heterogeneous TUs.

        ``generate_skeleton`` gives each backend's weights distinct symbols
        and may pack the same logical tensor in a different layout.  The
        Linux harness must therefore mirror the Zephyr harness: compile and
        link one weights object per (model, backend), while retaining exactly
        one shared intermediate-buffer object per model.
        """
        source = (_ROOT / "harness_xpurt_linux" / "CMakeLists.txt").read_text()

        self.assertIn(
            'add_library(${_weights_tgt} OBJECT "${_backend_dir}/weights.c")',
            source,
        )
        self.assertIn(
            'target_link_libraries(xpurt_harness PRIVATE ${_weights_tgt})',
            source,
        )
        self.assertIn(
            'add_library(${_buffers_tgt} OBJECT "${_primary_dir}/buffers.c")',
            source,
        )
        self.assertNotIn('foreach(_what weights buffers)', source)


if __name__ == "__main__":
    unittest.main()
