"""Extension hooks: MODELBLASTER_EXTENSIONS, multi-root curated lookup, emit_call."""
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG_PARENT = os.path.dirname(REPO)

EXTENSION = textwrap.dedent('''
    def _emit(op, ctx):
        return (f"kernel_toy_s8({ctx.ptr_for(op['inputs'][0], 'in')}, {ctx.out_ptr})",
                [f"kernel_toy_s8_{ctx.mid}(0, 0);"])

    def register(rk):
        rk.add_algorithm("add_q_s8", rk.AlgorithmCandidate(
            name="toy_accel", pin_only=True, description="", reference_impl=""))
        rk.register_kernel_spec(rk.KernelSpec(
            op="toy_s8", signature="void kernel_toy_s8(const int8_t *a, int8_t *b)",
            semantics="", reference_impl="", emit_call=_emit, has_model_init=True))
''')

PROBE = textwrap.dedent('''
    import json
    from modelblaster.pipeline import reference_kernels as rk
    from modelblaster.pipeline import generate_skeleton as gs
    print(json.dumps({
        "add_q_s8": [a.name for a in rk.KERNEL_SPECS["add_q_s8"].algorithms],
        "toy": "toy_s8" in rk.KERNEL_SPECS,
        "init_ops": sorted(gs._ops_with_model_init()),
    }))
''')


def _probe(env_ext):
    env = dict(os.environ, PYTHONPATH=PKG_PARENT)
    env.pop("MODELBLASTER_EXTENSIONS", None)
    if env_ext is not None:
        env["MODELBLASTER_EXTENSIONS"] = env_ext
    r = subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True)
    return r


@unittest.skipUnless(os.path.basename(REPO) == "modelblaster",
                     "the checkout directory must be named modelblaster")
class ExtensionTest(unittest.TestCase):
    def test_no_extension_changes_nothing(self):
        r = _probe(None)
        self.assertEqual(r.returncode, 0, r.stderr)
        out = json.loads(r.stdout)
        self.assertNotIn("toy_accel", out["add_q_s8"])
        self.assertFalse(out["toy"])
        self.assertEqual(out["init_ops"], [])

    def test_extension_registers_algorithm_spec_and_init(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "toy_ext.py")
            open(p, "w").write(EXTENSION)
            r = _probe(p)
        self.assertEqual(r.returncode, 0, r.stderr)
        out = json.loads(r.stdout)
        self.assertEqual(out["add_q_s8"][-1], "toy_accel")
        self.assertTrue(out["toy"])
        self.assertEqual(out["init_ops"], ["toy_s8"])

    def test_missing_extension_file_fails(self):
        r = _probe("/nonexistent/ext.py")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("MODELBLASTER_EXTENSIONS", r.stderr)


class CuratedRootsTest(unittest.TestCase):
    def test_first_root_with_the_file_wins(self):
        sys.path.insert(0, PKG_PARENT)
        try:
            from modelblaster.pipeline.generate_kernels import _curated_path
        except ImportError as e:  # checkout not named modelblaster
            self.skipTest(str(e))
        finally:
            sys.path.pop(0)
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            name = "scalar_add_q_s8_toy.c"
            os.makedirs(os.path.join(b, "scalar"))
            open(os.path.join(b, "scalar", name), "w").write("/* b */")
            roots = os.pathsep.join([a, b])
            self.assertEqual(_curated_path(roots, "scalar", "add_q_s8", "toy"),
                             os.path.join(b, "scalar", name))
            os.makedirs(os.path.join(a, "scalar"))
            open(os.path.join(a, "scalar", name), "w").write("/* a */")
            self.assertEqual(_curated_path(roots, "scalar", "add_q_s8", "toy"),
                             os.path.join(a, "scalar", name))
            self.assertIsNone(_curated_path(roots, "scalar", "add_q_s8", "other"))


if __name__ == "__main__":
    unittest.main()
