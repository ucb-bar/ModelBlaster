"""A5 helpers in kb_global: implementation pools over several impl stores (zero-LLM routing input for kb_reuse) and
structural priors for graph search (seed file + the work dir's own decisions)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))

from modelblaster.pipeline import kb_global  # noqa: E402


def _store_mod():
    kb_global._kb()
    from kernelblaster.knowledge.impl_store import ImplStore, signature_sha
    return ImplStore, signature_sha


class ImplPoolAcrossStores(unittest.TestCase):
    def test_pool_merges_dirs_and_routes(self):
        ImplStore, signature_sha = _store_mod()
        sig = signature_sha("void k(const half* x, half* y, int n)")
        with tempfile.TemporaryDirectory() as td:
            a, b = Path(td) / "run" / "impl", Path(td) / "seed" / "impl"
            sa, sb = ImplStore(a), ImplStore(b)
            kw = dict(target="cuda", arch="sm_89", op="relu_f16", op_class="elementwise", signature_sha=sig,
                      dtype="fp16", buckets=[], speedup_vs_seed=2.0)
            r1 = sa.add("kernel one", shapes=[{"n": 1024}], slot_us=5.0, **kw)
            r2 = sb.add("kernel two", shapes=[{"n": 1 << 20}], slot_us=40.0, **kw)
            sa.observe("relu_f16", sig, "sm_89", "fp16", r1.code_sha, {"n": 1 << 20}, us=60.0)
            p = kb_global.impl_pool([a, b, Path(td) / "missing"], "relu_f16", sig, "sm_89", "fp16")
            self.assertEqual(set(p.kernels), {r1.code_sha[:16], r2.code_sha[:16]})
            r = p.route([{"op_name": "x", "shape": {"n": 1024}}, {"op_name": "y", "shape": {"n": 1 << 20}}])
            self.assertEqual([c["choice"]["kernel"] for c in r["choices"]], [r1.code_sha[:16], r2.code_sha[:16]])
            self.assertEqual(r["sum_us"], 45.0)
            empty = kb_global.impl_pool([], "relu_f16", sig, "sm_89", "fp16")
            self.assertEqual(empty.kernels, {})


class StructPrior(unittest.TestCase):
    def test_seed_file_plus_own_decisions(self):
        with tempfile.TemporaryDirectory() as td:
            w = Path(td) / "work"
            (w / "gk_seed").mkdir(parents=True)
            rows = [{"id": f"s{i}", "part": "cuda:sm_89|fp16", "kind": "fuse", "context": ["batchnorm2d", "relu"],
                     "composition": "batchnorm2d+relu", "classes": "norm+ewise", "head": "norm",
                     "status": "accepted", "accepted": True, "gain": 0.4} for i in range(3)]
            (w / "gk_seed" / "struct_priors.json").write_text(json.dumps({"rows": rows}))
            (w / "state.json").write_text(json.dumps({"rounds": [{"round": 0, "decisions": [
                {"hint": "fuse:fused__batchnorm2d_f16__relu_f16", "status": "rejected", "gain": -0.05}]}]}))
            p = kb_global.struct_prior("fuse", ["batchnorm2d_f16", "relu_f16"], work=w, part="cuda:sm_89|fp16")
            self.assertEqual((p["level"], p["n"], p["accepted"], p["rejected"]), ("exact", 4, 3, 1))
            self.assertTrue(0 < p["lo"] < p["rate"] < p["hi"] <= 1 and p["alpha"] > p["beta"])
            none = kb_global.struct_prior("fold_layout", ["bmm"], work=Path(td) / "nowhere")
            self.assertEqual((none["level"], none["n"]), ("none", 0))


if __name__ == "__main__":
    unittest.main()
