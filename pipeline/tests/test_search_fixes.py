"""Search fixes (search_fixes.py; l3_farm/analysis/search_fixes.md), offline (no torch, no GPU):

* shape distance and the nearest-shape candidate pool (rejected codes never come back for that IR);
* routing (zero LLM) with a scripted measurement: per-shape choice, guard = exactly the routed shapes, provenance,
  nothing written when no candidate passes; the profile-fallback callback on a DenseNet-like conv graph;
* re-queue after a profile fallback (only instances still on a seed, at most N times per item, resume-safe) and
  the rejection note in the re-queued problem's slot semantics;
* the final-model guard on synthetic and on recorded sweep data (15_DenseNet121: the 960x7x7 conv left on the seed);
* nearest-shape seed candidates in the manifest; defaults (fallback repair + guard on, nothing else; round key).
"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sf = CudaCache = cache_plan = SPECS = None
DATA = Path(__file__).resolve().parent / "data"


def setUpModule():
    global sf, CudaCache, cache_plan, SPECS
    from modelblaster.pipeline import search_fixes as sf
    from modelblaster.pipeline.kb_ingest import CudaCache, cache_plan
    from modelblaster.pipeline.kb_proposals import _all_specs
    SPECS = _all_specs()


def conv_shape(ic, hw, oc=32, k=3):
    return {"N": 10, "IC": ic, "IH": hw, "IW": hw, "OC": oc, "OH": hw, "OW": hw, "KH": k, "KW": k, "SH": 1, "SW": 1,
            "PH": k // 2, "PW": k // 2, "DH": 1, "DW": 1}


def conv_graph(shapes, name="dn"):
    t, ops = {"x0": {"shape": [1]}}, []
    cur = "x0"
    for i, sh in enumerate(shapes):
        nm = f"conv{i}"
        for w in ("w", "b"):
            t[f"{nm}__{w}"] = {"shape": [1], "const": f"{nm}__{w}"}
        t[f"y{i}"] = {"shape": [1]}
        ops.append({"name": nm, "op": "conv2d_f16", "inputs": [cur, f"{nm}__w", f"{nm}__b"], "outputs": [f"y{i}"],
                    "shape": sh})
        cur = f"y{i}"
    return {"name": name, "tensors": t, "ops": ops, "output": {"tensor": cur}}


def add(cache, shapes, code, us, iid, kind="conv2d_f16"):
    from modelblaster.pipeline.kernel_abi import parse_signature
    spec = SPECS[kind]
    e, _ = cache.add(op=kind, entry=parse_signature(spec.signature)[0], signature=spec.signature, arch="sm_89",
                     shapes=shapes, code=code, slot_us=us, speedup=10.0, impl_id=iid, source={})
    return e


class DistanceTest(unittest.TestCase):
    def test_nearest_is_the_same_regime(self):
        t = conv_shape(960, 7)
        self.assertEqual(sf.shape_distance(t, t), 0.0)
        near, mid, far = conv_shape(928, 7), conv_shape(480, 28), conv_shape(64, 56)
        self.assertLess(sf.shape_distance(t, near), sf.shape_distance(t, mid))
        self.assertLess(sf.shape_distance(t, mid), sf.shape_distance(t, far))
        self.assertEqual(sf.set_distance([t], [far, near]), sf.shape_distance(t, near))
        self.assertEqual(sf.set_distance([], [near]), float("inf"))
        # fused shapes compare stage by stage; a missing / differing flag costs
        self.assertGreater(sf.shape_distance({"a": t, "b": {"n": 4}}, {"a": t, "b": {"n": 400}}), 0)
        self.assertGreater(sf.shape_distance({"n": 4, "reverse": 1}, {"n": 4}), 0)


class PoolAndRouteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.cache = CudaCache(self.d / "cache")
        self.t960, self.t928, self.t480, self.t64 = conv_shape(960, 7), conv_shape(928, 7), conv_shape(480, 28), \
            conv_shape(64, 56)
        self.e960 = add(self.cache, [self.t960], "// k960", 50.0, "w960")
        self.e928 = add(self.cache, [self.t928], "// k928", 40.0, "w928")
        self.e480 = add(self.cache, [self.t480], "// k480", 30.0, "w480")
        self.e64 = add(self.cache, [self.t64], "// k64", 20.0, "w64")
        self.e960["rejected"].append("dn")                          # failed in the whole model (profile fallback)
        self.cache.save()

    def tearDown(self):
        self.tmp.cleanup()

    def test_pool_nearest_first_and_rejected_never_offered(self):
        c = sf.nearest_pool("conv2d_f16", self.cache, arch="89", specs=SPECS, ir_name="dn", targets=[self.t960],
                            max_cands=10)
        self.assertEqual([x["from"] for x in c], ["cache:w928", "cache:w480", "cache:w64"])
        self.assertTrue(c[0]["dist"] < c[1]["dist"] < c[2]["dist"])
        # the rejected code is out for that IR even when the impl store offers it again
        impl = self.d / "impl"
        (impl / "code").mkdir(parents=True)
        import hashlib
        sha = hashlib.sha1(b"// k960").hexdigest()
        (impl / "code" / f"{sha}.cu").write_text("// k960")
        from modelblaster.pipeline.cuda_dtypes import cuda_signature
        from modelblaster.pipeline.kb_ingest import signature_sha
        rec = {"op": "conv2d_f16", "signature_sha": signature_sha(cuda_signature(SPECS["conv2d_f16"].signature)),
               "dtype": "fp16", "arch": "sm_89", "target": "cuda", "code_sha": sha, "shapes": [self.t960],
               "slot_us": 1.0, "speedup_vs_seed": 50.0, "id": "r1", "problem_id": "p"}
        (impl / "index.jsonl").write_text(json.dumps(rec) + "\n")
        c2 = sf.nearest_pool("conv2d_f16", self.cache, arch="89", specs=SPECS, ir_name="dn", targets=[self.t960],
                             impl_dirs=[impl], max_cands=10)
        self.assertNotIn("// k960", [x["code"] for x in c2])
        # ... but another IR may use it
        c3 = sf.nearest_pool("conv2d_f16", self.cache, arch="89", specs=SPECS, ir_name="other",
                             targets=[self.t960], max_cands=1)
        self.assertEqual(c3[0]["from"], "cache:w960")

    def _measure(self, results):
        """Scripted kb_reuse.measure_kind: {code: (ok, us)} per candidate, incumbent (seed) 1000 us."""
        def fn(kind, graph, src_ir, ops, incumbent, cands):
            shapes = []
            for o in graph["ops"]:
                if o["name"] in ops and o["shape"] not in shapes:
                    shapes.append(o["shape"])
            table = {}
            for k, sh in enumerate(shapes):
                row = {"n": 1, "shape": sh, "ops": [o["name"] for o in graph["ops"] if o["shape"] == sh],
                       "incumbent": {"ok": True, "us": 1000.0}, "cands": {}}
                for c in cands:
                    ok, us = results.get(c["code"], (False, None))
                    row["cands"][c["sha"][:12]] = {"ok": ok, "us": us if ok else None}
                table[k] = row
            return {"table": table}
        return fn

    def test_route_writes_a_guarded_entry_with_provenance(self):
        g = conv_graph([self.t960, self.t64])
        rep = sf.route_ops(kind="conv2d_f16", ops=["conv0"], graph=g, src_ir=self.d, cache=self.cache, specs=SPECS,
                           arch="89", ir_name="dn", work=self.d / "w",
                           measure_fn=self._measure({"// k928": (False, None), "// k480": (True, 60.0),
                                                     "// k64": (True, 90.0)}),
                           why={"rejected": ["w960"]})
        self.assertEqual(rep["result"], "routed")
        self.assertEqual([x["from"] for x in rep["written"]], ["cache:w480"])   # 928's launch failed on 960x7x7
        e = self.cache.entries[-1]
        self.assertTrue(e["reuse"])
        self.assertEqual(e["source"]["by"], sf.ROUTE_TAG)
        self.assertEqual(e["source"]["why"], {"rejected": ["w960"]})
        self.assertEqual(e["shape_guard"], [json.dumps(self.t960, sort_keys=True)])
        # cache_plan now deploys it for the 960 instance of that IR (the rejected winner stays out)
        plan = cache_plan(g, self.cache, "sm_89", SPECS)
        used = {k: v["impl_id"] for k, v in plan["used"].items()}
        self.assertIn(e["impl_id"], used.values())
        self.assertNotIn("w960", used.values())

    def test_route_keeps_the_incumbent_when_nothing_passes_or_beats_it(self):
        g = conv_graph([self.t960])
        n = len(self.cache.entries)
        rep = sf.route_ops(kind="conv2d_f16", ops=["conv0"], graph=g, src_ir=self.d, cache=self.cache, specs=SPECS,
                           arch="89", ir_name="dn", work=self.d / "w", measure_fn=self._measure({}))
        self.assertEqual(rep["result"], "no candidate passes and beats the incumbent")
        self.assertEqual(len(self.cache.entries), n)
        rep = sf.route_ops(kind="conv2d_f16", ops=["conv0"], graph=g, src_ir=self.d, cache=self.cache, specs=SPECS,
                           arch="89", ir_name="dn", work=self.d / "w",
                           measure_fn=self._measure({"// k480": (True, 990.0)}))     # < 3% better than the seed
        self.assertEqual(len(self.cache.entries), n)
        bad = sf.route_ops(kind="conv2d_f16", ops=["conv0"], graph=g, src_ir=self.d, cache=self.cache, specs=SPECS,
                           arch="89", ir_name="dn", work=self.d / "w",
                           measure_fn=lambda *a: (_ for _ in ()).throw(RuntimeError("no GPU")))
        self.assertEqual(bad["result"], "measure failed")

    def test_fallback_callback(self):
        g = conv_graph([self.t960, self.t64])
        log = []
        cb = sf.fallback_router(enabled=True, log=log,
                                measure_fn=self._measure({"// k928": (True, 70.0), "// k480": (True, 60.0)}))
        self.assertIsNone(sf.fallback_router(enabled=False))
        man = {"ir_name": "dn", "ops": [{"name": "conv0", "slot": "conv2d_f16__v0"},
                                        {"name": "conv1", "slot": "conv2d_f16__v1"}]}
        res = {"passed": False, "errors": ["conv0 (slot conv2d_f16__v0): invalid argument"]}
        out = cb(ir_dir=self.d, graph=g, bad={"conv2d_f16__v0": self.e960}, man=man, res=res, cache=self.cache,
                 specs=SPECS, arch="89", work=self.d / "prof", attempt=0)
        self.assertEqual(out[0]["result"], "routed")
        self.assertEqual(out[0]["ops"], ["conv0"])
        e = self.cache.entries[-1]
        self.assertEqual(e["source"]["why"]["errors"], res["errors"])
        self.assertEqual(log[0]["ir"], "dn")
        # a stage kernel of a fused slot is not routed (the fused slot re-seeds from the remaining stage winners)
        self.assertIsNone(cb(ir_dir=self.d, graph=g, bad={"fused__x/conv2d_f16": self.e960}, man=man, res=res,
                             cache=self.cache, specs=SPECS, arch="89", work=self.d / "prof", attempt=1))


class RequeueTest(unittest.TestCase):
    def _profile(self, seed_tag):
        from modelblaster.pipeline.kb_ingest import signature_sha
        from modelblaster.pipeline.kernel_abi import parse_signature
        spec = SPECS["conv2d_f16"]
        rec = {"name": "conv0", "op": "conv2d_f16", "slot": "conv2d_f16__v0",
               "entry": parse_signature(spec.signature)[0], "signature_sha": signature_sha(spec.signature),
               "shape": conv_shape(960, 7), "us": 954.0, "seed": seed_tag}
        return {"ir": "dn", "ir_dir": "/x", "ops": [rec],
                "cache_fallback": [{"attempt": 0, "rejected": {"conv2d_f16__v13": "w960"},
                                    "errors": ["conv0 (slot conv2d_f16__v13): invalid argument"],
                                    "ops": {"conv2d_f16__v13": ["conv0"]}}]}

    def test_requeue_only_stuck_items_and_at_most_n_times(self):
        from modelblaster.pipeline.kb_plan import rank
        p = self._profile("hand")
        key = rank([p])[0]["key"]
        state = {}
        keys, notes = sf.requeue([p], {key}, state, 1, max_per_key=1)
        self.assertEqual(keys, {key})
        self.assertEqual(notes[key]["rejected"], ["w960"])
        self.assertIn("invalid argument", notes[key]["errors"][0])
        k2, _ = sf.requeue([p], {key}, state, 1, max_per_key=1)        # the same round resumed: still re-queued
        self.assertEqual(k2, {key})
        k3, _ = sf.requeue([p], {key}, state, 2, max_per_key=1)        # a later round: the budget is spent
        self.assertEqual(k3, set())
        self.assertEqual(state["requeued"][key], {"n": 1, "rounds": [1]})
        # routed (now a cached kernel) or never sent: nothing to re-queue
        self.assertEqual(sf.requeue([self._profile("cache:route:abc")], {key}, {}, 1)[0], set())
        self.assertEqual(sf.requeue([p], set(), {}, 1)[0], set())

    def test_note_reaches_the_slot_semantics(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t) / "p0"
            d.mkdir()
            man = {"slots": [{"id": "conv2d_f16", "semantics": "conv", "instances": [{"op_name": "conv0"}]},
                             {"id": "relu_f16", "semantics": "relu", "instances": [{"op_name": "r"}]}]}
            (d / "manifest.json").write_text(json.dumps(man))
            notes = {"K": {"op": "conv0", "rejected": ["w960"], "errors": ["conv0: invalid argument"]}}
            out = sf.annotate_requeued([d], {"problems": [{"keys": ["K"]}]}, notes)
            m = json.loads((d / "manifest.json").read_text())
            self.assertEqual(out, {"p0": ["conv2d_f16"]})
            self.assertIn("FAILED when the whole model", m["slots"][0]["semantics"])
            self.assertIn("w960", m["slots"][0]["semantics"])
            self.assertEqual(m["slots"][1]["semantics"], "relu")
            self.assertEqual(m["requeued"][0]["op"], "conv0")


class GuardTest(unittest.TestCase):
    def test_seed_left_rules(self):
        ops = [{"name": "a", "op": "conv2d_f16", "us": 500.0, "seed_source": "hand"},          # on seed, 50%
               {"name": "b", "op": "relu_f16", "us": 200.0, "seed_source": "cache:x"},         # 1.0x its seed, 20%
               {"name": "c+d", "op": "fused__x", "us": 150.0, "seed_source": "cache:y"},       # fused, 2x
               {"name": "e", "op": "linear_f16", "us": 140.0, "seed_source": "cache:z"},       # 14%, no seed time
               {"name": "f", "op": "relu_f16", "us": 10.0, "seed_source": "hand"}]             # 1%: below share
        seeds = {"a": 600.0, "b": 210.0, "c": 200.0, "d": 100.0, "f": 10.0}
        fl = sf.seed_left(ops, seeds, share=0.10, min_speedup=1.25)
        self.assertEqual([f["name"] for f in fl], ["a", "b"])
        self.assertTrue(fl[0]["on_seed_kernel"])
        self.assertFalse(fl[1]["on_seed_kernel"])
        self.assertAlmostEqual(fl[1]["speedup_vs_seed"], 1.05)
        self.assertEqual(sf.seed_left(ops, seeds, share=0.6), [])
        self.assertEqual([f["name"] for f in sf.seed_left(ops, seeds, share=0.10, min_speedup=2.5)], ["a", "b", "c+d"])

    def test_densenet121_sweep_data(self):
        """The recorded final DenseNet121 (sweep) as the guard sees it: the per-op times of the last profile, the
        final slot tags, the first profile's seed times -> only the 960x7x7 conv is flagged (on the seed)."""
        d = json.loads((DATA / "seed_left_15_densenet121.json").read_text())
        fl = sf.seed_left(d["ops"], d["seeds"], share=0.10, min_speedup=1.25)
        self.assertEqual([f["name"] for f in fl], ["dense_blocks.3.layers.14.2"])
        self.assertTrue(fl[0]["on_seed_kernel"])
        self.assertGreater(fl[0]["share"], 0.2)

    def test_final_guard_report_and_reasons(self):
        with tempfile.TemporaryDirectory() as t:
            work = Path(t)
            (work / "r00" / "profile" / "m").mkdir(parents=True)
            (work / "r01" / "profile" / "m").mkdir(parents=True)
            (work / "r00" / "profile" / "m" / "profile.json").write_text(json.dumps(
                {"ops": [{"name": "a", "us": 900.0, "seed": "hand"}, {"name": "b", "us": 100.0, "seed": "hand"}]}))
            (work / "r01" / "profile" / "m" / "profile.json").write_text(json.dumps(
                {"ops": [], "cache_fallback": [{"ops": {"conv2d_f16__v1": ["a"]}}]}))
            final = {"m": {"ir": "m", "passed": True, "slots": {},
                           "ops": [{"name": "a", "op": "conv2d_f16", "us": 800.0, "seed_source": "hand"},
                                   {"name": "b", "op": "relu_f16", "us": 20.0, "seed_source": "cache:q"}]}}

            class C:
                arch, final_guard = "89", "warn"
            fx = sf.Fixes(C(), work)
            rep = fx.final_guard([work / "m"], final, {"sent": []}, None, SPECS)
            self.assertEqual(rep["n_flagged"], 1)
            f = rep["irs"]["m"]["flagged"][0]
            self.assertEqual(f["name"], "a")
            self.assertTrue(f["reason"].startswith("profile-fallback"))
            C.final_guard = "off"
            self.assertIsNone(sf.Fixes(C(), work).final_guard([work / "m"], final, {}, None, SPECS))


class NearestSeedTest(unittest.TestCase):
    def test_nearest_candidates_in_the_manifest(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            cache = CudaCache(d / "cache")
            add(cache, [conv_shape(928, 7)], "// k928", 40.0, "w928")
            add(cache, [conv_shape(64, 56)], "// k64", 20.0, "w64")
            add(cache, [conv_shape(960, 7)], "// mine", 20.0, "w960")
            p = d / "p"
            (p / "slots").mkdir(parents=True)
            (p / "slots" / "conv2d_f16.cu").write_text("// seed")
            (p / "slots" / "conv2d_f16.cand0.cu").write_text("// k64")         # A1's candidate (same code)
            man = {"ir_name": "dn", "slots": [
                {"id": "conv2d_f16", "op": "conv2d_f16", "file": "slots/conv2d_f16.cu", "seed_source": "hand",
                 "instances": [{"op_name": "c", "shape": conv_shape(896, 7)}]}],
                "seed_candidates": {"conv2d_f16": [{"file": "slots/conv2d_f16.cand0.cu", "from": "cache:w64"}]}}
            (p / "manifest.json").write_text(json.dumps(man))
            out = sf.add_nearest_seed_candidates(p, cache, arch="89", specs=SPECS, n=2)
            m = json.loads((p / "manifest.json").read_text())
            lst = m["seed_candidates"]["conv2d_f16"]
            self.assertEqual([c["from"] for c in lst], ["cache:w928", "cache:w960", "cache:w64"])
            self.assertEqual((p / lst[0]["file"]).read_text(), "// k928")
            self.assertEqual([x[0] for x in out["conv2d_f16"]], ["cache:w928", "cache:w960"])
            # a slot seeded from its own cached winner is left alone
            man["slots"][0]["seed_source"] = "cache:w960"
            man.pop("seed_candidates")
            (p / "manifest.json").write_text(json.dumps(man))
            self.assertEqual(sf.add_nearest_seed_candidates(p, cache, arch="89", specs=SPECS), {})


def _have_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


@unittest.skipUnless(_have_torch(), "needs torch (CPU) for the export")
class ProfileHookTest(unittest.TestCase):
    """kb_plan.profile_ir calls on_fallback after rejecting a failing cached winner, records the failed slot's ops,
    and the next attempt deploys whatever the callback cached."""

    def test_callback_and_ops_record(self):
        from modelblaster.pipeline.kb_plan import profile_ir
        from modelblaster.pipeline.tests.test_graphopt_remote import FakeEval, _make_cache
        from modelblaster.pipeline.tests.test_kb_model_run_eval import _tiny_ir

        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            ir = _tiny_ir(d)
            cache, specs, sid = _make_cache(ir, d, "c")

            class FailFirst(FakeEval):
                def run(self, jobs, **kw):
                    self.slots_bad = (sid,) if not self.calls else ()
                    return super().run(jobs, **kw)
            seen = []

            def cb(**kw):
                seen.append(kw)
                return [{"result": "routed", "kind": "x"}]
            p = profile_ir(ir, d / "prof", cache=cache, arch="89", runs=1, kbeval=FailFirst(), specs=specs,
                           on_fallback=cb)
            self.assertTrue(p["passed"])
            fb = p["cache_fallback"][0]
            self.assertEqual(list(fb["ops"]), [sid])
            self.assertTrue(fb["ops"][sid])
            self.assertEqual(fb["route"], [{"result": "routed", "kind": "x"}])
            self.assertEqual(set(seen[0]["bad"]), {sid})
            self.assertEqual(seen[0]["attempt"], 0)
            # stuck on the seed after the fallback -> a re-queue candidate
            self.assertEqual(set(sf.stuck_after_fallback(p)), set(fb["ops"][sid]))


class CacheBesideRejectedTest(unittest.TestCase):
    """A re-optimized winner of a shape whose faster winner was rejected in the whole model is kept beside it (the
    live run L1c: the re-queued problem's 1585 us winner was dropped as kept-existing next to the rejected 0.001 us
    entry, so the shape stayed on its seed); the same code again is no change; kill switch MB_CACHE_BESIDE_REJECTED=0."""

    def test_beside_rejected(self):
        t = conv_shape(960, 7)
        with tempfile.TemporaryDirectory() as d:
            cache = CudaCache(Path(d) / "c")
            bad = add(cache, [t], "// bad", 1.0, "bad")
            bad["rejected"].append("dn")
            with mock.patch.dict(os.environ, {"MB_CACHE_BESIDE_REJECTED": "0"}):
                _, st = cache.add(op="conv2d_f16", entry="kernel_conv2d_f16", signature=SPECS["conv2d_f16"].signature,
                                  arch="sm_89", shapes=[t], code="// new", slot_us=8.0, speedup=9.0, impl_id="n0",
                                  source={})
                self.assertEqual(st, "kept-existing")                 # legacy: dropped
            kw = dict(op="conv2d_f16", entry="kernel_conv2d_f16", signature=SPECS["conv2d_f16"].signature,
                      arch="sm_89", shapes=[t], speedup=9.0, source={})
            e, st = cache.add(code="// new", slot_us=8.0, impl_id="n1", **kw)
            self.assertEqual(st, "added-beside-rejected")
            self.assertEqual(cache.add(code="// new", slot_us=8.0, impl_id="n1", **kw)[1], "kept-existing")
            self.assertEqual(cache.add(code="// bad", slot_us=1.0, impl_id="bad", **kw)[1], "kept-existing")
            self.assertEqual(cache.add(code="// slower", slot_us=9.0, impl_id="n2", **kw)[1], "kept-existing")
            self.assertEqual(cache.add(code="// faster", slot_us=7.0, impl_id="n3", **kw)[1], "replaced")
            self.assertEqual(len(cache.entries), 2)
            sig = SPECS["conv2d_f16"].signature
            self.assertEqual(cache.lookup("kernel_conv2d_f16", sig, "sm_89", [t], ir_name="dn")["impl_id"], "n3")
            self.assertEqual(cache.lookup("kernel_conv2d_f16", sig, "sm_89", [t], ir_name="other")["impl_id"], "bad")


class IngestRequeuedTest(unittest.TestCase):
    """KernelBlaster's winners export keeps the fastest record per shape set -- the rejected kernel -- so a re-queued
    problem's slower working winner is ingested from its own result (best/slots) and kept beside the rejected entry."""

    def test_requeued_best_reaches_the_cache(self):
        t = conv_shape(960, 7)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cache = CudaCache(d / "c")
            bad = add(cache, [t], "// bad", 8.19, "bad")
            bad["rejected"].append("dn")
            pdir, rdir = d / "problems" / "p0", d / "run" / "p0__r01"
            pdir.mkdir(parents=True)
            (rdir / "best" / "slots").mkdir(parents=True)
            (pdir / "manifest.json").write_text(json.dumps({"dtype": "fp16", "slots": [
                {"id": "conv2d_f16__v61", "op": "conv2d_f16", "instances": [{"op_name": "c", "shape": t}]}]}))
            (rdir / "best" / "slots" / "conv2d_f16__v61.cu").write_text("// requeued winner")
            (rdir / "result.json").write_text(json.dumps({"problem_dir": str(pdir), "final_ok": True, "slots": {
                "conv2d_f16__v61": {"best_us": 12.66, "baseline_us": 955.0, "impl_id": "new1"}}}))

            class C:
                arch = "89"
            fx = sf.Fixes(C(), d)
            out = fx.ingest_requeued([{"_dir": str(rdir)}], cache, SPECS, 1)
            self.assertEqual([(x["impl_id"], x["status"]) for x in out], [("new1", "added-beside-rejected")])
            self.assertIsNone(fx.ingest_requeued([{"_dir": str(rdir)}], cache, SPECS, 1) and None)
            e = cache.lookup("kernel_conv2d_f16", SPECS["conv2d_f16"].signature, "sm_89", [t], ir_name="dn")
            # a live (not rejected) entry of these shapes exists now: the normal ingest path applies again
            self.assertIsNone(fx.ingest_requeued([{"_dir": str(rdir)}], cache, SPECS, 2))
            C.fallback_repair = "off"
            self.assertIsNone(sf.Fixes(C(), d).ingest_requeued([{"_dir": str(rdir)}], cache, SPECS, 1))
            self.assertEqual(e["impl_id"], "new1")


class DefaultsTest(unittest.TestCase):
    def test_defaults_and_round_key(self):
        from modelblaster.pipeline.kb_graph_opt import Config
        cfg = Config(kb=Path("."))
        fx = sf.Fixes(cfg, Path("/nonexistent"))
        self.assertEqual((fx.fallback, fx.guard, fx.nearest, fx.resync), ("on", "warn", "off", 0.0))
        self.assertTrue(fx.routing and fx.requeueing)
        self.assertEqual(fx.round_key(), {})                       # existing work dirs keep their key
        cfg.fallback_repair, cfg.seed_nearest, cfg.rollout_resync = "off", "on", 2.0
        fx = sf.Fixes(cfg, Path("/nonexistent"))
        self.assertFalse(fx.routing or fx.requeueing)
        self.assertIsNone(fx.on_fallback())
        self.assertEqual(fx.round_key(), {"fallback_repair": "off", "seed_nearest": "on", "rollout_resync": 2.0})
        self.assertEqual(fx.exclude({"k"}, [], {}, 0, {}, SPECS), {"k"})
        cfg.final_guard = "bogus"
        with self.assertRaises(ValueError):
            sf.Fixes(cfg, Path("."))

    def test_lineage_env(self):
        from modelblaster.pipeline.kb_graph_opt import Config
        cfg = Config(kb=Path("."))
        cfg.rollout_accept = "lineage"
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("KERNELBLASTER_MULTISLOT_ROLLOUT_ACCEPT", None)
            os.environ.pop("KERNELBLASTER_MULTISLOT_ROLLOUT_RESYNC", None)
            sf.Fixes(Config(kb=Path(".")), Path(".")).apply_env()
            self.assertNotIn("KERNELBLASTER_MULTISLOT_ROLLOUT_ACCEPT", os.environ)
            sf.Fixes(cfg, Path(".")).apply_env()
            self.assertEqual(os.environ["KERNELBLASTER_MULTISLOT_ROLLOUT_ACCEPT"], "lineage")
            self.assertNotIn("KERNELBLASTER_MULTISLOT_ROLLOUT_RESYNC", os.environ)


if __name__ == "__main__":
    unittest.main()
