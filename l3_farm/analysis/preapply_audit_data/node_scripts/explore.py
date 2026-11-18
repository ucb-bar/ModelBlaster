import json, sys
from pathlib import Path
from modelblaster.pipeline import graph_hints as gh, preapply_audit as pa
from modelblaster.pipeline.kb_ingest import CudaCache
from modelblaster.pipeline.kb_proposals import _all_specs
ir = Path("/tmp/krisd/pa/irs/8_ResNetBasicBlock")
g0 = json.loads((ir / "graph.json").read_text())
specs = _all_specs()
h = gh.fuse_hint("fused__conv2d_f16__batchnorm2d_f16", None, {"by": "t"}, "accepted")
inc, added, prov = gh.apply_hints(g0, [h], dict(specs))
print([ (o["name"], o["op"]) for o in inc["ops"]], prov["applied"][:1])
sp = {**specs, **added}
cache = CudaCache(Path("/tmp/krisd/pa/explore/cache"))
p = pa.compose_model(ir, inc, sp, cache, Path("/tmp/krisd/pa/explore/inc"), "89")
man = json.loads((Path(p) / "manifest.json").read_text())
print(p, [ (s["id"], s["op"], s["file"]) for s in man["slots"]])
for s in man["slots"]:
    if s["op"].startswith("fused__"):
        t = (Path(p) / s["file"]).read_text()
        print(len(t)); print(t[3000:])
