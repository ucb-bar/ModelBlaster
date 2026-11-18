"""Interleaved flow, is the hint-status difference halfway-on vs off larger than off vs off? (stub LLM, tiny model, GPU)"""
import json, sys
from modelblaster.pipeline.tests import test_kb_halfway_flow as T
T.HalfwayFlowTest.setUpClass()
t = T.HalfwayFlowTest("test_rounds_flow")
kw = dict(graph_search="interleaved", graph_priors="off", graph_characterize=1, graph_char_top=2)
res = {}
for name, hw in (("offA", False), ("onA", True), ("offB", False), ("onB", True), ("offC", False), ("onC", True)):
    s, w, st, dt = t._run(name, hw, **kw)
    ir = next(iter(s["irs"].values()))
    res[name] = {"status": st, "passed": ir["passed"], "dt": round(dt)}
    print(name, json.dumps(res[name]), flush=True)
T.HalfwayFlowTest.tearDownClass()
