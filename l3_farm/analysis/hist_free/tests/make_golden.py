"""Generate golden_off.json from a PRISTINE KernelBlaster tree:
   PYTHONPATH=<pristine>/src python make_golden.py <fixture_dir> out.json"""
import json, sys
from kernelblaster.agents.opt_multislot import build_codegen_messages
from kernelblaster.knowledge.slots import SlotProblem
import golden_cases
p = SlotProblem(sys.argv[1])
out = {name: build_codegen_messages(problem=p, sid=sid, **kw) for name, (sid, kw) in golden_cases.cases(p).items()}
json.dump(out, open(sys.argv[2], "w"), indent=1, sort_keys=True)
