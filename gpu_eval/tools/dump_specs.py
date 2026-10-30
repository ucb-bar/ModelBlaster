"""Export ModelBlaster's KERNEL_SPECS (signature + reference_impl) to the
specs.json the GPU harnesses read; the GPU run scripts call this when no
specs.json is given.

    python dump_specs.py OUT.json
"""
import json
import sys
from pathlib import Path

# import this checkout's modelblaster package (its parent dir is the import root)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from modelblaster.pipeline.reference_kernels import KERNEL_SPECS

json.dump({op: {"signature": s.signature, "reference_impl": s.reference_impl}
           for op, s in KERNEL_SPECS.items()}, open(sys.argv[1], "w"), indent=1)
print(len(KERNEL_SPECS), "specs ->", sys.argv[1])
