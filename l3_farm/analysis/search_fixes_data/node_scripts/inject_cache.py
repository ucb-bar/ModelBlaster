#!/usr/bin/env python3
"""Replace the cached winner of a work dir's largest linear op (by its round-0 profile) with a kernel that fails at
launch (4096 threads per block -> invalid argument) -- a stand-in for a winner that passed its own problem but fails in
the whole model. usage: inject_cache.py WORK [KIND]"""
import json
import sys
from pathlib import Path

from modelblaster.pipeline.cuda_dtypes import cuda_signature
from modelblaster.pipeline.kb_ingest import CudaCache
from modelblaster.pipeline.kb_proposals import _all_specs
from modelblaster.pipeline.kernel_abi import parse_signature

work = Path(sys.argv[1])
kind = sys.argv[2] if len(sys.argv) > 2 else "linear_f16"
prof = json.loads(next(work.glob("r00/profile/*/profile.json")).read_text())
op = max((o for o in prof["ops"] if o["op"] == kind), key=lambda o: o["us"] or 0)
specs = _all_specs()
sig = cuda_signature(specs[kind].signature)
entry, _ = parse_signature(specs[kind].signature)
bad = ("#include <cuda_fp16.h>\n#include <cuda_runtime.h>\n__global__ void mb_bad_k(int x) {}\n"
       f"{sig} {{\n  mb_bad_k<<<1, 4096>>>(0);   // deliberately invalid launch\n}}\n")
cache = CudaCache(work / "cache")
e, st = cache.add(op=kind, entry=entry, signature=specs[kind].signature, arch="sm_89", shapes=[op["shape"]], code=bad,
                  slot_us=0.001, speedup=1000.0, impl_id="inject:bad", source={"by": "a3-inject"})
cache.save()
print(json.dumps({"op": op["name"], "shape": op["shape"], "status": st}))
