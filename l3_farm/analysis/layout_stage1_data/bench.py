#!/usr/bin/env python3
"""Sanity timing (NOT a performance study): tiled seed vs naive seed vs torch copy+bmm (strided_copy of A and B to
contiguous, cuBLAS bmm, write-back through the C strides) at the real fold shapes. usage: bench.py NEW.so NAIVE.so REAL.json OUT.json"""
import json, sys, statistics, torch
import kernel_lib as KL

def tm(fn, n=10, warm=3):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); e.synchronize(); ts.append(s.elapsed_time(e) * 1000)
    return statistics.median(ts)

def main():
    new, naive = KL.load(sys.argv[1]), KL.load(sys.argv[2])
    cases = json.load(open(sys.argv[3]))
    rows = []
    for c in cases:
        v = KL.views(c)
        ext = {k: KL.extent(o, d, s)[1] for k, (d, s, o) in v.items()}
        A = (torch.rand(ext["A"], device="cuda") * 2 - 1).half()
        B = (torch.rand(ext["B"], device="cuda") * 2 - 1).half()
        C = torch.zeros(ext["C"], device="cuda", dtype=torch.float16)
        (da, sa, oa), (db, sb, ob), (dc, sc, oc) = v["A"], v["B"], v["C"]
        Av, Bv, Cv = torch.as_strided(A, da, sa, oa), torch.as_strided(B, db, sb, ob), torch.as_strided(C, dc, sc, oc)
        flops = 2.0 * c["B0"] * c["B1"] * c["M"] * c["K"] * c["N"]
        r = dict(problem=c["problem"], op=c["op"], gflop=flops / 1e9)
        r["tiled_us"] = tm(lambda: KL.call(new, A, B, C, c))
        try:
            r["naive_us"] = tm(lambda: KL.call(naive, A, B, C, c), n=3, warm=1)
        except Exception as e:
            r["naive_us"] = None
        def copies():
            return Av.reshape(-1, c["M"], c["K"]).contiguous(), Bv.reshape(-1, c["K"], c["N"]).contiguous()
        Ac, Bc = copies()
        r["copy_only_us"] = tm(copies)
        r["bmm_only_us"] = tm(lambda: torch.bmm(Ac, Bc))
        def full():
            a, b = copies(); o = torch.bmm(a, b); Cv.copy_(o.view(dc))
        r["copy_bmm_us"] = tm(full)
        r["tiled_over_copy_bmm"] = r["tiled_us"] / r["copy_bmm_us"]
        r["tiled_tflops"] = flops / r["tiled_us"] / 1e6
        rows.append(r); print(json.dumps(r), flush=True)
    json.dump(rows, open(sys.argv[4], "w"), indent=1)

if __name__ == "__main__":
    main()
