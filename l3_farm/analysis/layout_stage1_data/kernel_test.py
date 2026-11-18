#!/usr/bin/env python3
"""Standalone correctness of kernel_bmm_strided_f16 (the tiled seed) vs an fp64 torch reference.
usage: kernel_test.py LIB.so REAL.json [--small N] [--out results.json] [--seed S]
Bound: |got - ref| <= 2^-11 |ref| + 2^-25 + 1e-6 sum|a||b| (one fp16 rounding of the fp64-exact result + fp32 accumulation noise);
unwritten C elements (outside the image of the C map, incl. guard pads) must keep their sentinel."""
import json, random, sys, time, torch
import kernel_lib as KL

def rand_layout(rng, dims, names, unit=None, pad_ok=True, bcast=False):
    """random strides for dims: a random axis order (innermost first) with random padding; optional broadcast (stride 0
    on a batch dim of A/B); unit: force this axis innermost with stride 1 (None = any)."""
    n = len(dims)
    order = list(range(n)); rng.shuffle(order)
    if unit is not None:
        order.remove(unit); order.insert(0, unit)
    st, cur = [0] * n, 1
    for ax in order:
        st[ax] = cur
        cur *= max(dims[ax], 1) + (rng.choice([0, 0, 1, 3, 8, 16]) if pad_ok else 0)
    if bcast:
        for ax in (0, 1):
            if rng.random() < 0.3:
                st[ax] = 0
    return st

def small_cases(n, seed):
    rng = random.Random(seed)
    sizes = [1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 129, 200]
    out = []
    while len(out) < n:
        c = dict(B0=rng.choice([1, 1, 2, 3]), B1=rng.choice([1, 2, 5]), M=rng.choice(sizes), K=rng.choice(sizes + [0]),
                 N=rng.choice(sizes))
        # unit-stride patterns: mostly the row-major / col-major pairs; sometimes none (random order)
        pat = rng.choice(["rr", "rc", "cr", "cc", "any", "any"])
        ua = {"r": 3, "c": 2}.get(pat[0]) if pat != "any" else None          # A axes (b0,b1,m,k): k=3 row-major, m=2
        ub = {"r": 3, "c": 2}.get(pat[1]) if pat != "any" else None          # B axes (b0,b1,k,n): n=3 row-major, k=2
        uc = rng.choice([3, 2, None])
        sa = rand_layout(rng, [c["B0"], c["B1"], c["M"], c["K"]], "", ua, bcast=True)
        sb = rand_layout(rng, [c["B0"], c["B1"], c["K"], c["N"]], "", ub, bcast=True)
        sc = rand_layout(rng, [c["B0"], c["B1"], c["M"], c["N"]], "", uc)
        # C must be injective: pad_ok layouts with unit dims of size 1 are fine; require non-broadcast (done: bcast False)
        off = lambda: rng.choice([0, 0, 1, 3, 8, 9, 16, 24])
        c.update(a_off=off(), a_s0=sa[0], a_s1=sa[1], a_sm=sa[2], a_sk=sa[3],
                 b_off=off(), b_s0=sb[0], b_s1=sb[1], b_sk=sb[2], b_sn=sb[3],
                 c_off=off(), c_s0=sc[0], c_s1=sc[1], c_sm=sc[2], c_sn=sc[3])
        out.append(c)
    return out

def main():
    a = sys.argv[1:]
    so, real = a[0], ([] if "--no-real" in a else json.load(open(a[1])))
    n = int(a[a.index("--small") + 1]) if "--small" in a else 300
    seed = int(a[a.index("--seed") + 1]) if "--seed" in a else 1
    outp = a[a.index("--out") + 1] if "--out" in a else None
    lib = KL.load(so)
    res = {"real": [], "small": []}
    for c in real:
        t = time.time(); r = KL.run_case(lib, c, seed=seed)
        r.update(problem=c["problem"], op=c["op"], **{k: c[k] for k in KL.ARGS}); res["real"].append(r)
        print(f"REAL {c['problem']:28s} {c['op'][:28]:28s} M{c['M']} K{c['K']} N{c['N']} B{c['B0']}x{c['B1']}: "
              f"{'PASS' if r['ok'] else 'FAIL'} ratio={r['max_ratio']:.3f} oob={r.get('oob')} ({time.time()-t:.1f}s)", flush=True)
    fails = 0
    for i, c in enumerate(small_cases(n, seed)):
        try:
            r = KL.run_case(lib, c, seed=i)
        except AssertionError as e:       # an invalid random layout (negative extent): skip
            continue
        r.update(**{k: c[k] for k in KL.ARGS}); res["small"].append(r)
        if not r["ok"]:
            fails += 1
            print("FAIL small", c, r, flush=True)
    npass = sum(r["ok"] for r in res["small"])
    print(f"SMALL: {npass}/{len(res['small'])} pass; REAL: {sum(r['ok'] for r in res['real'])}/{len(res['real'])} pass; "
          f"max small ratio {max([r['max_ratio'] for r in res['small']] or [0]):.3f}")
    if outp:
        json.dump(res, open(outp, "w"), indent=1)
    sys.exit(0 if fails == 0 and all(r["ok"] for r in res["real"]) else 1)

if __name__ == "__main__":
    main()
