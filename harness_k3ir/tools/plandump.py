"""Dump a k3ir plan: op index, kind, ins, outs, ip[:24] (attention-relevant).
    python plandump.py PLAN [FIRST LAST]"""
import sys, json, numpy as np
KINDS = json.loads(open(sys.argv[1].replace('.plan','.names.json')).read())["op_kinds"]
w = np.fromfile(sys.argv[1], np.int64)
nt, no, ni, nout = w[1:5]; p = 7
T = w[p:p+5*nt].reshape(nt,5); p += 5*nt
ins = w[p:p+ni]; p += ni; outs = w[p:p+nout]; p += nout
OPW = 3+8+4+6+40+4
ops = w[p:p+OPW*no].reshape(no, OPW)
names = json.load(open(sys.argv[1].replace('.plan','.names.json')))
lo, hi = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, no)
for i in range(lo, min(hi, no)):
    r = ops[i]; k, a, b = r[0], r[1], r[2]
    ii = list(r[3:3+a]); oo = list(r[11:11+b]); ip = list(r[21:21+24]); fp = r[61:65].view(np.float64)
    print(i, names["op_kinds"][i], ii, oo, [T[t][2] for t in oo], ip[:22], fp[:2].tolist(), names["ops"][i][:50])
