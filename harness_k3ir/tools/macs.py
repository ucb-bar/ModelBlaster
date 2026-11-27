"""MACs per k3ir plan: int8 linears (M K N), bmm (B0 B1 M K N), sdpa (BH S S D x2), conv2d, im2col-linears counted as linears.
    python macs.py PLAN [PLAN ...]   (reads PLAN with .plan -> .names.json)"""
import sys, json, numpy as np
for f in sys.argv[1:]:
    w = np.fromfile(f, np.int64); nt, no = int(w[1]), int(w[2]); ni, nout = int(w[3]), int(w[4]); p = 7 + 5 * nt + ni + nout
    OPW = 3 + 8 + 4 + 6 + 40 + 4; ops = w[p:p + OPW * no].reshape(no, OPW); kinds = json.load(open(f.replace('.plan', '.names.json')))["op_kinds"]
    i8 = f32 = 0
    for k, r in zip(kinds, ops):
        ip = r[21:61]
        if k in ("linear_dq_s8", "linear_dq_add_s8"): i8 += ip[0] * ip[1] * ip[2]
        elif k == "linear3_dq_s8": i8 += ip[0] * ip[1] * (ip[2] + ip[3] + ip[4])
        elif k == "bmm_strided": f32 += ip[0] * ip[1] * ip[2] * ip[3] * ip[4]
        elif k == "sdpa": f32 += 2 * ip[0] * ip[1] * ip[1] * ip[2]
        elif k in ("linear", "matmul"): f32 += ip[0] * ip[1] * ip[2]
        elif k == "conv2d": f32 += ip[0] * ip[4] * ip[5] * ip[6] * ip[1] * ip[7] * ip[8]
        elif k == "conv_transpose2d": f32 += ip[0] * ip[1] * ip[2] * ip[3] * ip[4] * ip[7] * ip[8]
    print(f, f"int8_GMAC={i8/1e9:.3f} fp32/attn_GMAC={f32/1e9:.3f}")
