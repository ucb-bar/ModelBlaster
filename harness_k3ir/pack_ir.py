"""Pack a ModelBlaster IR (graph.json + weights.npz, gpu_eval's mbir format; the int8_mixed IRs of
smolvla_split and every fp32 FX extract) into the flat plan + weight blob that the K3 IR runtime
(harness_k3ir/src/k3ir.c) executes.

    python pack_ir.py IR_DIR OUT_PREFIX [--blob SHARED_BLOB_PREFIX]

Writes OUT_PREFIX.plan (int64 words), OUT_PREFIX.wblob (the ops' weights, 64-byte aligned; identical
for IRs that share weights, e.g. one graph extracted at several prompt lengths), OUT_PREFIX.cblob (the
IR's const tensors: masks, RoPE tables, smoothing factors, indices) and OUT_PREFIX.names.json (op /
tensor names, for diagnostics and feed / dump files).

Plan layout (all little-endian int64):
  magic 0x4d424b3349520001, n_tensors, n_ops, n_inputs, n_outputs, wblob_bytes, cblob_bytes
  tensors: n_tensors x [dtype, kind, nelem, a, b]          dtype 0 f32 / 1 i8 / 2 i32
           kind 0 = buffer, 1 = const (a = cblob byte offset), 2 = alias (a = base tensor, b = elem offset)
  inputs:  n_inputs tensor ids;  outputs: n_outputs tensor ids
  ops:     n_ops x OPW words: kind, n_in, n_out, in[8], out[4], w[6], ip[40], fp[4] (fp as float64 bits)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

MAGIC = 0x4D424B3349520001
KINDS = ["view", "strided_copy", "catn", "ewise1", "ewise2", "mul_scalar", "add", "softmax", "layer_norm",
         "rms_norm", "rope_half", "silu_mul", "gelu_exact", "linear", "bmm_strided", "conv2d", "index_select",
         "quantize_rows_s8", "linear_dq_s8", "linear3_dq_s8", "linear_dq_add_s8", "gelu", "ewise_scalar",
         "silu", "relu", "matmul", "layer_norm_noaffine", "attention", "softmax_dim", "mean_dim", "sum_dim",
         "softplus", "tanh", "mul", "sdpa", "conv_transpose2d", "sigmoid", "im2col",
         "cast", "maxpool2d", "upsample_nearest", "gather", "conv2d_grouped", "cumsum", "strided_scatter", "reduce"]
KID = {k: i for i, k in enumerate(KINDS)}
NIN, NOUT, NW, NIP, NFP = 8, 4, 6, 40, 4
OPW = 3 + NIN + NOUT + NW + NIP + NFP
DT = {"f32": 0, "i8": 1, "i32": 2, "i64": 2, "u8": 1, "b8": 1, "f16": 3}
ESZ = {0: 4, 1: 1, 2: 4, 3: 2}
IP_WN, IP_WF16 = 32, 39     # fp16 ops: weight element counts, binary16-weight bitmask (k3ir.h)


def _f64bits(x: float) -> int:
    return struct.unpack("<q", struct.pack("<d", float(x)))[0]


class Blob:
    def __init__(self):
        self.parts, self.size, self.index = [], 0, {}

    def add(self, arr: np.ndarray) -> int:
        b = np.ascontiguousarray(arr).tobytes()
        h = hashlib.sha1(b).hexdigest() + str(arr.dtype)
        if h in self.index:
            return self.index[h]
        pad = (-self.size) % 64
        if pad:
            self.parts.append(b"\0" * pad)
            self.size += pad
        off = self.size
        self.parts.append(b)
        self.size += len(b)
        self.index[h] = off
        return off


def pack(ir_dir: str, out: str) -> dict:
    d = Path(ir_dir)
    g = json.loads((d / "graph.json").read_text())
    wz = np.load(d / "weights.npz")
    W = {k: wz[k] for k in wz.files}
    T = g["tensors"]
    tid = {n: i for i, n in enumerate(T)}
    blob, cblob = Blob(), Blob()

    def nel(n):
        return int(np.prod(T[n]["shape"])) if T[n]["shape"] else 1

    trec = []
    for n, m in T.items():
        dt = DT[m.get("dtype", "f32")]
        if m.get("const"):
            arr = W[m["const"]]
            if arr.dtype.kind in "iu" and arr.dtype.itemsize > 1:
                arr, dt = arr.astype(np.int32), 2
            elif arr.dtype == np.int8 or arr.dtype == np.uint8 or arr.dtype == bool:
                arr, dt = arr.astype(np.int8), 1
            elif arr.dtype == np.float16:              # an fp16 op's constant (f16_ir.py)
                dt = 3
            else:
                arr, dt = arr.astype(np.float32), 0
            assert arr.size == nel(n), (n, arr.shape, T[n]["shape"])
            trec.append([dt, 1, nel(n), cblob.add(arr), 0])
        else:
            trec.append([dt, 0, nel(n), 0, 0])

    wmeta = {}                                     # blob offset -> (elements, binary16)

    def wofs(key, dtype=np.float32):
        if key is None:
            return -1
        arr = W[key]
        if arr.dtype == np.float16 and dtype == np.float32:     # an fp16 op's weight (f16_ir.py) stays binary16
            dtype = np.float16
        if dtype is not None:
            arr = arr.astype(dtype)
        off = blob.add(arr)
        wmeta[off] = (int(arr.size), arr.dtype == np.float16)
        return off

    ops = []
    for op in g["ops"]:
        k = op["op"]
        if k in ("cast_f32_to_f16", "cast_f16_to_f32"):
            k = "cast"
        elif k.endswith("_f16"):                   # fp16 storage: the tensors' and weights' dtypes say so
            k = k[:-4]
        sh = op.get("shape", {})
        ins = [tid[t] for t in op["inputs"]]
        outs = [tid[t] for t in op["outputs"]]
        w = [-1] * NW
        ip = [0] * NIP
        fp = [0.0] * NFP
        if k == "view":
            base = ins[0]
            off = int(op.get("offset", 0))
            while trec[base][1] == 2:              # resolve view of a view
                off += trec[base][4]
                base = trec[base][3]
            if trec[base][1] == 1:                 # view of a const: a const at an offset
                o = outs[0]
                trec[o] = [trec[base][0], 1, trec[o][2], trec[base][3] + off * ESZ[trec[base][0]], 0]
            else:
                trec[outs[0]] = [trec[base][0], 2, trec[outs[0]][2], base, off]
            continue                               # no kernel
        if k not in KID:
            raise SystemExit(f"pack_ir: op kind {k!r} ({op['name']}) has no K3 runtime kernel")
        if k == "strided_copy":
            ip[:13] = [sh["off"]] + [sh[f"d{i}"] for i in range(6)] + [sh[f"s{i}"] for i in range(6)]
        elif k == "catn":
            ip[:10] = [sh["n_in"], sh["outer"], sh["inner"]] + [sh.get(f"c{i}", 0) for i in range(8)][:7]
            ip[10] = sh.get("c7", 0)
        elif k == "ewise1":
            ip[:2] = [op["opc"], sh["n"]]
            fp[:2] = [op.get("p0", 0.0), op.get("p1", 0.0)]
        elif k == "ewise2":
            ip[:19] = ([op["opc"]] + [sh[f"d{i}"] for i in range(6)] + [sh[f"sa{i}"] for i in range(6)]
                       + [sh[f"sb{i}"] for i in range(6)])
            fp[0] = op.get("alpha", 1.0)
        elif k == "ewise_scalar":
            ip[:2] = [op["opc"], nel(op["outputs"][0])]
            fp[0] = op["c"]
        elif k == "mul_scalar":
            ip[0] = sh["n"]
            fp[0] = op["scalar"]
        elif k in ("add", "silu_mul", "gelu_exact", "gelu", "silu", "relu", "softplus", "tanh", "mul", "sigmoid"):
            ip[0] = sh.get("n", nel(op["outputs"][0]))
            if k in ("add", "silu_mul", "mul"):
                assert all(nel(t) == ip[0] for t in op["inputs"]), op["name"]
        elif k == "softmax":
            ip[:2] = [sh["M"], sh["K"]]
        elif k == "layer_norm":
            ip[:2] = [sh["M"], sh["K"]]
            fp[0] = op.get("eps", 1e-5)
            w[0], w[1] = wofs(op["gamma_key"]), wofs(op["beta_key"])
        elif k == "rms_norm":
            assert sh.get("inner", 1) == 1, op["name"]
            ip[:2] = [sh["outer"], sh["reduce"]]
            fp[0] = op["eps"]
        elif k == "rope_half":
            ip[:3] = [sh["R"], sh["H"], sh["half"]]
        elif k == "linear":
            ip[:3] = [sh["M"], sh["K"], sh["N"]]
            w[0], w[1] = wofs(op["weight"]), wofs(op.get("bias"))
        elif k == "matmul":
            ip[:3] = [sh["M"], sh["K"], sh["N"]]
        elif k == "bmm_strided":
            keys = ["B0", "B1", "M", "K", "N", "a_off", "a_s0", "a_s1", "a_sm", "a_sk", "b_off", "b_s0", "b_s1",
                    "b_sk", "b_sn", "c_off", "c_s0", "c_s1", "c_sm", "c_sn"]
            ip[:len(keys)] = [sh[x] for x in keys]
            ip[len(keys)] = nel(op["outputs"][0])
        elif k == "im2col":
            ip[:13] = [sh[x] for x in ("IC", "IH", "IW", "OH", "OW", "KH", "KW", "SH", "SW", "PH", "PW", "DH", "DW")]
        elif k == "sdpa":
            ip[:3] = [sh["BH"], sh["S"], sh["D"]]
        elif k == "conv_transpose2d":
            keys = ["N", "IC", "IH", "IW", "OC", "OH", "OW", "KH", "KW", "SH", "SW", "PH", "PW", "DH", "DW", "G"]
            ip[:16] = [sh.get(x, 1 if x in ("DH", "DW", "N", "G") else 0) for x in keys]
            w[0], w[1] = wofs(op["weight"]), wofs(op.get("bias"))
        elif k in ("conv2d", "conv2d_grouped"):
            keys = ["N", "IC", "IH", "IW", "OC", "OH", "OW", "KH", "KW", "SH", "SW", "PH", "PW", "DH", "DW", "G"]
            ip[:16] = [sh.get(x, 1 if x in ("DH", "DW", "N", "G") else 0) for x in keys]
            w[0], w[1] = wofs(op["weight"]), wofs(op.get("bias"))
        elif k == "index_select":
            ip[:4] = [sh["outer"], sh["A"], sh["J"], sh["inner"]]
            it = op["inputs"][1]
            if T[it].get("const") and trec[tid[it]][0] != 2:     # float-valued indices -> int32
                arr = W[T[it]["const"]].astype(np.int64).astype(np.int32)
                trec[tid[it]] = [2, 1, arr.size, cblob.add(arr), 0]
        elif k == "quantize_rows_s8":
            ip[:2] = [sh["M"], sh["K"]]
        elif k in ("linear_dq_s8", "linear_dq_add_s8"):
            ip[:3] = [sh["M"], sh["K"], sh["N"]]
            w[0], w[1], w[2] = wofs(op["weight"], np.int8), wofs(op.get("bias")), wofs(op["w_scale"])
        elif k == "linear3_dq_s8":
            ip[:5] = [sh["M"], sh["K"], sh["N0"], sh["N1"], sh["N2"]]
            w[0], w[1], w[2] = wofs(op["weight"], np.int8), wofs(op.get("bias")), wofs(op["w_scale"])
        elif k == "attention":
            keys = ["B0", "B1", "M", "N", "D", "DV", "q_off", "q_s0", "q_s1", "q_sm", "k_off", "k_s0", "k_s1",
                    "k_sn", "v_off", "v_s0", "v_s1", "v_sn", "o_off", "o_s0", "o_s1", "o_sm"]
            ip[:len(keys)] = [int(sh[x]) for x in keys]
            ip[len(keys)] = int(bool(sh.get("causal")))
            ip[len(keys) + 1] = int(sh.get("act", 0))
            ip[len(keys) + 2] = nel(op["outputs"][0])
            fp[0] = op["scale"]
        elif k in ("softmax_dim", "mean_dim", "sum_dim"):
            ip[:3] = [sh["outer"], sh.get("R", sh.get("reduce")), sh["inner"]]
        elif k == "gather":
            ip[0] = nel(op["outputs"][0])
            it = op["inputs"][1]
            if T[it].get("const") and trec[tid[it]][0] != 2:     # float-valued indices -> int32 (-1: zero)
                arr = W[T[it]["const"]].astype(np.int64).astype(np.int32)
                trec[tid[it]] = [2, 1, arr.size, cblob.add(arr), 0]
        elif k == "cumsum":
            ip[:3] = [sh["outer"], sh["axis"], sh["inner"]]
        elif k == "strided_scatter":
            ip[:14] = [sh["off"]] + [sh[f"d{i}"] for i in range(6)] + [sh[f"s{i}"] for i in range(6)] + [sh["n"]]
        elif k == "reduce":
            ip[:4] = [sh["outer"], sh["R"], sh["inner"], op["opc"]]
        elif k == "cast":
            ip[0] = nel(op["outputs"][0])
            assert nel(op["inputs"][0]) == ip[0], op["name"]
        elif k == "maxpool2d":
            keys = ["N", "C", "IH", "IW", "OH", "OW", "KH", "KW", "SH", "SW", "PH", "PW", "DH", "DW"]
            ip[:14] = [sh.get(x, 1 if x in ("N", "DH", "DW") else 0) for x in keys]
        elif k == "upsample_nearest":
            ip[:5] = [sh.get("N", 1) * sh["C"], sh["IH"], sh["IW"], sh.get("OH", sh["IH"] * sh.get("scale", 1)),
                      sh.get("OW", sh["IW"] * sh.get("scale", 1))]
        for j in range(NW):
            if w[j] >= 0:
                n_el, h = wmeta[w[j]]
                ip[IP_WN + j] = n_el
                ip[IP_WF16] |= (1 << j) if h else 0
        assert len(ins) <= NIN and len(outs) <= NOUT, op["name"]
        rec = [KID[k], len(ins), len(outs)] + ins + [-1] * (NIN - len(ins)) + outs + [-1] * (NOUT - len(outs))
        rec += w + [int(x) for x in ip] + [_f64bits(x) for x in fp]
        assert len(rec) == OPW
        ops.append((op["name"], rec))

    inp = g["input"]
    in_names = [p["name"] for p in inp.get("packed_inputs") or [{"name": inp["tensor"]}]]
    out_names = g["output"].get("tensors") or [g["output"]["tensor"]]
    words = [MAGIC, len(trec), len(ops), len(in_names), len(out_names), blob.size, cblob.size]
    for r in trec:
        words += r
    words += [tid[n] for n in in_names] + [tid[n] for n in out_names]
    for _, r in ops:
        words += r
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.asarray(words, dtype=np.int64).tofile(out + ".plan")
    for b, ext in ((blob, ".wblob"), (cblob, ".cblob")):
        with open(out + ext, "wb") as f:
            for p in b.parts:
                f.write(p)
    wh = hashlib.sha1()
    for p in blob.parts:
        wh.update(p)
    names = {"ir": str(d), "tensors": list(T), "ops": [n for n, _ in ops], "op_kinds": [KINDS[r[0]] for _, r in ops],
             "inputs": in_names, "outputs": out_names, "wblob_sha1": wh.hexdigest(),
             "shapes": {n: T[n]["shape"] for n in T}, "dtypes": {n: T[n].get("dtype", "f32") for n in T},
             "producer": {list(T)[i]: KINDS[r[0]] for _, r in ops for i in r[3 + NIN:3 + NIN + NOUT] if i >= 0}}
    Path(out + ".names.json").write_text(json.dumps(names))
    return {"ops": len(ops), "tensors": len(trec), "wblob_mb": blob.size / 1e6, "cblob_mb": cblob.size / 1e6,
            "wblob_sha1": wh.hexdigest()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ir_dir")
    ap.add_argument("out")
    a = ap.parse_args()
    print(json.dumps(pack(a.ir_dir, a.out)))
