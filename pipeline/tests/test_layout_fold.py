"""layout_fold: strided copies folded into bmm_strided give bit-identical
results (pure-python IR evaluator, small shapes; no torch needed)."""
import random

try:
    from modelblaster.pipeline import layout_fold as lf
except ImportError:  # running from the repo checkout
    import importlib.util, pathlib
    _p = pathlib.Path(__file__).resolve().parents[1] / "layout_fold.py"
    _s = importlib.util.spec_from_file_location("layout_fold", _p)
    lf = importlib.util.module_from_spec(_s); _s.loader.exec_module(lf)


def _cs(shape):
    return lf._contig(shape)


def _copy_shape(off, dims, strides):
    dims, strides = [1] * (6 - len(dims)) + list(dims), [0] * (6 - len(strides)) + list(strides)
    sh = {"off": off}
    sh.update({f"d{i}": dims[i] for i in range(6)})
    sh.update({f"s{i}": strides[i] for i in range(6)})
    return sh


class G:
    def __init__(self):
        self.ops, self.tensors, self.n = [], {}, 0

    def t(self, shape, name=None):
        name = name or f"t{self.n}"; self.n += 1
        self.tensors[name] = {"shape": list(shape), "dtype": "f16"}
        return name

    def permute(self, src, perm):
        sh = self.tensors[src]["shape"]; cs = _cs(sh)
        out = self.t([sh[p] for p in perm])
        self.ops.append({"name": f"perm{self.n}", "op": "strided_copy_f16", "inputs": [src], "outputs": [out],
                         "shape": _copy_shape(0, [sh[p] for p in perm], [cs[p] for p in perm])})
        return out

    def slice_last(self, src, lo, hi):
        sh = self.tensors[src]["shape"]; cs = _cs(sh)
        out = self.t(sh[:-1] + [hi - lo])
        self.ops.append({"name": f"slice{self.n}", "op": "strided_copy_f16", "inputs": [src], "outputs": [out],
                         "shape": _copy_shape(lo, sh[:-1] + [hi - lo], cs)})
        return out

    def view(self, src, shape):
        out = self.t(shape)
        self.ops.append({"name": f"view{self.n}", "op": "view", "inputs": [src], "outputs": [out], "shape": {"n": lf._prod(shape)}})
        return out

    def bmm(self, a, b, batch, M, K, N):
        out = self.t([batch, M, N])
        self.ops.append({"name": f"bmm{self.n}", "op": "bmm_f16", "inputs": [a, b], "outputs": [out],
                         "shape": {"batch": batch, "M": M, "K": K, "N": N}})
        return out

    def graph(self, inp, out):
        return {"ops": self.ops, "tensors": self.tensors, "input": {"tensor": inp}, "output": {"tensor": out}}


def run(graph, x):
    env = {graph["input"]["tensor"]: x}
    for o in graph["ops"]:
        sh = o["shape"]
        if o["op"] == "view":
            src = env[o["inputs"][0]]; off = o.get("offset", 0)
            env[o["outputs"][0]] = src[off:off + sh["n"]]
        elif o["op"] == "strided_copy_f16":
            off, dims = lf.copy_map(sh); src = env[o["inputs"][0]]
            ds = [d for d, _ in dims]
            env[o["outputs"][0]] = [src[lf._eval((off, dims), lf._unravel(i, ds))] for i in range(lf._prod(ds))]
        elif o["op"] == "bmm_f16":
            A, B = (env[t] for t in o["inputs"]); b, M, K, N = sh["batch"], sh["M"], sh["K"], sh["N"]
            env[o["outputs"][0]] = [sum(A[(i * M + m) * K + k] * B[(i * K + k) * N + n] for k in range(K))
                                    for i in range(b) for m in range(M) for n in range(N)]
        elif o["op"] == "bmm_strided_f16":
            A, B = (env[t] for t in o["inputs"])
            out = [None] * lf._prod(graph["tensors"][o["outputs"][0]]["shape"])
            for b0 in range(sh["B0"]):
                for b1 in range(sh["B1"]):
                    for m in range(sh["M"]):
                        for n in range(sh["N"]):
                            acc = sum(A[sh["a_off"] + b0 * sh["a_s0"] + b1 * sh["a_s1"] + m * sh["a_sm"] + k * sh["a_sk"]] *
                                      B[sh["b_off"] + b0 * sh["b_s0"] + b1 * sh["b_s1"] + k * sh["b_sk"] + n * sh["b_sn"]]
                                      for k in range(sh["K"]))
                            out[sh["c_off"] + b0 * sh["c_s0"] + b1 * sh["c_s1"] + m * sh["c_sm"] + n * sh["c_sn"]] = acc
            env[o["outputs"][0]] = out
        else:
            raise NotImplementedError(o["op"])
    return env[graph["output"]["tensor"]]


def _check(g, x, n_folds):
    g2, rep = lf.fold(g)
    assert len(rep["folded"]) == n_folds, rep
    assert not any(o["op"] == "strided_copy_f16" for o in g2["ops"]), [o["name"] for o in g2["ops"]]
    assert run(g2, x) == run(g, x)
    return g2, rep


def test_attention_heads():
    """MinGPT: qkv = x@W (B,T,3C) -> split -> view (B,T,nh,hs) -> transpose(1,2) -> q@k^T, att@v, y.transpose(1,2)."""
    B, T, nh, hs = 2, 3, 2, 4; C = nh * hs
    g = G(); qkv = g.t([B, T, 3 * C], "qkv")
    q, k, v = (g.view(g.slice_last(qkv, i * C, (i + 1) * C), [B, T, nh, hs]) for i in range(3))
    q, k, v = (g.permute(t, [0, 2, 1, 3]) for t in (q, k, v))
    kT = g.permute(k, [0, 1, 3, 2])
    att = g.bmm(q, kT, B * nh, T, hs, T)
    y = g.bmm(att, v, B * nh, T, T, hs)
    y = g.view(y, [B, nh, T, hs])
    y = g.permute(y, [0, 2, 1, 3])
    out = g.view(y, [B, T, C])
    rng = random.Random(1); x = [rng.randrange(-9, 9) for _ in range(B * T * 3 * C)]
    g2, rep = _check(g.graph("qkv", out), x, 2)
    s = next(o for o in g2["ops"] if o["op"] == "bmm_strided_f16" and o["inputs"][0] == "qkv")["shape"]
    assert (s["B0"], s["B1"], s["a_sm"], s["a_sk"], s["b_sk"], s["b_sn"]) == (B, nh, 3 * C, 1, 1, 3 * C)


def test_mamba_states():
    """states = einsum('bclhn,bclhp->bchpn'): X (b,c,l,h,p) permuted to (bch, p, l) before the bmm."""
    b, c, l, h, p, n = 2, 2, 3, 2, 4, 3
    g = G(); X = g.t([b, c, l, h, p], "X")
    Xp = g.view(g.permute(X, [0, 1, 3, 4, 2]), [b * c * h, p, l])
    D = g.t([b * c * h, l, n], "D")
    g.ops.insert(0, {"name": "mkD", "op": "view", "inputs": ["X"], "outputs": ["D"], "shape": {"n": b * c * h * l * n}})
    out = g.bmm(Xp, D, b * c * h, p, l, n)
    rng = random.Random(2); x = [rng.randrange(-9, 9) for _ in range(b * c * l * h * p)]
    g2, rep = _check(g.graph("X", out), x, 1)
    s = g2["ops"][-1]["shape"]
    assert s["a_sm"] == 1 and s["a_sk"] == h * p          # read X in place, transposed


def test_netvlad_out_transpose():
    """vlad = (assign^T @ x).transpose(1,2): the result transpose is folded into C's strides."""
    B, N, K, D = 2, 5, 3, 4
    g = G(); a = g.t([B, N, K], "a"); x = g.t([B, N, D], "x")
    g.ops.append({"name": "mkx", "op": "view", "inputs": ["a"], "outputs": ["x"], "shape": {"n": B * N * D}})
    aT = g.permute(a, [0, 2, 1])
    v = g.bmm(aT, x, B, K, N, D)
    out = g.permute(v, [0, 2, 1])
    rng = random.Random(3); xx = [rng.randrange(-9, 9) for _ in range(B * N * max(K, D))]
    g2, rep = _check(g.graph("a", out), xx, 1)
    s = g2["ops"][-1]["shape"]
    assert (s["c_sm"], s["c_sn"]) == (1, K) and (s["a_sm"], s["a_sk"]) == (1, K)


def test_gather_like_copy_not_folded():
    """A copy whose operand would have no unit inner stride stays a copy."""
    B, M, K, N = 2, 3, 4, 2
    g = G(); a = g.t([B, K, 2, M], "a")
    # a[:, :, 0, :] transposed -> (B, M, K) with strides (.., 1*?,..): inner strides 1 and 2*M; take every 2nd col
    sl = g.t([B, M, K]); g.ops.append({"name": "odd", "op": "strided_copy_f16", "inputs": [a], "outputs": [sl],
                                      "shape": _copy_shape(0, [B, M, K], [K * 2 * M, 2, 2 * M])})
    w = g.t([B, K, N], "w"); g.ops.insert(0, {"name": "mkw", "op": "view", "inputs": ["a"], "outputs": ["w"], "shape": {"n": B * K * N}})
    out = g.bmm(sl, w, B, M, K, N)
    g2, rep = lf.fold(g.graph("a", out))
    assert not rep["folded"] and any(o["op"] == "strided_copy_f16" for o in g2["ops"])


if __name__ == "__main__":
    for f in (test_attention_heads, test_mamba_states, test_netvlad_out_transpose, test_gather_like_copy_not_folded):
        f(); print("ok", f.__name__)
