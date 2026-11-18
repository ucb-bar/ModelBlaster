"""Shared helpers: build + load kernel_bmm_strided_f16 from a .cu, torch references (fp64) for strided bmm."""
import ctypes, subprocess, os, torch
from pathlib import Path

ARGS = ["B0", "B1", "M", "K", "N", "a_off", "a_s0", "a_s1", "a_sm", "a_sk", "b_off", "b_s0", "b_s1", "b_sk", "b_sn",
        "c_off", "c_s0", "c_s1", "c_sm", "c_sn"]


def build(cu: str, so: str, arch="sm_89", extra=()):
    cmd = ["nvcc", "-O3", f"-arch={arch}", "-shared", "-Xcompiler", "-fPIC", "-std=c++17", *extra, "-o", so, cu]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode:
        raise RuntimeError(cp.stderr[-3000:])
    return so


def load(so):
    lib = ctypes.CDLL(so)
    f = lib.kernel_bmm_strided_f16
    f.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 20
    f.restype = None
    lib.mb_last_error.restype = ctypes.c_int
    return lib


def call(lib, A, B, C, c):
    lib.kernel_bmm_strided_f16(A.data_ptr(), B.data_ptr(), C.data_ptr(), *[int(c[k]) for k in ARGS])


def extent(off, dims, strides):
    """1 + max linear index (0 if any dim is 0) and the min index, over the index box."""
    if any(d == 0 for d in dims):
        return 0, 0
    hi = off + sum((d - 1) * s for d, s in zip(dims, strides) if s > 0)
    lo = off + sum((d - 1) * s for d, s in zip(dims, strides) if s < 0)
    return lo, hi + 1


def views(c):
    return dict(
        A=((c["B0"], c["B1"], c["M"], c["K"]), (c["a_s0"], c["a_s1"], c["a_sm"], c["a_sk"]), c["a_off"]),
        B=((c["B0"], c["B1"], c["K"], c["N"]), (c["b_s0"], c["b_s1"], c["b_sk"], c["b_sn"]), c["b_off"]),
        C=((c["B0"], c["B1"], c["M"], c["N"]), (c["c_s0"], c["c_s1"], c["c_sm"], c["c_sn"]), c["c_off"]))


SENT = 0x5555   # fp16 bit pattern 21845 (~ 5.3e+2... finite) used to fill C before the call


def run_case(lib, c, dev="cuda", seed=0, pad=4096, dtype_ref=torch.float64):
    """Returns dict(ok, max_ratio, oob, nan). ratio = |got-ref| / (2^-11 |ref| + 2^-25 (fp16 subnormal half-ulp) + 1e-6 * sum|a||b|)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    v = views(c)
    ext = {k: extent(o, d, s) for k, (d, s, o) in v.items()}
    for k, (lo, hi) in ext.items():
        assert lo >= 0, (k, lo)
    na, nb, nc = (ext[k][1] for k in "ABC")
    A = torch.empty(na + 2 * pad, dtype=torch.float16)
    B = torch.empty(nb + 2 * pad, dtype=torch.float16)
    A.copy_((torch.rand(na + 2 * pad, generator=g) * 2 - 1).half())
    B.copy_((torch.rand(nb + 2 * pad, generator=g) * 2 - 1).half())
    C = torch.full((nc + 2 * pad,), 0, dtype=torch.int16)
    C.fill_(SENT)
    A, B, C = A.to(dev), B.to(dev), C.to(dev)
    # pads: poison A/B pads with NaN-like large values so an out-of-bounds read shows up
    A[:pad] = 1e4; A[pad + na:] = 1e4; B[:pad] = 1e4; B[pad + nb:] = 1e4
    c_before = C.clone()
    if min(ext[k][1] for k in "ABC") >= 0:
        call(lib, A[pad:], B[pad:], C[pad:], c)
    torch.cuda.synchronize()
    err = lib.mb_last_error()
    if any(d == 0 for d in (c["B0"], c["B1"], c["M"], c["N"])):
        return dict(ok=bool((C == c_before).all()) and err == 0, max_ratio=0.0, oob=False, err=err)
    (da, sa, oa), (db, sb, ob), (dc, sc, oc) = v["A"], v["B"], v["C"]
    Af = torch.as_strided(A.double(), da, sa, oa + pad)  # as_strided offsets are absolute
    Bf = torch.as_strided(B.double(), db, sb, ob + pad)
    flatA, flatB = Af.reshape(c["B0"] * c["B1"], c["M"], c["K"]), Bf.reshape(c["B0"] * c["B1"], c["K"], c["N"])
    if c["K"] == 0:
        ref = torch.zeros(flatA.shape[0], c["M"], c["N"], dtype=torch.float64, device=dev)
        absum = torch.zeros_like(ref)
    else:
        ref = torch.bmm(flatA, flatB)
        absum = torch.bmm(flatA.abs(), flatB.abs())
    ref = ref.reshape(dc)
    absum = absum.reshape(dc)
    Cg = torch.as_strided(C, dc, sc, oc + pad)        # int16 view of the written elements
    got = Cg.view(torch.float16).double() if Cg.is_contiguous() else Cg.contiguous().view(torch.float16).double()
    ratio = (got - ref).abs() / (2.0 ** -11 * ref.abs() + 2.0 ** -25 + 1e-6 * absum)
    max_ratio = float(ratio.max()) if ratio.numel() else 0.0
    bad_nan = bool(torch.isnan(got).any())
    # every other element of C (outside the image, incl. pads) must be untouched
    mask = torch.zeros(nc + 2 * pad, dtype=torch.bool, device=dev)
    torch.as_strided(mask, dc, sc, oc + pad).fill_(True)
    untouched = bool((C[~mask] == c_before[~mask]).all())
    ok = max_ratio <= 1.0 and not bad_nan and untouched and err == 0
    return dict(ok=ok, max_ratio=max_ratio, oob=not untouched, nan=bad_nan, err=err)
