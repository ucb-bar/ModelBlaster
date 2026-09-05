#!/usr/bin/env python3
"""Dronet-scoped post-generation bake of integer batchnorm params.

TACIT showed the three dronet BN layers spend ~2.2% of the inference in the
one-time per-channel float->fixed setup (soft-float on the misa.F=0 core). Those
scale/bias/scale_in/scale_out are compile-time constants, so this script does the
whole float->fixed decomposition ONCE in Python (bit-identical float32 math to
gemmini_q31_rvv_batchnorm2d_s8_direct.c) and bakes int32 Mc[], int32 Bc[], int8
S[] per BN node, swapping in the zero-float `_iq` kernel.

Scope: patches ONLY the generated dronet files (kernels.c/weights.c/weights.h/
model.c/kernels.h). It does NOT touch the shared ModelBlaster pipeline, so no
other model is affected. Idempotent. Run after generate_skeleton (i.e. after the
normal gen for the hetero build), before `west build`.

Bit-exactness: `direct` computes the SAME Mc/Bc/S at runtime and applies the
SAME acc=x*Mc+Bc; round_half_away(acc>>S). Baking the identical constants leaves
the output byte-identical (validated on FPGA: max_abs_err=0).

Usage: python bake_bn_int.py <generated/hetero dir>
"""
import sys, os, re, json
import numpy as np

f32 = np.float32

def mbcs(scale, bias, scale_in, scale_out):
    """Replicate the direct kernel's per-channel float->fixed setup in float32."""
    inv_so = f32(1.0) / f32(scale_out)
    Mc, Bc, Sh = [], [], []
    for c in range(len(scale)):
        cs = f32(f32(f32(scale[c]) * f32(scale_in)) * inv_so)
        cb = f32(f32(bias[c]) * inv_so)
        csa = f32(abs(cs)); cba = f32(abs(cb))
        S = 24
        while S > 1 and (f32(csa * f32(2.0**S)) >= f32(4194304.0)
                         or f32(cba * f32(2.0**S)) >= f32(1073741824.0)):
            S -= 1
        Mc.append(int(np.rint(np.float64(f32(cs * f32(2.0**S))))))
        Bc.append(int(np.rint(np.float64(f32(cb * f32(2.0**S))))))
        Sh.append(S)
    return Mc, Bc, Sh

def c_int_array(ctype, name, vals):
    body = ", ".join(str(v) for v in vals)
    return f"const {ctype} {name}[{len(vals)}] = {{ {body} }};\n"

def main():
    gen = os.path.abspath(sys.argv[1])
    # weights.npz + graph.json live in the parent of the backend subdir (.../generated)
    root = os.path.dirname(gen)  # .../generated
    npz = np.load(os.path.join(root, "weights.npz"))
    graph = json.load(open(os.path.join(root, "graph.json")))
    ops = graph.get("ops", graph.get("nodes", []))
    bn_ops = [o for o in ops if o.get("op") == "batchnorm2d_s8"]
    if not bn_ops:
        print("no batchnorm2d_s8 ops; nothing to bake"); return

    marker = "/* MB_BN_INTBAKED */"
    wc = os.path.join(gen, "weights.c"); wh = os.path.join(gen, "weights.h")
    mc = os.path.join(gen, "model.c");   kc = os.path.join(gen, "kernels.c")
    kh = os.path.join(gen, "kernels.h")
    weights_c = open(wc).read()
    if marker in weights_c:
        print("already baked (idempotent no-op)"); return

    weights_h = open(wh).read(); model_c = open(mc).read()
    kernels_c = open(kc).read(); kernels_h = open(kh).read()

    arr_c = [marker + "\n"]; arr_h = [marker + "\n"]
    for o in bn_ops:
        wkey = o["weight"]; bkey = o["bias"]                 # e.g. bn_modules.0.scale
        node = wkey.rsplit(".", 1)[0].replace(".", "_")      # bn_modules_0
        q = o["quant"]
        scale = npz[wkey]; bias = npz[bkey]
        Mc, Bc, Sh = mbcs(scale, bias, q["scale_in"], q["scale_out"])
        base = f"dronet_{node}"
        arr_c.append(c_int_array("int32_t", base + "_Mc", Mc))
        arr_c.append(c_int_array("int32_t", base + "_Bc", Bc))
        arr_c.append(c_int_array("int8_t",  base + "_S",  Sh))
        for nm, ct, n in ((base+"_Mc","int32_t",len(Mc)),(base+"_Bc","int32_t",len(Bc)),(base+"_S","int8_t",len(Sh))):
            arr_h.append(f"extern const {ct} {nm}[{n}];\n")
        # rewrite this node's dispatch call in model.c:
        # kernel_batchnorm2d_s8_dronet(IN, dronet_<node>_scale_..., dronet_<node>_bias_fused_..., OUT, N,C,H,W, Sin f, Sout f, MIN, MAX)
        #   -> kernel_batchnorm2d_s8_dronet(IN, dronet_<node>_Mc, _Bc, _S, OUT, N,C,H,W, MIN, MAX)
        pat = re.compile(
            r"kernel_batchnorm2d_s8_dronet\(\s*([^,]+?),\s*dronet_" + re.escape(node) +
            r"_scale_\w+,\s*dronet_" + re.escape(node) +
            r"_bias_fused_\w+,\s*([^,]+?),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),"
            r"\s*[-0-9.eE]+f,\s*[-0-9.eE]+f,\s*(-?\d+),\s*(-?\d+)\)")
        def repl(m):
            inp, outp, N, C, H, W, amin, amax = m.groups()
            return (f"kernel_batchnorm2d_s8_dronet({inp}, {base}_Mc, {base}_Bc, "
                    f"{base}_S, {outp}, {N}, {C}, {H}, {W}, {amin}, {amax})")
        model_c, nsub = pat.subn(repl, model_c)
        if nsub != 1:
            raise SystemExit(f"model.c: expected 1 bn call for {node}, patched {nsub}")

    # append the int arrays
    weights_c += "\n" + "".join(arr_c)
    weights_h = weights_h.rstrip("\n") + "\n\n" + "".join(arr_h)

    # swap the kernel body: replace the float bn function with the _iq one, renamed
    # modelblaster root = the ".../modelblaster" ancestor of this script
    mb_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))  # bake_bn_int.py is in examples/dronet/int8/
    iq_path = os.path.join(mb_root, "kernels", "gemmini_q31_rvv",
                           "gemmini_q31_rvv_batchnorm2d_s8_iq_intbaked.c")
    iq_src = open(iq_path).read()
    # take just the function (drop its includes/comments header)
    fstart = iq_src.index("void kernel_batchnorm2d_s8_iq(")
    iq_fn = iq_src[fstart:].replace("kernel_batchnorm2d_s8_iq", "kernel_batchnorm2d_s8_dronet", 1)
    # replace the existing float bn function definition in kernels.c
    kpat = re.compile(r"void kernel_batchnorm2d_s8_dronet\(.*?\n\}\n", re.DOTALL)
    kernels_c, ksub = kpat.subn(marker + "\n" + iq_fn, kernels_c, count=1)
    if ksub != 1:
        raise SystemExit(f"kernels.c: expected 1 bn function, patched {ksub}")

    # update the prototype in kernels.h
    hpat = re.compile(r"void kernel_batchnorm2d_s8_dronet\([^;]*?\);", re.DOTALL)
    new_proto = ("void kernel_batchnorm2d_s8_dronet(const int8_t *input, "
                 "const int32_t *Mc, const int32_t *Bc, const int8_t *Sh, "
                 "int8_t *output, int N, int C, int H, int W, "
                 "int activation_min, int activation_max);")
    kernels_h, hsub = hpat.subn(new_proto, kernels_h, count=1)
    if hsub != 1:
        raise SystemExit(f"kernels.h: expected 1 bn prototype, patched {hsub}")

    open(wc, "w").write(weights_c); open(wh, "w").write(weights_h)
    open(mc, "w").write(model_c);   open(kc, "w").write(kernels_c)
    open(kh, "w").write(kernels_h)
    print(f"baked int BN for {len(bn_ops)} nodes: " +
          ", ".join('dronet_'+o['weight'].rsplit('.',1)[0].replace('.','_') for o in bn_ops))

if __name__ == "__main__":
    main()
