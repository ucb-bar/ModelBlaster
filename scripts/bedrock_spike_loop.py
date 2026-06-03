#!/usr/bin/env python3
"""End-to-end loop: Bedrock generates → cross-compile → spike verify →
on failure, feed the error back to Bedrock and retry."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import pipeline.reference_kernels as rk
from pipeline.bedrock_client import BedrockClient

# Toolchain paths.
# spike-hetero wraps stock spike with both gemmini RoCC (custom-3, 0x7B)
# and Saturn-OPU (OP-V, 0x57) extensions loaded as runtime .so plugins.
# This is REQUIRED because the AlgorithmCandidate descriptions emit
# OPMVINBCAST / VOPACC / VMV_VR (OPU) and gemmini RoCC ops; the stock
# spike traps these as illegal instructions.
CHIPYARD_ROOT = "/scratch2/agustin/chipyard"
RISCV_GCC = "/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc"
SPIKE_HETERO = "/scratch2/agustin/merlin/build_tools/spike-hetero/spike-hetero"
PK        = "/scratch2/agustin/chipyard/.conda-env/riscv-tools/riscv64-unknown-elf/bin/pk"
# rv64gcv_zicntr matches the spike-hetero default (counters needed for
# the perf-counter MMIO some test harnesses use).
# For Saturn-OPU: varch must be vlen=512 elen=64 to get OPU dim=64.
ISA = "rv64gcv_zicntr"
ABI = "lp64d"
VARCH = "vlen:512,elen:64"


def build_initial_prompt(spec, algo):
    return f"""You will write a C implementation of the fused kernel below for the
RVV 1.0 target (rv64gcv with vector). Output must be:
  * A SINGLE C function with EXACTLY the signature below.
  * Self-contained: include any helpers as `static inline` above the function.
  * Bit-exact match vs the reference impl on spike (max_abs_err == 0).
  * No `#include` statements — they are added by the harness (which already
    pulls in <stdint.h>, <stddef.h>, <math.h>, <riscv_vector.h>).
  * RVV 1.0 intrinsic names: `__riscv_v*` prefix. Important conversions:
    - int8→int16 widen: `__riscv_vsext_vf2_i16m2(v_int8, vl)`
    - int16→float32 widen-convert: `__riscv_vfwcvt_f_x_v_f32m4(v_int16, vl)`
      (NOT `vfcvt` — that's same-width; for widening to f32 from i16 use `vfwcvt`)
    - float32→int32 same-width: `__riscv_vfcvt_x_f_v_i32m4(v_f32, vl)`
    - int32→int16 narrow: `__riscv_vnsra_wx_i16m2(v_int32, 0, vl)`
    - int16→int8 narrow: `__riscv_vnsra_wx_i8m1(v_int16, 0, vl)`
    - Set VL: `__riscv_vsetvl_e8m1(n)` for e8m1, etc.

SIGNATURE (must match exactly):
{spec.signature}

REFERENCE IMPLEMENTATION (must produce bit-identical results on spike):
{spec.reference_impl}

ALGORITHM (target_affinity={algo.target_affinity}):
{algo.description}

OUTPUT FORMAT:
Wrap your final code in ```c ... ``` fences. Output ONLY the code."""


def build_retry_prompt(prior_code, error_text, spec):
    return f"""The previous kernel candidate did not compile or did not match
the reference. Here is the compiler / spike output:

```
{error_text[:2000]}
```

Your previous code was:

```c
{prior_code}
```

The expected signature is:
{spec.signature}

Fix the bug and output ONLY the corrected kernel inside ```c ... ``` fences."""


def extract_code(text):
    m = re.search(r"```(?:c|C)?\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else None


def harness_bn_silu(shape):
    rng = np.random.default_rng(42)
    N, C, H, W = shape["N"], shape["C"], shape["H"], shape["W"]
    n_el = N * C * H * W
    x = rng.integers(-128, 128, size=n_el, dtype=np.int8)
    s = rng.uniform(0.5, 1.5, size=C).astype(np.float32)
    b = rng.uniform(-1.0, 1.0, size=C).astype(np.float32)
    inp = ",".join(str(int(v)) for v in x)
    sc  = ",".join(f"{float(v):.9g}f" for v in s)
    bi  = ",".join(f"{float(v):.9g}f" for v in b)
    return f"""
extern void kernel_batchnorm2d_silu_s8(
    const int8_t *input, const float *scale, const float *bias,
    int8_t *output,
    int N, int C, int H, int W,
    float bn_scale_in, float bn_scale_out,
    int bn_activation_min, int bn_activation_max,
    float silu_scale_in, float silu_scale_out,
    int silu_activation_min, int silu_activation_max);

static const int8_t INPUT[{n_el}] = {{ {inp} }};
static const float SCALE[{C}] = {{ {sc} }};
static const float BIAS[{C}]  = {{ {bi} }};
static int8_t OUT[{n_el}];

int main(void) {{
    kernel_batchnorm2d_silu_s8(
        INPUT, SCALE, BIAS, OUT,
        {N}, {C}, {H}, {W},
        0.01f, 0.01f, -128, 127,
        0.01f, 0.01f, -128, 127);
    for (int i = 0; i < {n_el}; i++) printf("%02x\\n", (unsigned char)OUT[i]);
    return 0;
}}
""", n_el


def xc_and_run(kernel_c, harness_c, workdir, label):
    src = Path(workdir) / f"{label}.c"
    elf = Path(workdir) / f"{label}.elf"
    prologue = ("#include <stdint.h>\n#include <stddef.h>\n#include <stdio.h>\n"
                "#include <math.h>\n#include <riscv_vector.h>\n")
    src.write_text(prologue + kernel_c + "\n\n" + harness_c)
    cp = subprocess.run(
        [RISCV_GCC, "-march=" + ISA, "-mabi=" + ABI, "-O2",
         str(src), "-o", str(elf), "-lm"],
        capture_output=True, text=True, timeout=60,
    )
    if cp.returncode != 0:
        return None, f"compile fail:\n{cp.stderr[-1500:]}"
    # Use spike-hetero which loads libgemmini.so + libsaturn_opu.so.
    # The kernel here doesn't emit gemmini/OPU custom ops (it's pure RVV),
    # but using spike-hetero universally is harmless — extensions only
    # decode their own opcode spaces and pass through everything else.
    spike_env = os.environ.copy()
    spike_env["CHIPYARD_ROOT"] = CHIPYARD_ROOT
    spike_env["SPIKE_ISA"] = ISA
    # Need single hart for pk; spike-hetero defaults to 2 harts for the
    # DualSaturnOPUGemmini topology — override for one-shot kernel test.
    spike_env["SPIKE_HARTS"] = "1"
    # spike-hetero on this build doesn't accept --varch as a CLI flag;
    # ISA string drives v config. Defaults match the FireSim
    # DualSaturnOPUGemmini topology (vlen=512, elen=64 → OPU dim=64).
    cp = subprocess.run(
        [SPIKE_HETERO, PK, str(elf)],
        capture_output=True, text=True, timeout=180, env=spike_env,
    )
    if cp.returncode != 0:
        return None, f"spike fail rc={cp.returncode}:\n{cp.stderr[-500:]}"
    vals = []
    for line in cp.stdout.splitlines():
        line = line.strip()
        if not line: continue
        try:
            vals.append(int(line, 16))
        except ValueError:
            continue
    return np.array(vals, dtype=np.uint8).view(np.int8), "ok"


def harness_conv_bn(shape):
    """Harness for conv2d_batchnorm2d_s8 spike verify."""
    rng = np.random.default_rng(42)
    N, IC, IH, IW = shape["N"], shape["IC"], shape["IH"], shape["IW"]
    OC, KH, KW = shape["OC"], shape["KH"], shape["KW"]
    SH, SW, PH, PW = shape["SH"], shape["SW"], shape["PH"], shape["PW"]
    OH = (IH + 2*PH - KH) // SH + 1
    OW = (IW + 2*PW - KW) // SW + 1
    n_in = N * IC * IH * IW
    n_w  = OC * IC * KH * KW
    n_out = N * OC * OH * OW
    x = rng.integers(-128, 128, size=n_in, dtype=np.int8)
    w = rng.integers(-128, 128, size=n_w,  dtype=np.int8)
    b = rng.integers(-1000, 1000, size=OC, dtype=np.int32)
    bn_scale = rng.uniform(0.5, 1.5, size=OC).astype(np.float32)
    bn_bias  = rng.uniform(-1.0, 1.0, size=OC).astype(np.float32)
    inp = ",".join(str(int(v)) for v in x)
    wgt = ",".join(str(int(v)) for v in w)
    bia = ",".join(str(int(v)) for v in b)
    sc  = ",".join(f"{float(v):.9g}f" for v in bn_scale)
    bb  = ",".join(f"{float(v):.9g}f" for v in bn_bias)
    return f"""
extern void kernel_conv2d_batchnorm2d_s8(
    const int8_t *input, const int8_t *weight, const int32_t *bias,
    const float *bn_scale, const float *bn_bias, int8_t *output,
    int N, int IC, int IH, int IW, int OC,
    int KH, int KW, int SH, int SW, int PH, int PW,
    int input_offset, int filter_offset, int conv_output_offset,
    int conv_output_multiplier, int conv_output_shift,
    int conv_activation_min, int conv_activation_max,
    float bn_scale_in, float bn_scale_out,
    int bn_activation_min, int bn_activation_max);

static const int8_t INPUT[{n_in}] = {{ {inp} }};
static const int8_t WEIGHT[{n_w}] = {{ {wgt} }};
static const int32_t BIAS[{OC}] = {{ {bia} }};
static const float BNSCALE[{OC}] = {{ {sc} }};
static const float BNBIAS[{OC}] = {{ {bb} }};
static int8_t OUT[{n_out}];

int main(void) {{
    kernel_conv2d_batchnorm2d_s8(
        INPUT, WEIGHT, BIAS, BNSCALE, BNBIAS, OUT,
        {N}, {IC}, {IH}, {IW}, {OC},
        {KH}, {KW}, {SH}, {SW}, {PH}, {PW},
        0, 0, 0, 1073741824, 1, -128, 127,
        0.01f, 0.01f, -128, 127);
    for (int i = 0; i < {n_out}; i++) printf("%02x\\n", (unsigned char)OUT[i]);
    return 0;
}}
""", n_out


def verify_shape(candidate_c, spec, shape, workdir):
    if spec.op == "batchnorm2d_silu_s8":
        harness, n_el = harness_bn_silu(shape)
    elif spec.op == "conv2d_batchnorm2d_s8":
        harness, n_el = harness_conv_bn(shape)
    else:
        return {"status": "unsupported_spec", "shape": shape}
    y_cand, st_c = xc_and_run(candidate_c, harness, workdir, "cand")
    if y_cand is None:
        return {"status": "candidate_" + st_c, "shape": shape}
    y_ref, st_r = xc_and_run(spec.reference_impl, harness, workdir, "ref")
    if y_ref is None:
        return {"status": "reference_" + st_r, "shape": shape}
    if len(y_cand) != n_el or len(y_ref) != n_el:
        return {"status": "size_mismatch", "shape": shape,
                "n_cand": int(len(y_cand)), "n_ref": int(len(y_ref))}
    diff = np.abs(y_cand.astype(np.int32) - y_ref.astype(np.int32))
    return {"status": "ok", "shape": shape,
            "max_abs_err": int(diff.max()),
            "n_diff": int((diff > 0).sum()), "n_el": int(n_el)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="batchnorm2d_silu_s8")
    ap.add_argument("--algo-idx", type=int, default=0)
    ap.add_argument("--max-attempts", type=int, default=5)
    ap.add_argument("--shape-indices", default="4,5",
                    help="indices into spec.extra_shapes to test")
    ap.add_argument("--out-dir",
                    default=str(REPO / "artifacts" / "kernels"))
    args = ap.parse_args()

    spec_map = {
        "batchnorm2d_silu_s8": rk.BATCHNORM2D_SILU_S8,
        "conv2d_batchnorm2d_s8": rk.CONV2D_BATCHNORM2D_S8,
    }
    spec = spec_map[args.spec]
    algo = spec.algorithms[args.algo_idx]
    out_dir = Path(args.out_dir) / args.spec / algo.name
    out_dir.mkdir(parents=True, exist_ok=True)
    shape_indices = [int(i) for i in args.shape_indices.split(",")]
    shapes = [spec.extra_shapes[i] for i in shape_indices]

    client = BedrockClient(log_path=str(out_dir / "spike_loop.jsonl"))
    prompt = build_initial_prompt(spec, algo)
    (out_dir / "prompt_spike.txt").write_text(prompt)

    history = []
    prior_code = None
    last_error = None
    total_usd = 0.0

    for attempt in range(1, args.max_attempts + 1):
        print(f"\n=== Attempt {attempt} ===")
        if attempt == 1:
            cur_prompt = prompt
        else:
            cur_prompt = build_retry_prompt(prior_code, last_error, spec)

        t0 = time.perf_counter()
        try:
            r = client.converse(user=cur_prompt, max_tokens=4096, temperature=0.2)
        except Exception as e:
            print(f"Bedrock error: {e}")
            history.append({"attempt": attempt, "bedrock_error": str(e)})
            break
        wall = time.perf_counter() - t0
        cost = getattr(r, "usd", None) or 0.0
        total_usd += cost
        print(f"  bedrock wall={wall:.1f}s tokens_out={getattr(r,'output_tokens','?')} cost=${cost:.4f}")

        code = extract_code(r.text)
        if not code:
            print("  no code fence")
            (out_dir / f"spike_attempt_{attempt}_raw.txt").write_text(r.text)
            last_error = "no code fence in response"
            history.append({"attempt": attempt, "no_code": True})
            continue
        (out_dir / f"spike_attempt_{attempt}.c").write_text(code)
        prior_code = code

        with tempfile.TemporaryDirectory(prefix=f"spike_a{attempt}_") as wd:
            shape_results = []
            for shape in shapes:
                print(f"  spike-verify shape={shape}")
                t0 = time.perf_counter()
                res = verify_shape(code, spec, shape, wd)
                verify_wall = time.perf_counter() - t0
                res["verify_wall_s"] = round(verify_wall, 2)
                shape_results.append(res)
                print(f"    -> {res['status']}", end="")
                if res.get("max_abs_err") is not None:
                    print(f"  max_abs_err={res['max_abs_err']}", end="")
                print(f"  wall={verify_wall:.1f}s")
                if res["status"] != "ok" or res.get("max_abs_err") != 0:
                    last_error = res["status"]
                    break

        history.append({
            "attempt": attempt,
            "bedrock_cost_usd": cost,
            "shapes": shape_results,
        })

        all_ok = all(r["status"] == "ok" and r.get("max_abs_err") == 0
                     for r in shape_results)
        if all_ok:
            print(f"\nPASS on attempt {attempt}")
            (out_dir / "spike_final_kernel.c").write_text(code)
            (out_dir / "spike_loop_summary.json").write_text(json.dumps({
                "status": "PASS",
                "winning_attempt": attempt,
                "total_cost_usd": total_usd,
                "history": history,
            }, indent=2))
            return 0
        # Build a precise error to feed back.
        for r in shape_results:
            if r["status"] != "ok":
                last_error = (r["status"] + "\n" +
                              json.dumps(r, indent=2)[-1500:])
                break
            elif r.get("max_abs_err") != 0:
                last_error = f"shape {r['shape']} max_abs_err={r['max_abs_err']}"
                break

    print(f"\nFAIL after {args.max_attempts} attempts; total ${total_usd:.4f}")
    (out_dir / "spike_loop_summary.json").write_text(json.dumps({
        "status": "FAIL", "attempts": args.max_attempts,
        "total_cost_usd": total_usd, "history": history,
    }, indent=2))
    return 1


if __name__ == "__main__":
    sys.exit(main())
