"""Generate two Conv1x1 ONNX variants for the weight-rebind feasibility test.

Variant A (control): weight is a graph INITIALIZER (baked-in constant).
   This is what extract_hta_convs.py currently produces.

Variant B (test):    weight is a graph INPUT.
   If QAIRT v66 lets us rebind this tensor's clientBuf per graphExecute call,
   we can share one DLC across many same-shape segments (saves DSP context-
   table slots).

Both variants implement the same fp32 op:
   y[1, C_out, M, 1] = Conv(x[1, C_in, M, 1], w[C_out, C_in, 1, 1])

Shape choice: 768 / 1024 matches the SmolVLA B-type output_proj — same as
what we'd use in production. Generates both variants so we can compare
side-by-side (does the constant-weight variant build with HTA? does the
input-weight variant? do the outputs match given equivalent weights?).

Also generates calibration .raw files for qairt-quantizer.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from pathlib import Path

HERE = Path(__file__).parent

# Test op shape (matches B-type output_proj: 768 → 768, sequence length 1024)
C_IN = 768
C_OUT = 768
M = 1024
N_CALIB = 5
RNG_SEED = 42


def build_variant_a(out_path: Path, rng: np.random.Generator):
    """Conv1x1 with weight as initializer."""
    weight = rng.standard_normal((C_OUT, C_IN, 1, 1)).astype(np.float32) * 0.05
    bias = np.zeros((C_OUT,), dtype=np.float32)

    X = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, C_IN, M, 1])
    Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, C_OUT, M, 1])
    W_init = numpy_helper.from_array(weight, name="weight")
    B_init = numpy_helper.from_array(bias, name="bias")

    conv = helper.make_node("Conv",
        inputs=["x", "weight", "bias"], outputs=["y"],
        kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0])

    g = helper.make_graph([conv], "variant_a_const_weight",
        inputs=[X], outputs=[Y], initializer=[W_init, B_init])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m, full_check=True)
    onnx.save(m, str(out_path))
    print(f"  wrote {out_path}: x[1,{C_IN},{M},1] @ const-w[{C_OUT},{C_IN},1,1] → y[1,{C_OUT},{M},1]")
    return weight


def build_variant_b(out_path: Path):
    """Conv1x1 with weight as graph input (rebindable)."""
    X = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, C_IN, M, 1])
    W = helper.make_tensor_value_info("weight", TensorProto.FLOAT, [C_OUT, C_IN, 1, 1])
    Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, C_OUT, M, 1])

    # Bias still as initializer (small; less interesting to rebind).
    bias = np.zeros((C_OUT,), dtype=np.float32)
    B_init = numpy_helper.from_array(bias, name="bias")

    conv = helper.make_node("Conv",
        inputs=["x", "weight", "bias"], outputs=["y"],
        kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0])

    g = helper.make_graph([conv], "variant_b_input_weight",
        inputs=[X, W], outputs=[Y], initializer=[B_init])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    try:
        onnx.checker.check_model(m, full_check=True)
    except Exception as e:
        print(f"  warn: variant_b checker: {e}")
    onnx.save(m, str(out_path))
    print(f"  wrote {out_path}: x[1,{C_IN},{M},1] + input-w[{C_OUT},{C_IN},1,1] → y[1,{C_OUT},{M},1]")


def gen_calibration(cal_dir: Path, variant: str, includes_weight: bool, rng: np.random.Generator):
    """Random fp32 calibration data, in the format qairt-quantizer expects:
       one line per sample, `tensor_name:=abs/path/to/file.raw`."""
    cal_dir.mkdir(parents=True, exist_ok=True)
    list_path = cal_dir / f"{variant}_cal_list.txt"
    with open(list_path, "w") as f:
        for i in range(N_CALIB):
            x_data = rng.standard_normal((1, C_IN, M, 1)).astype(np.float32) * 0.5
            x_path = cal_dir / f"x_{i:02d}.raw"
            x_data.tofile(str(x_path))
            tokens = [f"x:={x_path.absolute()}"]
            if includes_weight:
                w_data = rng.standard_normal((C_OUT, C_IN, 1, 1)).astype(np.float32) * 0.05
                w_path = cal_dir / f"w_{i:02d}.raw"
                w_data.tofile(str(w_path))
                tokens.append(f"weight:={w_path.absolute()}")
            f.write(" ".join(tokens) + "\n")
    print(f"  cal list: {list_path}  ({N_CALIB} samples, includes_weight={includes_weight})")


def main():
    rng = np.random.default_rng(RNG_SEED)
    print("Generating test ONNX models...")
    weight_a = build_variant_a(HERE / "variant_a_const_weight.onnx", rng)
    build_variant_b(HERE / "variant_b_input_weight.onnx")
    print()
    print("Generating calibration data...")
    cal_dir = HERE / "calibration"
    rng_cal = np.random.default_rng(RNG_SEED + 1)
    gen_calibration(cal_dir, "variant_a", includes_weight=False, rng=rng_cal)
    rng_cal = np.random.default_rng(RNG_SEED + 1)
    gen_calibration(cal_dir, "variant_b", includes_weight=True,  rng=rng_cal)
    print()
    # Save weight_a as a .raw so the C++ test can also bind it as variant-B input
    # to verify outputs match between the two variants.
    np.save(HERE / "variant_a_weight.npy", weight_a)
    weight_a.tofile(str(HERE / "variant_a_weight.raw"))
    print(f"  saved variant-A's baked-in weight to variant_a_weight.{{npy,raw}}")


if __name__ == "__main__":
    main()
