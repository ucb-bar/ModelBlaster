"""Per-op kernel specs: signature, semantics, reference C impl, test shapes.

This is the single source of truth for:
  * the reference backend in generate_kernels (just emit `reference_impl`)
  * the LLM backend's prompt (signature + semantics + reference shown to LLM)
  * the verify harness (calls both .so's against shapes from `extra_shapes`
    plus shapes pulled from the actual model IR)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class AlgorithmCandidate:
    """An algorithmic approach for implementing an op.

    Multiple algorithms can be valid for the same op (e.g. conv2d via direct
    sliding-window vs im2col+GEMM). Each algorithm is a separate seed shown
    to the LLM during the correctness phase — the LLM is asked to write a
    kernel matching the algorithm's structure. Verify still uses the
    KernelSpec.reference_impl as the (algorithm-agnostic) oracle, so any
    algorithm whose output is numerically equivalent passes.
    """
    name: str
    # Plain-English description of the algorithm — what shape of code to
    # produce and why. Goes into the LLM correctness prompt.
    description: str
    # A correct implementation written in this algorithmic style. Used as the
    # seed example in the LLM prompt. NOT the verify oracle (that's
    # KernelSpec.reference_impl); this can be slow / VLA-using / whatever.
    reference_impl: str
    # Optional shape filter: lambda shape_dict -> bool. If set and returns
    # False, the algorithm is skipped for that shape during verify.
    applicable: Optional[Callable[[dict], bool]] = None
    # Backends this algorithm makes sense for. Empty means all.
    target_affinity: tuple[str, ...] = ()


@dataclass
class KernelSpec:
    op: str
    # Function signature, exactly as it should appear in kernels.h.
    signature: str
    # Plain-English semantics for the LLM prompt.
    semantics: str
    # Trusted reference impl. Used as `--backend reference` output AND as the
    # ground-truth oracle that the LLM-generated kernel is compared against
    # for correctness — algorithm-agnostic; should be the simplest, most
    # obviously-correct version.
    reference_impl: str
    # Extra shape combinations to verify against, beyond what's in the IR.
    # Each entry is the kwargs that the verify harness will use to allocate
    # buffers and call the kernel.
    extra_shapes: list[dict[str, int]] = field(default_factory=list)
    # ctypes argtypes for verify (in order).
    argtypes_factory: Callable[[], list[Any]] = None  # type: ignore
    # Optional list of algorithmic styles the LLM can be seeded with. If
    # empty (the default), a single "direct" algorithm wrapping
    # `reference_impl` is synthesized, preserving the previous single-seed
    # behavior.
    algorithms: list[AlgorithmCandidate] = field(default_factory=list)

    def __post_init__(self):
        if not self.algorithms:
            self.algorithms = [
                AlgorithmCandidate(
                    name="direct",
                    description=(
                        "Direct, naive implementation. Loop over every output "
                        "element and compute it explicitly."
                    ),
                    reference_impl=self.reference_impl,
                )
            ]


def _linear_argtypes():
    import ctypes
    fp = ctypes.POINTER(ctypes.c_float)
    return [fp, fp, fp, fp, ctypes.c_int, ctypes.c_int, ctypes.c_int]


def _relu_argtypes():
    import ctypes
    fp = ctypes.POINTER(ctypes.c_float)
    return [fp, fp, ctypes.c_int]


def _elu_argtypes():
    import ctypes
    fp = ctypes.POINTER(ctypes.c_float)
    # input, output, n, alpha
    return [fp, fp, ctypes.c_int, ctypes.c_float]


def _conv2d_argtypes():
    import ctypes
    fp = ctypes.POINTER(ctypes.c_float)
    # input, weight, bias, output, N, IC, IH, IW, OC, KH, KW, SH, SW, PH, PW
    return [fp, fp, fp, fp] + [ctypes.c_int] * 11


def _maxpool2d_argtypes():
    import ctypes
    fp = ctypes.POINTER(ctypes.c_float)
    # input, output, N, C, IH, IW, KH, KW, SH, SW
    return [fp, fp] + [ctypes.c_int] * 8


def _add_argtypes():
    import ctypes
    fp = ctypes.POINTER(ctypes.c_float)
    return [fp, fp, fp, ctypes.c_int]


def _batchnorm2d_argtypes():
    import ctypes
    fp = ctypes.POINTER(ctypes.c_float)
    # input, scale, bias, output, N, C, H, W
    return [fp, fp, fp, fp] + [ctypes.c_int] * 4


def _sigmoid_argtypes():
    import ctypes
    fp = ctypes.POINTER(ctypes.c_float)
    return [fp, fp, ctypes.c_int]


def _conv2d_dw_argtypes():
    import ctypes
    fp = ctypes.POINTER(ctypes.c_float)
    # input, weight, bias, output, N, C, IH, IW, KH, KW, SH, SW, PH, PW
    # OC == IC == groups (depthwise) — folded into a single C param.
    return [fp, fp, fp, fp] + [ctypes.c_int] * 10


def _relu6_argtypes():
    import ctypes
    fp = ctypes.POINTER(ctypes.c_float)
    return [fp, fp, ctypes.c_int]


def _adaptive_avg_pool2d_argtypes():
    import ctypes
    fp = ctypes.POINTER(ctypes.c_float)
    # input, output, N, C, IH, IW (output is always [N, C, 1, 1] for now)
    return [fp, fp] + [ctypes.c_int] * 4


def _linear_s8_argtypes():
    import ctypes
    i8p = ctypes.POINTER(ctypes.c_int8)
    i32p = ctypes.POINTER(ctypes.c_int32)
    # input(i8*), weight(i8*), bias(i32*), output(i8*),
    # M, K, N,
    # input_offset, filter_offset, output_offset,
    # output_multiplier, output_shift,
    # activation_min, activation_max
    return [i8p, i8p, i32p, i8p] + [ctypes.c_int] * 10


def _relu_s8_argtypes():
    import ctypes
    i8p = ctypes.POINTER(ctypes.c_int8)
    return [i8p, i8p, ctypes.c_int]


def _conv2d_s8_argtypes():
    import ctypes
    i8p = ctypes.POINTER(ctypes.c_int8)
    i32p = ctypes.POINTER(ctypes.c_int32)
    # input(i8*), weight(i8*), bias(i32*), output(i8*),
    # N, IC, IH, IW, OC, KH, KW, SH, SW, PH, PW,
    # input_offset, filter_offset, output_offset,
    # output_multiplier, output_shift,
    # activation_min, activation_max
    return [i8p, i8p, i32p, i8p] + [ctypes.c_int] * 18


def _maxpool2d_s8_argtypes():
    import ctypes
    i8p = ctypes.POINTER(ctypes.c_int8)
    return [i8p, i8p] + [ctypes.c_int] * 8


def _add_s8_argtypes():
    import ctypes
    i8p = ctypes.POINTER(ctypes.c_int8)
    # a, b, output, n, scale_a, scale_b, scale_out, activation_min, activation_max
    return [i8p, i8p, i8p, ctypes.c_int,
            ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_int, ctypes.c_int]


def _batchnorm2d_s8_argtypes():
    import ctypes
    i8p = ctypes.POINTER(ctypes.c_int8)
    fp = ctypes.POINTER(ctypes.c_float)
    # input, scale, bias, output, N, C, H, W, scale_in, scale_out,
    # activation_min, activation_max
    return [i8p, fp, fp, i8p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_float, ctypes.c_float,
            ctypes.c_int, ctypes.c_int]


def _sigmoid_s8_argtypes():
    import ctypes
    i8p = ctypes.POINTER(ctypes.c_int8)
    # input, output, n, scale_in, scale_out, activation_min, activation_max
    return [i8p, i8p, ctypes.c_int,
            ctypes.c_float, ctypes.c_float,
            ctypes.c_int, ctypes.c_int]


LINEAR = KernelSpec(
    op="linear",
    signature=(
        "void kernel_linear(const float *input, const float *weight, "
        "const float *bias, float *output, int M, int K, int N)"
    ),
    semantics=(
        "Computes a fully-connected (matmul + bias) layer matching PyTorch "
        "nn.Linear semantics:\n"
        "  output[m, n] = bias[n] + sum_{k=0..K-1} input[m, k] * weight[n, k]\n"
        "Shapes (row-major contiguous):\n"
        "  input:  [M, K]\n"
        "  weight: [N, K]   (note: out_features outer, in_features inner — "
        "PyTorch's storage order)\n"
        "  bias:   [N]      (may be NULL — treat as zeros)\n"
        "  output: [M, N]\n"
        "All tensors are float32."
    ),
    reference_impl="""\
void kernel_linear(const float *input, const float *weight, const float *bias,
                   float *output, int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = bias ? bias[n] : 0.0f;
            for (int k = 0; k < K; k++) {
                acc += input[m * K + k] * weight[n * K + k];
            }
            output[m * N + n] = acc;
        }
    }
}
""",
    extra_shapes=[
        {"M": 1, "K": 1, "N": 1},
        {"M": 1, "K": 7, "N": 13},      # primes, no padding alignment
        {"M": 4, "K": 17, "N": 23},     # batch > 1
        {"M": 1, "K": 64, "N": 64},     # power-of-two
    ],
    argtypes_factory=_linear_argtypes,
)


ELU = KernelSpec(
    op="elu",
    signature="void kernel_elu(const float *input, float *output, int n, float alpha)",
    semantics=(
        "Elementwise ELU on a contiguous float32 buffer:\n"
        "  output[i] = input[i]                       if input[i] > 0\n"
        "  output[i] = alpha * (expf(input[i]) - 1)   otherwise\n"
        "It must be safe for `input` and `output` to alias. The most common "
        "value of alpha is 1.0 — that's what nn.ELU defaults to."
    ),
    reference_impl="""\
void kernel_elu(const float *input, float *output, int n, float alpha) {
    for (int i = 0; i < n; i++) {
        float v = input[i];
        output[i] = v > 0.0f ? v : alpha * (expf(v) - 1.0f);
    }
}
""",
    extra_shapes=[
        {"n": 1},
        {"n": 16},
        {"n": 256},
        {"n": 1024},
    ],
    argtypes_factory=_elu_argtypes,
)


RELU = KernelSpec(
    op="relu",
    signature="void kernel_relu(const float *input, float *output, int n)",
    semantics=(
        "Elementwise ReLU on a contiguous float32 buffer:\n"
        "  output[i] = max(0.0f, input[i])  for i in [0, n)\n"
        "It must be safe for `input` and `output` to alias."
    ),
    reference_impl="""\
void kernel_relu(const float *input, float *output, int n) {
    for (int i = 0; i < n; i++) {
        float v = input[i];
        output[i] = v > 0.0f ? v : 0.0f;
    }
}
""",
    extra_shapes=[
        {"n": 1},
        {"n": 17},
        {"n": 1024},
    ],
    argtypes_factory=_relu_argtypes,
)


CONV2D = KernelSpec(
    op="conv2d",
    signature=(
        "void kernel_conv2d(const float *input, const float *weight, "
        "const float *bias, float *output, "
        "int N, int IC, int IH, int IW, int OC, "
        "int KH, int KW, int SH, int SW, int PH, int PW)"
    ),
    semantics=(
        "2D convolution matching torch.nn.Conv2d semantics with groups=1, "
        "dilation=1.\n"
        "Layout (row-major, NCHW for activations / OIHW for weights):\n"
        "  input:  [N, IC, IH, IW]\n"
        "  weight: [OC, IC, KH, KW]\n"
        "  bias:   [OC] (may be NULL — treat as zeros)\n"
        "  output: [N, OC, OH, OW]  with\n"
        "    OH = (IH + 2*PH - KH) / SH + 1\n"
        "    OW = (IW + 2*PW - KW) / SW + 1\n"
        "Definition (zero-padding implied by PH, PW):\n"
        "  output[n, oc, oh, ow] = bias[oc] + sum over ic, kh, kw of\n"
        "    input[n, ic, oh*SH - PH + kh, ow*SW - PW + kw]\n"
        "    * weight[oc, ic, kh, kw]\n"
        "  with input reads outside [0, IH) x [0, IW) treated as 0.\n"
        "All tensors are float32."
    ),
    reference_impl="""\
void kernel_conv2d(const float *input, const float *weight, const float *bias,
                   float *output,
                   int N, int IC, int IH, int IW, int OC,
                   int KH, int KW, int SH, int SW, int PH, int PW) {
    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;
    for (int n = 0; n < N; n++) {
        for (int oc = 0; oc < OC; oc++) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    float acc = bias ? bias[oc] : 0.0f;
                    for (int ic = 0; ic < IC; ic++) {
                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh * SH - PH + kh;
                            if (ih < 0 || ih >= IH) continue;
                            for (int kw = 0; kw < KW; kw++) {
                                int iw = ow * SW - PW + kw;
                                if (iw < 0 || iw >= IW) continue;
                                float v = input[((n*IC + ic)*IH + ih)*IW + iw];
                                float w = weight[((oc*IC + ic)*KH + kh)*KW + kw];
                                acc += v * w;
                            }
                        }
                    }
                    output[((n*OC + oc)*OH + oh)*OW + ow] = acc;
                }
            }
        }
    }
}
""",
    extra_shapes=[
        # LeNet
        {"N": 1, "IC": 1, "IH": 28, "IW": 28, "OC": 6,
         "KH": 5, "KW": 5, "SH": 1, "SW": 1, "PH": 0, "PW": 0},
        {"N": 1, "IC": 6, "IH": 12, "IW": 12, "OC": 16,
         "KH": 5, "KW": 5, "SH": 1, "SW": 1, "PH": 0, "PW": 0},
        # DroNet-style (3x3 stride=2 padding=1, 1x1 stride=2, 3x3 padding=1)
        {"N": 1, "IC": 32, "IH": 16, "IW": 16, "OC": 64,
         "KH": 3, "KW": 3, "SH": 2, "SW": 2, "PH": 1, "PW": 1},
        {"N": 1, "IC": 64, "IH": 8, "IW": 8, "OC": 64,
         "KH": 3, "KW": 3, "SH": 1, "SW": 1, "PH": 1, "PW": 1},
        {"N": 1, "IC": 32, "IH": 16, "IW": 16, "OC": 64,
         "KH": 1, "KW": 1, "SH": 2, "SW": 2, "PH": 0, "PW": 0},
        # Trained DroNet first conv (3-channel 112x112).
        {"N": 1, "IC": 3, "IH": 112, "IW": 112, "OC": 32,
         "KH": 3, "KW": 3, "SH": 2, "SW": 2, "PH": 1, "PW": 1},
        # Asymmetric / small / generalization
        {"N": 1, "IC": 3, "IH": 7, "IW": 5, "OC": 4,
         "KH": 3, "KW": 3, "SH": 1, "SW": 1, "PH": 1, "PW": 1},
    ],
    argtypes_factory=_conv2d_argtypes,
    algorithms=[
        AlgorithmCandidate(
            name="direct",
            description=(
                "Direct sliding-window convolution. Six nested loops over "
                "(n, oc, oh, ow, ic, kh, kw) reading input pixels with "
                "explicit bounds checks for padding. No extra memory. Fine "
                "for small spatial dims and 1x1 kernels but the boundary "
                "checks make vectorization awkward."
            ),
            reference_impl="""\
void kernel_conv2d(const float *input, const float *weight, const float *bias,
                   float *output,
                   int N, int IC, int IH, int IW, int OC,
                   int KH, int KW, int SH, int SW, int PH, int PW) {
    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;
    for (int n = 0; n < N; n++) {
        for (int oc = 0; oc < OC; oc++) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    float acc = bias ? bias[oc] : 0.0f;
                    for (int ic = 0; ic < IC; ic++) {
                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh * SH - PH + kh;
                            if (ih < 0 || ih >= IH) continue;
                            for (int kw = 0; kw < KW; kw++) {
                                int iw = ow * SW - PW + kw;
                                if (iw < 0 || iw >= IW) continue;
                                float v = input[((n*IC + ic)*IH + ih)*IW + iw];
                                float w = weight[((oc*IC + ic)*KH + kh)*KW + kw];
                                acc += v * w;
                            }
                        }
                    }
                    output[((n*OC + oc)*OH + oh)*OW + ow] = acc;
                }
            }
        }
    }
}
""",
        ),
        AlgorithmCandidate(
            name="im2col_gemm",
            description=(
                "im2col + GEMM lowering of 2D convolution. This is a "
                "two-stage algorithm — your kernel MUST keep the stages "
                "separate.\n\n"
                "STAGE 1 (im2col gather; no math, just data movement):\n"
                "  Allocate `float im2col_buf[OH*OW * IC*KH*KW];` as a "
                "stack VLA (the harness main stack is sized for this).\n"
                "  For each output position (oh, ow), gather the IC*KH*KW "
                "input values that contribute to that output into row "
                "(oh*OW + ow) of im2col_buf. Pad reads outside "
                "[0, IH) x [0, IW) with 0.0. This stage produces NO "
                "arithmetic ops — just loads, conditional zero-fills, and "
                "stores into im2col_buf.\n\n"
                "STAGE 2 (GEMM; all the arithmetic):\n"
                "  Compute output[n, oc, oh*OW + ow] = bias[oc] + "
                "sum_k im2col_buf[oh*OW + ow, k] * weight[oc, k] where "
                "weight is treated as an [OC, IC*KH*KW] matrix (identical "
                "memory layout to OIHW). Stage 2 reads exclusively from "
                "im2col_buf and weight; it MUST NOT touch the original "
                "`input` array.\n\n"
                "Why this lowering matters: stage 2 is a standard "
                "[OH*OW, K] x [OC, K]^T matmul where K = IC*KH*KW. The "
                "reduction loop over k vectorizes cleanly without any "
                "padding-aware indexing, regardless of the conv's stride or "
                "padding. The padding/stride logic is fully isolated in the "
                "non-arithmetic gather of stage 1.\n\n"
                "Numerical equivalence: as long as im2col_buf is populated "
                "in the order ic-major then kh, then kw, the matmul "
                "summation order matches direct convolution and the result "
                "is bit-equivalent.\n\n"
                "DO NOT fuse the two stages into one loop nest. DO NOT "
                "skip the im2col_buf allocation. DO NOT read from `input` "
                "inside stage 2. If you write a loop that reads from "
                "`input` and `weight` in the same iteration, you have "
                "fused the stages — that is the *direct* algorithm, not "
                "the im2col_gemm algorithm.\n\n"
                "VECTORIZATION GUIDANCE (scalar target backend should ignore "
                "this): Keep stage 1 (the gather) **scalar** for now — its "
                "memory access pattern is irregular and vectorizing it is "
                "bug-prone. Vectorize ONLY stage 2 (the matmul) over the "
                "reduction dim k. The canonical per-output-element pattern "
                "for the rvv backend is:\n\n"
                "  size_t vl;\n"
                "  size_t vlmax = __riscv_vsetvlmax_e32m1();\n"
                "  vfloat32m1_t vacc = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);\n"
                "  for (int k = 0; k < K; k += vl) {\n"
                "      vl = __riscv_vsetvl_e32m1(K - k);\n"
                "      vfloat32m1_t va = __riscv_vle32_v_f32m1(\n"
                "          &im2col_buf[row * K + k], vl);\n"
                "      vfloat32m1_t vw = __riscv_vle32_v_f32m1(\n"
                "          &weight[oc * K + k], vl);\n"
                "      vacc = __riscv_vfmacc_vv_f32m1(vacc, va, vw, vl);\n"
                "  }\n"
                "  vfloat32m1_t vinit = __riscv_vfmv_s_f_f32m1(0.0f, 1);\n"
                "  vfloat32m1_t vsum = __riscv_vfredusum_vs_f32m1_f32m1(\n"
                "      vacc, vinit, vlmax);\n"
                "  float acc = __riscv_vfmv_f_s_f32m1_f32(vsum)\n"
                "      + (bias ? bias[oc] : 0.0f);\n"
                "  output[(n * OC + oc) * M + row] = acc;\n\n"
                "Common bugs to avoid:\n"
                "  * vfmv_v_f (broadcast to ALL lanes) is the right "
                "init for the accumulator; vfmv_s_f (single-lane move) is "
                "for the reduction's scalar init slot only.\n"
                "  * vle32_v_f32m1 takes a POINTER, not a value: "
                "`&im2col_buf[row * K + k]` or `im2col_buf + row * K + k`. "
                "Writing `im2col_buf[row * K + k]` as the first arg passes "
                "a float and fails to compile.\n"
                "  * Add bias as a scalar AFTER the horizontal reduction. "
                "Don't try to broadcast bias into vacc and then reduce — "
                "that double-counts."
            ),
            reference_impl="""\
void kernel_conv2d(const float *input, const float *weight, const float *bias,
                   float *output,
                   int N, int IC, int IH, int IW, int OC,
                   int KH, int KW, int SH, int SW, int PH, int PW) {
    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;
    int M = OH * OW;
    int K = IC * KH * KW;

    /* Stack VLA — sized at runtime, harness stack is sized for it. */
    float im2col_buf[M * K];

    for (int n = 0; n < N; n++) {
        /* Stage 1: im2col gather, padding-aware. */
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                int row = oh * OW + ow;
                for (int ic = 0; ic < IC; ic++) {
                    for (int kh = 0; kh < KH; kh++) {
                        int ih = oh * SH - PH + kh;
                        for (int kw = 0; kw < KW; kw++) {
                            int iw = ow * SW - PW + kw;
                            int col = (ic * KH + kh) * KW + kw;
                            float v = 0.0f;
                            if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                                v = input[((n*IC + ic)*IH + ih)*IW + iw];
                            }
                            im2col_buf[row * K + col] = v;
                        }
                    }
                }
            }
        }

        /* Stage 2: GEMM. weight is [OC, K] (OIHW flattened), output is
         * NCHW which equals [N, OC, M] when M = OH*OW. */
        for (int oc = 0; oc < OC; oc++) {
            float b = bias ? bias[oc] : 0.0f;
            for (int row = 0; row < M; row++) {
                float acc = b;
                for (int k = 0; k < K; k++) {
                    acc += im2col_buf[row * K + k] * weight[oc * K + k];
                }
                output[(n * OC + oc) * M + row] = acc;
            }
        }
    }
}
""",
        ),
    ],
)


MAXPOOL2D = KernelSpec(
    op="maxpool2d",
    signature=(
        "void kernel_maxpool2d(const float *input, float *output, "
        "int N, int C, int IH, int IW, "
        "int KH, int KW, int SH, int SW)"
    ),
    semantics=(
        "2D max pooling matching torch.nn.MaxPool2d semantics with no padding "
        "and dilation=1.\n"
        "Layout (NCHW, row-major):\n"
        "  input:  [N, C, IH, IW]\n"
        "  output: [N, C, OH, OW]  with\n"
        "    OH = (IH - KH) / SH + 1\n"
        "    OW = (IW - KW) / SW + 1\n"
        "  output[n, c, oh, ow] = max over kh in [0,KH), kw in [0,KW) of\n"
        "    input[n, c, oh*SH + kh, ow*SW + kw].\n"
        "All tensors are float32."
    ),
    reference_impl="""\
void kernel_maxpool2d(const float *input, float *output,
                     int N, int C, int IH, int IW,
                     int KH, int KW, int SH, int SW) {
    int OH = (IH - KH) / SH + 1;
    int OW = (IW - KW) / SW + 1;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    int ih0 = oh * SH;
                    int iw0 = ow * SW;
                    float m = input[((n*C + c)*IH + ih0)*IW + iw0];
                    for (int kh = 0; kh < KH; kh++) {
                        for (int kw = 0; kw < KW; kw++) {
                            float v = input[((n*C + c)*IH + ih0+kh)*IW + iw0+kw];
                            if (v > m) m = v;
                        }
                    }
                    output[((n*C + c)*OH + oh)*OW + ow] = m;
                }
            }
        }
    }
}
""",
    extra_shapes=[
        # LeNet
        {"N": 1, "C": 6, "IH": 24, "IW": 24, "KH": 2, "KW": 2, "SH": 2, "SW": 2},
        {"N": 1, "C": 16, "IH": 8, "IW": 8, "KH": 2, "KW": 2, "SH": 2, "SW": 2},
        # DroNet
        {"N": 1, "C": 32, "IH": 64, "IW": 64, "KH": 3, "KW": 3, "SH": 2, "SW": 2},
        # Generalization
        {"N": 1, "C": 4, "IH": 9, "IW": 7, "KH": 3, "KW": 3, "SH": 2, "SW": 2},
    ],
    argtypes_factory=_maxpool2d_argtypes,
)


ADD = KernelSpec(
    op="add",
    signature=(
        "void kernel_add(const float *a, const float *b, float *output, int n)"
    ),
    semantics=(
        "Elementwise float32 add over a contiguous buffer:\n"
        "  output[i] = a[i] + b[i]   for i in [0, n)\n"
        "It must be safe for output to alias either input."
    ),
    reference_impl="""\
void kernel_add(const float *a, const float *b, float *output, int n) {
    for (int i = 0; i < n; i++) {
        output[i] = a[i] + b[i];
    }
}
""",
    extra_shapes=[
        {"n": 1},
        {"n": 17},
        {"n": 1024},
        {"n": 8192},          # DroNet residual block 0 output
    ],
    argtypes_factory=_add_argtypes,
)


BATCHNORM2D = KernelSpec(
    op="batchnorm2d",
    signature=(
        "void kernel_batchnorm2d(const float *input, const float *scale, "
        "const float *bias, float *output, int N, int C, int H, int W)"
    ),
    semantics=(
        "Per-channel affine on a NCHW float32 activation.\n"
        "  output[n, c, h, w] = scale[c] * input[n, c, h, w] + bias[c]\n"
        "scale and bias are length-C arrays. This is a fused form of an\n"
        "eval-mode torch.nn.BatchNorm2d: scale = gamma / sqrt(var + eps),\n"
        "bias  = beta - mean * scale. The kernel only sees the fused values."
    ),
    reference_impl="""\
void kernel_batchnorm2d(const float *input, const float *scale,
                        const float *bias, float *output,
                        int N, int C, int H, int W) {
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            float s = scale[c];
            float b = bias[c];
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    int idx = ((n*C + c)*H + h)*W + w;
                    output[idx] = s * input[idx] + b;
                }
            }
        }
    }
}
""",
    extra_shapes=[
        {"N": 1, "C": 32, "H": 16, "W": 16},
        {"N": 1, "C": 64, "H": 8, "W": 8},
        {"N": 1, "C": 128, "H": 4, "W": 4},
        {"N": 1, "C": 7, "H": 5, "W": 3},
    ],
    argtypes_factory=_batchnorm2d_argtypes,
)


SIGMOID = KernelSpec(
    op="sigmoid",
    signature=(
        "void kernel_sigmoid(const float *input, float *output, int n)"
    ),
    semantics=(
        "Elementwise sigmoid on a contiguous float32 buffer:\n"
        "  output[i] = 1.0f / (1.0f + expf(-input[i]))   for i in [0, n)\n"
        "expf comes from <math.h> which is already in scope."
    ),
    reference_impl="""\
void kernel_sigmoid(const float *input, float *output, int n) {
    for (int i = 0; i < n; i++) {
        output[i] = 1.0f / (1.0f + expf(-input[i]));
    }
}
""",
    extra_shapes=[
        {"n": 1},
        {"n": 16},
        {"n": 256},
    ],
    argtypes_factory=_sigmoid_argtypes,
)


# ---------------------------------------------------------------------------
# int8 op kinds (muRISCV-NN convention)
# ---------------------------------------------------------------------------

LINEAR_S8 = KernelSpec(
    op="linear_s8",
    signature=(
        "void kernel_linear_s8(const int8_t *input, const int8_t *weight, "
        "const int32_t *bias, int8_t *output, "
        "int M, int K, int N, "
        "int input_offset, int filter_offset, int output_offset, "
        "int output_multiplier, int output_shift, "
        "int activation_min, int activation_max)"
    ),
    semantics=(
        "Quantized fully-connected layer (per-tensor symmetric, "
        "muRISCV-NN / CMSIS-NN convention).\n"
        "Layout (row-major):\n"
        "  input:  int8  [M, K]\n"
        "  weight: int8  [N, K]   (out_features outer, like nn.Linear)\n"
        "  bias:   int32 [N]      (pre-scaled to s_in*s_w; may be NULL)\n"
        "  output: int8  [M, N]\n"
        "Compute (per output element):\n"
        "  acc = bias[n]   if bias != NULL else 0\n"
        "  for k in 0..K-1:\n"
        "    acc += (input[m, k] + input_offset)\n"
        "         * (weight[n, k] + filter_offset)\n"
        "  acc = requantize(acc, output_multiplier, output_shift)\n"
        "  acc += output_offset\n"
        "  acc = clamp(acc, activation_min, activation_max)\n"
        "  output[m, n] = (int8_t) acc\n\n"
        "requantize(x, mult, shift) is the canonical Q0.31 fixed-point\n"
        "rounding-multiply-and-shift used in CMSIS-NN / muRISCV-NN:\n"
        "  int64 prod = (int64) x * (int64) mult;\n"
        "  prod = (prod + (1LL << 30)) >> 31;          // round to even\n"
        "  if (shift > 0) {\n"
        "    int32 round = (1 << (shift - 1));\n"
        "    return ((int32) prod + round) >> shift;\n"
        "  } else {\n"
        "    return (int32) prod << -shift;\n"
        "  }\n\n"
        "Notes:\n"
        "  * activation_min/max fold ReLU into the requantize tail. For a\n"
        "    fused ReLU pass activation_min = output_offset (often 0 in\n"
        "    symmetric quant, where it just becomes 0).\n"
        "  * Symmetric per-tensor quant (the only mode the IR emits today)\n"
        "    means input_offset = filter_offset = output_offset = 0; the\n"
        "    kernel must still accept the parameters for API parity with\n"
        "    asymmetric variants we'll add later.\n"
    ),
    reference_impl="""\
void kernel_linear_s8(const int8_t *input, const int8_t *weight,
                      const int32_t *bias, int8_t *output,
                      int M, int K, int N,
                      int input_offset, int filter_offset, int output_offset,
                      int output_multiplier, int output_shift,
                      int activation_min, int activation_max) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t acc = bias ? bias[n] : 0;
            for (int k = 0; k < K; k++) {
                int32_t in_v = (int32_t)input[m * K + k] + input_offset;
                int32_t w_v  = (int32_t)weight[n * K + k] + filter_offset;
                acc += in_v * w_v;
            }
            /* Q0.31 rounding multiply. */
            int64_t prod = (int64_t)acc * (int64_t)output_multiplier;
            prod = (prod + (1LL << 30)) >> 31;
            int32_t scaled = (int32_t)prod;
            if (output_shift > 0) {
                int32_t round = (1 << (output_shift - 1));
                scaled = (scaled + round) >> output_shift;
            } else if (output_shift < 0) {
                scaled = scaled << (-output_shift);
            }
            scaled += output_offset;
            if (scaled < activation_min) scaled = activation_min;
            if (scaled > activation_max) scaled = activation_max;
            output[m * N + n] = (int8_t)scaled;
        }
    }
}
""",
    extra_shapes=[
        # MLP shapes
        {"M": 1, "K": 16, "N": 32},
        {"M": 1, "K": 32, "N": 32},
        {"M": 1, "K": 32, "N": 10},
        # Generalization
        {"M": 1, "K": 7,  "N": 13},
        {"M": 4, "K": 17, "N": 23},
    ],
    argtypes_factory=_linear_s8_argtypes,
)


RELU_S8 = KernelSpec(
    op="relu_s8",
    signature="void kernel_relu_s8(const int8_t *input, int8_t *output, int n)",
    semantics=(
        "Elementwise ReLU on a contiguous int8 buffer with symmetric "
        "quantization (zero_point = 0):\n"
        "  output[i] = max(0, input[i])  for i in [0, n)\n"
        "Standalone version; for fused linear→relu we use linear_s8 with "
        "activation_min = 0."
    ),
    reference_impl="""\
void kernel_relu_s8(const int8_t *input, int8_t *output, int n) {
    for (int i = 0; i < n; i++) {
        int8_t v = input[i];
        output[i] = v > 0 ? v : 0;
    }
}
""",
    extra_shapes=[
        {"n": 1},
        {"n": 17},
        {"n": 256},
    ],
    argtypes_factory=_relu_s8_argtypes,
)


CONV2D_S8 = KernelSpec(
    op="conv2d_s8",
    signature=(
        "void kernel_conv2d_s8(const int8_t *input, const int8_t *weight, "
        "const int32_t *bias, int8_t *output, "
        "int N, int IC, int IH, int IW, int OC, "
        "int KH, int KW, int SH, int SW, int PH, int PW, "
        "int input_offset, int filter_offset, int output_offset, "
        "int output_multiplier, int output_shift, "
        "int activation_min, int activation_max)"
    ),
    semantics=(
        "Quantized 2D convolution (per-tensor symmetric, muRISCV-NN / "
        "CMSIS-NN convention). Same dataflow as conv2d but in int8 with "
        "an int32 accumulator and a Q0.31 requantize tail.\n"
        "Layout (NCHW for activations, OIHW for weights):\n"
        "  input:  int8  [N, IC, IH, IW]\n"
        "  weight: int8  [OC, IC, KH, KW]\n"
        "  bias:   int32 [OC] (may be NULL)\n"
        "  output: int8  [N, OC, OH, OW]  with OH/OW per the conv2d formula\n"
        "Compute (per output element):\n"
        "  acc = bias[oc] if bias != NULL else 0\n"
        "  for ic, kh, kw with ih = oh*SH-PH+kh, iw = ow*SW-PW+kw:\n"
        "    if (ih, iw) in bounds:\n"
        "      acc += (input[n,ic,ih,iw] + input_offset)\n"
        "           * (weight[oc,ic,kh,kw] + filter_offset)\n"
        "    (input reads outside [0,IH) x [0,IW) treated as 0 — "
        "i.e. their contribution to acc is just `input_offset * "
        "(weight[..] + filter_offset)`. For the symmetric quant case "
        "input_offset = 0 those padded contributions are zero.)\n"
        "  acc = requantize(acc, output_multiplier, output_shift)\n"
        "  acc += output_offset\n"
        "  acc = clamp(acc, activation_min, activation_max)\n"
        "  output[n, oc, oh, ow] = (int8_t) acc\n"
        "Requantize is the same Q0.31 fixed-point rounding-multiply-and-"
        "shift as kernel_linear_s8."
    ),
    reference_impl="""\
void kernel_conv2d_s8(const int8_t *input, const int8_t *weight,
                      const int32_t *bias, int8_t *output,
                      int N, int IC, int IH, int IW, int OC,
                      int KH, int KW, int SH, int SW, int PH, int PW,
                      int input_offset, int filter_offset, int output_offset,
                      int output_multiplier, int output_shift,
                      int activation_min, int activation_max) {
    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;
    for (int n = 0; n < N; n++) {
        for (int oc = 0; oc < OC; oc++) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    int32_t acc = bias ? bias[oc] : 0;
                    for (int ic = 0; ic < IC; ic++) {
                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh * SH - PH + kh;
                            for (int kw = 0; kw < KW; kw++) {
                                int iw = ow * SW - PW + kw;
                                int32_t in_v;
                                if (ih < 0 || ih >= IH || iw < 0 || iw >= IW) {
                                    in_v = input_offset;
                                } else {
                                    in_v = (int32_t)input[((n*IC + ic)*IH + ih)*IW + iw]
                                         + input_offset;
                                }
                                int32_t w_v = (int32_t)weight[((oc*IC + ic)*KH + kh)*KW + kw]
                                            + filter_offset;
                                acc += in_v * w_v;
                            }
                        }
                    }
                    int64_t prod = (int64_t)acc * (int64_t)output_multiplier;
                    prod = (prod + (1LL << 30)) >> 31;
                    int32_t scaled = (int32_t)prod;
                    if (output_shift > 0) {
                        int32_t round = (1 << (output_shift - 1));
                        scaled = (scaled + round) >> output_shift;
                    } else if (output_shift < 0) {
                        scaled = scaled << (-output_shift);
                    }
                    scaled += output_offset;
                    if (scaled < activation_min) scaled = activation_min;
                    if (scaled > activation_max) scaled = activation_max;
                    output[((n*OC + oc)*OH + oh)*OW + ow] = (int8_t)scaled;
                }
            }
        }
    }
}
""",
    extra_shapes=[
        # LeNet shapes
        {"N": 1, "IC": 1, "IH": 28, "IW": 28, "OC": 6,
         "KH": 5, "KW": 5, "SH": 1, "SW": 1, "PH": 0, "PW": 0},
        {"N": 1, "IC": 6, "IH": 12, "IW": 12, "OC": 16,
         "KH": 5, "KW": 5, "SH": 1, "SW": 1, "PH": 0, "PW": 0},
        # DroNet-style
        {"N": 1, "IC": 32, "IH": 16, "IW": 16, "OC": 64,
         "KH": 3, "KW": 3, "SH": 2, "SW": 2, "PH": 1, "PW": 1},
        {"N": 1, "IC": 32, "IH": 16, "IW": 16, "OC": 64,
         "KH": 1, "KW": 1, "SH": 2, "SW": 2, "PH": 0, "PW": 0},
    ],
    argtypes_factory=_conv2d_s8_argtypes,
    algorithms=[
        AlgorithmCandidate(
            name="direct",
            description=(
                "Direct sliding-window int8 conv. Six nested loops over "
                "(n, oc, oh, ow, ic, kh, kw); per-element bounds checks; "
                "i32 accumulator; Q0.31 requantize tail per output element. "
                "No vectorization, no precomputation. The simplest correct "
                "form — useful as a baseline but on RVV it leaves all the "
                "vector lanes idle."
            ),
            reference_impl="""\
void kernel_conv2d_s8(const int8_t *input, const int8_t *weight,
                      const int32_t *bias, int8_t *output,
                      int N, int IC, int IH, int IW, int OC,
                      int KH, int KW, int SH, int SW, int PH, int PW,
                      int input_offset, int filter_offset, int output_offset,
                      int output_multiplier, int output_shift,
                      int activation_min, int activation_max) {
    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;
    for (int n = 0; n < N; n++) {
        for (int oc = 0; oc < OC; oc++) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    int32_t acc = bias ? bias[oc] : 0;
                    for (int ic = 0; ic < IC; ic++) {
                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh * SH - PH + kh;
                            for (int kw = 0; kw < KW; kw++) {
                                int iw = ow * SW - PW + kw;
                                int32_t in_v;
                                if (ih < 0 || ih >= IH || iw < 0 || iw >= IW) {
                                    in_v = input_offset;
                                } else {
                                    in_v = (int32_t)input[((n*IC + ic)*IH + ih)*IW + iw]
                                         + input_offset;
                                }
                                int32_t w_v = (int32_t)weight[((oc*IC + ic)*KH + kh)*KW + kw]
                                            + filter_offset;
                                acc += in_v * w_v;
                            }
                        }
                    }
                    int64_t prod = (int64_t)acc * (int64_t)output_multiplier;
                    prod = (prod + (1LL << 30)) >> 31;
                    int32_t scaled = (int32_t)prod;
                    if (output_shift > 0) {
                        int32_t round = (1 << (output_shift - 1));
                        scaled = (scaled + round) >> output_shift;
                    } else if (output_shift < 0) {
                        scaled = scaled << (-output_shift);
                    }
                    scaled += output_offset;
                    if (scaled < activation_min) scaled = activation_min;
                    if (scaled > activation_max) scaled = activation_max;
                    output[((n*OC + oc)*OH + oh)*OW + ow] = (int8_t)scaled;
                }
            }
        }
    }
}
""",
        ),
        AlgorithmCandidate(
            name="rvv_widening_oc",
            target_affinity=("rvv",),
            description=(
                "Vectorize over OUTPUT CHANNELS using RVV's widening "
                "integer intrinsics — the int8 analogue of the XNNPACK "
                "f32 OC-broadcast pattern (this same shape was the 16x "
                "win on fp32 conv2d). Stay in plain int across the "
                "accumulation hot path; only convert to fp32 for the "
                "requantize tail.\n\n"
                "WHY THIS HELPS:\n"
                "  Each i8 MAC, when written scalar, is 6+ instructions "
                "(load, sign-extend twice, add offsets, multiply, add). "
                "The widening-multiply path collapses i8xi8 -> i16 to "
                "one vwmul.vv that produces a vector of i16 partial "
                "products, and vwadd.wv folds that into an i32 vector "
                "accumulator. With LMUL=2 on i32 (so VLEN-bytes worth "
                "of accumulators in one register group), one inner "
                "iteration replaces a scalar inner iteration's worth of "
                "OC outputs.\n\n"
                "DATA LAYOUT (unchanged from input/output of "
                "kernel_conv2d_s8):\n"
                "  input:  int8 [N, IC, IH, IW]\n"
                "  weight: int8 [OC, IC, KH, KW]\n"
                "  output: int8 [N, OC, OH, OW]\n\n"
                "ALGORITHM (for each (n, oh, ow) — outer 3 loops "
                "scalar, oc loop is the vectorized one):\n"
                "  for oc_base in [0, OC) step vl:\n"
                "      vl = vsetvl_e32m2(OC - oc_base);\n"
                "      // bias[oc_base..oc_base+vl] -> i32 vector acc\n"
                "      vint32m2_t vacc;\n"
                "      if (bias) vacc = __riscv_vle32_v_i32m2(\n"
                "          bias + oc_base, vl);\n"
                "      else      vacc = __riscv_vmv_v_x_i32m2(0, vl);\n"
                "      for ic, kh, kw with bounds-checked ih/iw:\n"
                "          int8_t in_byte =\n"
                "              (in-bounds) ? input[((n*IC+ic)*IH+ih)*IW+iw]\n"
                "                          : 0;\n"
                "          int32_t in_v = (int32_t)in_byte + input_offset;\n"
                "          // Strided load of OC weight bytes. Stride =\n"
                "          // IC*KH*KW (bytes) since weight is OIHW.\n"
                "          vint8m1_t vw8 = __riscv_vlse8_v_i8m1(\n"
                "              weight + (oc_base*IC*KH*KW)\n"
                "                     + (ic*KH + kh)*KW + kw,\n"
                "              (ptrdiff_t)IC*KH*KW, vl);\n"
                "          // Widen i8 -> i16 and add filter_offset.\n"
                "          vint16m2_t vw16 = __riscv_vwadd_vx_i16m2(\n"
                "              vw8, filter_offset, vl);\n"
                "          // Widen-multiply by the input scalar:\n"
                "          // vwmul.vx i32 = i16 * scalar (sign-ext).\n"
                "          // We have vacc (i32) += vw16 * in_v\n"
                "          // implemented as widening-multiply-accumulate:\n"
                "          vacc = __riscv_vwmacc_vx_i32m2(\n"
                "              vacc, (int16_t)in_v, vw16, vl);\n"
                "      }}}\n"
                "      // Q0.31 requantize tail. Keep this in scalar i32 "
                "math (matches the reference bit-exactly). Pull each "
                "lane out, requantize, store.\n"
                "      int32_t accs[vl_max];\n"
                "      __riscv_vse32_v_i32m2(accs, vacc, vl);\n"
                "      for j in [0, vl):\n"
                "          int32_t acc = accs[j];\n"
                "          int64_t prod = (int64_t)acc * output_multiplier;\n"
                "          prod = (prod + (1LL<<30)) >> 31;\n"
                "          int32_t s = (int32_t)prod;\n"
                "          if (output_shift > 0)\n"
                "              s = (s + (1 << (output_shift-1))) >> output_shift;\n"
                "          else if (output_shift < 0)\n"
                "              s <<= -output_shift;\n"
                "          s += output_offset;\n"
                "          if (s < activation_min) s = activation_min;\n"
                "          if (s > activation_max) s = activation_max;\n"
                "          output[((n*OC + oc_base + j)*OH + oh)*OW + ow]\n"
                "              = (int8_t)s;\n\n"
                "INTRINSIC NAMES (RVV V1.0, exact spellings):\n"
                "  __riscv_vsetvl_e32m2(size_t avl) -> size_t\n"
                "  __riscv_vle32_v_i32m2(const int32_t*, size_t)\n"
                "  __riscv_vmv_v_x_i32m2(int32_t, size_t)\n"
                "  __riscv_vlse8_v_i8m1(const int8_t*, ptrdiff_t bstride, size_t)\n"
                "  __riscv_vwadd_vx_i16m2(vint8m1_t, int16_t, size_t)\n"
                "  __riscv_vwmacc_vx_i32m2(vint32m2_t, int16_t, vint16m2_t, size_t)\n"
                "  __riscv_vse32_v_i32m2(int32_t*, vint32m2_t, size_t)\n\n"
                "DO NOT use any vfXXX (fp) intrinsics — the requantize "
                "must be Q0.31 fixed-point exactly (the verify oracle "
                "checks bit-equal to the scalar reference).\n"
                "DO NOT vectorize across (ic, kh, kw) — they're a "
                "reduction; the vector dim is OC. The strided weight "
                "load is what feeds OC-many filter elements per inner "
                "iteration; do not flatten OIHW into a contiguous read.\n"
                "DO handle the boundary case where vl_max may be < OC "
                "(loop on oc_base += vl).\n"
                "DO put the bounds check (ih, iw padding) outside the "
                "vector op — the in/out of bounds decision is on the "
                "scalar input pixel, not on the OC fan-out."
            ),
            reference_impl="""\
void kernel_conv2d_s8(const int8_t *input, const int8_t *weight,
                      const int32_t *bias, int8_t *output,
                      int N, int IC, int IH, int IW, int OC,
                      int KH, int KW, int SH, int SW, int PH, int PW,
                      int input_offset, int filter_offset, int output_offset,
                      int output_multiplier, int output_shift,
                      int activation_min, int activation_max) {
    /* OC-vectorized i8 conv, RVV V1.0.
     *
     * LMUL choice: i32 acc at LMUL=2 means we need i16 ops at LMUL=1
     * and i8 ops at LMUL=1/2 (fractional) to stay element-count-matched
     * across widening intrinsics. vsetvl_e32m2(remaining_OC) gives one
     * vl that's reused for the i8 strided load (vlse8_v_i8mf2), the
     * widening offset add (vwadd_vx_i16m1), and the widening MAC
     * (vwmacc_vx_i32m2). VLEN=128 -> vlmax for e32m2 == 8 OC elems per
     * chunk. */
    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;
    const ptrdiff_t oc_stride_bytes =
        (ptrdiff_t)IC * (ptrdiff_t)KH * (ptrdiff_t)KW * (ptrdiff_t)sizeof(int8_t);

    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                int oc_base = 0;
                while (oc_base < OC) {
                    size_t vl = __riscv_vsetvl_e32m2((size_t)(OC - oc_base));
                    vint32m2_t vacc;
                    if (bias != NULL) {
                        vacc = __riscv_vle32_v_i32m2(bias + oc_base, vl);
                    } else {
                        vacc = __riscv_vmv_v_x_i32m2(0, vl);
                    }
                    for (int ic = 0; ic < IC; ic++) {
                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh * SH - PH + kh;
                            int row_in_bounds = (ih >= 0 && ih < IH);
                            for (int kw = 0; kw < KW; kw++) {
                                int iw = ow * SW - PW + kw;
                                int8_t in_byte = 0;
                                if (row_in_bounds && iw >= 0 && iw < IW) {
                                    in_byte = input[((n*IC + ic)*IH + ih)*IW + iw];
                                }
                                int32_t in_v = (int32_t)in_byte + input_offset;
                                const int8_t *wp = weight
                                    + (size_t)oc_base * (size_t)IC * KH * KW
                                    + ((size_t)ic * KH + kh) * KW + kw;
                                /* strided i8 load at LMUL=1/2 so the
                                 * subsequent widen produces i16 LMUL=1. */
                                vint8mf2_t vw8 = __riscv_vlse8_v_i8mf2(
                                    wp, oc_stride_bytes, vl);
                                /* i8mf2 -> i16m1, fold filter_offset in. */
                                vint16m1_t vw16 = __riscv_vwadd_vx_i16m1(
                                    vw8, (int16_t)filter_offset, vl);
                                /* i32m2 += i16m1 * i16-scalar-extended-from-int. */
                                vacc = __riscv_vwmacc_vx_i32m2(
                                    vacc, (int16_t)in_v, vw16, vl);
                            }
                        }
                    }
                    /* Pull lanes out and run the Q0.31 requantize
                     * scalar — bit-exact match to the reference. */
                    int32_t lane[64];   /* vlmax for e32m2 at VLEN=512 is 32; 64 is generous */
                    __riscv_vse32_v_i32m2(lane, vacc, vl);
                    for (size_t j = 0; j < vl; j++) {
                        int32_t acc = lane[j];
                        int64_t prod = (int64_t)acc * (int64_t)output_multiplier;
                        prod = (prod + (1LL << 30)) >> 31;
                        int32_t scaled = (int32_t)prod;
                        if (output_shift > 0) {
                            int32_t round = (1 << (output_shift - 1));
                            scaled = (scaled + round) >> output_shift;
                        } else if (output_shift < 0) {
                            scaled = scaled << (-output_shift);
                        }
                        scaled += output_offset;
                        if (scaled < activation_min) scaled = activation_min;
                        if (scaled > activation_max) scaled = activation_max;
                        output[((n*OC + oc_base + (int)j)*OH + oh)*OW + ow] =
                            (int8_t)scaled;
                    }
                    oc_base += (int)vl;
                }
            }
        }
    }
}
""",
        ),
    ],
)


MAXPOOL2D_S8 = KernelSpec(
    op="maxpool2d_s8",
    signature=(
        "void kernel_maxpool2d_s8(const int8_t *input, int8_t *output, "
        "int N, int C, int IH, int IW, "
        "int KH, int KW, int SH, int SW)"
    ),
    semantics=(
        "Quantized 2D max-pool. Identical dataflow to fp32 maxpool2d but "
        "operating on int8 lanes. No requantize is needed — max is just a "
        "compare, and selecting an int8 input directly produces an int8 "
        "output at the same scale.\n"
        "Layout (NCHW):\n"
        "  input:  int8 [N, C, IH, IW]\n"
        "  output: int8 [N, C, OH, OW]  with OH = (IH-KH)/SH + 1, OW analog\n"
        "  output[n, c, oh, ow] = max over kh in [0,KH), kw in [0,KW) of\n"
        "    input[n, c, oh*SH + kh, ow*SW + kw]"
    ),
    reference_impl="""\
void kernel_maxpool2d_s8(const int8_t *input, int8_t *output,
                         int N, int C, int IH, int IW,
                         int KH, int KW, int SH, int SW) {
    int OH = (IH - KH) / SH + 1;
    int OW = (IW - KW) / SW + 1;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    int ih0 = oh * SH;
                    int iw0 = ow * SW;
                    int8_t m = input[((n*C + c)*IH + ih0)*IW + iw0];
                    for (int kh = 0; kh < KH; kh++) {
                        for (int kw = 0; kw < KW; kw++) {
                            int8_t v = input[((n*C + c)*IH + ih0+kh)*IW + iw0+kw];
                            if (v > m) m = v;
                        }
                    }
                    output[((n*C + c)*OH + oh)*OW + ow] = m;
                }
            }
        }
    }
}
""",
    extra_shapes=[
        # LeNet
        {"N": 1, "C": 6, "IH": 24, "IW": 24, "KH": 2, "KW": 2, "SH": 2, "SW": 2},
        {"N": 1, "C": 16, "IH": 8, "IW": 8, "KH": 2, "KW": 2, "SH": 2, "SW": 2},
        # DroNet
        {"N": 1, "C": 32, "IH": 64, "IW": 64, "KH": 3, "KW": 3, "SH": 2, "SW": 2},
    ],
    argtypes_factory=_maxpool2d_s8_argtypes,
)


ADD_S8 = KernelSpec(
    op="add_s8",
    signature=(
        "void kernel_add_s8(const int8_t *a, const int8_t *b, int8_t *output, "
        "int n, float scale_a, float scale_b, float scale_out, "
        "int activation_min, int activation_max)"
    ),
    semantics=(
        "Quantized elementwise add. The two inputs may have different "
        "scales — they are dequantized to float, summed, and re-quantized "
        "into the output's scale. Internal float32 math; result is\n"
        "  output[i] = clamp(roundf((a[i]*scale_a + b[i]*scale_b)/scale_out),\n"
        "                    activation_min, activation_max)\n"
        "Use roundf (round-to-nearest, ties to even on most platforms; the\n"
        "verify simulator matches by using numpy.float32 + numpy.round)."
    ),
    reference_impl="""\
void kernel_add_s8(const int8_t *a, const int8_t *b, int8_t *output, int n,
                   float scale_a, float scale_b, float scale_out,
                   int activation_min, int activation_max) {
    for (int i = 0; i < n; i++) {
        float fa = (float)a[i] * scale_a;
        float fb = (float)b[i] * scale_b;
        float fout = (fa + fb) / scale_out;
        int32_t v = (int32_t)roundf(fout);
        if (v < activation_min) v = activation_min;
        if (v > activation_max) v = activation_max;
        output[i] = (int8_t)v;
    }
}
""",
    extra_shapes=[{"n": 1}, {"n": 17}, {"n": 8192}],
    argtypes_factory=_add_s8_argtypes,
)


BATCHNORM2D_S8 = KernelSpec(
    op="batchnorm2d_s8",
    signature=(
        "void kernel_batchnorm2d_s8(const int8_t *input, const float *scale, "
        "const float *bias, int8_t *output, int N, int C, int H, int W, "
        "float scale_in, float scale_out, int activation_min, int activation_max)"
    ),
    semantics=(
        "Quantized BN: per-channel float affine on int8 activations.\n"
        "  output[n, c, h, w] = clamp(\n"
        "      roundf((scale[c] * (input[n, c, h, w] * scale_in) + bias[c])\n"
        "             / scale_out),\n"
        "      activation_min, activation_max)\n"
        "scale and bias are the eval-mode-folded BatchNorm parameters\n"
        "(scale = gamma / sqrt(var + eps), bias = beta - mean * scale).\n"
        "Per-channel float arrays of length C."
    ),
    reference_impl="""\
void kernel_batchnorm2d_s8(const int8_t *input, const float *scale,
                           const float *bias, int8_t *output,
                           int N, int C, int H, int W,
                           float scale_in, float scale_out,
                           int activation_min, int activation_max) {
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            float s = scale[c];
            float b = bias[c];
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    int idx = ((n*C + c)*H + h)*W + w;
                    float fv = (float)input[idx] * scale_in;
                    float y = s * fv + b;
                    int32_t v = (int32_t)roundf(y / scale_out);
                    if (v < activation_min) v = activation_min;
                    if (v > activation_max) v = activation_max;
                    output[idx] = (int8_t)v;
                }
            }
        }
    }
}
""",
    extra_shapes=[
        {"N": 1, "C": 32, "H": 16, "W": 16},
        {"N": 1, "C": 64, "H": 8, "W": 8},
        {"N": 1, "C": 128, "H": 4, "W": 4},
    ],
    argtypes_factory=_batchnorm2d_s8_argtypes,
)


SIGMOID_S8 = KernelSpec(
    op="sigmoid_s8",
    signature=(
        "void kernel_sigmoid_s8(const int8_t *input, int8_t *output, int n, "
        "float scale_in, float scale_out, int activation_min, int activation_max)"
    ),
    semantics=(
        "Elementwise quantized sigmoid:\n"
        "  output[i] = clamp(\n"
        "      roundf( (1 / (1 + expf(-input[i] * scale_in))) / scale_out ),\n"
        "      activation_min, activation_max)\n"
        "Tiny tensors (often n=1 at a model output head) — scalar loop is\n"
        "fine."
    ),
    reference_impl="""\
void kernel_sigmoid_s8(const int8_t *input, int8_t *output, int n,
                       float scale_in, float scale_out,
                       int activation_min, int activation_max) {
    for (int i = 0; i < n; i++) {
        float fv = (float)input[i] * scale_in;
        float sig = 1.0f / (1.0f + expf(-fv));
        int32_t v = (int32_t)roundf(sig / scale_out);
        if (v < activation_min) v = activation_min;
        if (v > activation_max) v = activation_max;
        output[i] = (int8_t)v;
    }
}
""",
    extra_shapes=[{"n": 1}, {"n": 16}],
    argtypes_factory=_sigmoid_s8_argtypes,
)


CONV2D_DW = KernelSpec(
    op="conv2d_dw",
    signature=(
        "void kernel_conv2d_dw(const float *input, const float *weight, "
        "const float *bias, float *output, "
        "int N, int C, int IH, int IW, "
        "int KH, int KW, int SH, int SW, int PH, int PW)"
    ),
    semantics=(
        "Depthwise 2D convolution — each input channel has its OWN filter,\n"
        "applied independently. Equivalent to torch.nn.Conv2d with\n"
        "groups=in_channels=out_channels=C and dilation=1.\n"
        "Layout (row-major):\n"
        "  input:  [N, C, IH, IW]\n"
        "  weight: [C, 1, KH, KW]   (one KHxKW filter per channel)\n"
        "  bias:   [C] (may be NULL — treat as zeros)\n"
        "  output: [N, C, OH, OW]   with\n"
        "    OH = (IH + 2*PH - KH) / SH + 1\n"
        "    OW = (IW + 2*PW - KW) / SW + 1\n"
        "Definition (zero-padding):\n"
        "  output[n, c, oh, ow] = bias[c] + sum over kh, kw of\n"
        "    input[n, c, oh*SH - PH + kh, ow*SW - PW + kw]\n"
        "    * weight[c, 0, kh, kw]\n"
        "  with input reads outside [0, IH) x [0, IW) treated as 0.\n"
        "All tensors are float32."
    ),
    reference_impl="""\
void kernel_conv2d_dw(const float *input, const float *weight, const float *bias,
                      float *output,
                      int N, int C, int IH, int IW,
                      int KH, int KW, int SH, int SW, int PH, int PW) {
    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    float acc = bias ? bias[c] : 0.0f;
                    for (int kh = 0; kh < KH; kh++) {
                        int ih = oh * SH - PH + kh;
                        if (ih < 0 || ih >= IH) continue;
                        for (int kw = 0; kw < KW; kw++) {
                            int iw = ow * SW - PW + kw;
                            if (iw < 0 || iw >= IW) continue;
                            float v = input[((n*C + c)*IH + ih)*IW + iw];
                            float w = weight[(c*KH + kh)*KW + kw];
                            acc += v * w;
                        }
                    }
                    output[((n*C + c)*OH + oh)*OW + ow] = acc;
                }
            }
        }
    }
}
""",
    extra_shapes=[
        # MobileNetV2 width_mult=0.25 first-block 3x3 stride=1 dw conv
        {"N": 1, "C": 8, "IH": 56, "IW": 56,
         "KH": 3, "KW": 3, "SH": 1, "SW": 1, "PH": 1, "PW": 1},
        # MobileNetV2 stride=2 dw transition
        {"N": 1, "C": 24, "IH": 56, "IW": 56,
         "KH": 3, "KW": 3, "SH": 2, "SW": 2, "PH": 1, "PW": 1},
        # Asymmetric / odd shapes
        {"N": 1, "C": 4, "IH": 7, "IW": 5,
         "KH": 3, "KW": 3, "SH": 1, "SW": 1, "PH": 1, "PW": 1},
    ],
    argtypes_factory=_conv2d_dw_argtypes,
)


RELU6 = KernelSpec(
    op="relu6",
    signature="void kernel_relu6(const float *input, float *output, int n)",
    semantics=(
        "Elementwise ReLU6 on a contiguous float32 buffer:\n"
        "  output[i] = min(max(0.0f, input[i]), 6.0f)  for i in [0, n)\n"
        "It must be safe for `input` and `output` to alias."
    ),
    reference_impl="""\
void kernel_relu6(const float *input, float *output, int n) {
    for (int i = 0; i < n; i++) {
        float v = input[i];
        if (v < 0.0f) v = 0.0f;
        if (v > 6.0f) v = 6.0f;
        output[i] = v;
    }
}
""",
    extra_shapes=[
        {"n": 1},
        {"n": 17},
        {"n": 1024},
    ],
    argtypes_factory=_relu6_argtypes,
)


ADAPTIVE_AVG_POOL2D = KernelSpec(
    op="adaptive_avg_pool2d",
    signature=(
        "void kernel_adaptive_avg_pool2d(const float *input, float *output, "
        "int N, int C, int IH, int IW)"
    ),
    semantics=(
        "Global average pooling — collapses [N, C, IH, IW] to [N, C, 1, 1]\n"
        "by averaging over each channel's HxW plane. Matches\n"
        "torch.nn.AdaptiveAvgPool2d(output_size=1).\n"
        "Definition:\n"
        "  output[n, c, 0, 0] = (1 / (IH*IW)) * sum over ih, iw of\n"
        "    input[n, c, ih, iw]\n"
        "All tensors are float32. (Generic AdaptiveAvgPool2d to non-1x1\n"
        "outputs would partition the spatial dims into output-sized chunks;\n"
        "we only support 1x1 outputs since that's what classifiers use.)"
    ),
    reference_impl="""\
void kernel_adaptive_avg_pool2d(const float *input, float *output,
                                int N, int C, int IH, int IW) {
    int n_per_chan = IH * IW;
    float inv = 1.0f / (float)n_per_chan;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const float *src = input + ((n*C + c) * IH) * IW;
            float acc = 0.0f;
            for (int i = 0; i < n_per_chan; i++) {
                acc += src[i];
            }
            output[n*C + c] = acc * inv;
        }
    }
}
""",
    extra_shapes=[
        {"N": 1, "C": 16, "IH": 7, "IW": 7},
        {"N": 1, "C": 320, "IH": 7, "IW": 7},
        {"N": 1, "C": 4, "IH": 1, "IW": 1},
    ],
    argtypes_factory=_adaptive_avg_pool2d_argtypes,
)


KERNEL_SPECS: dict[str, KernelSpec] = {
    "linear": LINEAR,
    "relu": RELU,
    "relu6": RELU6,
    "elu": ELU,
    "conv2d": CONV2D,
    "conv2d_dw": CONV2D_DW,
    "maxpool2d": MAXPOOL2D,
    "adaptive_avg_pool2d": ADAPTIVE_AVG_POOL2D,
    "add": ADD,
    "batchnorm2d": BATCHNORM2D,
    "sigmoid": SIGMOID,
    "linear_s8": LINEAR_S8,
    "relu_s8": RELU_S8,
    "conv2d_s8": CONV2D_S8,
    "maxpool2d_s8": MAXPOOL2D_S8,
    "add_s8": ADD_S8,
    "batchnorm2d_s8": BATCHNORM2D_S8,
    "sigmoid_s8": SIGMOID_S8,
}


def shapes_from_ir(ir: dict, op: str) -> list[dict[str, int]]:
    """Pull every distinct shape combo for `op` out of an IR graph."""
    seen: set[tuple] = set()
    out: list[dict[str, int]] = []
    for node in ir.get("ops", []):
        if node["op"] != op:
            continue
        shape = node.get("shape", {})
        key = tuple(sorted(shape.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(shape))
    return out
