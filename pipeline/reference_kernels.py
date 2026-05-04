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


def _matmul_argtypes():
    import ctypes
    fp = ctypes.POINTER(ctypes.c_float)
    # A, B, C, M, K, N
    return [fp, fp, fp, ctypes.c_int, ctypes.c_int, ctypes.c_int]


def _bmm_argtypes():
    import ctypes
    fp = ctypes.POINTER(ctypes.c_float)
    # A, B, C, batch, M, K, N
    return [fp, fp, fp, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]


def _matmul_f16_argtypes():
    import ctypes
    h = ctypes.POINTER(ctypes.c_uint16)
    # A, B, C, M, K, N
    return [h, h, h, ctypes.c_int, ctypes.c_int, ctypes.c_int]


def _bmm_f16_argtypes():
    import ctypes
    h = ctypes.POINTER(ctypes.c_uint16)
    # A, B, C, batch, M, K, N
    return [h, h, h, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]


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
    # input, output, N, C, IH, IW, KH, KW, SH, SW, PH, PW, DH, DW
    return [fp, fp] + [ctypes.c_int] * 12


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


# KernelBench Phase 2 activations — all pointwise, signatures shaped
# like the existing relu / sigmoid / elu kernels. The parametric ones
# (leaky_relu, hardtanh) take their constants as float scalar args; the
# rest match (input, output, n).

def _pointwise_argtypes():
    """Standard signature for parameter-free pointwise activations:
    void kernel(const float *in, float *out, int n)."""
    import ctypes
    fp = ctypes.POINTER(ctypes.c_float)
    return [fp, fp, ctypes.c_int]


def _leaky_relu_argtypes():
    import ctypes
    fp = ctypes.POINTER(ctypes.c_float)
    # input, output, n, negative_slope
    return [fp, fp, ctypes.c_int, ctypes.c_float]


def _hardtanh_argtypes():
    import ctypes
    fp = ctypes.POINTER(ctypes.c_float)
    # input, output, n, min_val, max_val
    return [fp, fp, ctypes.c_int, ctypes.c_float, ctypes.c_float]


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
    # input, output, N, C, IH, IW, KH, KW, SH, SW, PH, PW, DH, DW
    return [i8p, i8p] + [ctypes.c_int] * 12


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


# ---------------------------------------------------------------------------
# fp16 (half-precision) argtypes. ctypes has no native _Float16, so the host
# verify path uses c_uint16 as a 16-bit opaque blob — bit-identical layout
# to _Float16, just with the math done in numpy. Only relevant for
# BACKEND=llm verify; BACKEND=reference dumps the C reference impl directly
# and never touches ctypes.
# ---------------------------------------------------------------------------

def _relu_f16_argtypes():
    import ctypes
    h = ctypes.POINTER(ctypes.c_uint16)
    return [h, h, ctypes.c_int]


def _elu_f16_argtypes():
    import ctypes
    h = ctypes.POINTER(ctypes.c_uint16)
    # input, output, n, alpha (passed as fp32 — gets converted in the kernel)
    return [h, h, ctypes.c_int, ctypes.c_float]


def _sigmoid_f16_argtypes():
    import ctypes
    h = ctypes.POINTER(ctypes.c_uint16)
    return [h, h, ctypes.c_int]


def _conv2d_f16_argtypes():
    import ctypes
    h = ctypes.POINTER(ctypes.c_uint16)
    return [h, h, h, h] + [ctypes.c_int] * 11


def _maxpool2d_f16_argtypes():
    import ctypes
    h = ctypes.POINTER(ctypes.c_uint16)
    return [h, h] + [ctypes.c_int] * 12


def _batchnorm2d_f16_argtypes():
    import ctypes
    h = ctypes.POINTER(ctypes.c_uint16)
    # input, scale, bias, output, N, C, H, W
    return [h, h, h, h] + [ctypes.c_int] * 4


# Phase 2 fp16 argtypes — same C-side ABI shapes as the fp32 specs,
# with c_uint16 standing in for _Float16 since Python ctypes has no
# native half. The host-verify (LLM optimize loop) is by-bits compares
# anyway; only the reference_impl C body sees the actual _Float16.

def _pointwise_f16_argtypes():
    import ctypes
    h = ctypes.POINTER(ctypes.c_uint16)
    return [h, h, ctypes.c_int]


def _leaky_relu_f16_argtypes():
    import ctypes
    h = ctypes.POINTER(ctypes.c_uint16)
    return [h, h, ctypes.c_int, ctypes.c_float]


def _hardtanh_f16_argtypes():
    import ctypes
    h = ctypes.POINTER(ctypes.c_uint16)
    return [h, h, ctypes.c_int, ctypes.c_float, ctypes.c_float]


def _reduce_f16_argtypes():
    import ctypes
    h = ctypes.POINTER(ctypes.c_uint16)
    return [h, h, ctypes.c_int, ctypes.c_int, ctypes.c_int]


def _argreduce_f16_argtypes():
    import ctypes
    h = ctypes.POINTER(ctypes.c_uint16)
    i64p = ctypes.POINTER(ctypes.c_int64)
    return [h, i64p, ctypes.c_int, ctypes.c_int, ctypes.c_int]


def _frobenius_norm_f16_argtypes():
    import ctypes
    h = ctypes.POINTER(ctypes.c_uint16)
    return [h, h, ctypes.c_int]


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


_MATMUL_EXTRA_SHAPES = [
    {"M": 1,  "K": 1,  "N": 1},
    {"M": 1,  "K": 7,  "N": 13},
    {"M": 4,  "K": 17, "N": 23},
    {"M": 1,  "K": 64, "N": 64},
    {"M": 32, "K": 32, "N": 32},
]

MATMUL = KernelSpec(
    op="matmul",
    signature=(
        "void kernel_matmul(const float *A, const float *B, float *C, "
        "int M, int K, int N)"
    ),
    semantics=(
        "Dense matrix multiply C = A @ B (row-major contiguous):\n"
        "  A: [M, K],  B: [K, N],  C: [M, N]\n"
        "  C[m, n] = sum_{k=0..K-1} A[m*K+k] * B[k*N+n]\n"
        "All tensors float32."
    ),
    reference_impl="""\
void kernel_matmul(const float *A, const float *B, float *C,
                   int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++)
                acc += A[m * K + k] * B[k * N + n];
            C[m * N + n] = acc;
        }
    }
}
""",
    extra_shapes=_MATMUL_EXTRA_SHAPES,
    argtypes_factory=_matmul_argtypes,
)

MATMUL_TA = KernelSpec(
    op="matmul_ta",
    signature=(
        "void kernel_matmul_ta(const float *A, const float *B, float *C, "
        "int M, int K, int N)"
    ),
    semantics=(
        "Dense matrix multiply C = A.T @ B (A stored in (K,M) order):\n"
        "  A stored: [K, M] — A[k, m] = A[k*M+m]\n"
        "  B: [K, N],  C: [M, N]\n"
        "  C[m, n] = sum_{k=0..K-1} A[k*M+m] * B[k*N+n]\n"
        "All tensors float32."
    ),
    reference_impl="""\
void kernel_matmul_ta(const float *A, const float *B, float *C,
                      int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++)
                acc += A[k * M + m] * B[k * N + n];
            C[m * N + n] = acc;
        }
    }
}
""",
    extra_shapes=_MATMUL_EXTRA_SHAPES,
    argtypes_factory=_matmul_argtypes,
)

MATMUL_TB = KernelSpec(
    op="matmul_tb",
    signature=(
        "void kernel_matmul_tb(const float *A, const float *B, float *C, "
        "int M, int K, int N)"
    ),
    semantics=(
        "Dense matrix multiply C = A @ B.T (B stored in (N,K) order):\n"
        "  A: [M, K],  B stored: [N, K] — B[n, k] = B[n*K+k]\n"
        "  C: [M, N]\n"
        "  C[m, n] = sum_{k=0..K-1} A[m*K+k] * B[n*K+k]\n"
        "All tensors float32."
    ),
    reference_impl="""\
void kernel_matmul_tb(const float *A, const float *B, float *C,
                      int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++)
                acc += A[m * K + k] * B[n * K + k];
            C[m * N + n] = acc;
        }
    }
}
""",
    extra_shapes=_MATMUL_EXTRA_SHAPES,
    argtypes_factory=_matmul_argtypes,
)

MATMUL_TATB = KernelSpec(
    op="matmul_tatb",
    signature=(
        "void kernel_matmul_tatb(const float *A, const float *B, float *C, "
        "int M, int K, int N)"
    ),
    semantics=(
        "Dense matrix multiply C = A.T @ B.T (both stored transposed):\n"
        "  A stored: [K, M] — A[k, m] = A[k*M+m]\n"
        "  B stored: [N, K] — B[n, k] = B[n*K+k]\n"
        "  C: [M, N]\n"
        "  C[m, n] = sum_{k=0..K-1} A[k*M+m] * B[n*K+k]\n"
        "All tensors float32."
    ),
    reference_impl="""\
void kernel_matmul_tatb(const float *A, const float *B, float *C,
                        int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++)
                acc += A[k * M + m] * B[n * K + k];
            C[m * N + n] = acc;
        }
    }
}
""",
    extra_shapes=_MATMUL_EXTRA_SHAPES,
    argtypes_factory=_matmul_argtypes,
)

BMM = KernelSpec(
    op="bmm",
    signature=(
        "void kernel_bmm(const float *A, const float *B, float *C, "
        "int batch, int M, int K, int N)"
    ),
    semantics=(
        "Batched matrix multiply C[b] = A[b] @ B[b] (row-major contiguous):\n"
        "  A: [batch, M, K],  B: [batch, K, N],  C: [batch, M, N]\n"
        "  C[b,m,n] = sum_{k=0..K-1} A[b*M*K + m*K+k] * B[b*K*N + k*N+n]\n"
        "All tensors float32."
    ),
    reference_impl="""\
void kernel_bmm(const float *A, const float *B, float *C,
                int batch, int M, int K, int N) {
    for (int b = 0; b < batch; b++) {
        const float *Ab = A + b * M * K;
        const float *Bb = B + b * K * N;
        float *Cb = C + b * M * N;
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float acc = 0.0f;
                for (int k = 0; k < K; k++)
                    acc += Ab[m * K + k] * Bb[k * N + n];
                Cb[m * N + n] = acc;
            }
        }
    }
}
""",
    extra_shapes=[
        {"batch": 1, "M": 1,  "K": 1,  "N": 1},
        {"batch": 2, "M": 4,  "K": 8,  "N": 4},
        {"batch": 4, "M": 8,  "K": 16, "N": 8},
        {"batch": 1, "M": 32, "K": 32, "N": 32},
    ],
    argtypes_factory=_bmm_argtypes,
)


# ---------------------------------------------------------------------------
# fp16 matmul / bmm variants — same semantics, _Float16 storage.
# The accumulator widens to float for numerics then narrows back.
# ---------------------------------------------------------------------------

MATMUL_F16 = KernelSpec(
    op="matmul_f16",
    signature=(
        "void kernel_matmul_f16(const _Float16 *A, const _Float16 *B, "
        "_Float16 *C, int M, int K, int N)"
    ),
    semantics=(
        "Dense matrix multiply C = A @ B, _Float16 storage:\n"
        "  A: [M, K],  B: [K, N],  C: [M, N]\n"
        "  C[m, n] = sum_{k=0..K-1} A[m*K+k] * B[k*N+n]  (fp16 arithmetic)"
    ),
    reference_impl="""\
void kernel_matmul_f16(const _Float16 *A, const _Float16 *B,
                       _Float16 *C, int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++)
                acc += (float)A[m * K + k] * (float)B[k * N + n];
            C[m * N + n] = (_Float16)acc;
        }
    }
}
""",
    extra_shapes=_MATMUL_EXTRA_SHAPES,
    argtypes_factory=_matmul_f16_argtypes,
)

MATMUL_TA_F16 = KernelSpec(
    op="matmul_ta_f16",
    signature=(
        "void kernel_matmul_ta_f16(const _Float16 *A, const _Float16 *B, "
        "_Float16 *C, int M, int K, int N)"
    ),
    semantics=(
        "Dense matrix multiply C = A.T @ B, _Float16 storage:\n"
        "  A stored: [K, M],  B: [K, N],  C: [M, N]"
    ),
    reference_impl="""\
void kernel_matmul_ta_f16(const _Float16 *A, const _Float16 *B,
                          _Float16 *C, int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++)
                acc += (float)A[k * M + m] * (float)B[k * N + n];
            C[m * N + n] = (_Float16)acc;
        }
    }
}
""",
    extra_shapes=_MATMUL_EXTRA_SHAPES,
    argtypes_factory=_matmul_f16_argtypes,
)

MATMUL_TB_F16 = KernelSpec(
    op="matmul_tb_f16",
    signature=(
        "void kernel_matmul_tb_f16(const _Float16 *A, const _Float16 *B, "
        "_Float16 *C, int M, int K, int N)"
    ),
    semantics=(
        "Dense matrix multiply C = A @ B.T, _Float16 storage:\n"
        "  A: [M, K],  B stored: [N, K],  C: [M, N]"
    ),
    reference_impl="""\
void kernel_matmul_tb_f16(const _Float16 *A, const _Float16 *B,
                          _Float16 *C, int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++)
                acc += (float)A[m * K + k] * (float)B[n * K + k];
            C[m * N + n] = (_Float16)acc;
        }
    }
}
""",
    extra_shapes=_MATMUL_EXTRA_SHAPES,
    argtypes_factory=_matmul_f16_argtypes,
)

MATMUL_TATB_F16 = KernelSpec(
    op="matmul_tatb_f16",
    signature=(
        "void kernel_matmul_tatb_f16(const _Float16 *A, const _Float16 *B, "
        "_Float16 *C, int M, int K, int N)"
    ),
    semantics=(
        "Dense matrix multiply C = A.T @ B.T, _Float16 storage:\n"
        "  A stored: [K, M],  B stored: [N, K],  C: [M, N]"
    ),
    reference_impl="""\
void kernel_matmul_tatb_f16(const _Float16 *A, const _Float16 *B,
                            _Float16 *C, int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++)
                acc += (float)A[k * M + m] * (float)B[n * K + k];
            C[m * N + n] = (_Float16)acc;
        }
    }
}
""",
    extra_shapes=_MATMUL_EXTRA_SHAPES,
    argtypes_factory=_matmul_f16_argtypes,
)

BMM_F16 = KernelSpec(
    op="bmm_f16",
    signature=(
        "void kernel_bmm_f16(const _Float16 *A, const _Float16 *B, "
        "_Float16 *C, int batch, int M, int K, int N)"
    ),
    semantics=(
        "Batched matrix multiply C[b] = A[b] @ B[b], _Float16 storage:\n"
        "  A: [batch, M, K],  B: [batch, K, N],  C: [batch, M, N]"
    ),
    reference_impl="""\
void kernel_bmm_f16(const _Float16 *A, const _Float16 *B,
                   _Float16 *C, int batch, int M, int K, int N) {
    for (int b = 0; b < batch; b++) {
        const _Float16 *Ab = A + b * M * K;
        const _Float16 *Bb = B + b * K * N;
        _Float16 *Cb = C + b * M * N;
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float acc = 0.0f;
                for (int k = 0; k < K; k++)
                    acc += (float)Ab[m * K + k] * (float)Bb[k * N + n];
                Cb[m * N + n] = (_Float16)acc;
            }
        }
    }
}
""",
    extra_shapes=[
        {"batch": 1, "M": 1,  "K": 1,  "N": 1},
        {"batch": 2, "M": 4,  "K": 8,  "N": 4},
        {"batch": 4, "M": 8,  "K": 16, "N": 8},
        {"batch": 1, "M": 32, "K": 32, "N": 32},
    ],
    argtypes_factory=_bmm_f16_argtypes,
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
        AlgorithmCandidate(
            name="oc_blocked",
            description=(
                "Cache-blocked direct convolution. Same arithmetic as the "
                "`direct` algorithm — six nested loops over (n, oc, oh, "
                "ow, ic, kh, kw) — but the OUTPUT-CHANNEL dimension is "
                "tiled so that one slab of the weight tensor stays "
                "resident in L1D across the entire OH*OW spatial sweep. "
                "This is purely a loop-restructuring optimization: the "
                "compiler emits the same FMACs in the same order; we just "
                "change which dimension is outermost so weight reuse "
                "happens at L1D rather than L2/LLC.\n\n"
                "STRUCTURE (the LLM MUST follow this exactly):\n"
                "  Define TILE_OC at the top of the function. Aim for "
                "TILE_OC * IC * KH * KW * 4 bytes <= ~24 KB so the slab "
                "fits in 32 KB L1D with room for input/output. For "
                "OC>=128 IC>=128 K=3 use TILE_OC=4. For smaller IC use "
                "TILE_OC=8 or 16.\n\n"
                "  for (oc_outer = 0; oc_outer < OC; oc_outer += TILE_OC) {\n"
                "      // Hot weight slab: weight[oc_outer .. oc_outer + tile - 1]\n"
                "      // is now reused across the (oh, ow) sweep below.\n"
                "      for (n = 0; n < N; n++)\n"
                "      for (oh = 0; oh < OH; oh++)\n"
                "      for (ow = 0; ow < OW; ow++)\n"
                "      for (oc_inner = 0; oc_inner < tile; oc_inner++) {\n"
                "          int oc = oc_outer + oc_inner;\n"
                "          float acc = bias ? bias[oc] : 0.0f;\n"
                "          // standard 3-deep ic/kh/kw reduction here\n"
                "          output[((n*OC + oc)*OH + oh)*OW + ow] = acc;\n"
                "      }\n"
                "  }\n\n"
                "Why this matters: a direct conv2d with the OC loop in "
                "the middle pulls the entire weight tensor through L1D "
                "OH*OW times. With OC blocked outermost, each weight "
                "loaded into L1D once is reused OH*OW times before "
                "eviction. For dronet's heavy 3x3 conv (OC=128, IC=128, "
                "OH=4, OW=4) that's a 16x reduction in weight traffic to "
                "L2/LLC.\n\n"
                "TAIL HANDLING: when OC is not a multiple of TILE_OC, "
                "the final outer iteration uses tile = OC - oc_outer "
                "(may be < TILE_OC). Compute it inside the outer loop:\n"
                "  int tile = TILE_OC;\n"
                "  if (oc_outer + tile > OC) tile = OC - oc_outer;\n\n"
                "RVV vectorization (rvv backend only): you may vectorize "
                "the IC reduction inside the inner per-output-element "
                "block exactly the same way as the `direct` algorithm — "
                "this transformation only restructures the OC loop, not "
                "the inner reduction.\n\n"
                "DO NOT remove the `oc_outer += TILE_OC` outer loop — "
                "it's the entire point of this algorithm. DO NOT pick "
                "TILE_OC = 1 (that's just the direct algorithm with "
                "extra overhead). DO NOT pick TILE_OC = OC (that's also "
                "just the direct algorithm). Pick a fixed TILE_OC "
                "constant; do not compute it from runtime IC — the "
                "compiler needs the constant to keep the inner-loop "
                "indexing cheap."
            ),
            reference_impl="""\
void kernel_conv2d(const float *input, const float *weight, const float *bias,
                   float *output,
                   int N, int IC, int IH, int IW, int OC,
                   int KH, int KW, int SH, int SW, int PH, int PW) {
    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;
    /* Tile so one OC-slab of weights fits in ~24 KB of L1D.
     * For OC=128 IC=128 K=3: 4 * 128 * 9 * 4 = 18 KB. */
    const int TILE_OC = 4;

    for (int oc_outer = 0; oc_outer < OC; oc_outer += TILE_OC) {
        int tile = TILE_OC;
        if (oc_outer + tile > OC) tile = OC - oc_outer;
        for (int n = 0; n < N; n++) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    for (int oc_inner = 0; oc_inner < tile; oc_inner++) {
                        int oc = oc_outer + oc_inner;
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
}
""",
            # OC-blocked is only worth the loop-restructure cost when
            # the weight tensor is too large to stay in L1D. Skip on
            # tiny channel counts where direct already wins easily.
            applicable=lambda s: s.get("OC", 0) >= 32 and s.get("IC", 0) >= 16,
        ),
    ],
)


MAXPOOL2D = KernelSpec(
    op="maxpool2d",
    signature=(
        "void kernel_maxpool2d(const float *input, float *output, "
        "int N, int C, int IH, int IW, "
        "int KH, int KW, int SH, int SW, "
        "int PH, int PW, int DH, int DW)"
    ),
    semantics=(
        "2D max pooling matching torch.nn.MaxPool2d semantics with padding\n"
        "and dilation.\n"
        "Layout (NCHW, row-major):\n"
        "  input:  [N, C, IH, IW]\n"
        "  output: [N, C, OH, OW]  with\n"
        "    OH = (IH + 2*PH - DH*(KH-1) - 1) / SH + 1\n"
        "    OW = (IW + 2*PW - DW*(KW-1) - 1) / SW + 1\n"
        "  output[n, c, oh, ow] = max over kh in [0,KH), kw in [0,KW) of\n"
        "    val(n, c, oh*SH - PH + kh*DH, ow*SW - PW + kw*DW)\n"
        "  where val(...) returns input[n, c, ih, iw] when 0<=ih<IH and\n"
        "  0<=iw<IW, else -INF (out-of-bounds lanes never win the max).\n"
        "All tensors are float32. PH/PW are zero-padding amounts on the\n"
        "spatial dims; DH/DW are kernel-element strides (dilation)."
    ),
    reference_impl="""\
#include <float.h>

void kernel_maxpool2d(const float *input, float *output,
                     int N, int C, int IH, int IW,
                     int KH, int KW, int SH, int SW,
                     int PH, int PW, int DH, int DW) {
    int OH = (IH + 2*PH - DH*(KH-1) - 1) / SH + 1;
    int OW = (IW + 2*PW - DW*(KW-1) - 1) / SW + 1;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    float m = -FLT_MAX;
                    for (int kh = 0; kh < KH; kh++) {
                        int ih = oh*SH - PH + kh*DH;
                        if (ih < 0 || ih >= IH) continue;
                        for (int kw = 0; kw < KW; kw++) {
                            int iw = ow*SW - PW + kw*DW;
                            if (iw < 0 || iw >= IW) continue;
                            float v = input[((n*C + c)*IH + ih)*IW + iw];
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
        {"N": 1, "C": 6, "IH": 24, "IW": 24, "KH": 2, "KW": 2, "SH": 2, "SW": 2,
         "PH": 0, "PW": 0, "DH": 1, "DW": 1},
        {"N": 1, "C": 16, "IH": 8, "IW": 8, "KH": 2, "KW": 2, "SH": 2, "SW": 2,
         "PH": 0, "PW": 0, "DH": 1, "DW": 1},
        # DroNet
        {"N": 1, "C": 32, "IH": 64, "IW": 64, "KH": 3, "KW": 3, "SH": 2, "SW": 2,
         "PH": 0, "PW": 0, "DH": 1, "DW": 1},
        # Generalization
        {"N": 1, "C": 4, "IH": 9, "IW": 7, "KH": 3, "KW": 3, "SH": 2, "SW": 2,
         "PH": 0, "PW": 0, "DH": 1, "DW": 1},
        # KernelBench 42: padding=1, dilation=3
        {"N": 1, "C": 8, "IH": 16, "IW": 16, "KH": 2, "KW": 2, "SH": 2, "SW": 2,
         "PH": 1, "PW": 1, "DH": 3, "DW": 3},
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
        "int KH, int KW, int SH, int SW, "
        "int PH, int PW, int DH, int DW)"
    ),
    semantics=(
        "Quantized 2D max-pool. Identical dataflow to fp32 maxpool2d but "
        "operating on int8 lanes. No requantize is needed — max is just a "
        "compare, and selecting an int8 input directly produces an int8 "
        "output at the same scale. Padding is filled with INT8_MIN so OOB "
        "lanes never win the max (the int8 analogue of -INF).\n"
        "Layout (NCHW):\n"
        "  input:  int8 [N, C, IH, IW]\n"
        "  output: int8 [N, C, OH, OW]  with\n"
        "    OH = (IH + 2*PH - DH*(KH-1) - 1) / SH + 1\n"
        "    OW = (IW + 2*PW - DW*(KW-1) - 1) / SW + 1\n"
        "  output[n, c, oh, ow] = max over kh in [0,KH), kw in [0,KW) of\n"
        "    val(n, c, oh*SH - PH + kh*DH, ow*SW - PW + kw*DW)\n"
        "  where val(...) returns input[n,c,ih,iw] when in bounds, else\n"
        "  INT8_MIN."
    ),
    reference_impl="""\
#include <stdint.h>

void kernel_maxpool2d_s8(const int8_t *input, int8_t *output,
                         int N, int C, int IH, int IW,
                         int KH, int KW, int SH, int SW,
                         int PH, int PW, int DH, int DW) {
    int OH = (IH + 2*PH - DH*(KH-1) - 1) / SH + 1;
    int OW = (IW + 2*PW - DW*(KW-1) - 1) / SW + 1;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    int8_t m = INT8_MIN;
                    for (int kh = 0; kh < KH; kh++) {
                        int ih = oh*SH - PH + kh*DH;
                        if (ih < 0 || ih >= IH) continue;
                        for (int kw = 0; kw < KW; kw++) {
                            int iw = ow*SW - PW + kw*DW;
                            if (iw < 0 || iw >= IW) continue;
                            int8_t v = input[((n*C + c)*IH + ih)*IW + iw];
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
        {"N": 1, "C": 6, "IH": 24, "IW": 24, "KH": 2, "KW": 2, "SH": 2, "SW": 2,
         "PH": 0, "PW": 0, "DH": 1, "DW": 1},
        {"N": 1, "C": 16, "IH": 8, "IW": 8, "KH": 2, "KW": 2, "SH": 2, "SW": 2,
         "PH": 0, "PW": 0, "DH": 1, "DW": 1},
        # DroNet
        {"N": 1, "C": 32, "IH": 64, "IW": 64, "KH": 3, "KW": 3, "SH": 2, "SW": 2,
         "PH": 0, "PW": 0, "DH": 1, "DW": 1},
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


# ---------------------------------------------------------------------------
# KernelBench Phase 2 activations (10 ops). All pointwise, fp32. The
# matching fp16 variants live below alongside the other _f16 specs so
# the existing fp16 mode picks them up via the op-suffix mechanism.
# ---------------------------------------------------------------------------

LEAKY_RELU = KernelSpec(
    op="leaky_relu",
    signature=(
        "void kernel_leaky_relu(const float *input, float *output, "
        "int n, float negative_slope)"
    ),
    semantics=(
        "Elementwise LeakyReLU on a contiguous float32 buffer:\n"
        "  output[i] = input[i]                if input[i] >= 0\n"
        "  output[i] = input[i] * negative_slope otherwise\n"
        "Same shape as ReLU; the slope on the negative half-line is the\n"
        "only difference. Safe to alias input/output."
    ),
    reference_impl="""\
void kernel_leaky_relu(const float *input, float *output,
                       int n, float negative_slope) {
    for (int i = 0; i < n; i++) {
        float v = input[i];
        output[i] = v >= 0.0f ? v : v * negative_slope;
    }
}
""",
    extra_shapes=[{"n": 1}, {"n": 17}, {"n": 1024}],
    argtypes_factory=_leaky_relu_argtypes,
)


TANH = KernelSpec(
    op="tanh",
    signature="void kernel_tanh(const float *input, float *output, int n)",
    semantics=(
        "Elementwise tanh on a contiguous float32 buffer.\n"
        "Uses libm's tanhf — already accurate to <1ulp on Zephyr's\n"
        "newlib/picolibc."
    ),
    reference_impl="""\
#include <math.h>

void kernel_tanh(const float *input, float *output, int n) {
    for (int i = 0; i < n; i++) {
        output[i] = tanhf(input[i]);
    }
}
""",
    extra_shapes=[{"n": 1}, {"n": 17}, {"n": 1024}],
    argtypes_factory=_pointwise_argtypes,
)


SWISH = KernelSpec(
    op="swish",
    signature="void kernel_swish(const float *input, float *output, int n)",
    semantics=(
        "Elementwise Swish (also called SiLU): output[i] = x * sigmoid(x).\n"
        "sigmoid(x) = 1 / (1 + expf(-x))."
    ),
    reference_impl="""\
#include <math.h>

void kernel_swish(const float *input, float *output, int n) {
    for (int i = 0; i < n; i++) {
        float x = input[i];
        float s = 1.0f / (1.0f + expf(-x));
        output[i] = x * s;
    }
}
""",
    extra_shapes=[{"n": 1}, {"n": 17}, {"n": 1024}],
    argtypes_factory=_pointwise_argtypes,
)


GELU = KernelSpec(
    op="gelu",
    signature="void kernel_gelu(const float *input, float *output, int n)",
    semantics=(
        "Elementwise GELU using the EXACT erf formulation, matching\n"
        "torch.nn.GELU and torch.nn.functional.gelu's `approximate='none'`\n"
        "default (PyTorch 1.10+):\n"
        "  output = 0.5 * x * (1 + erf(x / sqrt(2)))\n"
        "The tanh approximation lives under the separate `gelu_exact`\n"
        "op (used by the MinGPT-style hand-rolled expression). The two\n"
        "agree to ~5e-4 absolute on randn inputs — close but distinct,\n"
        "and PyTorch ships the erf form as default so we use it here."
    ),
    reference_impl="""\
#include <math.h>

void kernel_gelu(const float *input, float *output, int n) {
    /* 1 / sqrt(2) = 0.70710678118654752440f. */
    const float inv_sqrt2 = 0.7071067811865475f;
    for (int i = 0; i < n; i++) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + erff(x * inv_sqrt2));
    }
}
""",
    extra_shapes=[{"n": 1}, {"n": 17}, {"n": 1024}],
    argtypes_factory=_pointwise_argtypes,
)


# The tanh-based GELU approximation — what MinGPT (and the original
# BERT / GPT-2 papers) hand-rolled. PyTorch reaches this form via
# `F.gelu(x, approximate='tanh')` or by tracing the explicit
# expression. We keep a separate op for it so the IR cleanly captures
# which surface the bench used; numerically the two agree to ~5e-4.
GELU_EXACT = KernelSpec(
    op="gelu_exact",
    signature=(
        "void kernel_gelu_exact(const float *input, float *output, int n)"
    ),
    semantics=(
        "Elementwise GELU computed via the MinGPT / BERT formula:\n"
        "  output = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))\n"
        "Approximation of the exact erf form; the two agree to ~5e-4 on\n"
        "randn inputs. Used by 88_MinGPTNewGelu and `F.gelu(approximate='tanh')`."
    ),
    reference_impl="""\
#include <math.h>

void kernel_gelu_exact(const float *input, float *output, int n) {
    /* sqrtf(2.0f / M_PI) = 0.79788456080286535588f. */
    const float k = 0.7978845608028654f;
    for (int i = 0; i < n; i++) {
        float x = input[i];
        float u = k * (x + 0.044715f * x * x * x);
        output[i] = 0.5f * x * (1.0f + tanhf(u));
    }
}
""",
    extra_shapes=[{"n": 1}, {"n": 17}, {"n": 1024}],
    argtypes_factory=_pointwise_argtypes,
)


SELU = KernelSpec(
    op="selu",
    signature="void kernel_selu(const float *input, float *output, int n)",
    semantics=(
        "Scaled Exponential Linear Unit. PyTorch uses fixed constants:\n"
        "  alpha  = 1.6732632423543772\n"
        "  scale  = 1.0507009873554805\n"
        "  output = scale * (x       if x > 0\n"
        "                    alpha * (expf(x) - 1)   otherwise)"
    ),
    reference_impl="""\
#include <math.h>

void kernel_selu(const float *input, float *output, int n) {
    const float alpha = 1.6732632423543772f;
    const float scale = 1.0507009873554805f;
    for (int i = 0; i < n; i++) {
        float x = input[i];
        float y = x > 0.0f ? x : alpha * (expf(x) - 1.0f);
        output[i] = scale * y;
    }
}
""",
    extra_shapes=[{"n": 1}, {"n": 17}, {"n": 1024}],
    argtypes_factory=_pointwise_argtypes,
)


HARDSIGMOID = KernelSpec(
    op="hardsigmoid",
    signature=(
        "void kernel_hardsigmoid(const float *input, float *output, int n)"
    ),
    semantics=(
        "Piecewise-linear sigmoid approximation matching\n"
        "torch.nn.functional.hardsigmoid:\n"
        "  output[i] =   0          if x <= -3\n"
        "              1            if x >=  3\n"
        "              x/6 + 0.5    otherwise"
    ),
    reference_impl="""\
void kernel_hardsigmoid(const float *input, float *output, int n) {
    for (int i = 0; i < n; i++) {
        float x = input[i];
        float y;
        if (x <= -3.0f)      y = 0.0f;
        else if (x >=  3.0f) y = 1.0f;
        else                 y = x * (1.0f / 6.0f) + 0.5f;
        output[i] = y;
    }
}
""",
    extra_shapes=[{"n": 1}, {"n": 17}, {"n": 1024}],
    argtypes_factory=_pointwise_argtypes,
)


SOFTPLUS = KernelSpec(
    op="softplus",
    signature=(
        "void kernel_softplus(const float *input, float *output, int n)"
    ),
    semantics=(
        "Elementwise softplus: output = log(1 + expf(x)).\n"
        "For numerical stability we pivot on the sign of x so we never\n"
        "compute expf of a large positive: large-x reduces to x +\n"
        "log(1 + expf(-x)) ~= x. PyTorch defaults to beta=1, threshold=20\n"
        "but the threshold-passthrough only matters for extreme inputs;\n"
        "the unconditional formula via log1pf(expf(...)) is fine in fp32."
    ),
    reference_impl="""\
#include <math.h>

void kernel_softplus(const float *input, float *output, int n) {
    for (int i = 0; i < n; i++) {
        float x = input[i];
        /* Pivot keeps expf's argument <= 0, avoiding overflow. */
        float y = x > 0.0f
            ? x + log1pf(expf(-x))
            : log1pf(expf(x));
        output[i] = y;
    }
}
""",
    extra_shapes=[{"n": 1}, {"n": 17}, {"n": 1024}],
    argtypes_factory=_pointwise_argtypes,
)


SOFTSIGN = KernelSpec(
    op="softsign",
    signature=(
        "void kernel_softsign(const float *input, float *output, int n)"
    ),
    semantics=(
        "Elementwise softsign: output = x / (1 + |x|). Smooth, bounded\n"
        "alternative to tanh; no transcendental cost."
    ),
    reference_impl="""\
void kernel_softsign(const float *input, float *output, int n) {
    for (int i = 0; i < n; i++) {
        float x = input[i];
        float ax = x < 0.0f ? -x : x;
        output[i] = x / (1.0f + ax);
    }
}
""",
    extra_shapes=[{"n": 1}, {"n": 17}, {"n": 1024}],
    argtypes_factory=_pointwise_argtypes,
)


HARDTANH = KernelSpec(
    op="hardtanh",
    signature=(
        "void kernel_hardtanh(const float *input, float *output, "
        "int n, float min_val, float max_val)"
    ),
    semantics=(
        "Pointwise clamp to [min_val, max_val]. PyTorch's hardtanh\n"
        "defaults to min=-1, max=+1. Same dataflow as relu6 with\n"
        "configurable bounds."
    ),
    reference_impl="""\
void kernel_hardtanh(const float *input, float *output,
                     int n, float min_val, float max_val) {
    for (int i = 0; i < n; i++) {
        float x = input[i];
        if (x < min_val) x = min_val;
        if (x > max_val) x = max_val;
        output[i] = x;
    }
}
""",
    extra_shapes=[{"n": 1}, {"n": 17}, {"n": 1024}],
    argtypes_factory=_hardtanh_argtypes,
)


# ---------------------------------------------------------------------------
# KernelBench Phase 2 reductions over a single dimension. Each kernel
# treats the input as logically 3D — [outer, reduce, inner] — so any
# Nd input with `dim=k` flattens to outer=prod(shape[:k]),
# reduce=shape[k], inner=prod(shape[k+1:]). The output is [outer, inner]
# (or [outer, 1, inner] for keepdim) — same flat layout for both, so
# the kernels don't need to know about keepdim.
# ---------------------------------------------------------------------------

def _reduce_argtypes():
    import ctypes
    fp = ctypes.POINTER(ctypes.c_float)
    return [fp, fp, ctypes.c_int, ctypes.c_int, ctypes.c_int]


def _argreduce_argtypes():
    import ctypes
    fp = ctypes.POINTER(ctypes.c_float)
    i64p = ctypes.POINTER(ctypes.c_int64)
    return [fp, i64p, ctypes.c_int, ctypes.c_int, ctypes.c_int]


SUM_DIM = KernelSpec(
    op="sum_dim",
    signature=(
        "void kernel_sum_dim(const float *input, float *output, "
        "int outer, int reduce, int inner)"
    ),
    semantics=(
        "Reduce-sum along the middle axis of a logically [outer, reduce,\n"
        "inner] flattened tensor. Output is [outer, inner].\n"
        "  output[o, i] = sum over r in [0, reduce) of\n"
        "    input[(o * reduce + r) * inner + i]"
    ),
    reference_impl="""\
void kernel_sum_dim(const float *input, float *output,
                    int outer, int reduce, int inner) {
    for (int o = 0; o < outer; o++) {
        for (int i = 0; i < inner; i++) {
            float acc = 0.0f;
            for (int r = 0; r < reduce; r++) {
                acc += input[(o * reduce + r) * inner + i];
            }
            output[o * inner + i] = acc;
        }
    }
}
""",
    extra_shapes=[
        {"outer": 1,  "reduce": 17,  "inner": 1},
        {"outer": 4,  "reduce": 33,  "inner": 8},
        {"outer": 16, "reduce": 256, "inner": 256},
    ],
    argtypes_factory=_reduce_argtypes,
)


MEAN_DIM = KernelSpec(
    op="mean_dim",
    signature=(
        "void kernel_mean_dim(const float *input, float *output, "
        "int outer, int reduce, int inner)"
    ),
    semantics=(
        "Reduce-mean along the middle axis. Same dataflow as sum_dim,\n"
        "with a final divide by `reduce`."
    ),
    reference_impl="""\
void kernel_mean_dim(const float *input, float *output,
                     int outer, int reduce, int inner) {
    float inv = 1.0f / (float)reduce;
    for (int o = 0; o < outer; o++) {
        for (int i = 0; i < inner; i++) {
            float acc = 0.0f;
            for (int r = 0; r < reduce; r++) {
                acc += input[(o * reduce + r) * inner + i];
            }
            output[o * inner + i] = acc * inv;
        }
    }
}
""",
    extra_shapes=[
        {"outer": 1,  "reduce": 17,  "inner": 1},
        {"outer": 4,  "reduce": 33,  "inner": 8},
        {"outer": 16, "reduce": 256, "inner": 256},
    ],
    argtypes_factory=_reduce_argtypes,
)


MAX_DIM = KernelSpec(
    op="max_dim",
    signature=(
        "void kernel_max_dim(const float *input, float *output, "
        "int outer, int reduce, int inner)"
    ),
    semantics=(
        "Reduce-max (values only, not indices) along the middle axis.\n"
        "Initialized to -FLT_MAX so a single-element reduce returns the\n"
        "input value directly."
    ),
    reference_impl="""\
#include <float.h>

void kernel_max_dim(const float *input, float *output,
                    int outer, int reduce, int inner) {
    for (int o = 0; o < outer; o++) {
        for (int i = 0; i < inner; i++) {
            float m = -FLT_MAX;
            for (int r = 0; r < reduce; r++) {
                float v = input[(o * reduce + r) * inner + i];
                if (v > m) m = v;
            }
            output[o * inner + i] = m;
        }
    }
}
""",
    extra_shapes=[
        {"outer": 1,  "reduce": 17,  "inner": 1},
        {"outer": 4,  "reduce": 33,  "inner": 8},
        {"outer": 16, "reduce": 256, "inner": 256},
    ],
    argtypes_factory=_reduce_argtypes,
)


MIN_DIM = KernelSpec(
    op="min_dim",
    signature=(
        "void kernel_min_dim(const float *input, float *output, "
        "int outer, int reduce, int inner)"
    ),
    semantics=(
        "Reduce-min (values only) along the middle axis. Mirror of\n"
        "max_dim with FLT_MAX init and `<` compare."
    ),
    reference_impl="""\
#include <float.h>

void kernel_min_dim(const float *input, float *output,
                    int outer, int reduce, int inner) {
    for (int o = 0; o < outer; o++) {
        for (int i = 0; i < inner; i++) {
            float m = FLT_MAX;
            for (int r = 0; r < reduce; r++) {
                float v = input[(o * reduce + r) * inner + i];
                if (v < m) m = v;
            }
            output[o * inner + i] = m;
        }
    }
}
""",
    extra_shapes=[
        {"outer": 1,  "reduce": 17,  "inner": 1},
        {"outer": 4,  "reduce": 33,  "inner": 8},
        {"outer": 16, "reduce": 256, "inner": 256},
    ],
    argtypes_factory=_reduce_argtypes,
)


PROD_DIM = KernelSpec(
    op="prod_dim",
    signature=(
        "void kernel_prod_dim(const float *input, float *output, "
        "int outer, int reduce, int inner)"
    ),
    semantics=(
        "Reduce-product along the middle axis. Init=1.0, multiply-fold."
    ),
    reference_impl="""\
void kernel_prod_dim(const float *input, float *output,
                     int outer, int reduce, int inner) {
    for (int o = 0; o < outer; o++) {
        for (int i = 0; i < inner; i++) {
            float acc = 1.0f;
            for (int r = 0; r < reduce; r++) {
                acc *= input[(o * reduce + r) * inner + i];
            }
            output[o * inner + i] = acc;
        }
    }
}
""",
    extra_shapes=[
        {"outer": 1,  "reduce": 17,  "inner": 1},
        {"outer": 4,  "reduce": 33,  "inner": 8},
        {"outer": 16, "reduce": 256, "inner": 256},
    ],
    argtypes_factory=_reduce_argtypes,
)


ARGMAX_DIM = KernelSpec(
    op="argmax_dim",
    signature=(
        "void kernel_argmax_dim(const float *input, int64_t *output, "
        "int outer, int reduce, int inner)"
    ),
    semantics=(
        "Argmax along the middle axis. Returns int64 indices (same dtype\n"
        "as torch.argmax). On ties, returns the FIRST index (matches\n"
        "torch's behavior on CPU)."
    ),
    reference_impl="""\
#include <stdint.h>
#include <float.h>

void kernel_argmax_dim(const float *input, int64_t *output,
                       int outer, int reduce, int inner) {
    for (int o = 0; o < outer; o++) {
        for (int i = 0; i < inner; i++) {
            float m = -FLT_MAX;
            int64_t idx = 0;
            for (int r = 0; r < reduce; r++) {
                float v = input[(o * reduce + r) * inner + i];
                if (v > m) { m = v; idx = (int64_t)r; }
            }
            output[o * inner + i] = idx;
        }
    }
}
""",
    extra_shapes=[
        {"outer": 1,  "reduce": 17,  "inner": 1},
        {"outer": 4,  "reduce": 33,  "inner": 8},
        {"outer": 16, "reduce": 256, "inner": 256},
    ],
    argtypes_factory=_argreduce_argtypes,
)


ARGMIN_DIM = KernelSpec(
    op="argmin_dim",
    signature=(
        "void kernel_argmin_dim(const float *input, int64_t *output, "
        "int outer, int reduce, int inner)"
    ),
    semantics=(
        "Argmin along the middle axis. Returns int64 indices. Same\n"
        "tie-breaking as torch (first index)."
    ),
    reference_impl="""\
#include <stdint.h>
#include <float.h>

void kernel_argmin_dim(const float *input, int64_t *output,
                       int outer, int reduce, int inner) {
    for (int o = 0; o < outer; o++) {
        for (int i = 0; i < inner; i++) {
            float m = FLT_MAX;
            int64_t idx = 0;
            for (int r = 0; r < reduce; r++) {
                float v = input[(o * reduce + r) * inner + i];
                if (v < m) { m = v; idx = (int64_t)r; }
            }
            output[o * inner + i] = idx;
        }
    }
}
""",
    extra_shapes=[
        {"outer": 1,  "reduce": 17,  "inner": 1},
        {"outer": 4,  "reduce": 33,  "inner": 8},
        {"outer": 16, "reduce": 256, "inner": 256},
    ],
    argtypes_factory=_argreduce_argtypes,
)


# ---------------------------------------------------------------------------
# KernelBench Phase 2 norms (subset). Each divides the input by some
# reduction-derived scalar/vector. L1Norm / L2Norm share the
# (outer, reduce, inner) shape with the reductions; FrobeniusNorm
# uses a single global denominator over the whole tensor.
# ---------------------------------------------------------------------------

L1_NORM = KernelSpec(
    op="l1_norm",
    signature=(
        "void kernel_l1_norm(const float *input, float *output, "
        "int outer, int reduce, int inner)"
    ),
    semantics=(
        "Per-(outer, inner) L1 normalization:\n"
        "  denom[o, i] = sum over r of |input[(o*reduce + r)*inner + i]|\n"
        "  output[o, r, i] = input[o, r, i] / denom[o, i]\n"
        "Same flat layout as the reduction kernels, but the output\n"
        "preserves the reduce axis (broadcast division)."
    ),
    reference_impl="""\
void kernel_l1_norm(const float *input, float *output,
                    int outer, int reduce, int inner) {
    for (int o = 0; o < outer; o++) {
        for (int i = 0; i < inner; i++) {
            float denom = 0.0f;
            for (int r = 0; r < reduce; r++) {
                float v = input[(o * reduce + r) * inner + i];
                denom += v < 0.0f ? -v : v;
            }
            float inv = 1.0f / denom;
            for (int r = 0; r < reduce; r++) {
                int idx = (o * reduce + r) * inner + i;
                output[idx] = input[idx] * inv;
            }
        }
    }
}
""",
    extra_shapes=[
        {"outer": 1, "reduce": 17, "inner": 1},
        {"outer": 16, "reduce": 16384, "inner": 1},
    ],
    argtypes_factory=_reduce_argtypes,
)


L2_NORM = KernelSpec(
    op="l2_norm",
    signature=(
        "void kernel_l2_norm(const float *input, float *output, "
        "int outer, int reduce, int inner)"
    ),
    semantics=(
        "Per-(outer, inner) L2 normalization:\n"
        "  denom[o, i] = sqrt(sum over r of input[o, r, i]^2)\n"
        "  output[o, r, i] = input[o, r, i] / denom[o, i]"
    ),
    reference_impl="""\
#include <math.h>

void kernel_l2_norm(const float *input, float *output,
                    int outer, int reduce, int inner) {
    for (int o = 0; o < outer; o++) {
        for (int i = 0; i < inner; i++) {
            float ssq = 0.0f;
            for (int r = 0; r < reduce; r++) {
                float v = input[(o * reduce + r) * inner + i];
                ssq += v * v;
            }
            float inv = 1.0f / sqrtf(ssq);
            for (int r = 0; r < reduce; r++) {
                int idx = (o * reduce + r) * inner + i;
                output[idx] = input[idx] * inv;
            }
        }
    }
}
""",
    extra_shapes=[
        {"outer": 1, "reduce": 17, "inner": 1},
        {"outer": 16, "reduce": 16384, "inner": 1},
    ],
    argtypes_factory=_reduce_argtypes,
)


def _frobenius_norm_argtypes():
    import ctypes
    fp = ctypes.POINTER(ctypes.c_float)
    return [fp, fp, ctypes.c_int]


FROBENIUS_NORM = KernelSpec(
    op="frobenius_norm",
    signature=(
        "void kernel_frobenius_norm(const float *input, float *output, int n)"
    ),
    semantics=(
        "Global Frobenius normalization — whole tensor flattened.\n"
        "  denom = sqrt(sum over i of input[i]^2)\n"
        "  output[i] = input[i] / denom\n"
        "torch.norm(x, p='fro') == torch.norm(x, p=2) on a flattened\n"
        "tensor; we compute it as a sum-of-squares + sqrtf."
    ),
    reference_impl="""\
#include <math.h>

void kernel_frobenius_norm(const float *input, float *output, int n) {
    float ssq = 0.0f;
    for (int i = 0; i < n; i++) {
        float v = input[i];
        ssq += v * v;
    }
    float inv = 1.0f / sqrtf(ssq);
    for (int i = 0; i < n; i++) {
        output[i] = input[i] * inv;
    }
}
""",
    extra_shapes=[{"n": 17}, {"n": 1024}],
    argtypes_factory=_frobenius_norm_argtypes,
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


# ---------------------------------------------------------------------------
# fp16 (half-precision) kernel specs. Mirror the fp32 pointwise / pool /
# conv / batchnorm shapes but use _Float16 storage. For accumulating ops
# (conv2d, batchnorm-fold) we accumulate in fp32 then cast back — same
# pattern Tensor Cores use, and the accuracy gap to a pure-fp16 accumulator
# is significant enough that torch.float16 conv goes through this path too.
# Element-wise transcendentals (sigmoid, elu) round-trip through float for
# the math kernel since libm has no half-precision expf yet.
# ---------------------------------------------------------------------------

RELU_F16 = KernelSpec(
    op="relu_f16",
    signature="void kernel_relu_f16(const _Float16 *input, _Float16 *output, int n)",
    semantics=(
        "Elementwise ReLU on a contiguous _Float16 buffer:\n"
        "  output[i] = max(0, input[i])  for i in [0, n)\n"
        "Same semantics as the fp32 RELU kernel — only the storage type "
        "changes. It must be safe for `input` and `output` to alias."
    ),
    reference_impl="""\
void kernel_relu_f16(const _Float16 *input, _Float16 *output, int n) {
    for (int i = 0; i < n; i++) {
        _Float16 v = input[i];
        output[i] = v > (_Float16)0.0f ? v : (_Float16)0.0f;
    }
}
""",
    extra_shapes=[
        {"n": 1}, {"n": 17}, {"n": 1024},
    ],
    argtypes_factory=_relu_f16_argtypes,
)


SIGMOID_F16 = KernelSpec(
    op="sigmoid_f16",
    signature="void kernel_sigmoid_f16(const _Float16 *input, _Float16 *output, int n)",
    semantics=(
        "Elementwise sigmoid on a contiguous _Float16 buffer:\n"
        "  output[i] = 1 / (1 + expf(-input[i]))\n"
        "Math is done in float (libm has no expf16), then the result is\n"
        "cast back to _Float16. This is what torch.float16 sigmoid does\n"
        "internally on CPU."
    ),
    reference_impl="""\
#include <math.h>

void kernel_sigmoid_f16(const _Float16 *input, _Float16 *output, int n) {
    for (int i = 0; i < n; i++) {
        float v = (float)input[i];
        output[i] = (_Float16)(1.0f / (1.0f + expf(-v)));
    }
}
""",
    extra_shapes=[
        {"n": 1}, {"n": 17}, {"n": 1024},
    ],
    argtypes_factory=_sigmoid_f16_argtypes,
)


ELU_F16 = KernelSpec(
    op="elu_f16",
    signature=("void kernel_elu_f16(const _Float16 *input, _Float16 *output, "
               "int n, float alpha)"),
    semantics=(
        "Elementwise ELU on a contiguous _Float16 buffer:\n"
        "  output[i] = input[i]                       if input[i] > 0\n"
        "  output[i] = alpha * (expf(input[i]) - 1)   otherwise\n"
        "alpha is passed as float and used directly by expf — the cast back\n"
        "to _Float16 happens at store. nn.ELU defaults to alpha=1.0."
    ),
    reference_impl="""\
#include <math.h>

void kernel_elu_f16(const _Float16 *input, _Float16 *output,
                    int n, float alpha) {
    for (int i = 0; i < n; i++) {
        float v = (float)input[i];
        float r = v > 0.0f ? v : alpha * (expf(v) - 1.0f);
        output[i] = (_Float16)r;
    }
}
""",
    extra_shapes=[
        {"n": 1}, {"n": 16}, {"n": 256}, {"n": 1024},
    ],
    argtypes_factory=_elu_f16_argtypes,
)


BATCHNORM2D_F16 = KernelSpec(
    op="batchnorm2d_f16",
    signature=(
        "void kernel_batchnorm2d_f16(const _Float16 *input, "
        "const _Float16 *scale, const _Float16 *bias, _Float16 *output, "
        "int N, int C, int H, int W)"
    ),
    semantics=(
        "Apply pre-folded affine BatchNorm to a [N, C, H, W] _Float16 input:\n"
        "  output[n, c, h, w] = scale[c] * input[n, c, h, w] + bias[c]\n"
        "scale and bias are the gamma/(running_var+eps) and beta-mean*scale\n"
        "fold pre-computed in fp32 by extract_graph and cast to _Float16 at\n"
        "save time."
    ),
    reference_impl="""\
void kernel_batchnorm2d_f16(const _Float16 *input,
                            const _Float16 *scale, const _Float16 *bias,
                            _Float16 *output,
                            int N, int C, int H, int W) {
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            _Float16 s = scale[c];
            _Float16 b = bias[c];
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
        {"N": 1, "C": 4, "H": 8, "W": 8},
        {"N": 1, "C": 64, "H": 7, "W": 7},
    ],
    argtypes_factory=_batchnorm2d_f16_argtypes,
)


MAXPOOL2D_F16 = KernelSpec(
    op="maxpool2d_f16",
    signature=(
        "void kernel_maxpool2d_f16(const _Float16 *input, _Float16 *output, "
        "int N, int C, int IH, int IW, "
        "int KH, int KW, int SH, int SW, "
        "int PH, int PW, int DH, int DW)"
    ),
    semantics=(
        "Half-precision 2D max pool. Same dataflow as the fp32 MAXPOOL2D\n"
        "kernel; only the storage type differs. OOB padding lanes are\n"
        "initialized with -65504 (the most-negative finite _Float16) so they\n"
        "never win the max — equivalent to -INF for the representable range."
    ),
    reference_impl="""\
void kernel_maxpool2d_f16(const _Float16 *input, _Float16 *output,
                          int N, int C, int IH, int IW,
                          int KH, int KW, int SH, int SW,
                          int PH, int PW, int DH, int DW) {
    int OH = (IH + 2*PH - DH*(KH-1) - 1) / SH + 1;
    int OW = (IW + 2*PW - DW*(KW-1) - 1) / SW + 1;
    /* -FLT16_MAX equivalent: -65504 is the most-negative finite half. */
    const _Float16 NEG_HALF_INF = (_Float16)-65504.0f;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    _Float16 m = NEG_HALF_INF;
                    for (int kh = 0; kh < KH; kh++) {
                        int ih = oh*SH - PH + kh*DH;
                        if (ih < 0 || ih >= IH) continue;
                        for (int kw = 0; kw < KW; kw++) {
                            int iw = ow*SW - PW + kw*DW;
                            if (iw < 0 || iw >= IW) continue;
                            _Float16 v = input[((n*C + c)*IH + ih)*IW + iw];
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
        {"N": 1, "C": 6, "IH": 24, "IW": 24, "KH": 2, "KW": 2, "SH": 2, "SW": 2,
         "PH": 0, "PW": 0, "DH": 1, "DW": 1},
        # KernelBench 42 shape (padding+dilation)
        {"N": 1, "C": 8, "IH": 16, "IW": 16, "KH": 2, "KW": 2, "SH": 2, "SW": 2,
         "PH": 1, "PW": 1, "DH": 3, "DW": 3},
    ],
    argtypes_factory=_maxpool2d_f16_argtypes,
)


CONV2D_F16 = KernelSpec(
    op="conv2d_f16",
    signature=(
        "void kernel_conv2d_f16(const _Float16 *input, const _Float16 *weight, "
        "const _Float16 *bias, _Float16 *output, "
        "int N, int IC, int IH, int IW, int OC, "
        "int KH, int KW, int SH, int SW, int PH, int PW)"
    ),
    semantics=(
        "Half-precision 2D convolution. groups=1, dilation=1.\n"
        "Storage is _Float16 for input/weight/bias/output, but the inner\n"
        "accumulator is fp32 to avoid catastrophic cancellation when summing\n"
        "many partial products — same pattern as Tensor Cores and what\n"
        "torch.float16 conv2d does on CPU. The accumulator is cast back to\n"
        "_Float16 only at the final store."
    ),
    reference_impl="""\
void kernel_conv2d_f16(const _Float16 *input, const _Float16 *weight,
                       const _Float16 *bias, _Float16 *output,
                       int N, int IC, int IH, int IW, int OC,
                       int KH, int KW, int SH, int SW, int PH, int PW) {
    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;
    for (int n = 0; n < N; n++) {
        for (int oc = 0; oc < OC; oc++) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    float acc = bias ? (float)bias[oc] : 0.0f;
                    for (int ic = 0; ic < IC; ic++) {
                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh * SH - PH + kh;
                            if (ih < 0 || ih >= IH) continue;
                            for (int kw = 0; kw < KW; kw++) {
                                int iw = ow * SW - PW + kw;
                                if (iw < 0 || iw >= IW) continue;
                                float v = (float)input[((n*IC + ic)*IH + ih)*IW + iw];
                                float w = (float)weight[((oc*IC + ic)*KH + kh)*KW + kw];
                                acc += v * w;
                            }
                        }
                    }
                    output[((n*OC + oc)*OH + oh)*OW + ow] = (_Float16)acc;
                }
            }
        }
    }
}
""",
    extra_shapes=[
        # LeNet-shape, KernelBench 63
        {"N": 1, "IC": 3, "IH": 16, "IW": 16, "OC": 4,
         "KH": 3, "KW": 3, "SH": 1, "SW": 1, "PH": 0, "PW": 0},
    ],
    argtypes_factory=_conv2d_f16_argtypes,
)


KERNEL_SPECS: dict[str, KernelSpec] = {
    "linear": LINEAR,
    "matmul": MATMUL,
    "matmul_ta": MATMUL_TA,
    "matmul_tb": MATMUL_TB,
    "matmul_tatb": MATMUL_TATB,
    "bmm": BMM,
    "relu": RELU,
    "relu6": RELU6,
    "elu": ELU,
    # KernelBench Phase 2 activations.
    "leaky_relu": LEAKY_RELU,
    "tanh": TANH,
    "swish": SWISH,
    "gelu": GELU,
    "gelu_exact": GELU_EXACT,
    "selu": SELU,
    "hardsigmoid": HARDSIGMOID,
    "softplus": SOFTPLUS,
    "softsign": SOFTSIGN,
    "hardtanh": HARDTANH,
    # KernelBench Phase 2 reductions over a dim.
    "sum_dim": SUM_DIM,
    "mean_dim": MEAN_DIM,
    "max_dim": MAX_DIM,
    "min_dim": MIN_DIM,
    "prod_dim": PROD_DIM,
    "argmax_dim": ARGMAX_DIM,
    "argmin_dim": ARGMIN_DIM,
    # KernelBench Phase 2 norms (subset — see Tier 3 follow-on for the
    # affine-bearing nn.Module ones).
    "l1_norm": L1_NORM,
    "l2_norm": L2_NORM,
    "frobenius_norm": FROBENIUS_NORM,
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
    "relu_f16": RELU_F16,
    "sigmoid_f16": SIGMOID_F16,
    "elu_f16": ELU_F16,
    "batchnorm2d_f16": BATCHNORM2D_F16,
    "maxpool2d_f16": MAXPOOL2D_F16,
    "conv2d_f16": CONV2D_F16,
    "matmul_f16": MATMUL_F16,
    "matmul_ta_f16": MATMUL_TA_F16,
    "matmul_tb_f16": MATMUL_TB_F16,
    "matmul_tatb_f16": MATMUL_TATB_F16,
    "bmm_f16": BMM_F16,
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
