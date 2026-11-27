// Shared prelude of the ModelBlaster Vulkan compute shaders.
// Pointers arrive in push constants as GPU addresses (buffer_reference), all of
// them first, then 4-byte scalars -- the order mb_pc packs them in.
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_scalar_block_layout : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer F32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer I32 { int v[]; };
layout(buffer_reference, std430, buffer_reference_align = 8) buffer I64 { int64_t v[]; };
// 16-byte view for the VEC = 4 elementwise variants (the launcher checks alignment)
layout(buffer_reference, std430, buffer_reference_align = 16) buffer F32x4 { vec4 v[]; };
#define MB_V4(ptr, elem) F32x4(uint64_t(ptr) + uint64_t(elem) * 4ul).v[0]
#ifdef MB_INT8
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
layout(buffer_reference, std430, buffer_reference_align = 1) buffer I8 { int8_t v[]; };
// C roundf / round: half away from zero, exactly (x - trunc(x) is exact)
float mb_roundf(float x) { float t = trunc(x); return abs(x - t) >= 0.5 ? t + sign(x) : t; }
double mb_round(double x) { double t = trunc(x); return abs(x - t) >= 0.5LF ? t + sign(x) : t; }
int mb_clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
// a / b rounded as IEEE fp32 division (the reference's float division): the
// double quotient of two floats rounds to the correctly rounded float
float mb_fdiv(float a, float b) { return float(double(a) / double(b)); }
// linear_dq_s8 / linear3_dq_s8 output column n of row m: columns [0, N0) -> y0 [M, N0], [N0, N0+N1) -> y1,
// the rest -> y2 [M, N - N0 - N1] (linear_dq_s8: N0 = N)
#define MB_DQ_STORE(y0, y1, y2, N, N0, N1, m, n, val) { \
    if ((n) < (N0)) (y0).v[(m) * (N0) + (n)] = (val); \
    else if ((n) < (N0) + (N1)) (y1).v[(m) * (N1) + (n) - (N0)] = (val); \
    else (y2).v[(m) * ((N) - (N0) - (N1)) + (n) - (N0) - (N1)] = (val); }
#endif

// 1-D global id over a grid folded into (x, y) by mb_vk_launch1d
#define MB_GID (int64_t(gl_WorkGroupID.y) * int64_t(gl_NumWorkGroups.x) * 256 + int64_t(gl_WorkGroupID.x) * 256 + int64_t(gl_LocalInvocationID.x))

bool mb_nonnull(F32 p) { return uint64_t(p) != 0ul; }

// fp32 math the GLSL builtins do not cover (or cover loosely)
float mb_tanh(float x) {
    float t = exp(-2.0 * abs(x));
    float r = (1.0 - t) / (1.0 + t);
    return x < 0.0 ? -r : r;
}
// erf (GLSL has none): Abramowitz-Stegun 7.1.26 (|abs err| <= 1.5e-7) for
// |x| >= 0.5, and the Maclaurin series through x^11 below that, where the A&S
// form's RELATIVE error is worst (truncation < 2e-8).
float mb_erf(float x) {
    float a = abs(x);
    float t = 1.0 / (1.0 + 0.3275911 * a);
    float y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
    if (a < 0.5) {
        float x2 = a * a;
        y = a * (1.1283791671 + x2 * (-0.3761263890 + x2 * (0.1128379167 + x2 * (-0.0268661706 + x2 * (0.0052239776 + x2 * -0.0008548327)))));
    }
    return x < 0.0 ? -y : y;
}
float mb_log1p(float x) {
    float u = 1.0 + x;
    return u == 1.0 ? x : log(u) * (x / (u - 1.0));
}
float mb_expm1(float x) {
    if (abs(x) < 1e-5) return x + 0.5 * x * x;
    return exp(x) - 1.0;
}
float mb_sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }
