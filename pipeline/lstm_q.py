"""Integer LSTM cell (`lstm_q_s8`): build-time constants, tables, numpy model, C text.

The op is `lstm_s8` with its float cell replaced by integer arithmetic (pipeline/lower_lstm_int.py rewrites
lstm_s8 -> lstm_q_s8). Same int8 weights, int32 biases and int8 h/c state buffers; only the cell differs:

  A_x[j] = b_ih[j] + b_hh[j] + sum_k W_ih[j,k] x[k]      A_h[j] = sum_k W_hh[j,k] h_prev[k]     (int32, exact)
  G[j]   = clamp((A_x*Mx + A_h*Mh + 2^(S-1)) >> S, -2^19, 2^19-1)                             (Q.16, int64 products)
  i,f,o  = lut(SIG, G);  g = lut(TANH, G)                                                      (Q0.15)
  Cn     = ((f * c_prev*256 + 2^14) >> 15) + ((i*g*127 + 2^24) >> 25)                         (units (8/127)/256)
  c_new  = sat8((Cn + 128) >> 8)
  h_new  = sat8((o * lut(TANH, clamp((Cn*KC + 2^15) >> 16)) * 127 + 2^29) >> 30)
  lut(T, G): u = G + 2^19; k = u >> 11; r = u & 2047; T[k] + (((T[k+1]-T[k])*r + 1024) >> 11)
  T: 513 entries, x_k = (k-256)/32; SIG = min(32767, rha(2^15 sigma(x))), TANH = clamp(rha(2^15 tanh(x)), +-32767)

Requires scale_h = 1/127 and scale_c = 8/127 (what extract_graph emits). The bias scale is scale_in*scale_w_ih by
construction, so the bias adds into A_x. Constants: Mx = rha(scale_in*scale_w_ih*2^(16+S)), Mh likewise with
scale_h*scale_w_hh, S the largest shift keeping both <= 2^23-1.
"""
from __future__ import annotations

import math

F_G = 16
G_MIN, G_MAX = -(1 << 19), (1 << 19) - 1
LUT_N = 513
LUT_FRAC = 11
M_MAX = (1 << 23) - 1
KC = 1056831
S_H, S_C = 1.0 / 127.0, 8.0 / 127.0


def _rha(v: float) -> int:
    return math.floor(v + 0.5) if v >= 0 else -math.floor(-v + 0.5)


def tables() -> tuple[list[int], list[int]]:
    """(sigmoid, tanh) tables as Python ints (no numpy needed to emit C)."""
    xs = [(k - 256) / 32.0 for k in range(LUT_N)]
    sig = [min(32767, _rha(32768.0 / (1.0 + math.exp(-x)))) for x in xs]
    tnh = [max(-32767, min(32767, _rha(32768.0 * math.tanh(x)))) for x in xs]
    return sig, tnh


SIG, TANH = tables()


def layer_consts(q: dict) -> tuple[int, int, int]:
    if abs(float(q["scale_h"]) - S_H) > 1e-12 or abs(float(q["scale_c"]) - S_C) > 1e-12:
        raise ValueError(f"lstm_q_s8 needs scale_h=1/127 and scale_c=8/127, got "
                         f"{q['scale_h']}, {q['scale_c']}")
    sx = float(q["scale_in"]) * float(q["scale_w_ih"])
    sh = float(q["scale_h"]) * float(q["scale_w_hh"])
    best = None
    for S in range(63):
        mx, mh = _rha(sx * 2.0 ** (F_G + S)), _rha(sh * 2.0 ** (F_G + S))
        if max(mx, mh) > M_MAX:
            break
        best = (mx, mh, S)
    if best is None or best[2] < 1:
        raise ValueError("lstm_q_s8: no shift S >= 1 keeps Mx, Mh within 24 bits")
    return best


def _lut(T, G):
    import numpy as np
    T = np.asarray(T, np.int64)
    u = np.asarray(G, np.int64) + (1 << 19)
    k = u >> LUT_FRAC
    r = u & ((1 << LUT_FRAC) - 1)
    return T[k] + (((T[k + 1] - T[k]) * r + (1 << (LUT_FRAC - 1))) >> LUT_FRAC)


def step(x, w_ih, w_hh, b_ih, b_hh, h, c, Mx, Mh, S, has_bias):
    """numpy model of kernel_lstm_q_s8 (one tick, one layer). Returns (h_new, c_new) int8."""
    import numpy as np
    H = h.size
    A_x = w_ih.astype(np.int64) @ x.astype(np.int64)
    if has_bias:
        A_x = A_x + b_ih.astype(np.int64) + b_hh.astype(np.int64)
    A_h = w_hh.astype(np.int64) @ h.astype(np.int64)
    G = np.clip((A_x * Mx + A_h * Mh + (1 << (S - 1))) >> S, G_MIN, G_MAX)
    i, f = _lut(SIG, G[:H]), _lut(SIG, G[H:2 * H])
    g, o = _lut(TANH, G[2 * H:3 * H]), _lut(SIG, G[3 * H:])
    Cn = ((f * (c.astype(np.int64) * 256) + (1 << 14)) >> 15) + ((i * g * 127 + (1 << 24)) >> 25)
    c_new = np.clip((Cn + 128) >> 8, -128, 127)
    tc = _lut(TANH, np.clip((Cn * KC + (1 << 15)) >> 16, G_MIN, G_MAX))
    h_new = np.clip((o * tc * 127 + (1 << 29)) >> 30, -128, 127)
    return h_new.astype(np.int8), c_new.astype(np.int8)


def c_tables(guard: str = "MB_LSTM_Q_TABLES") -> str:
    def arr(name, T):
        body = ",".join(str(int(v)) for v in T)
        return f"static const int16_t {name}[{LUT_N}] = {{{body}}};\n"
    return (f"#ifndef {guard}\n#define {guard}\n" + arr("mb_lstm_q_sig", SIG)
            + arr("mb_lstm_q_tanh", TANH) + "#endif\n")


# The integer post stage, shared by every lstm_q_s8 kernel: the matvec part differs per backend
# (scalar loops, Gemmini full_C, RVV dot products), the cell does not. A_x already includes the bias.
C_POST = r"""
static inline int32_t mb_lstm_q_lut(const int16_t *T, int32_t G) {
    const int32_t u = G + (1 << 19);
    const int32_t k = u >> 11, r = u & 2047;
    return (int32_t)T[k] + ((((int32_t)T[k + 1] - (int32_t)T[k]) * r + 1024) >> 11);
}

static inline int32_t mb_lstm_q_gate(int32_t ax, int32_t ah, int32_t Mx, int32_t Mh, int S) {
    int64_t v = (int64_t)ax * Mx + (int64_t)ah * Mh + ((int64_t)1 << (S - 1));
    v >>= S;
    if (v < -(1 << 19)) v = -(1 << 19);
    if (v > (1 << 19) - 1) v = (1 << 19) - 1;
    return (int32_t)v;
}

/* Cell update for unit t from the four gates' accumulators. Writes h_out[t], c_state[t]; the caller
 * copies h_out into h_state after every unit is done (h_prev must stay intact for the matvecs). */
static inline void mb_lstm_q_cell(int32_t ai, int32_t af, int32_t ag, int32_t ao,
                                  int32_t hi, int32_t hf, int32_t hg, int32_t ho,
                                  int32_t Mx, int32_t Mh, int S,
                                  int8_t *c_state, int8_t *h_out, int t) {
    const int32_t i = mb_lstm_q_lut(mb_lstm_q_sig, mb_lstm_q_gate(ai, hi, Mx, Mh, S));
    const int32_t f = mb_lstm_q_lut(mb_lstm_q_sig, mb_lstm_q_gate(af, hf, Mx, Mh, S));
    const int32_t g = mb_lstm_q_lut(mb_lstm_q_tanh, mb_lstm_q_gate(ag, hg, Mx, Mh, S));
    const int32_t o = mb_lstm_q_lut(mb_lstm_q_sig, mb_lstm_q_gate(ao, ho, Mx, Mh, S));
    const int32_t cp = (int32_t)c_state[t] * 256;
    const int32_t fc = (f * cp + (1 << 14)) >> 15;
    const int32_t ig = (int32_t)(((int64_t)(i * g) * 127 + (1 << 24)) >> 25);
    const int32_t cn = fc + ig;
    int32_t cq = (cn + 128) >> 8;
    if (cq < -128) cq = -128;
    if (cq > 127) cq = 127;
    int64_t gc = ((int64_t)cn * 1056831 + (1 << 15)) >> 16;
    if (gc < -(1 << 19)) gc = -(1 << 19);
    if (gc > (1 << 19) - 1) gc = (1 << 19) - 1;
    const int32_t tc = mb_lstm_q_lut(mb_lstm_q_tanh, (int32_t)gc);
    int64_t hq = ((int64_t)(o * tc) * 127 + (1 << 29)) >> 30;
    if (hq < -128) hq = -128;
    if (hq > 127) hq = 127;
    c_state[t] = (int8_t)cq;
    h_out[t] = (int8_t)hq;
}
"""
