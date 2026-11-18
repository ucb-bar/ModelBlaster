"""The output-row split of conv2d_s8 (k3-work TASK-016).

* `conv_split.split_axis` follows cores/codegen_contract.json: K3 backends only,
  rows when OH >= width and OH*OW >= min_pixels, else channels.
* `parallel_conv2d_s8_rows` (generate_skeleton's emitted wrapper) is bit-exact
  with the serial call: compiled on the host with the real modelblaster_pool and a
  scalar reference kernel, over strides, paddings, kernel sizes and widths.
* codegen: a row-split conv calls the rows wrapper with its full weight (no shard
  arrays); a channel-split one keeps the sharded call; K1 backends are unchanged.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))
from modelblaster.pipeline import conv_split, generate_skeleton as gs  # noqa: E402

CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")


def _shape(ih, oc=8, k=3, s=1, p=1, ic=4):
    return {"N": 1, "IC": ic, "IH": ih, "IW": ih, "OC": oc, "KH": k, "KW": k,
            "SH": s, "SW": s, "PH": p, "PW": p}


def test_axis_rule():
    assert [conv_split.split_axis("conv2d_s8", _shape(20), w, "ime_k3_x100")
            for w in (1, 2, 4, 8)] == ["oc", "row", "row", "row"]
    # 5x5 output: 25 pixels < min_pixels -> channels; 10x10 at width 16: OH < width
    assert conv_split.split_axis("conv2d_s8", _shape(5), 4, "rvv_k3") == "oc"
    assert conv_split.split_axis("conv2d_s8", _shape(10), 16, "ime_k3_a100") == "oc"
    # not a K3 backend, no backend, or a fused conv: channels, as before
    for be in ("rvv_x60", "rvv", "scalar", None):
        assert conv_split.split_axis("conv2d_s8", _shape(20), 4, be) == "oc"
    assert conv_split.split_axis("conv2d_batchnorm2d_silu_s8", _shape(20), 4, "rvv_k3") == "oc"


def test_buildable_widths():
    odd = _shape(20, oc=30)
    assert conv_split.width_buildable("conv2d_s8", odd, 4, ("rvv_k3", "ime_k3_x100"))
    assert not conv_split.width_buildable("conv2d_s8", odd, 4, ())            # no backend: OC rule
    assert not conv_split.width_buildable("conv2d_s8", odd, 4, ("rvv_x60",))
    assert not conv_split.width_buildable("conv2d_s8", _shape(5, oc=30), 4, ("rvv_k3",))


def test_ab_switch(monkeypatch):
    monkeypatch.setenv("MB_CONV_SPLIT_AXIS", "oc")
    assert conv_split.split_axis("conv2d_s8", _shape(20), 4, "rvv_k3") == "oc"


REF = r"""
#include <stddef.h>
#include <stdint.h>
void kernel_conv2d_s8_t(const int8_t *input, const int8_t *weight, const int32_t *bias,
        int8_t *output, int N, int IC, int IH, int IW, int OC, int KH, int KW, int SH, int SW,
        int PH, int PW, int input_offset, int filter_offset, int output_offset,
        int output_multiplier, int output_shift, int activation_min, int activation_max) {
    int OH = (IH + 2*PH - KH) / SH + 1, OW = (IW + 2*PW - KW) / SW + 1;
    for (int oc = 0; oc < OC; oc++) for (int oh = 0; oh < OH; oh++) for (int ow = 0; ow < OW; ow++) {
        int32_t acc = bias ? bias[oc] : 0;
        for (int ic = 0; ic < IC; ic++) for (int kh = 0; kh < KH; kh++) for (int kw = 0; kw < KW; kw++) {
            int ih = oh*SH - PH + kh, iw = ow*SW - PW + kw;
            int32_t x = (ih < 0 || ih >= IH || iw < 0 || iw >= IW) ? input_offset
                        : input[((size_t)ic*IH + ih)*IW + iw] + input_offset;
            acc += x * (weight[((size_t)oc*IC + ic)*KH*KW + kh*KW + kw] + filter_offset);
        }
        int64_t p = ((int64_t)acc * output_multiplier + (1LL << 30)) >> 31;
        int32_t v = (int32_t)p;
        if (output_shift > 0) v = (int32_t)(((int64_t)v + (1LL << (output_shift - 1))) >> output_shift);
        v += output_offset;
        output[((size_t)oc*OH + oh)*OW + ow] = (int8_t)(v < activation_min ? activation_min : v > activation_max ? activation_max : v);
    }
}
"""

MAIN = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "modelblaster_pool.h"
void kernel_conv2d_s8_t(const int8_t *, const int8_t *, const int32_t *, int8_t *, int, int, int, int, int,
        int, int, int, int, int, int, int, int, int, int, int, int, int);
#include "rows.inc"
static unsigned r = 7;
static int rnd(void) { r = r * 1103515245u + 12345u; return (int)((r >> 16) & 0xff) - 128; }
int main(void) {
    static const int S[][7] = { /* IC IH OC K S P IOFF */
        {3,16,8,3,1,1,0},{4,15,5,3,2,1,0},{8,9,16,1,1,0,0},{2,20,4,5,1,2,-3},{5,11,7,3,2,0,0},
        {6,8,8,3,1,1,0},{3,33,4,3,2,1,5},{4,7,3,1,2,0,0},{1,40,2,3,1,1,0},{16,10,16,3,1,1,0} };
    static const int W[] = {2, 3, 4, 8};
    int bad = 0, runs = 0;
    for (unsigned wi = 0; wi < sizeof W / sizeof W[0]; wi++) {
        int harts[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int i = 0; i < W[wi]; i++) harts[i] = i % 2;
        modelblaster_pool_t pool = modelblaster_pool_create_on_harts(W[wi], harts);
        if (!pool) { printf("pool failed\n"); return 2; }
        for (unsigned si = 0; si < sizeof S / sizeof S[0]; si++) {
            const int *s = S[si];
            int IC = s[0], IH = s[1], OC = s[2], K = s[3], SH = s[4], P = s[5], ioff = s[6];
            int OH = (IH + 2*P - K) / SH + 1;
            size_t ni = (size_t)IC*IH*IH, nw = (size_t)OC*IC*K*K, no = (size_t)OC*OH*OH;
            int8_t *in = malloc(ni), *w = malloc(nw), *a = malloc(no), *b = malloc(no);
            int32_t *bias = malloc(4 * (size_t)OC);
            for (size_t x = 0; x < ni; x++) in[x] = (int8_t)rnd();
            for (size_t x = 0; x < nw; x++) w[x] = (int8_t)rnd();
            for (int x = 0; x < OC; x++) bias[x] = rnd() * 50;
            kernel_conv2d_s8_t(in, w, bias, a, 1, IC, IH, IH, OC, K, K, SH, SH, P, P, ioff, 0, 0,
                               1300000000, 7, -128, 127);
            memset(b, 0x55, no);
            parallel_conv2d_s8_rows(pool, in, w, bias, b, 1, IC, IH, IH, OC, K, K, SH, SH, P, P, ioff, 0, 0,
                                    1300000000, 7, -128, 127);
            if (memcmp(a, b, no)) { bad++; printf("MISMATCH w=%d shape %u\n", W[wi], si); }
            runs++;
            free(in); free(w); free(a); free(b); free(bias);
        }
        modelblaster_pool_destroy(pool);
    }
    printf("ROWS_RESULT runs=%d bad=%d\n", runs, bad);
    return bad != 0;
}
"""


@pytest.mark.skipif(CC is None, reason="no C compiler on PATH")
def test_rows_wrapper_is_bit_exact(tmp_path):
    (tmp_path / "rows.inc").write_text(gs._CONV2D_S8_ROWS_WRAPPER.format(mid="t"))
    (tmp_path / "ref.c").write_text(REF)
    (tmp_path / "main.c").write_text(MAIN)
    exe = tmp_path / "rows"
    rt = ROOT / "runtime"
    cmd = [CC, "-O1", "-pthread", "-DMB_PLATFORM_POSIX", "-D_GNU_SOURCE", "-DMODELBLASTER_USE_POOL",
           f"-I{rt}", f"-I{rt / 'modelblaster_pool'}", f"-I{tmp_path}", "-o", str(exe),
           str(tmp_path / "main.c"), str(tmp_path / "ref.c"),
           str(rt / "modelblaster_pool" / "modelblaster_pool.c"), str(rt / "mb_placement.c")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    out = subprocess.run([str(exe)], capture_output=True, text=True, timeout=120)
    assert "ROWS_RESULT runs=40 bad=0" in out.stdout, out.stdout[-2000:] + out.stderr[-2000:]


def _ir(ih, oc, w):
    return {"name": "m", "input": {"tensor": "x"}, "output": {"tensor": "y"},
            "ops": [{"op": "conv2d_s8", "dispatch_id": 0, "name": "c0", "weight": "c0.w",
                     "shape": dict(_shape(ih, oc=oc), OH=ih, OW=ih), "shard_factor": w,
                     "quant": {"input_offset": 0, "filter_offset": 0}}]}


def test_plans():
    # 20x20 at width 4: rows, no shard arrays; 5x5: channels (4 | 8)
    assert gs.row_split_convs(_ir(20, 8, 4), "ime_k3_x100", 1) == {0: 4}
    assert gs.shard_conv_weights(_ir(20, 8, 4), "ime_k3_x100", 1) == {}
    assert gs.row_split_convs(_ir(5, 8, 4), "ime_k3_x100", 1) == {}
    assert set(gs.shard_conv_weights(_ir(5, 8, 4), "ime_k3_x100", 1)) == {0}
    # K1: the channel split, unchanged
    assert gs.row_split_convs(_ir(20, 8, 4), "rvv_x60", 1) == {}
    assert set(gs.shard_conv_weights(_ir(20, 8, 4), "rvv_x60", 1)) == {0}


def test_shareable_ops_match_code():
    """cores/codegen_contract.json's shareable_ops is what generate_skeleton
    can actually spread over a pool; a new parallel_<op> must update both."""
    import json
    contract = json.loads((ROOT / "cores" / "codegen_contract.json").read_text())
    assert contract["rules"]["shareable_ops"]["ops"] == gs.shareable_ops()
    assert "conv2d_s8" in gs.shareable_ops() and "silu_s8" not in gs.shareable_ops()
