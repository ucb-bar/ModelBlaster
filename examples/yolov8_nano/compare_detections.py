"""Four-way accuracy comparison: ultralytics upstream / our fx-traceable
wrapper / int8 ceiling / spike actual.

Mirrors `modelblaster/examples/vint/compare_waypoints.py`. YOLOv8n outputs
three detection feature maps (P3 at stride 8, P4 at 16, P5 at 32) with
channel layout `[reg_max*4=64 (box) | nc=80 (cls)]` per scale. The four
versions compared:

  1. **Ultralytics upstream** — load `yolov8n.pt` via the ultralytics
     package and run `yolo.model(x)` directly. This is the
     vendor-canonical reference; if our wrapper diverges from this on
     fp32 inputs we have an FX-tracing bug, not a quantization issue.
  2. **Our wrapper at fp32** — `modelblaster.models.yolov8_nano.get_model()`
     returns the same arithmetic in fx-traceable form. Difference vs
     row 1 isolates whether the static-unroll + flat-forward refactor
     preserved numerics.
  3. **int8 ceiling** — `io.npz["output"]` is what the
     PyTorch-side integer simulator emits at extract time (running the
     same int8 ops in float arithmetic with our extracted scales). If
     our int8 forward were lossless, spike output would match this
     exactly.
  4. **Spike actual** — what the compiled int8 binary produces. Read
     from the `=== MODELBLASTER_OUTPUT_BEGIN/END ===` block in spike
     stdout.

Metrics per pair:
  - raw L∞ / L2 / cosine similarity (per-scale and combined)
  - top-K detection comparison after DFL decode + sigmoid + NMS

Run (from zephyr-chipyard-sw root, conda zephyr env active):
    PYTHONPATH=. python modelblaster/examples/yolov8_nano/compare_detections.py
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# YOLOv8n head defaults
_NC = 80
_REG_MAX = 16
_STRIDES = (8, 16, 32)


def _run_spike(elf: str, spike_bin: str, timeout: int = 1800) -> str:
    """Run spike on the elf, return stdout+stderr text."""
    res = subprocess.run(
        [spike_bin, "--isa=rv64gcv_zicntr", elf],
        capture_output=True, text=True, timeout=timeout,
    )
    return res.stdout + res.stderr


def _parse_output_block(text: str) -> Optional[np.ndarray]:
    """Pull fp32 lines between MODELBLASTER_OUTPUT_BEGIN/END markers.

    Returns None when the harness was built with in-binary verify
    instead of stdout-dump output (the yolov8n case — dumping 75600
    floats over HTIF is too slow, so the harness compares bit-exact
    against io.npz internally and prints just a VERIFY marker)."""
    m = re.search(
        r"=== (?:MODELBLASTER|AGENTS)_OUTPUT_BEGIN ===\n(.*?)\n"
        r"=== (?:MODELBLASTER|AGENTS)_OUTPUT_END ===",
        text, re.DOTALL,
    )
    if not m:
        return None
    vals = [float(x) for x in m.group(1).strip().split("\n") if x.strip()]
    return np.asarray(vals, dtype=np.float32)


def _parse_verify_marker(text: str) -> Optional[dict]:
    """Pull the in-binary verify result: max_abs_err / max_rel_err."""
    m = re.search(
        r"=== (?:MODELBLASTER|AGENTS)_VERIFY ===\s*"
        r"max_abs_err=([\d.eE+-]+)\s+max_rel_err=([\d.eE+-]+)\s+n=(\d+)",
        text,
    )
    if not m:
        return None
    return {
        "max_abs_err": float(m.group(1)),
        "max_rel_err": float(m.group(2)),
        "n": int(m.group(3)),
    }


def _split_into_scales(flat: np.ndarray, img: int) -> list[np.ndarray]:
    """Reshape the harness's flat output into a list of [1, C, OH, OW]
    detection feature maps at strides (8, 16, 32). The harness emits the
    three scales concatenated; the codegen orders them P3, P4, P5."""
    chans = 4 * _REG_MAX + _NC  # 64 + 80 = 144
    scales = []
    off = 0
    for s in _STRIDES:
        oh = ow = img // s
        n = chans * oh * ow
        scales.append(flat[off:off + n].reshape(1, chans, oh, ow))
        off += n
    if off != flat.size:
        raise RuntimeError(
            f"flat output size {flat.size} != sum of scales {off}; "
            f"check img={img} or codegen ordering"
        )
    return scales


def _compare_tensors(a: np.ndarray, b: np.ndarray, label: str) -> dict:
    """L∞ / L2 / cosine on flattened tensors."""
    a_flat = a.astype(np.float64).ravel()
    b_flat = b.astype(np.float64).ravel()
    diff = a_flat - b_flat
    l_inf = float(np.max(np.abs(diff)))
    l2 = float(np.sqrt(np.mean(diff ** 2)))
    denom = float(np.linalg.norm(a_flat) * np.linalg.norm(b_flat))
    cos = float(np.dot(a_flat, b_flat) / denom) if denom > 0 else 0.0
    return {"label": label, "L_inf": l_inf, "RMSE": l2, "cosine": cos,
            "n": a_flat.size}


def _print_table(rows: list[dict], pair_label: str):
    print(f"  {pair_label}")
    hdr = f"    {'scale':>8s} | {'L_inf':>9s} {'RMSE':>9s} {'cosine':>9s} "
    hdr += f"{'n elems':>10s}"
    print(hdr)
    for r in rows:
        print(f"    {r['label']:>8s} | {r['L_inf']:>9.4f} {r['RMSE']:>9.4f} "
              f"{r['cosine']:>9.6f} {r['n']:>10d}")


def _decode_dfl_and_nms(scales: list[np.ndarray], img: int,
                        score_thresh: float = 0.10,
                        iou_thresh: float = 0.45,
                        top_k: int = 50) -> list[tuple]:
    """Decode raw detection feature maps -> NMS'd boxes.

    Returns a list of (cls_id, score, x1, y1, x2, y2) tuples sorted by
    descending score, up to top_k entries.

    Box channels: [reg_max*4]  — split into 4 groups of reg_max=16,
    softmax along the reg_max dim, dot with [0..15] to get the
    per-side distance in cells (left, top, right, bottom).
    Class channels: [nc] — sigmoid for confidence.
    """
    dets = []  # (cls, score, x1, y1, x2, y2)
    arange = np.arange(_REG_MAX, dtype=np.float32)
    for s, fmap in zip(_STRIDES, scales):
        _, _, oh, ow = fmap.shape
        # Build the (oh, ow) anchor grid in pixel coords.
        # YOLOv8: anchor centre = (cx + 0.5, cy + 0.5) * stride.
        grid_y, grid_x = np.meshgrid(
            np.arange(oh, dtype=np.float32),
            np.arange(ow, dtype=np.float32),
            indexing="ij",
        )
        anchor_x = (grid_x + 0.5) * s
        anchor_y = (grid_y + 0.5) * s
        # box: reg_max*4 channels reshape to (4, reg_max, oh, ow), softmax
        # along the reg_max axis, dot with arange -> per-side distances
        # in cells. Multiply by stride to get pixels.
        box = fmap[0, : 4 * _REG_MAX].reshape(4, _REG_MAX, oh, ow)
        box_max = np.max(box, axis=1, keepdims=True)
        box_exp = np.exp(box - box_max)
        box_sm = box_exp / np.sum(box_exp, axis=1, keepdims=True)
        # distance per side (l, t, r, b) in cells:
        d = np.einsum("ksij,s->kij", box_sm, arange) * s
        x1 = anchor_x - d[0]
        y1 = anchor_y - d[1]
        x2 = anchor_x + d[2]
        y2 = anchor_y + d[3]
        # cls: sigmoid
        cls = fmap[0, 4 * _REG_MAX:]  # (nc, oh, ow)
        cls_sig = 1.0 / (1.0 + np.exp(-cls))
        cls_best_id = np.argmax(cls_sig, axis=0)
        cls_best_score = np.max(cls_sig, axis=0)
        # Threshold and collect.
        mask = cls_best_score > score_thresh
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        for yi, xi in zip(ys, xs):
            dets.append((
                int(cls_best_id[yi, xi]),
                float(cls_best_score[yi, xi]),
                float(x1[yi, xi]), float(y1[yi, xi]),
                float(x2[yi, xi]), float(y2[yi, xi]),
            ))

    if not dets:
        return []
    # Simple per-class NMS.
    dets.sort(key=lambda d: -d[1])
    keep = []
    used = [False] * len(dets)
    for i, d_i in enumerate(dets):
        if used[i]:
            continue
        keep.append(d_i)
        if len(keep) >= top_k:
            break
        cls_i, _, ax1, ay1, ax2, ay2 = d_i
        area_i = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        for j in range(i + 1, len(dets)):
            if used[j]:
                continue
            cls_j, _, bx1, by1, bx2, by2 = dets[j]
            if cls_j != cls_i:
                continue
            ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            area_j = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            union = area_i + area_j - inter + 1e-9
            if inter / union > iou_thresh:
                used[j] = True
    return keep[:top_k]


def _detection_iou(da: tuple, db: tuple) -> float:
    ax1, ay1, ax2, ay2 = da[2:6]
    bx1, by1, bx2, by2 = db[2:6]
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def _detection_agreement(dets_a: list[tuple], dets_b: list[tuple],
                         iou_match: float = 0.5) -> dict:
    """Match each detection in A to its best-IoU counterpart in B with
    the same class. Returns count of matched / unmatched on each side."""
    if not dets_a and not dets_b:
        return {"a": 0, "b": 0, "matched": 0, "mean_iou": float("nan")}
    used_b = [False] * len(dets_b)
    matches = []
    for da in dets_a:
        best_iou = 0.0
        best_j = -1
        for j, db in enumerate(dets_b):
            if used_b[j] or db[0] != da[0]:
                continue
            iou = _detection_iou(da, db)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_match:
            used_b[best_j] = True
            matches.append(best_iou)
    return {
        "a": len(dets_a), "b": len(dets_b),
        "matched": len(matches),
        "mean_iou": float(np.mean(matches)) if matches else float("nan"),
    }


def _print_detections(label: str, dets: list[tuple], max_show: int = 6):
    print(f"  {label} (n={len(dets)}, top-{min(max_show, len(dets))}):")
    if not dets:
        print("    (none above threshold)")
        return
    for cls, score, x1, y1, x2, y2 in dets[:max_show]:
        print(f"    cls={cls:>2d} score={score:.3f} "
              f"box=({x1:7.2f},{y1:7.2f})-({x2:7.2f},{y2:7.2f})")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build-dir",
                   default="modelblaster/examples/yolov8_nano/int8/build/scalar")
    p.add_argument("--ir-dir",
                   default="modelblaster/examples/yolov8_nano/int8/generated")
    p.add_argument("--spike",
                   default="/scratch2/dima/miniforge3/envs/zephyr/bin/spike")
    p.add_argument("--img", type=int,
                   default=int(os.environ.get("MODELBLASTER_YOLOV8N_INPUT", "160")))
    p.add_argument("--score-thresh", type=float, default=0.10)
    p.add_argument("--iou-thresh", type=float, default=0.45)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--skip-ultralytics", action="store_true",
                   help="skip the upstream ultralytics row (e.g. when "
                        "running offline / no ultralytics installed)")
    args = p.parse_args()

    elf = Path(args.build_dir) / "zephyr" / "zephyr.elf"
    if not elf.exists():
        sys.exit(f"build the binary first (missing {elf}). e.g.:\n"
                 f"  bash modelblaster/examples/yolov8_nano/run.sh")

    print(f"== input: 1x3x{args.img}x{args.img} (seed=1 synthetic frame, "
          f"matches io.npz baked-in input) ==")
    print()

    # ---- (1) Our fx-traceable wrapper at fp32 -----------------------------
    # The PyTorch flow uses get_sample_input() with seed=1 — same input
    # the io.npz golden was made from.
    os.environ.setdefault("MODELBLASTER_YOLOV8N_INPUT", str(args.img))
    from modelblaster.models import yolov8_nano as ymod
    x = ymod.get_sample_input(seed=1)
    model = ymod.get_model()
    with torch.no_grad():
        wrap_outs = model(x)
    # wrap_outs is a tuple of 3 tensors (B, 144, OH, OW)
    wrap_scales = [t.cpu().numpy() for t in wrap_outs]
    print("[1] our fx-traceable wrapper @ fp32:")
    print(f"    P3 {wrap_scales[0].shape}, P4 {wrap_scales[1].shape}, "
          f"P5 {wrap_scales[2].shape}")
    print()

    # ---- (2) Ultralytics upstream (optional) -----------------------------
    ultra_scales: Optional[list[np.ndarray]] = None
    if not args.skip_ultralytics:
        try:
            from ultralytics import YOLO  # noqa: PLC0415
            yolo = YOLO("yolov8n.pt")
            yolo.model.eval()
            with torch.no_grad():
                # ultralytics' DetectionModel.forward returns a tuple
                # (preds, raw_features) in eval; raw is what matches our
                # wrapper. To extract the same raw 3-scale features we
                # call the head directly. Easier: monkey-patch detect to
                # return raw, or call the backbone+neck+head manually.
                # Simplest hack: temporarily replace the head's "export"
                # flag so it returns raw features.
                detect = yolo.model.model[-1]
                _saved_training = detect.training
                detect.training = True   # training mode returns raw maps
                ultra_outs_t = yolo.model(x)
                detect.training = _saved_training
            # Ultralytics' detect head in training mode returns either a
            # list of 3 tensors OR a dict {"one_img": [...]} on newer
            # ultralytics versions (8.4+). Handle both shapes.
            maps = None
            if isinstance(ultra_outs_t, (list, tuple)) and \
                    len(ultra_outs_t) == 3 and torch.is_tensor(ultra_outs_t[0]):
                maps = ultra_outs_t
            elif isinstance(ultra_outs_t, dict):
                # Pick the value that's a 3-list of tensors with our shapes.
                for v in ultra_outs_t.values():
                    if isinstance(v, (list, tuple)) and len(v) == 3 \
                            and torch.is_tensor(v[0]):
                        maps = v
                        break
            if maps is not None:
                ultra_scales = [t.cpu().numpy() for t in maps]
                print("[2] ultralytics upstream @ fp32 (yolov8n.pt direct):")
                print(f"    P3 {ultra_scales[0].shape}, "
                      f"P4 {ultra_scales[1].shape}, "
                      f"P5 {ultra_scales[2].shape}")
            else:
                print("[2] ultralytics: unexpected output shape, skipping.")
                print(f"    got: {type(ultra_outs_t)}")
        except Exception as e:
            print(f"[2] ultralytics row skipped ({type(e).__name__}: {e})")
        print()

    # ---- (3) int8 representation ceiling ---------------------------------
    io = np.load(Path(args.ir_dir) / "io.npz")
    ceil_flat = np.asarray(io["output"], dtype=np.float32)
    ceil_scales = _split_into_scales(ceil_flat, args.img)
    print("[3] int8 ceiling (PyTorch int8 simulator forward):")
    print(f"    P3 {ceil_scales[0].shape}, P4 {ceil_scales[1].shape}, "
          f"P5 {ceil_scales[2].shape}")
    print()

    # ---- (4) Spike actual ----------------------------------------------
    # The yolov8 harness uses in-binary verify (no stdout output dump —
    # 75600 floats over HTIF would dominate spike runtime). Parse the
    # MODELBLASTER_VERIFY marker; if max_abs_err==0 the spike output is
    # bit-identical to io.npz["output"] and the "ceiling" row above
    # is also the spike row.
    print(f"[4] running spike on {elf} ...", flush=True)
    spike_text = _run_spike(str(elf), args.spike)
    spike_flat = _parse_output_block(spike_text)
    spike_scales: Optional[list[np.ndarray]] = None
    spike_verify = _parse_verify_marker(spike_text)
    if spike_flat is not None:
        spike_scales = _split_into_scales(spike_flat, args.img)
        print(f"    P3 {spike_scales[0].shape}, "
              f"P4 {spike_scales[1].shape}, "
              f"P5 {spike_scales[2].shape}")
    elif spike_verify is not None:
        print(f"    in-binary verify: max_abs_err="
              f"{spike_verify['max_abs_err']} max_rel_err="
              f"{spike_verify['max_rel_err']} n={spike_verify['n']}")
        if spike_verify["max_abs_err"] == 0.0:
            print(f"    -> spike output is bit-identical to io.npz "
                  f"['output'] (the int8 ceiling row above is also "
                  f"the spike row).")
            spike_scales = ceil_scales
        else:
            print(f"    -> spike drifts from io.npz; can't extract "
                  f"per-element spike output without rebuilding the "
                  f"harness with stdout dump enabled. Skipping the "
                  f"spike-vs-ceiling row.")
    else:
        print("    !! could not find any verify or output marker in "
              "spike stdout. Build state probably stale (pre-rename "
              "AGENTS_* vs the renamed MODELBLASTER_* — wipe and "
              "rebuild the example).")
    print()

    # ---- Raw-tensor comparisons -----------------------------------------
    def _scale_rows(name: str, a: list[np.ndarray], b: list[np.ndarray]):
        rows = [_compare_tensors(a[i], b[i], f"P{3+i}")
                for i in range(3)]
        rows.append(_compare_tensors(np.concatenate([s.ravel() for s in a]),
                                     np.concatenate([s.ravel() for s in b]),
                                     "all"))
        _print_table(rows, name)

    print("== raw-tensor pairwise (vs our wrapper @ fp32) ==")
    if ultra_scales is not None:
        _scale_rows("ultralytics  vs  wrapper", ultra_scales, wrap_scales)
        print()
    _scale_rows("ceiling      vs  wrapper", ceil_scales, wrap_scales)
    print()
    if spike_scales is not None:
        if spike_scales is ceil_scales:
            print("  spike actual vs wrapper / ceiling: identical to "
                  "ceiling (max_abs_err=0 in-binary).\n")
        else:
            _scale_rows("spike actual vs  wrapper", spike_scales, wrap_scales)
            print()
            _scale_rows("spike actual vs  ceiling", spike_scales, ceil_scales)
            print()

    # ---- Post-NMS detection comparison ----------------------------------
    print("== post-DFL+NMS detection comparison ==")
    print(f"   score_thresh={args.score_thresh} iou_thresh={args.iou_thresh} "
          f"top_k={args.top_k}")
    print()
    wrap_dets = _decode_dfl_and_nms(wrap_scales, args.img,
                                     args.score_thresh, args.iou_thresh,
                                     args.top_k)
    _print_detections("wrapper @ fp32", wrap_dets)
    print()
    if ultra_scales is not None:
        ultra_dets = _decode_dfl_and_nms(ultra_scales, args.img,
                                          args.score_thresh, args.iou_thresh,
                                          args.top_k)
        _print_detections("ultralytics @ fp32", ultra_dets)
        ag = _detection_agreement(ultra_dets, wrap_dets, iou_match=0.5)
        print(f"  matched (cls+IoU≥0.5): {ag['matched']}/{ag['a']} "
              f"(mean IoU when matched: {ag['mean_iou']:.3f})")
        print()
    ceil_dets = _decode_dfl_and_nms(ceil_scales, args.img,
                                     args.score_thresh, args.iou_thresh,
                                     args.top_k)
    _print_detections("int8 ceiling", ceil_dets)
    ag = _detection_agreement(wrap_dets, ceil_dets, iou_match=0.5)
    print(f"  matched vs wrapper (cls+IoU≥0.5): {ag['matched']}/{ag['a']} "
          f"(mean IoU when matched: {ag['mean_iou']:.3f})")
    print()
    if spike_scales is not None:
        spike_dets = _decode_dfl_and_nms(spike_scales, args.img,
                                          args.score_thresh,
                                          args.iou_thresh,
                                          args.top_k)
        _print_detections("spike actual", spike_dets)
        ag_w = _detection_agreement(wrap_dets, spike_dets, iou_match=0.5)
        ag_c = _detection_agreement(ceil_dets, spike_dets, iou_match=0.5)
        print(f"  matched vs wrapper (cls+IoU≥0.5): "
              f"{ag_w['matched']}/{ag_w['a']} "
              f"(mean IoU: {ag_w['mean_iou']:.3f})")
        if spike_scales is ceil_scales:
            print("  matched vs ceiling: identical (spike output is "
                  "bit-exact to io.npz).")
        else:
            print(f"  matched vs ceiling (cls+IoU≥0.5): "
                  f"{ag_c['matched']}/{ag_c['a']} "
                  f"(mean IoU: {ag_c['mean_iou']:.3f})")

    # ---- Verdict / interpretation ---------------------------------------
    print()
    print("== verdict ==")
    # HW correctness:
    if spike_verify is not None and spike_verify["max_abs_err"] == 0.0:
        print("  ✓ HW correctness: spike output bit-exact to PyTorch int8")
        print("    simulator (in-binary verify max_abs_err=0, n="
              f"{spike_verify['n']}). The C kernels + scalar/RVV/OPU")
        print("    paths reproduce the simulator exactly.")
    # PTQ quality:
    rows_ceil = [_compare_tensors(ceil_scales[i], wrap_scales[i],
                                   f"P{3+i}") for i in range(3)]
    ceil_linf = max(r["L_inf"] for r in rows_ceil)
    ceil_cos_min = min(r["cosine"] for r in rows_ceil)
    if ceil_linf > 5.0 or ceil_cos_min < 0.5:
        print()
        print(f"  ⚠ PTQ quality: int8 ceiling diverges substantially from")
        print(f"    fp32 wrapper (L∞={ceil_linf:.2f}, "
              f"min(cosine)={ceil_cos_min:.3f}).")
        print(f"    This is a CALIBRATION issue, not a HW or quantization-")
        print(f"    scheme bug. The default extract uses a single random")
        print(f"    torch.randn frame for activation-range calibration; for")
        print(f"    a vision detection model trained on natural images, that")
        print(f"    samples activations far out of distribution. The")
        print(f"    Q0.31 multipliers / shifts end up over-narrow, saturating")
        print(f"    the cls logits and producing nonsense detections (n="
              f"{len(ceil_dets)} top-confidence boxes vs {len(wrap_dets)}")
        print(f"    from fp32). To get meaningful int8 accuracy on yolov8n:")
        print(f"      1. Calibrate on real images (a handful of COCO val")
        print(f"         frames or similar). Plumb via")
        print(f"         modelblaster/datasets/<your_loader>.py.")
        print(f"      2. Use --per-channel for the conv weight scales.")
        print(f"      3. Bump --num-calibration to >= 16.")
    elif ceil_linf > 0.5:
        print()
        print(f"  ✓ PTQ quality: int8 ceiling tracks fp32 wrapper within "
              f"reasonable bounds (L∞={ceil_linf:.2f}, cos≥{ceil_cos_min:.3f}).")


if __name__ == "__main__":
    main()
