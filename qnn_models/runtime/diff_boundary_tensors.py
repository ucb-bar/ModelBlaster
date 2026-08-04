"""Per-boundary numeric diff between the runtime's intermediates and
onnxruntime's golden captures.

Reads:
  * runtime cache dump (run with QNN_RUNTIME_DUMP_ALL=1) — one .raw
    file per intermediate tensor, named after the sanitized tensor.
  * the boundary-calibration capture from
    sub_onnx/calib/<network>/tensors/<safe_name>/sample_0000.raw —
    these are fp32 onnxruntime intermediates at the same tensor names.

For each tensor present in both: dequantise the runtime bytes using
each per-tensor scale/zero-point pulled from the sub-DLC info (we
parse snpe-dlc-info output offline), then compute (max_abs_err,
mean_abs_err, cosine) against the fp32 capture. Per-boundary error
breakdown shows where the per-segment chain accumulates drift.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

import numpy as np


def _safe(name: str) -> str:
    s = re.sub(r"[/.]", "_", name)
    while s.startswith("_"): s = s[1:]
    return s


def _read_dlc_encodings(dlc_dir: str) -> dict[str, tuple[float, int]]:
    """Run snpe-dlc-info on every quantized sub-DLC under dlc_dir and
    parse `<tensor_name> encoding : ... scale X, offset Y` lines into
    {tensor_name: (scale, signed_offset)} pairs."""
    out: dict[str, tuple[float, int]] = {}
    for fn in sorted(os.listdir(dlc_dir)):
        if not fn.endswith("_quantized.dlc"):
            continue
        try:
            text = subprocess.check_output([
                "docker", "run", "--rm", "-v", f"{os.getcwd()}:/work",
                "qnn-convert",
                "/qnn/bin/x86_64-linux-clang/snpe-dlc-info",
                "-i", f"/work/{os.path.relpath(os.path.join(dlc_dir, fn))}",
            ], stderr=subprocess.DEVNULL, timeout=30).decode()
        except Exception as e:
            print(f"  snpe-dlc-info {fn}: {e}", file=sys.stderr); continue
        # Match lines like:
        #   <tensor_name> encoding : bitwidth 8, min ..., max ..., scale 0.0xxx, offset -129.0
        for m in re.finditer(
                r"([\w/.\-]+)\s+encoding\s*:\s*bitwidth\s+\d+,"
                r"\s*min\s*[-\d.eE]+,\s*max\s*[-\d.eE]+,"
                r"\s*scale\s+([-\d.eE]+),\s*offset\s+([-\d.eE]+)", text):
            name = m.group(1)
            scale = float(m.group(2))
            offset = int(float(m.group(3)))
            # Prefer the first occurrence — the same tensor can appear
            # in both producer's "out" and consumer's "in", and the
            # producer's encoding is what got serialised.
            if name not in out:
                out[name] = (scale, offset)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runtime-dump",   required=True,
                    help="dir with one .raw per cache entry (run with QNN_RUNTIME_DUMP_ALL=1)")
    ap.add_argument("--golden-dir",     required=True,
                    help="sub_onnx/calib/<network>/tensors/ — onnxruntime fp32 captures")
    ap.add_argument("--sample-idx",     type=int, default=0,
                    help="which calibration sample's golden captures to diff against")
    ap.add_argument("--dlc-dir",        required=True,
                    help="dir containing the *_quantized.dlc files for scale/offset extraction")
    args = ap.parse_args()

    print(f"==> reading sub-DLC quantization encodings from {args.dlc_dir}")
    enc = _read_dlc_encodings(args.dlc_dir)
    print(f"   parsed {len(enc)} per-tensor encodings")

    # Build a map: tensor_name → (golden_path, runtime_path). golden uses
    # onnxruntime sanitized name, runtime uses our sanitized name.
    pairs = []
    for safe_dir in sorted(os.listdir(args.golden_dir)):
        gpath = os.path.join(args.golden_dir, safe_dir, f"sample_{args.sample_idx:04d}.raw")
        if not os.path.exists(gpath): continue
        # The runtime's safe-name strips a leading '/' (turns '_x' →
        # 'x' after the trim-leading-underscore step). The golden's
        # safe-name keeps it (the sanitizer there used a single
        # `re.sub` without trim). Try both forms.
        cands = [safe_dir.lstrip("_"), safe_dir]
        for c in cands:
            rpath = os.path.join(args.runtime_dump, c + ".raw")
            if os.path.exists(rpath): break
        else:
            continue
        # Reverse the safe-name to recover the original tensor name
        # for encoding lookup. Try a few candidates: leading "/" or not.
        orig_candidates = [
            "/" + safe_dir.lstrip("_").replace("_", "/"),
            safe_dir.lstrip("_").replace("_", "/"),
            "/" + safe_dir.lstrip("_").replace("_", "."),
            "/" + safe_dir.replace("_", "/"),
        ]
        # A simpler heuristic: in our encoding map, look for any key
        # whose `_safe()` matches `safe_dir`.
        match = next((k for k in enc if _safe(k) == safe_dir
                       or _safe(k) == safe_dir.lstrip("_")), None)
        pairs.append((safe_dir, gpath, rpath, match))

    print(f"\n==> per-boundary diff ({len(pairs)} tensors)")
    print(f"    {'tensor':<45s}  {'g_range':>22s}  {'max_abs':>10s}  {'mean_abs':>10s}  {'cosine':>8s}  enc")
    print(f"    {'-'*45}  {'-'*22}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*15}")
    for safe_dir, gpath, rpath, enc_key in pairs:
        g = np.fromfile(gpath, dtype=np.float32)
        rb = np.fromfile(rpath, dtype=np.uint8)
        if rb.size != g.size:
            # Sometimes rb is twice as big (shape diff) or vice versa —
            # take the common prefix.
            n = min(rb.size, g.size)
            rb = rb[:n]; g = g[:n]
        if enc_key and enc_key in enc:
            scale, off = enc[enc_key]
            zp = -off
            cand = (rb.astype(np.int32) - zp) * scale
            ekey = f"{scale:.4g}/{zp}"
        else:
            cand = rb.astype(np.float32)
            ekey = "RAW"
        diff = np.abs(g - cand)
        cos = (np.dot(g, cand) /
               (np.linalg.norm(g) * np.linalg.norm(cand) + 1e-12))
        print(f"    {safe_dir:<45s}  "
              f"[{g.min():>9.4g},{g.max():>9.4g}]  "
              f"{diff.max():>10.4g}  {diff.mean():>10.4g}  "
              f"{cos:>8.4f}  {ekey}")


if __name__ == "__main__":
    main()
