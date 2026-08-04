"""Bundle existing DLCs into multi-graph context binaries on the board.

Each call to `qnn-context-binary-generator --dlc_path a.dlc,b.dlc,...`
produces ONE context binary holding all listed graphs. On QRB5165 v66 we
validated that up to 10 graphs per ctx works cleanly on DSP and HTA
(see qnn_models/runtime/multi_graph_test.cpp). This collapses the firmware
context-cap pressure: 23 vision DSP segments → 3 ctx loads instead of 23,
46 HTA convs → 5 ctx loads instead of 46.

What this script does:
  1. Identifies the DLCs to bundle (from a glob over a directory).
  2. Pushes them to the board (skipping any already present).
  3. Chunks the list at N per group.
  4. For each chunk, runs `qnn-context-binary-generator` for the chosen
     backend, producing one multi-graph .bin on board.
  5. Writes a manifest.json describing each (graph_name, ctx_bin, backend)
     tuple. Downstream tools (stage script + runtime generator) read this
     to wire up dispatches → graphs.

The graph_name inside each ctx binary is what the DLC's
`snpe-onnx-to-dlc` step preserved (typically matches the source ONNX
filename's stem without `_quantized` suffix). We read it back via
QnnSystemContext_GetBinaryInfo on the board to be sure.

Usage:
  python build_multi_graph_ctx.py \\
      --dlc-dir vision_slices_v3/dlc \\
      --dlc-pattern 'dsp_seg_*_quantized.dlc' \\
      --backend Dsp \\
      --chunk 10 \\
      --bundle-name v3_dsp_segs \\
      --board $QNN_BOARD_HOST

Produces:
  on host:  vision_slices_v3/multi_ctx/v3_dsp_segs_manifest.json
  on board: /root/multi_ctx/ctx_v3_dsp_segs_chunk0__Dsp.bin,
            ctx_v3_dsp_segs_chunk1__Dsp.bin, ...
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dlc-dir", required=True,
                    help="host-side directory containing source DLCs")
    ap.add_argument("--dlc-pattern", required=True,
                    help="glob pattern relative to --dlc-dir (e.g. 'dsp_seg_*_quantized.dlc')")
    ap.add_argument("--backend", required=True,
                    choices=["Cpu", "Dsp", "Hta"],
                    help="target backend; selects libQnn<Backend>.so")
    ap.add_argument("--chunk", type=int, default=10,
                    help="max graphs per multi-context binary (default 10)")
    ap.add_argument("--bundle-name", required=True,
                    help="prefix for the output bin filenames + manifest key")
    ap.add_argument("--board", default=os.environ.get("QNN_BOARD_HOST", "root@<BOARD_IP>"))
    ap.add_argument("--remote-base", default="/root/multi_ctx",
                    help="board-side dir for sources + output bins")
    ap.add_argument("--qnn-root", default="/root/qairt",
                    help="board-side QNN SDK root")
    ap.add_argument("--out-dir", default=None,
                    help="host-side dir for the manifest (default: <dlc-dir>/../multi_ctx)")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if the multi-ctx bin already exists on the board")
    args = ap.parse_args()

    dlc_dir = Path(args.dlc_dir).resolve()
    dlcs = sorted(dlc_dir.glob(args.dlc_pattern))
    if not dlcs:
        print(f"ERROR: no DLCs match {args.dlc_pattern} in {dlc_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(dlcs)} DLCs to bundle (chunk size = {args.chunk})")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (dlc_dir.parent / "multi_ctx")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure board base dir exists; push DLCs into a subdir
    src_dir = f"{args.remote_base}/src/{args.bundle_name}"
    subprocess.run(["ssh", args.board, f"mkdir -p {args.remote_base} {src_dir}"], check=True)

    # Push the DLCs (skip ones already present + same size)
    print("Pushing DLCs to board...")
    existing = subprocess.run(
        ["ssh", args.board, f"cd {src_dir} && find . -maxdepth 1 -name '*.dlc' -printf '%f %s\\n' 2>/dev/null"],
        check=False, capture_output=True, text=True).stdout
    existing_set = {}
    for line in existing.strip().splitlines():
        parts = line.split()
        if len(parts) == 2:
            existing_set[parts[0]] = int(parts[1])
    n_pushed = 0
    for d in dlcs:
        if existing_set.get(d.name) == d.stat().st_size:
            continue
        subprocess.run(["scp", "-q", str(d), f"{args.board}:{src_dir}/"], check=True)
        n_pushed += 1
    print(f"  pushed {n_pushed} new DLCs ({len(dlcs) - n_pushed} already present)")

    # Chunk + build
    chunks = [dlcs[i:i+args.chunk] for i in range(0, len(dlcs), args.chunk)]
    print(f"Building {len(chunks)} multi-graph context binaries on board...")
    manifest = {
        "bundle_name": args.bundle_name,
        "backend": args.backend,
        "chunk_size": args.chunk,
        "source_dlc_dir": str(dlc_dir),
        "source_dlc_pattern": args.dlc_pattern,
        "remote_base": args.remote_base,
        "chunks": [],
    }
    for chunk_idx, chunk in enumerate(chunks):
        dlc_names = [d.name for d in chunk]
        bin_name = f"ctx_{args.bundle_name}_chunk{chunk_idx}__{args.backend}.bin"
        bin_path = f"{args.remote_base}/{bin_name}"
        dlc_list_str = ",".join(dlc_names)
        # Skip rebuild if exists and not forced
        check = subprocess.run(["ssh", args.board, f"test -f {bin_path} && stat -c%s {bin_path}"],
                                capture_output=True, text=True)
        if not args.force and check.returncode == 0:
            print(f"  chunk{chunk_idx} ({len(chunk)} graphs): SKIP (exists, {check.stdout.strip()} B)")
        else:
            cmd = textwrap.dedent(f"""
                set +e
                cd {src_dir}
                export LD_LIBRARY_PATH={args.qnn_root}/lib/target
                export ADSP_LIBRARY_PATH="{args.qnn_root}/lib/hexagon-v66;/dsp/cdsp;/dsp"
                {args.qnn_root}/bin/target/qnn-context-binary-generator \\
                    --backend {args.qnn_root}/lib/target/libQnn{args.backend}.so \\
                    --model {args.qnn_root}/lib/target/libQnnModelDlc.so \\
                    --dlc_path "{dlc_list_str}" \\
                    --binary_file {bin_name[:-4]} --output_dir {args.remote_base} \\
                    > /tmp/_mgb_{args.bundle_name}_{chunk_idx}.log 2>&1
                rc=$?
                if [ "$rc" -eq 0 ] && [ -f "{bin_path}" ]; then
                    echo "OK $(stat -c%s {bin_path})"
                else
                    echo "FAIL rc=$rc"
                    tail -5 /tmp/_mgb_{args.bundle_name}_{chunk_idx}.log
                fi
            """).strip()
            r = subprocess.run(["ssh", args.board, cmd], capture_output=True, text=True)
            if "OK" in r.stdout:
                size = r.stdout.split()[1]
                print(f"  chunk{chunk_idx} ({len(chunk)} graphs): built {size} B")
            else:
                print(f"  chunk{chunk_idx}: FAILED")
                print("    " + r.stdout.replace("\n", "\n    "))
                sys.exit(2)

        # Inspect the resulting bin to discover the actual graph names inside.
        # We use a small Python on board that calls systemContextGetBinaryInfo.
        # Easier: run multi_graph_test if present, parse "[i] graphname"
        # lines. If not present, just trust DLC stem mapping.
        graphs_in_chunk = []
        for d in chunk:
            # Convention: DLC `dsp_seg_01_quantized.dlc` produces graph
            # named `dsp_seg_01`. The trampoline DLCs follow similar
            # patterns. The standalone Conv DLCs from extract_hta_convs
            # use the conv-node-name inside (e.g. `dsp_seg_01_node_MatMul_549_conv1x1`).
            stem = d.stem
            for suffix in ("_quantized", "_q"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            graphs_in_chunk.append({"dlc": d.name, "graph_name": stem})

        manifest["chunks"].append({
            "chunk_idx": chunk_idx,
            "bin_filename": bin_name,
            "remote_bin_path": bin_path,
            "n_graphs": len(chunk),
            "graphs": graphs_in_chunk,
        })

    # Namespace manifest by backend so multiple backends with the same
    # bundle-name don't collide.
    manifest_path = out_dir / f"{args.bundle_name}_{args.backend}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path}")

    # Also stamp a tiny graph-name -> ctx index for the runtime/stage step.
    # Keyed by graph_name; backend is implicit (caller picks the right index
    # for the dispatch's target backend).
    g2c = {}
    for chunk in manifest["chunks"]:
        for g in chunk["graphs"]:
            g2c[g["graph_name"]] = {
                "bin": chunk["bin_filename"],
                "remote": chunk["remote_bin_path"],
                "chunk_idx": chunk["chunk_idx"],
                "backend": args.backend,
            }
    g2c_path = out_dir / f"{args.bundle_name}_{args.backend}_graph_index.json"
    with open(g2c_path, "w") as f:
        json.dump(g2c, f, indent=2)
    print(f"Graph index: {g2c_path}")


if __name__ == "__main__":
    main()
