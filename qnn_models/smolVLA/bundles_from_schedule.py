#!/usr/bin/env python3
"""Generate multi-graph bundle definitions + an on-board build script for
*any* scheduled JSON. Generalizes build_sched23_bundles.py.

WARNING — multi-graph context binaries silently crash the cDSP user-PD on
QRB5165 v66 firmware. See qnn_models/QRB5165_MULTIGRAPH_CDSP_CRASH_FORENSICS.md.
This script is preserved for future use (custom op-package, newer chip,
etc.); do not currently wire its output into a production runtime. The
canonical v3 bundle-aware run continues to use *single-graph* contexts.

Given a scheduled JSON (and the same segment_perf.json the scheduler used
for conv-name resolution), this script:

  1. Walks every dispatch in the schedule.
  2. Groups dispatches by (target_backend, bundle_name) where bundle_name
     is derived from the dispatch's module pattern (e.g. dsp_seg_*_tramp_*
     → "<schedule_stem>_dsp_tramps", dsp_seg_*_convN → "<schedule_stem>_hta_convs",
     etc.).
  3. Chunks each bundle's graphs at N (default 10) and emits:
       a) <schedule_stem>_<bundle>_<be>_graph_index.json
          (consumed by generate_runtime.py --graph-index)
       b) build_bundles_on_board.sh (single script that builds every
          chunk for every bundle; safe to re-run, skips existing)

Replaces:
  - build_sched23_bundles.py (hardcoded sched23 layer list)
  - /tmp/build_all_bundles.sh and friends (hand-listed DLCs)
  - the duplicated chunking logic in stage_multi_graph_pipeline.sh
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

# Map a schedule's hardware_target (the abstract name like "CPU_E#0") to
# the QNN backend identifier ("Dsp" / "Hta" / "Cpu"). This mirrors the
# --backend-map CLI of generate_runtime.py.
DEFAULT_BACKEND_MAP = {
    "CPU_E": "Dsp",
    "CPU_P": "Hta",
    "CPU_X": "Cpu",
}

# Pattern → bundle name fragment. The matcher runs against the
# dispatch's `module_name` and picks the first matching label. Each
# match also defines the corresponding DLC source naming convention:
#   - dlc_dir relative to the slice root
#   - dlc_suffix:  what gets appended to a graph_name to form the DLC
#                  basename. None means same-as-graph-name.
#   - graph_name_fn(module_name): how to derive the on-disk graph name.
#                                 Default is identity.
BUNDLE_RULES = [
    # (bundle_fragment, regex on module_name, dlc_subdir, dlc_suffix)
    ("dsp_tramps", re.compile(r"^dsp_seg_\d+_tramp_p[012]$"),
        "trampolines/dlc_dsp", "_q.dlc"),
    ("cpu_tramps", re.compile(r"^dsp_seg_\d+_tramp_p[012]$"),
        "trampolines/dlc",     ".dlc"),
    ("hta_convs",  re.compile(r"^dsp_seg_\d+_conv\d+$"),
        "hta_convs/dlc",       "_q.dlc"),
    ("dsp_segs",   re.compile(r"^dsp_seg_\d+$"),
        "dlc",                 "_quantized.dlc"),
    ("cpu_segs",   re.compile(r"^cpu_seg_\d+$"),
        "dlc",                 ".dlc"),
]


def load_segment_perf(p: Path) -> dict:
    return json.load(open(p)) if p and p.exists() else {}


def resolve_conv_name(module: str, seg_perf: dict) -> str | None:
    """dsp_seg_05_conv2 → dsp_seg_05_node_MatMul_794_conv1x1
       using the segment_perf.json mapping (Hta.convs[1].name).
       Returns None if unmappable."""
    m = re.match(r"^(dsp_seg_\d+)_conv(\d+)$", module)
    if not m:
        return module
    seg, idx = m.group(1), int(m.group(2))
    convs = seg_perf.get(seg, {}).get("Hta", {}).get("convs", [])
    if 1 <= idx <= len(convs):
        return convs[idx - 1]["name"]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("schedule",
                    help="path to scheduled_*.json")
    ap.add_argument("--seg-perf",
                    help="segment_perf.json (for conv1/conv2 → real DLC name)")
    ap.add_argument("--out-dir", default=None,
                    help="output dir for graph_index JSONs (default: "
                         "<slice_root>/multi_ctx_<schedule_stem>)")
    ap.add_argument("--slice-root",
                    default="qnn_models/smolVLA/vision_slices_v3",
                    help="root dir holding dlc/, trampolines/, hta_convs/")
    ap.add_argument("--build-script", default=None,
                    help="output path for board-side build script (default: "
                         "<out_dir>/build_bundles_on_board.sh)")
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--out-ctx-dir", default="/root/multi_ctx",
                    help="on-board dir where ctx binaries will live")
    ap.add_argument("--backend-map", default=None,
                    help="comma-separated overrides, e.g. CPU_E=Dsp,CPU_P=Hta,CPU_X=Cpu")
    args = ap.parse_args()

    backend_map = dict(DEFAULT_BACKEND_MAP)
    if args.backend_map:
        for tok in args.backend_map.split(","):
            k, v = tok.split("=")
            backend_map[k.strip()] = v.strip()

    sched = json.load(open(args.schedule))
    sched_stem = Path(args.schedule).stem
    # Trim noise prefix to get a short identifier:
    short = sched_stem.replace("scheduled_networks_", "")\
                     .replace("_greedy_profiled", "")\
                     .replace("_milp_profiled", "")
    # Compact further to avoid 200-char filenames
    if len(short) > 40:
        short = re.sub(r"[^a-z0-9_]", "_", short.lower())[:40]

    seg_perf = load_segment_perf(Path(args.seg_perf)) if args.seg_perf else {}

    # Group dispatches: (bundle_fragment, backend) → ordered list of (module, graph_name, dlc_basename)
    groups: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    for d in sched["dispatches"].values():
        hw = d["hardware_target"].split("#", 1)[0]  # CPU_E#0 → CPU_E
        be = backend_map.get(hw)
        if be is None:
            print(f"  skipping {d['module_name']} on unknown hw {hw}")
            continue
        module = d["module_name"]
        # Find which bundle rule matches
        matched = None
        for frag, rx, dlc_sub, suffix in BUNDLE_RULES:
            if rx.match(module):
                # cpu_tramps vs dsp_tramps disambiguates by backend
                if frag == "dsp_tramps" and be != "Dsp": continue
                if frag == "cpu_tramps" and be != "Cpu": continue
                if frag == "hta_convs"  and be != "Hta": continue
                matched = (frag, rx, dlc_sub, suffix)
                break
        if matched is None:
            # Mono dispatches that go through the v3_dsp_segs CPU fallback path
            # still need entries — fall back to dsp_segs/cpu_segs by name
            continue
        frag, rx, dlc_sub, suffix = matched

        # Resolve graph name (which may differ from module for synthetic convN)
        if frag == "hta_convs":
            graph = resolve_conv_name(module, seg_perf)
            if graph is None:
                print(f"  WARN: cannot resolve conv name for {module}; skip")
                continue
        else:
            graph = module
        dlc_basename = f"{graph}{suffix}"
        key = (frag, be, dlc_sub)
        # Dedup within bundle (a graph might be dispatched multiple times)
        if any(g == graph for _, g, _ in groups[key]):
            continue
        groups[key].append((module, graph, dlc_basename))

    # Output dir
    if args.out_dir is None:
        out_dir = Path(args.slice_root) / f"multi_ctx_{short}"
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Emit per-bundle graph_index.json + collect build commands
    print(f"Schedule:   {args.schedule}")
    print(f"Stem:       {short}")
    print(f"Slice root: {args.slice_root}")
    print(f"Out dir:    {out_dir}")
    print()

    build_lines: list[str] = []
    chunk_summaries = []
    for (frag, be, dlc_sub), entries in sorted(groups.items()):
        n = len(entries)
        n_chunks = (n + args.chunk - 1) // args.chunk
        print(f"  {short}_{frag}_{be}: {n} graphs → {n_chunks} chunks "
              f"(DLC dir: {dlc_sub})")

        # Graph index
        idx = {}
        for i, (_module, graph, _dlc) in enumerate(entries):
            ci = i // args.chunk
            idx[graph] = {
                "bin": f"ctx_{short}_{frag}_chunk{ci}__{be}.bin",
                "chunk_idx": ci,
                "backend": be,
            }
        idx_path = out_dir / f"{short}_{frag}_{be}_graph_index.json"
        idx_path.write_text(json.dumps(idx, indent=2))

        # Build script lines per chunk
        for ci in range(n_chunks):
            chunk = entries[ci*args.chunk:(ci+1)*args.chunk]
            dlcs = " ".join(dlc for _, _, dlc in chunk)
            out_bin = f"ctx_{short}_{frag}_chunk{ci}__{be}.bin"
            build_lines.append(
                f'build_one {out_bin} {be} "$SLICE_ROOT/{dlc_sub}" {dlcs}'
            )
            chunk_summaries.append((short, frag, be, ci, len(chunk)))

    # Emit board-side build script
    build_script = Path(args.build_script) if args.build_script else \
                   out_dir / "build_bundles_on_board.sh"
    header = [
        "#!/usr/bin/env bash",
        f"# Auto-generated by bundles_from_schedule.py from {Path(args.schedule).name}.",
        "# Builds every multi-graph context binary the schedule references.",
        "# Safe to re-run: skips chunks whose .bin already exists.",
        "set +e",
        "QNN=/root/qairt",
        "export LD_LIBRARY_PATH=$QNN/lib/target",
        'export ADSP_LIBRARY_PATH="$QNN/lib/hexagon-v66/unsigned;$QNN/lib/hexagon-v66;/dsp/cdsp;/dsp"',
        f'OUT={args.out_ctx_dir}',
        'SLICE_ROOT="${SLICE_ROOT:-/root/v3}"',
        "[ -d /root/models/smolvlm_vision_v3/dlc ] && SLICE_ROOT=/root/models/smolvlm_vision_v3 || true",
        '# (Common board layouts: cloud uses /root/v3; physical uses /root/models/smolvlm_vision_v3)',
        'mkdir -p "$OUT"; cd "$OUT"',
        "",
        "build_one() {",
        "    local out_name=$1 backend=$2 src_dir=$3; shift 3",
        "    local dlcs=()",
        '    for d in "$@"; do',
        '        if [ -f "$src_dir/$d" ]; then dlcs+=("$d")',
        '        else echo "  MISSING: $src_dir/$d"; return 1; fi',
        "    done",
        '    local list=$(IFS=,; echo "${dlcs[*]}")',
        '    if [ -f "$OUT/$out_name" ]; then',
        '        echo "  $out_name: SKIP (exists, $(stat -c%s $OUT/$out_name) B)"',
        "        return 0",
        "    fi",
        '    (cd "$src_dir" && $QNN/bin/target/qnn-context-binary-generator \\',
        "        --backend $QNN/lib/target/libQnn${backend}.so \\",
        "        --model $QNN/lib/target/libQnnModelDlc.so \\",
        '        --dlc_path "$list" \\',
        '        --binary_file ${out_name%.bin} --output_dir "$OUT" \\',
        "        > /tmp/_bld_${out_name%.bin}.log 2>&1)",
        "    local rc=$?",
        '    if [ "$rc" -eq 0 ] && [ -f "$OUT/$out_name" ]; then',
        '        echo "  $out_name: OK ($(stat -c%s $OUT/$out_name) B, ${#dlcs[@]} graphs)"',
        "    else",
        '        echo "  $out_name: FAIL rc=$rc"',
        '        tail -3 /tmp/_bld_${out_name%.bin}.log | sed "s/^/    /"',
        "    fi",
        "}",
        "",
    ]
    body_lines = []
    last_frag = None
    for line in build_lines:
        # Insert a header before each new bundle group
        m = re.search(r"ctx_[^ ]+_(?P<frag>[a-z_]+)_chunk", line)
        frag = m.group("frag") if m else None
        if frag != last_frag:
            body_lines.append("")
            body_lines.append(f'echo "=== {frag} bundles ==="')
            last_frag = frag
        body_lines.append(line)
    footer = [
        "",
        'echo ""',
        'echo "=== Summary ==="',
        'ls $OUT/ctx_'+short+'_*.bin 2>/dev/null | wc -l | xargs -I{} echo "  bundles: {}"',
        'du -sh "$OUT"',
    ]
    build_script.write_text("\n".join(header + body_lines + footer) + "\n")
    os.chmod(build_script, 0o755)

    print()
    print(f"  wrote {build_script}")
    print(f"  graph_index JSONs in {out_dir}/")
    print()
    print("Next steps:")
    print(f"  scp {build_script} <board>:/tmp/")
    print(f"  ssh <board> bash /tmp/{build_script.name}")
    print(f"  # then pass the graph_index JSONs to generate_runtime.py:")
    for (frag, be, _), _ in sorted(groups.items()):
        print(f"    --graph-index {out_dir}/{short}_{frag}_{be}_graph_index.json \\")


if __name__ == "__main__":
    main()
