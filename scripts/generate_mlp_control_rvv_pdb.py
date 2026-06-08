"""Generate the missing mlp_control V256D128_rvv PDB CSV for sweep_v8.

The solver's profile_loader.py looks for the sweep_v8 V256D128_rvv PDB row for
mlp_control on the rvv_opu backend. Without it, profile_loader falls back to
``rng.uniform(2.0, 10.0)`` synthetic times per op, inflating predicted load.

This script synthesises that missing CSV from two sources:

  (a) v20b's FireSim run.log MODELBLASTER_PROFILE_BEGIN [mlp_control] block
      (per-dispatch rvv_opu cycles, freshly measured).

  (b) The older non-sweep_v8 RVV PDB at .../V256D128_rvv/.../RVV_mlp_control.../
      results.csv, used as a fallback for any dispatch_id that didn't appear
      under rvv_opu in v20b (e.g. dispatch_0, routed to gemmini_q31 there).

It reads the gemmini_q31 sweep_v8 mlp_control PDB to learn the full ordered
dispatch_id list and ops/shapes, then writes the new V256D128_rvv CSV with the
same schema as existing rvv PDBs (e.g. dronet, yolov8_nano).
"""
from __future__ import annotations

import csv
import re
import shutil
import sys
from pathlib import Path

NET = "mlp_control"
QUANT = "fp32"

PDB_ROOT = Path("/scratch2/agustin/XPU-RT/zephyr-chipyard-sw/gen/profile/sweep_v8")
GEMMINI_PDB = (PDB_ROOT / "gemmini_q31" / "firesim_rocket_saturn" / NET / f"{NET}.{QUANT}"
               / f"{NET}_firesim_rocket_saturn_gemmini_q31_{NET}.{QUANT}" / "topo_0"
               / "results.csv")
RVV_PDB = (PDB_ROOT / "V256D128_rvv" / "firesim_rocket_saturn" / NET / f"{NET}.{QUANT}"
           / f"{NET}_firesim_rocket_saturn_V256D128_rvv_{NET}.{QUANT}" / "topo_0"
           / "results.csv")

# Older alt-path PDB used as fallback for any did not measured under rvv_opu
# in the v20b run (e.g. dispatch_0 which v20b routed to gemmini_q31).
ALT_RVV_PDB = Path(
    "/scratch2/agustin/XPU-RT/gen/profile/V256D128_rvv/firesim_rocket_saturn/"
    f"{NET}/{NET}.{QUANT}/"
    f"{NET}_firesim_rocket_saturn_RVV_{NET}.{QUANT}/topo_0/results.csv"
)

RUNLOG = Path("/scratch2/agustin/ModelBlaster/artifacts/runtime_optimization/"
              "v20b_transpose_elim_retry/run.log")

FIELDNAMES = ["dispatch_id", "module_name", "vmfb_path", "mlir_path",
              "mean_time", "mean_unit", "mean_time_ns", "returncode",
              "log_path", "source", "op", "shape", "cycles"]


def parse_runlog_block(text: str, net: str) -> dict[int, tuple[str, str, int, str, str]]:
    """Return {did: (backend, op, shape, cycles, name)} from the [net] block."""
    begin_re = re.compile(rf"=== MODELBLASTER_PROFILE_BEGIN \[{re.escape(net)}\] ===")
    end_re = re.compile(rf"=== MODELBLASTER_PROFILE_END \[{re.escape(net)}\] ===")
    in_block = False
    out: dict[int, tuple[str, str, int, str, str]] = {}
    for line in text.splitlines():
        if begin_re.search(line):
            in_block = True
            continue
        if in_block and end_re.search(line):
            break
        if not in_block:
            continue
        if not (line.startswith("gemmini_q31,") or line.startswith("rvv_opu,")):
            continue
        parts = line.split(",")
        if len(parts) < 6:
            continue
        backend = parts[0]
        try:
            did = int(parts[1])
            cyc = int(parts[-1])
        except ValueError:
            continue
        name, op, shape = parts[2], parts[3], parts[4]
        out[did] = (backend, op, shape, cyc, name)
    return out


def shape_to_modname(shape: str) -> str:
    """Convert 'M=1;K=16;N=256' style shape to 'M1xK16xN256'."""
    parts = shape.split(";")
    return "x".join(p.replace("=", "") for p in parts if p)


def load_alt_rvv_rows() -> dict[int, dict]:
    """Load alt-path RVV PDB indexed by dispatch_id."""
    if not ALT_RVV_PDB.is_file():
        return {}
    out: dict[int, dict] = {}
    for r in csv.DictReader(open(ALT_RVV_PDB)):
        try:
            did = int(r["dispatch_id"])
        except (KeyError, ValueError):
            continue
        out[did] = r
    return out


def main() -> int:
    if not GEMMINI_PDB.is_file():
        print(f"missing gemmini PDB: {GEMMINI_PDB}", file=sys.stderr)
        return 1
    if not RUNLOG.is_file():
        print(f"missing runlog: {RUNLOG}", file=sys.stderr)
        return 1

    gemmini_rows = list(csv.DictReader(open(GEMMINI_PDB)))
    print(f"[gen-rvv-pdb] gemmini PDB: {len(gemmini_rows)} rows")

    measured = parse_runlog_block(RUNLOG.read_text(), NET)
    rvv_meas = {d: m for d, m in measured.items() if m[0] == "rvv_opu"}
    print(f"[gen-rvv-pdb] runlog: {len(measured)} dispatches "
          f"({len(rvv_meas)} on rvv_opu)")

    alt_rvv = load_alt_rvv_rows()
    print(f"[gen-rvv-pdb] alt RVV PDB: {len(alt_rvv)} rows (fallback only)")

    out_rows: list[dict] = []
    for gr in gemmini_rows:
        did = int(gr["dispatch_id"])
        op = gr["op"]
        shape = gr["shape"]
        # Append _s8 suffix on op to match the dronet/yolov8 rvv PDB convention
        op_s8 = op if op.endswith("_s8") else f"{op}_s8"
        mod = (f"{NET}$dispatch_{did}_V256D128_rvv_curated_"
               f"{op_s8}_{shape_to_modname(shape)}")
        if did in rvv_meas:
            _, m_op, m_shape, cyc, _ = rvv_meas[did]
            # Trust the runlog op/shape (already _s8-suffixed in v20b output)
            op_field = m_op
            shape_field = m_shape
            mod = (f"{NET}$dispatch_{did}_V256D128_rvv_curated_"
                   f"{op_field}_{shape_to_modname(shape_field)}")
            source = "firesim"
        elif did in alt_rvv:
            alt = alt_rvv[did]
            try:
                cyc = int(alt["cycles"])
            except (KeyError, ValueError):
                cyc = 0
            op_field = op_s8
            shape_field = shape
            source = "firesim"
        else:
            # No measurement available; mark as missing
            cyc = 0
            op_field = op_s8
            shape_field = shape
            source = ""
        row = {
            "dispatch_id": str(did),
            "module_name": mod,
            "vmfb_path": "",
            "mlir_path": "",
            "mean_time": f"{cyc/1e6:.6f}" if cyc else "",
            "mean_unit": "ms" if cyc else "",
            "mean_time_ns": f"{cyc:.6f}" if cyc else "",
            "returncode": "0" if cyc else "",
            "log_path": "",
            "source": source,
            "op": op_field,
            "shape": shape_field,
            "cycles": str(cyc) if cyc else "",
        }
        out_rows.append(row)

    RVV_PDB.parent.mkdir(parents=True, exist_ok=True)
    if RVV_PDB.exists():
        backup = RVV_PDB.with_suffix(".csv.pregen_backup")
        if not backup.exists():
            shutil.copy(RVV_PDB, backup)
            print(f"[gen-rvv-pdb] backed up existing CSV -> {backup}")

    with open(RVV_PDB, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(out_rows)
    print(f"[gen-rvv-pdb] wrote {len(out_rows)} rows -> {RVV_PDB}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
