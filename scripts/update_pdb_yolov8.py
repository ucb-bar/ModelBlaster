import csv, sys, shutil
from pathlib import Path
UART = Path("/scratch2/agustin/ModelBlaster/artifacts/audit/firesim_yolov8_gemmini.uartlog")
PDB_ROOT = Path("/scratch2/agustin/XPU-RT/zephyr-chipyard-sw/gen/profile/sweep_v8")
measured = {}
for line in UART.read_text().splitlines():
    if line.startswith(("gemmini_q31,", "rvv_opu,")):
        parts = line.split(",")
        if len(parts) >= 6:
            backend, did, name, op, shape, cyc = parts[0], int(parts[1]), parts[2], parts[3], parts[4], int(parts[-1])
            measured[(backend, did)] = (op, shape, cyc, name)
print(f"loaded {len(measured)} measurements")
def update_pdb(pdb_csv: Path, backend_filter: str) -> int:
    if not pdb_csv.is_file():
        print(f"  missing: {pdb_csv}")
        return 0
    rows = list(csv.DictReader(open(pdb_csv)))
    backup = pdb_csv.with_suffix(".csv.prebitexact_backup")
    if not backup.exists():
        shutil.copy(pdb_csv, backup)
    n_updated = 0
    for r in rows:
        did = int(r["dispatch_id"])
        key = (backend_filter, did)
        if key not in measured:
            continue
        _, _, cyc, _ = measured[key]
        r["mean_time"] = f"{cyc/1e6:.6f}"; r["mean_unit"] = "ms"
        r["mean_time_ns"] = f"{cyc:.6f}"; r["cycles"] = str(cyc); r["source"] = "firesim"
        n_updated += 1
    with open(pdb_csv, "w") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"updated {n_updated} rows in {pdb_csv}")
    return n_updated
g = PDB_ROOT / "gemmini_q31/firesim_rocket_saturn/yolov8_nano/yolov8_nano.int8/yolov8_nano_firesim_rocket_saturn_gemmini_q31_yolov8_nano.int8/topo_0/results.csv"
r = PDB_ROOT / "V256D128_rvv/firesim_rocket_saturn/yolov8_nano/yolov8_nano.int8/yolov8_nano_firesim_rocket_saturn_V256D128_rvv_yolov8_nano.int8/topo_0/results.csv"
update_pdb(g, "gemmini_q31")
update_pdb(r, "rvv_opu")
