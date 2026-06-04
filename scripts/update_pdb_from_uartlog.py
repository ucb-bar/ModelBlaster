"""Re-ingest measured bit-exact cycles into the profile DB."""
import csv, re, sys, shutil
from pathlib import Path

UART = Path("/scratch2/agustin/ModelBlaster/artifacts/audit/firesim_dronet_gemmini_bit_exact.uartlog")
# Profile DB roots
PDB_ROOT = Path("/scratch2/agustin/XPU-RT/zephyr-chipyard-sw/gen/profile/sweep_v8")

# Parse uartlog: each line "<backend>,<did>,<name>,<op>,<shape>,<cycles>"
measured = {}  # (backend, dispatch_id) -> cycles
for line in UART.read_text().splitlines():
    if line.startswith(("gemmini_q31,", "rvv_opu,")):
        parts = line.split(",")
        if len(parts) >= 6:
            backend, did, name, op, shape, cyc = parts[0], int(parts[1]), parts[2], parts[3], parts[4], int(parts[-1])
            measured[(backend, did)] = (op, shape, cyc, name)

print(f"Loaded {len(measured)} measured (backend, did) entries")

def update_pdb(pdb_csv: Path, backend_filter: str) -> int:
    """Update mean_time / mean_time_ns / cycles for matching dispatch_ids."""
    if not pdb_csv.is_file():
        return 0
    rows = list(csv.DictReader(open(pdb_csv)))
    fieldnames = rows[0].keys()
    backup = pdb_csv.with_suffix(".csv.prebitexact_backup")
    if not backup.exists():
        shutil.copy(pdb_csv, backup)
    n_updated = 0
    for r in rows:
        did = int(r["dispatch_id"])
        key = (backend_filter, did)
        if key not in measured:
            continue
        op, shape, cyc, name = measured[key]
        old = float(r.get("mean_time_ns", "0"))
        r["mean_time"] = f"{cyc/1e6:.6f}"
        r["mean_unit"] = "ms"
        r["mean_time_ns"] = f"{cyc:.6f}"
        r["cycles"] = str(cyc)
        r["source"] = "firesim"
        n_updated += 1
        print(f"  did={did} {r['op']}  {old/1e6:.3f}ms -> {cyc/1e6:.3f}ms")
    with open(pdb_csv, "w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"updated {n_updated} rows in {pdb_csv} (backup: {backup})")
    return n_updated

# Update gemmini_q31 + V256D128_rvv dronet PDBs
g = PDB_ROOT / "gemmini_q31/firesim_rocket_saturn/dronet/dronet.int8/dronet_firesim_rocket_saturn_gemmini_q31_dronet.int8/topo_0/results.csv"
r = PDB_ROOT / "V256D128_rvv/firesim_rocket_saturn/dronet/dronet.int8/dronet_firesim_rocket_saturn_V256D128_rvv_dronet.int8/topo_0/results.csv"
print("--- gemmini_q31 PDB ---")
update_pdb(g, "gemmini_q31")
print("--- V256D128_rvv PDB ---")
update_pdb(r, "rvv_opu")
