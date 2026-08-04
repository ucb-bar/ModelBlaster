"""SUPERSEDED — use bundles_from_schedule.py instead.

This script's hardcoded sched23 layer list has been generalized into
bundles_from_schedule.py which derives the same chunks from the actual
scheduled JSON. Kept as a one-line shim so existing references resolve.
"""
import sys, subprocess
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parent.parent

print("[deprecated] build_sched23_bundles.py is superseded by "
      "bundles_from_schedule.py. Forwarding...")
subprocess.run([
    sys.executable, str(HERE / "bundles_from_schedule.py"),
    str(REPO / "schedules" / "scheduled_networks_smolvla_vision_v3_bundles_qrb5165_greedy_profiled.json"),
    "--seg-perf", str(REPO / "qnn_models/boards/qrb5165_v66/profiles/smolvlm_vision_v3/segment_perf.json"),
    "--slice-root", str(HERE / "vision_slices_v3"),
], check=True)
