"""FireSim-side counterpart to runners/spike.py.

Same translation shape — parse the harness stdout via
`modelblaster.validation.runner_common`, write the aggregator's
artifact files — but the profile CSV is named `profile_firesim.csv`.

The reason these are separate runner modules (rather than one with a
runner name parameter) is that the per-runner artifacts differ in
provenance: spike profile cycles are not authoritative on accelerator
targets (extensions execute atomically), and the aggregator uses the
filename suffix to enforce the cycle-source-honesty policy. Keeping
the writers separate keeps that policy explicit at the producer side
too.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

from modelblaster.validation import runner_common


RUNNER_NAME = "firesim"


def parse_stdout(stdout: str, *, tag: Optional[str] = None
                 ) -> dict[str, Any]:
    verify = runner_common.parse_verify(stdout, tag=tag)
    profile = runner_common.parse_profile(stdout, tag=tag) or []
    wall = runner_common.parse_wall_cycles(stdout, tag=tag)
    return {
        "verify": verify,
        "profile": profile,
        "wall_cycles": wall,
    }


def write_accuracy(out_dir: Path, verify: Optional[dict[str, Any]],
                   atol: Optional[float] = None,
                   rtol: Optional[float] = None) -> None:
    if verify is None:
        return
    data: dict[str, Any] = {
        "schema_version": 1,
        "linf": float(verify["max_abs_err"]),
        "rmse": None,
        "cosine": None,
        "n_samples": int(verify["n"]),
        "bit_exact": float(verify["max_abs_err"]) == 0.0,
        "verify_pass": True,
    }
    if atol is not None:
        data["atol_used"] = float(atol)
        data["verify_pass"] = float(verify["max_abs_err"]) <= float(atol)
    if rtol is not None:
        data["rtol_used"] = float(rtol)
    with open(out_dir / "accuracy.json", "w") as f:
        json.dump(data, f, indent=2)


def write_profile_csv(out_dir: Path, profile: list[dict[str, Any]]
                      ) -> None:
    if not profile:
        return
    path = out_dir / f"profile_{RUNNER_NAME}.csv"
    header = list(profile[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for row in profile:
            w.writerow(row)


def write_wall_cycles(out_dir: Path, wall_cycles: Optional[int]) -> None:
    if wall_cycles is None:
        return
    (out_dir / "wall_cycles.txt").write_text(f"{wall_cycles}\n")
