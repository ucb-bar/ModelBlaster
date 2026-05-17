"""Emit per-dispatch profile data in IREE-compatible CSV format.

XPU-RT consumes results.csv files produced by IREE's benchmark flow —
see e.g.
    .../gen/profile/RVV/spacemit_x60/dronet/dronet.q.int8/
        dronet_spacemit_x60_RVV_dronet.q.int8/topo_0_1_2_3/results.csv

This module produces a CSV of the same shape from any per-op timing
source so XPU-RT can ingest data from our modelblaster flow on spike today,
and from FireSim or RTL sim runs later, without changing format. The
provenance is encoded in:

  1. The path component `<cpu>` — `spike` for ISA simulator,
     `firesim` / `rtlsim` / `<chip>` for higher-fidelity backends.
  2. A `source` extension column on every row, repeating that tag.

IREE's columns are reproduced verbatim. We tack on a few extras after
the IREE block — consumers that follow the IREE schema strictly ignore
trailing columns; consumers that want our additions can read them.

Directory layout (matches IREE's):
    <out_root>/<backend>/<cpu>/<model>/<model>.<quant>/<spec>/topo_<cores>/results.csv

where `<spec> = <model>_<cpu>_<backend>_<model>.<quant>` (mirror of IREE
naming) and `<cores> = "0_1_2_3"` listing the harts the run used.

Pluggable harnesses
-------------------
A non-spike harness (FireSim, RTL sim, real silicon) plugs in by:
  * Producing an iterable of dicts, each with at minimum the four keys
    {name, op, shape, cycles}. cycles is the per-dispatch wall time in
    cycles of whatever clock the harness reports.
  * Calling write_profile(records, ProfileMeta(..., source="firesim",
    cpu="firesim", clock_mhz=<chip-clock>, ...)).

The clock_mhz field is what converts cycles → ns. For spike we use a
placeholder (1 GHz default) since spike cycles are retired-instruction
counts on a flat memory model; for FireSim/RTL we use the actual
target clock.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Iterable, Sequence


# IREE's results.csv schema, in exact column order.
_IREE_COLUMNS = [
    "dispatch_id",
    "module_name",
    "vmfb_path",
    "mlir_path",
    "mean_time",
    "mean_unit",
    "mean_time_ns",
    "returncode",
    "log_path",
]

# Our extensions — consumers that follow the IREE schema strictly ignore
# trailing columns; ours read them when present.
_EXTRA_COLUMNS = [
    "source",   # provenance tag: spike / firesim / rtlsim / <chip>
    "op",       # raw IR op kind, e.g. linear, conv2d_s8
    "shape",    # the IR-side shape descriptor, e.g. M=1;K=256;N=128
    "cycles",   # raw cycle count from the harness (pre-ns conversion)
]

CSV_COLUMNS = _IREE_COLUMNS + _EXTRA_COLUMNS


@dataclass
class ProfileMeta:
    """Per-run metadata used to label rows and build the output path."""

    model: str            # e.g. "mlp_control" — top-level network name
    quant: str            # "fp32" or "int8"
    backend: str          # HW backend: "scalar", "rvv", ...
    cores: Sequence[int]  # hart layout this run used, e.g. (0, 1, 2, 3)
    source: str           # "spike", "firesim", "rtlsim", ...
    cpu: str              # CPU label (path component); often == source
    clock_mhz: float      # cycles → ns conversion factor
    artifacts_dir: str = ""  # optional: where the per-dispatch C lives


def _shape_concise(shape: str) -> str:
    """Convert "M=1;K=256;N=128" → "M1xK256xN128" — compact, hyphen-free,
    safe to embed in module_name and filenames."""
    parts = []
    for kv in (shape or "").split(";"):
        kv = kv.strip()
        if not kv:
            continue
        if "=" in kv:
            k, v = kv.split("=", 1)
            parts.append(f"{k}{v}")
        else:
            parts.append(kv)
    return "x".join(parts) if parts else "scalar"


def _module_name(model: str, dispatch_id: int, backend: str,
                 op: str, shape: str) -> str:
    """Build an IREE-style module_name. IREE uses
    `<model>$async_dispatch_<id>_embedded_elf_riscv_64_<descriptor>`.
    We mirror with a `<source-target>` segment for our backends."""
    return (
        f"{model}$dispatch_{dispatch_id}_{backend}_"
        f"{op}_{_shape_concise(shape)}"
    )


def build_records(op_records: Iterable[dict], meta: ProfileMeta) -> list[dict]:
    """Convert our model_op_record_t rows into per-dispatch profile rows.

    op_records is the list parse_profile() returns (one dict per op,
    with keys name/op/shape/cycles)."""
    out: list[dict] = []
    if meta.clock_mhz <= 0.0:
        raise ValueError(f"clock_mhz must be positive, got {meta.clock_mhz}")
    for i, r in enumerate(op_records):
        cycles = int(r["cycles"])
        # Prefer the IR-assigned dispatch_id (extract_graph._annotate_dispatches);
        # fall back to enumeration order for harnesses that don't propagate it.
        dispatch_id = int(r["dispatch_id"]) if "dispatch_id" in r else i
        # cycles / (cycles per ns) = ns. cycles_per_ns = clock_mhz / 1000.
        ns = cycles * 1000.0 / meta.clock_mhz
        ms = ns / 1.0e6
        out.append({
            "dispatch_id": dispatch_id,
            "module_name": _module_name(
                meta.model, dispatch_id, meta.backend, r["op"], r["shape"]
            ),
            "vmfb_path": "",
            "mlir_path": "",
            "mean_time": f"{ms:.6f}",
            "mean_unit": "ms",
            "mean_time_ns": f"{ns:.6f}",
            "returncode": 0,
            "log_path": "",
            "source": meta.source,
            "op": r["op"],
            "shape": r["shape"],
            "cycles": cycles,
        })
    return out


def output_path(out_root: str, meta: ProfileMeta) -> str:
    """IREE-style nested output path. Caller is responsible for creating
    parent directories (write_profile() does that)."""
    spec = f"{meta.model}_{meta.cpu}_{meta.backend}_{meta.model}.{meta.quant}"
    topo = "topo_" + "_".join(str(c) for c in meta.cores)
    return os.path.join(
        out_root,
        meta.backend,
        meta.cpu,
        meta.model,
        f"{meta.model}.{meta.quant}",
        spec,
        topo,
        "results.csv",
    )


def write_profile(op_records: Iterable[dict], meta: ProfileMeta,
                  out_root: str) -> str:
    """Build per-dispatch records, write CSV, return the path written."""
    records = build_records(op_records, meta)
    path = output_path(out_root, meta)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return path
