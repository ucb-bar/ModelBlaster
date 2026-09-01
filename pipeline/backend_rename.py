"""Compute -D symbol-rename flags for heterogeneous-binary builds.

When `harness_xpurt` links kernels for multiple HW backends (e.g.
scalar + rvv) into one binary, it needs distinct symbols for each
backend. Rather than threading a `--backend-suffix` knob through the
codegen, we keep the source unchanged and rename the externally-visible
symbols at compile time via `-Dold=new` flags.

The renames cover everything that:
  * has external linkage in `model.c` or `kernels.c`, AND
  * is keyed only on the model name (not the backend) in the source.

Specifically:
  * `kernel_<op>_<mid>`            -> `kernel_<op>_<mid>_<bs>`
  * `run_model_<mid>`              -> `run_model_<mid>_<bs>`
  * `model_<mid>_reset_profile`    -> ...`_<bs>`
  * `model_<mid>_wall_cycles`      -> ...`_<bs>`
  * `model_<mid>_set_wall_cycles`  -> ...`_<bs>`
  * `model_<mid>_profile_records`  -> ...`_<bs>`
  * `MODEL_<UMID>_DISPATCH_FNS`    -> `MODEL_<UMID>_DISPATCH_FNS_<BS>`

What we DON'T rename (each backend's TU has its own copy, file-static
or struct-shape):
  * `dispatch_<mid>_<id>` (file-static in model.c)
  * `records_`, `n_`, `wall_cycles_` (file-static)
  * `parallel_<op>` (static inline in model.c)
  * `model_<mid>_state_t`, `model_<mid>_op_record_t`, `model_<mid>_dispatch_fn`
    (struct/typedef definitions; identical across backends)
  * `model_<mid>_input_t`, `model_<mid>_output_t` (typedefs)
  * `MODEL_<UMID>_*` macros (#defines; same value across backends)

Weights (`<mid>_<weight>`) ARE renamed. They are NOT backend-agnostic:
each backend packs the same tensor in its own layout (gemmini keeps
OIHW, rvv permutes to IHWOC), so for dronet 290489/312098 weight
elements differ between the gemmini_q31 and rvv copies. Emitting them
under one symbol and linking only the primary backend's `weights.c`
silently fed one backend the other's layout. Each backend now compiles
its own `weights.c` with `-D<sym>=<sym>_<bs>`, so the layouts coexist.

The symbol list is read from the backend's generated `weights.h` rather
than re-derived from the IR, so it always matches exactly what
`generate_skeleton.py` emitted (tile weights from IR-level sharding
included).
"""

from __future__ import annotations

import re
from typing import Iterable


def _c_ident(name: str) -> str:
    return name.replace(".", "_").replace("-", "_")


_EXTERN_RE = re.compile(
    r"^\s*extern\s+(?:const\s+)?[A-Za-z_][\w\s]*?\b(\w+)\s*(?:\[[^;]*\])?\s*;",
    re.M)


def weight_symbols_from_header(path: str) -> list[str]:
    """Externally-visible symbols declared in a generated `weights.h`.

    Parsing the header (rather than re-deriving names from the IR) keeps
    this exactly in step with `generate_skeleton.py`'s `_weight_name`,
    including split-tile weights whose names have no IR-level analogue.
    """
    with open(path) as f:
        return _EXTERN_RE.findall(f.read())


def weight_rename_defs(symbols: Iterable[str], backend: str) -> list[str]:
    """`-Dsym=sym_<bs>` for every weight symbol, so each backend's
    `weights.c` defines storage under its own name and its `model.c`
    reads back the layout that backend was packed for."""
    return [f"-D{sym}={sym}_{backend}" for sym in symbols]


def rename_defs(model_name: str, used_ops: Iterable[str], backend: str,
                weight_symbols: Iterable[str] = ()) -> list[str]:
    """Return a list of `-Dold=new` flags for one (model, backend) compile.
    Pass these as compile_definitions on model.c + kernels.c + weights.c —
    all three must agree on the renamed weight symbols."""
    mid = _c_ident(model_name)
    umid = mid.upper()
    bs = backend
    BS = backend.upper()

    defs: list[str] = []
    # Per-op kernel definitions in kernels.c + call sites in model.c.
    for op in used_ops:
        if op == "view":
            continue
        defs.append(f"-Dkernel_{op}_{mid}=kernel_{op}_{mid}_{bs}")
    # Standard model symbols.
    for fn in (
        f"run_model_{mid}",
        f"model_{mid}_reset_profile",
        f"model_{mid}_wall_cycles",
        f"model_{mid}_set_wall_cycles",
        f"model_{mid}_profile_records",
    ):
        defs.append(f"-D{fn}={fn}_{bs}")
    # Dispatch table — keep MODEL_<UMID>_DISPATCH_FNS as the prefix and
    # append _<BS> so the walker can pick by core_kind without parsing.
    defs.append(
        f"-DMODEL_{umid}_DISPATCH_FNS=MODEL_{umid}_DISPATCH_FNS_{BS}"
    )
    # Weight storage — see module docstring.
    defs.extend(weight_rename_defs(weight_symbols, bs))
    return defs


def renamed_dispatch_fn_table(model_name: str, backend: str) -> str:
    """The renamed external symbol for one (model, backend)'s dispatch
    table. The walker uses this to invoke per-entry."""
    umid = _c_ident(model_name).upper()
    return f"MODEL_{umid}_DISPATCH_FNS_{backend.upper()}"


# CLI for shell-script consumption: emit a semicolon-joined list of -D
# flags so CMake's COMPILE_OPTIONS / target_compile_definitions can
# splice them straight in.
def _main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--ir", required=True, help="path to graph.json")
    ap.add_argument("--backend", required=True, help="backend tag, e.g. rvv")
    ap.add_argument("--weights-header", default=None,
                    help="path to the backend's generated weights.h; its "
                         "symbols get -D renamed per backend")
    ap.add_argument("--separator", default=";",
                    help="separator between flags (default ';' for CMake lists)")
    args = ap.parse_args()

    with open(args.ir) as f:
        ir = json.load(f)
    used = {op["op"] for op in ir.get("ops", []) if op.get("op") != "view"}
    wsyms = (weight_symbols_from_header(args.weights_header)
             if args.weights_header else [])
    flags = rename_defs(ir["name"], used, args.backend, wsyms)
    print(args.separator.join(flags))


if __name__ == "__main__":
    _main()
