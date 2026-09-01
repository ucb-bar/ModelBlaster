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

Weights (`<mid>_<weight>`) are NOT renamed — they're backend-agnostic
const data and link once per model.
"""

from __future__ import annotations

from typing import Iterable


def _c_ident(name: str) -> str:
    return name.replace(".", "_").replace("-", "_")


def rename_defs(model_name: str, used_ops: Iterable[str], backend: str
                ) -> list[str]:
    """Return a list of `-Dold=new` flags for one (model, backend) compile.
    Pass these as compile_definitions on the model.c + kernels.c sources
    (NOT weights.c)."""
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
    ap.add_argument("--separator", default=";",
                    help="separator between flags (default ';' for CMake lists)")
    args = ap.parse_args()

    with open(args.ir) as f:
        ir = json.load(f)
    used = {op["op"] for op in ir.get("ops", []) if op.get("op") != "view"}
    flags = rename_defs(ir["name"], used, args.backend)
    print(args.separator.join(flags))


if __name__ == "__main__":
    _main()
