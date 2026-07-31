#!/usr/bin/env python3
"""Emit a per-variant rename header for the shared-input harness.

For a given (MODEL_DIR, TAG) pair, walks the generated model.h /
model.c / kernels.h and enumerates every mangled runtime symbol —
functions and global arrays that would collide if two copies of the
same model's TUs were linked into one binary. Emits a ``#define
<sym> <sym>_<tag>`` line for each, wrapped in ``#pragma once``. The
harness_shared_input CMakeLists ``-include``s this header into each
variant's OBJECT library, which is what actually gives every variant
its own symbol namespace at compile time.

We intentionally DO NOT rename symbols that live in weights.c or
test_io.S — those are shared across all variants (same weights, same
input) and the linker merges the identical .rodata contents. Renaming
them would multiply the ELF size by N.

Symbol classes we rename:
  * ``run_model_<mid>`` (and its unmangled alias when
    MODELBLASTER_DISABLE_UNMANGLED is not defined — but the harness
    always defines it).
  * ``MODEL_<MID>_DISPATCH_FNS`` — extern const array in model.h,
    defined in model.c.
  * ``dispatch_<mid>_<N>`` — static-linkage dispatch fns in model.c;
    though static, we rename for parity with future non-static
    variants.
  * ``kernel_<op>_<mid>`` — extern declared in kernels.h, defined in
    kernels.c. This is where the LLM's kernel body lives; renaming
    lets N different kernels.c coexist.
  * ``model_<mid>_profile_records`` / ``_wall_cycles`` /
    ``_set_wall_cycles`` / ``_reset_profile`` — runtime introspection
    helpers.

Any symbol we miss will surface as a link error like "multiple
definition of ``foo``" — the fix is to add it to _MANGLED_PATTERNS.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


# Enumerated by scanning generate_skeleton.py's emit-site templates. If
# generate_skeleton grows a new mangled symbol, add its regex here.
_MANGLED_DECL_RE = re.compile(
    # `void run_model_<mid>(` or `const model_<mid>_dispatch_fn` etc.
    r"\b(?P<sym>"
    r"run_model_[A-Za-z0-9_]+|"
    r"MODEL_[A-Z0-9_]+_DISPATCH_FNS|"
    r"dispatch_[A-Za-z0-9_]+_\d+|"
    r"kernel_[A-Za-z_][A-Za-z0-9_]*_(?:kb_)?[A-Za-z0-9_]+|"
    r"model_[A-Za-z0-9_]+?_(?:profile_records|wall_cycles|set_wall_cycles|reset_profile)"
    r")\b"
)

# Symbols in these files stay shared (rodata contents merged by linker).
# We open them only to EXCLUDE their exported symbols from the rename
# set — a naive walk would catch e.g. ``weights_kb_19_ReLU`` and rename
# it, breaking the shared-storage model.
_SHARED_FILES = ("weights.c", "weights.h", "test_io.S", "test_io.h")


_UNMANGLED_BLOCK_RE = re.compile(
    r"#ifndef\s+MODELBLASTER_DISABLE_UNMANGLED\b"
    r".*?"
    r"#endif",
    re.DOTALL,
)


def _strip_unmangled_aliases(text: str) -> str:
    """Drop the ``#ifndef MODELBLASTER_DISABLE_UNMANGLED ... #endif`` block.

    That block defines convenience aliases like ``model_set_wall_cycles
    -> model_kb_19_ReLU_set_wall_cycles`` for single-model use. We
    reject those from the rename set — they don't correspond to actual
    linker-visible symbols, and even if they did, the harness always
    defines MODELBLASTER_DISABLE_UNMANGLED so they never get emitted.
    """
    return _UNMANGLED_BLOCK_RE.sub("", text)


def _collect_symbols(model_dir: Path) -> set[str]:
    syms: set[str] = set()
    for name in ("model.h", "model.c", "kernels.h"):
        p = model_dir / name
        if not p.exists():
            continue
        text = _strip_unmangled_aliases(p.read_text(errors="replace"))
        for m in _MANGLED_DECL_RE.finditer(text):
            syms.add(m.group("sym"))
    return syms


def _collect_shared_symbols(model_dir: Path) -> set[str]:
    """Symbols we must NOT rename (shared rodata). We enumerate top-
    level identifiers in weights.h / test_io.h so anything named there
    is protected."""
    shared: set[str] = set()
    top_id = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]{2,})\b")
    for name in _SHARED_FILES:
        p = model_dir / name
        if not p.exists():
            continue
        text = p.read_text(errors="replace")
        for m in top_id.finditer(text):
            tok = m.group(1)
            # Skip obvious C keywords / types — cheap allow-listy filter.
            if tok in {"const", "extern", "static", "void", "int",
                       "float", "double", "char", "unsigned", "long",
                       "short", "signed", "size_t", "uint8_t",
                       "uint16_t", "uint32_t", "uint64_t", "int8_t",
                       "int16_t", "int32_t", "int64_t", "stdint",
                       "stddef", "pragma", "once", "include", "define",
                       "ifndef", "endif", "typedef", "struct", "union",
                       "enum", "return", "sizeof", "if", "else",
                       "while", "for", "do", "switch", "case", "break",
                       "continue", "goto", "default", "true", "false",
                       "NULL"}:
                continue
            shared.add(tok)
    return shared


def emit_header(model_dir: Path, tag: str, out: Path) -> None:
    syms = _collect_symbols(model_dir)
    shared = _collect_shared_symbols(model_dir)
    to_rename = sorted(syms - shared)

    lines: list[str] = []
    lines.append(f"/* Auto-generated by gen_variant_renames.py. DO NOT EDIT. */")
    lines.append(f"/* Variant tag: {tag}  Model dir: {model_dir} */")
    lines.append("#pragma once")
    lines.append("")
    lines.append(f"/* {len(to_rename)} mangled runtime symbols renamed to <sym>_{tag}. */")
    for sym in to_rename:
        lines.append(f"#define {sym} {sym}_{tag}")
    lines.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True, type=Path,
                    help="Path to modelblaster generated/<target>/ dir.")
    ap.add_argument("--tag", required=True,
                    help="Variant tag suffix (e.g. 'v0', 'cand3').")
    ap.add_argument("--out", required=True, type=Path,
                    help="Rename header path to emit.")
    args = ap.parse_args()
    emit_header(args.model_dir, args.tag, args.out)


if __name__ == "__main__":
    main()
