"""Namespace shim.

The repo's Python sources live as top-level directories under the
repo root (``pipeline/``, ``optimize/``, ``validation/``, ``models/``,
``datasets/``, ``benchmarks/``, ...) but are imported as
``modelblaster.<subpackage>``. This ``__init__.py`` lives under
``src/modelblaster/`` so hatchling's editable install puts ``src/``
on ``sys.path`` (not the repo root). That keeps our top-level
``datasets/`` directory from shadowing the HuggingFace ``datasets``
pip package when SmolVLA / lerobot import it directly.

``__path__`` is then extended back to the repo root so
``modelblaster.<subpackage>`` resolves to the existing top-level
directories without moving the 459 source files behind them.

Editable install only -- a built wheel would need the sources actually
relocated under ``modelblaster/``; we don't ship wheels today, so the
shim is sufficient.
"""

from os.path import abspath, dirname

# src/modelblaster/__init__.py -> src/modelblaster -> src -> <repo-root>
_REPO_ROOT = dirname(dirname(dirname(abspath(__file__))))
__path__ = [_REPO_ROOT]
