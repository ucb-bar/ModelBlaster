"""Namespace shim.

The repo's Python sources live as top-level directories (`pipeline/`,
`optimize/`, `validation/`, `models/`, `datasets/`, `benchmarks/`, ...)
but are imported as `modelblaster.<subpackage>`. This `__init__.py`
extends `__path__` to the repo root so the import system finds those
sibling directories under the `modelblaster` namespace.

Editable install only: `__path__` is computed at runtime from this
file's location. A built wheel installed into site-packages would
need the sources actually relocated under `modelblaster/`; we don't
build wheels for distribution today, so the shim is sufficient.
"""

from os.path import abspath, dirname

__path__ = [dirname(dirname(abspath(__file__)))]
