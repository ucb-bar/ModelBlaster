"""Explicit external inputs and the owning ModelBlaster checkout."""
from pathlib import Path
import importlib.util
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
# ModelBlaster uses a namespace package; isolated worktrees need not be named
# modelblaster. Pin this process to the checkout owning these scripts.
spec = importlib.util.spec_from_loader("modelblaster", loader=None, is_package=True)
package = importlib.util.module_from_spec(spec)
package.__path__ = [str(ROOT)]
sys.modules["modelblaster"] = package


def required_path(variable, *, directory=False, default=None):
    value = os.environ.get(variable, default)
    if not value:
        raise SystemExit(f"set {variable} to the required {'directory' if directory else 'file'}")
    path = Path(value).expanduser().resolve()
    exists = path.is_dir() if directory else path.is_file()
    if not exists:
        raise SystemExit(f"{variable}: missing {'directory' if directory else 'file'}: {path}")
    return str(path)
