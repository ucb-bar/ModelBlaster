#!/bin/bash
# Usage: run_tests.sh [KB_TREE]   KB_TREE = a KernelBlaster tree with the patch applied (default: the mb_hist_v1 worktree)
KB=${1:-/Users/krisd/Documents/ModelBlaster-clone/mb_hist_v1/third_party/KernelBlaster}
PY=${PY:-/tmp/mbvenv_es/bin/python}
cd "$(dirname "$0")"
PYTHONPATH=$KB/src:$KB:. KB_FIXTURES=$KB/tests/fixtures $PY -m pytest -q -p no:cacheprovider test_hist_free.py "${@:2}"
