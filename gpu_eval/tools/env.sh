# Source (don't run) to get a shell that can run the ModelBlaster pipeline:
#   source tools/env.sh
# Paths come from the environment, defaulting to this file's place in a
# ModelBlaster checkout (<MB_ROOT>/gpu_eval/tools/env.sh):
#   MB_ROOT          ModelBlaster checkout (its dir must be named modelblaster)
#   ZEPHYR_SW_ROOT   zephyr-chipyard-sw checkout (default: MB_ROOT's parent)
#   MB_PYTHON_ENV    python env to put first on PATH
#                    (default: ZEPHYR_SW_ROOT/tools/miniforge3/envs/zephyr, if present)
#   MB_PATH_EXCLUDE  grep -E pattern of PATH entries to drop (e.g. a broken cmake)
#   MB_EXTRA_PATH    extra dirs to prepend to PATH (e.g. a RISC-V toolchain)
#   MB_TMPDIR        TMPDIR for the run (keep large builds off a small /tmp)
_mb_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
export MB_ROOT=$(cd "${MB_ROOT:-$_mb_here/../..}" && pwd -P)
if [ ! -f "$MB_ROOT/pipeline/extract_graph.py" ]; then
  echo "env.sh: $MB_ROOT is not a ModelBlaster checkout; set MB_ROOT" >&2
  return 1
fi
export ZEPHYR_SW_ROOT=${ZEPHYR_SW_ROOT:-$(dirname "$MB_ROOT")}
[ -z "${MB_PATH_EXCLUDE:-}" ] || export PATH=$(echo "$PATH" | tr : '\n' | grep -Ev "$MB_PATH_EXCLUDE" | paste -sd:)
[ ! -f "$ZEPHYR_SW_ROOT/scripts/set_envvars_sdk.sh" ] || source "$ZEPHYR_SW_ROOT/scripts/set_envvars_sdk.sh" >/dev/null 2>&1
MB_PYTHON_ENV=${MB_PYTHON_ENV:-$ZEPHYR_SW_ROOT/tools/miniforge3/envs/zephyr}
if [ -x "$MB_PYTHON_ENV/bin/python" ]; then
  export PATH="$MB_PYTHON_ENV/bin:$PATH"; export WEST_PYTHON="$MB_PYTHON_ENV/bin/python"
fi
[ -z "${MB_EXTRA_PATH:-}" ] || export PATH="$MB_EXTRA_PATH:$PATH"
# `python -m modelblaster.…` imports the package from MB_ROOT's parent
export PYTHONPATH="$(dirname "$MB_ROOT")${PYTHONPATH:+:$PYTHONPATH}"
export MB_KB_DATA_ROOT=
[ -z "${MB_TMPDIR:-}" ] || { mkdir -p "$MB_TMPDIR"; export TMPDIR=$MB_TMPDIR; }
unset _mb_here
