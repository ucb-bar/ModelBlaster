#!/usr/bin/env bash
# Assert that the current shell can run a benchmark cell end-to-end.
# Exits 0 with "OK: benchmark env ready" or 1 with the first failing
# precondition + a one-line fix suggestion.
#
# Safe to run from any cwd; no side effects.

set -u

_fail() {
    echo "FAIL: $1" >&2
    echo "fix:  $2" >&2
    exit 1
}

# 1) Build tooling on PATH.
command -v west >/dev/null 2>&1 \
    || _fail "west not on PATH" \
             "source scripts/setup_benchmark_env.sh"
command -v spike >/dev/null 2>&1 \
    || _fail "spike not on PATH" \
             "source scripts/setup_benchmark_env.sh"

# 2) Zephyr workspace + SDK reachable.
[ -n "${ZEPHYR_BASE:-}" ] && [ -d "${ZEPHYR_BASE}" ] \
    || _fail "ZEPHYR_BASE not set or directory missing (${ZEPHYR_BASE:-<unset>})" \
             "source scripts/setup_benchmark_env.sh"
[ -n "${ZEPHYR_SDK_INSTALL_DIR:-}" ] && [ -d "${ZEPHYR_SDK_INSTALL_DIR}" ] \
    || _fail "ZEPHYR_SDK_INSTALL_DIR not set or missing (${ZEPHYR_SDK_INSTALL_DIR:-<unset>})" \
             "source scripts/setup_benchmark_env.sh"

# 3) RISC-V Zephyr toolchain binary exists (sanity-check the SDK install).
_GCC="${ZEPHYR_SDK_INSTALL_DIR}/gnu/riscv64-zephyr-elf/bin/riscv64-zephyr-elf-gcc"
[ -x "${_GCC}" ] \
    || _fail "RISC-V Zephyr GCC not found at ${_GCC}" \
             "verify ZEPHYR_SDK_INSTALL_DIR points at a complete SDK tree"

# 4) Bedrock creds for Arm B (warning only -- Arm A doesn't need them).
if [ -z "${AWS_BEARER_TOKEN_BEDROCK:-}" ]; then
    echo "warning: AWS_BEARER_TOKEN_BEDROCK not set -- Arm B-bedrock runs will fail." >&2
    echo "         (Arm A curated runs are unaffected.)" >&2
    echo "         Add the token to .env at the repo root."  >&2
fi

echo "OK: benchmark env ready"
echo "    west   = $(command -v west)"
echo "    spike  = $(command -v spike)"
echo "    ZEPHYR_BASE             = ${ZEPHYR_BASE}"
echo "    ZEPHYR_SDK_INSTALL_DIR  = ${ZEPHYR_SDK_INSTALL_DIR}"
if [ -n "${AWS_BEARER_TOKEN_BEDROCK:-}" ]; then
    echo "    AWS_BEARER_TOKEN_BEDROCK = (set, ${#AWS_BEARER_TOKEN_BEDROCK} chars)"
fi
# FireSim is opt-in (only needed for the accelerator-cycle baseline);
# report status but don't fail when it's not set up.
if command -v firesim >/dev/null 2>&1; then
    if ssh-add -l 2>/dev/null | grep -q firesim; then
        echo "    firesim = $(command -v firesim)  (ssh-agent has firesim key)"
    else
        echo "    firesim = $(command -v firesim)  (warning: firesim ssh key not in agent;"
        echo "              run \`ssh-add ~/.ssh/firesim\` before infrasetup/kill/runworkload)"
    fi
else
    echo "    firesim = not on PATH (spike-only session; ok for non-accelerator captures)"
fi
exit 0
