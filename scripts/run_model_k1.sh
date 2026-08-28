#!/usr/bin/env bash
# One command: generate, build, deploy, execute, verify and profile a
# ModelBlaster model on a physical SpaceMiT K1.
#
#   scripts/run_model_k1.sh mlp_control int8 scalar 0
#                           <model>     <quant> <target> <cpu>
#
# Why this is not a `k1)` case inside examples/_run_lib.sh: that library is
# built around `west build -b <board>` and Zephyr Kconfig overlays. The K1 runs
# Linux, so there is no board target, no prj.conf and no west -- the build is
# five C files and a link. Bending the Zephyr flow around that would make both
# paths harder to read than keeping them separate. The kernel ABI, the dispatch
# identities and the stdout marker protocol stay common, which is what actually
# needs to be shared.
#
# Env:
#   MODELBLASTER_K1_HOST         board ssh host      (default: k1)
#   MODELBLASTER_K1_REMOTE_ROOT  board staging dir   (default: /root/mb_k1)
#   CROSS                        cross toolchain prefix
#   PROFILE_OUT_ROOT             emit IREE-shaped results.csv under here
#   OUT_ROOT                     local build/staging dir
#
# No credentials are read or written; ssh does whatever it is already
# configured to do.

set -euo pipefail

MODEL="${1:-mlp_control}"
QUANT="${2:-int8}"
TARGET="${3:-scalar}"
CPU="${4:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/build/k1}"
CROSS="${CROSS:-riscv64-unknown-linux-gnu-}"
HOST="${MODELBLASTER_K1_HOST:-k1}"
PY="${PY:-python3}"
# Kernel synthesis on the K1 path goes to Codex, never Bedrock, and there is no
# fallback: if Codex is unavailable the kernel step must fail loudly rather than
# quietly produce kernels from another provider. Only consulted when
# BACKEND=llm; the default (reference) uses curated kernels and calls no model.
export LLM_PROVIDER="${LLM_PROVIDER:-codex}"

GEN="${OUT_ROOT}/${MODEL}/${QUANT}"
BIN="${OUT_ROOT}/${MODEL}_${QUANT}_${TARGET}_harness"
mkdir -p "${GEN}"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "=== 1/5 extract ${MODEL} (${QUANT}) ==="
"${PY}" -m modelblaster.pipeline.extract_graph \
    --model "${MODEL}" --quant "${QUANT}" --out-dir "${GEN}"

echo "=== 2/5 generate skeleton (platform=linux) ==="
# platform=linux matters: the Zephyr flavour emits rdcycle, which raises SIGILL
# from userspace on this board. A zephyr-platform binary does not run slowly
# here, it dies on the first timed dispatch.
"${PY}" -m modelblaster.pipeline.generate_skeleton \
    --ir "${GEN}/graph.json" --weights "${GEN}/weights.npz" --io "${GEN}/io.npz" \
    --out-dir "${GEN}/generated" --backend "${TARGET}" --platform linux

echo "=== 3/5 generate kernels (${TARGET}) ==="
"${PY}" -m modelblaster.pipeline.generate_kernels \
    --ir "${GEN}/graph.json" --out-dir "${GEN}/generated" \
    --target "${TARGET}" --backend "${BACKEND:-reference}" --quant "${QUANT}" \
    --global-curated-dir "${REPO_ROOT}/kernels"

echo "=== 4/5 build linux harness ==="
# repo_root must be the ABSOLUTE repo path, not ".". The flags can contain
# include paths (-I<repo_root>/kernels/rvv for the RVV intrinsics compat
# header, -isystem<repo_root>/cores/... for gemmini), and `make -C harness_linux`
# below runs from a different directory -- so a relative path resolves against
# the wrong place and the header is not found.
KERNEL_CFLAGS="$("${PY}" - "$TARGET" "${REPO_ROOT}" <<'PYEOF'
import sys
# src first: `modelblaster` is also installed editable from a sibling checkout
# in the venv commonly used here, and would otherwise shadow this one silently.
sys.path.insert(0, sys.argv[2] + "/src")
sys.path.insert(0, sys.argv[2])
from modelblaster.pipeline import backends
b = backends.get(sys.argv[1])
print(" ".join(b.resolved_kernel_cflags(sys.argv[2])))
PYEOF
)" || KERNEL_CFLAGS=""
make -s -C "${REPO_ROOT}/harness_linux" \
    MODEL_DIR="${GEN}/generated" CROSS="${CROSS}" \
    KERNEL_CFLAGS="${KERNEL_CFLAGS}" OUT="${BIN}"

# Gate: refuse to deploy a binary carrying an instruction the board will
# refuse. GCC 13.2 does not reliably carry vtype across a kernel's width
# changes, and the result is not a build error -- it is a SIGILL on the first
# dispatch that reaches the bad instruction. Three curated kernels shipped one.
# Each was found by decoding badaddr out of dmesg, one model at a time; this
# finds them at build time instead.
if command -v "${CROSS}objdump" >/dev/null 2>&1; then
    if ! "${PY}" "${REPO_ROOT}/scripts/check_rvv_vtype.py" \
            --objdump "${CROSS}objdump" "${BIN}"; then
        echo "refusing to deploy ${BIN}" >&2
        exit 1
    fi
fi

echo "=== 5/5 deploy + run on ${HOST} (pinned to cpu ${CPU}) ==="
PROFILE_ARGS=()
if [[ -n "${PROFILE_OUT_ROOT:-}" ]]; then
    PROFILE_ARGS+=(--profile-out-root "${PROFILE_OUT_ROOT}"
                   --profile-backend "${TARGET}")
fi
"${PY}" -m modelblaster.validation.k1_runner \
    --elf "${BIN}" --host "${HOST}" --cpu "${CPU}" \
    --io "${GEN}/io.npz" --quant "${QUANT}" \
    --profile-csv "${GEN}/profile_k1.csv" --model-name "${MODEL}" \
    --repo-root "${REPO_ROOT}" "${PROFILE_ARGS[@]}"
