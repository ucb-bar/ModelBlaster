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
#   MB_IR                        profile this graph.json instead of re-extracting
#                                (a fuse/split rewrite); see the note at step 1/5
#   MB_CORES                     core SET for a multi-core run, e.g. "0,1,2,3"
#                                or "0-3". Builds the harness with POOL=1, pins
#                                the process to that mask, sizes the pool from
#                                it, and files the profile under topo_0_1_2_3
#                                instead of topo_0. Unset = the single-core
#                                path, byte-identical to before.
#   MB_SHARD_FACTOR              OC shards per conv, at CODEGEN time (the
#                                weights are re-packed per shard, so it cannot
#                                be a runtime knob). Only meaningful with
#                                MB_CORES; 1 = off. Convs whose OC is not
#                                divisible by it stay whole.
#
# No credentials are read or written; ssh does whatever it is already
# configured to do.

set -euo pipefail

MODEL="${1:-mlp_control}"
QUANT="${2:-int8}"
TARGET="${3:-scalar}"
CPU="${4:-0}"

# MB_CORES turns this into a multi-core run. Everything it changes is derived
# here, once, so the pool width, the affinity mask and the topo tag in the
# profile tree cannot disagree with each other -- a run tagged topo_0_1_2_3
# whose pool was actually 1 thread is a serial measurement filed as a parallel
# one, and nothing downstream could tell.
POOL_ARGS=()
PROFILE_CORES_ARGS=()
RUN_CPU="${CPU}"
if [[ -n "${MB_CORES:-}" ]]; then
    RUN_CPU="${MB_CORES}"
    POOL_ARGS+=(POOL=1)
    # "0-3" and "0,1,2,3" both become the tag "0_1_2_3": the profile tree keys
    # on the hart LIST, and two spellings of one core set must not produce two
    # trees.
    TOPO_CORES="$("${PY:-python3}" - "${MB_CORES}" <<'PYEOF'
import sys
out = []
for part in sys.argv[1].replace(" ", "").split(","):
    if not part:
        continue
    if "-" in part:
        lo, hi = part.split("-", 1)
        out.extend(range(int(lo), int(hi) + 1))
    else:
        out.append(int(part))
seen = []
for c in out:
    if c not in seen:
        seen.append(c)
print("_".join(str(c) for c in sorted(seen)))
PYEOF
)"
    PROFILE_CORES_ARGS+=(--profile-cores "${TOPO_CORES}")
    BIN_SUFFIX="_pool${TOPO_CORES}"
    echo "=== multi-core run: cores ${MB_CORES} -> topo_${TOPO_CORES}, MB_SHARD_FACTOR=${MB_SHARD_FACTOR:-1} ==="
fi

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
# The pool build and the serial build get DIFFERENT binaries and different
# object dirs. Sharing one would relink a stale main.o -- built without
# MODELBLASTER_USE_POOL, so `pool` is NULL and every parallel_ wrapper takes
# its serial arm -- into a run that reports itself as multi-core. Same class of
# defect as the stale model.h sizes the Makefile documents, and just as silent.
BIN="${OUT_ROOT}/${MODEL}_${QUANT}_${TARGET}_harness${BIN_SUFFIX:-}"
mkdir -p "${GEN}"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# MB_IR: profile a PRE-STAGED IR instead of re-extracting the model.
#
# This exists because step 1/5 below is unconditional, and that made a
# granularity rung impossible to run correctly. The obvious recipe for
# profiling an apply_split_hint / apply_fusion_hint rewrite is
#
#     cp round/graph.split.json build/k1/<model>/int8/graph.json
#     bash scripts/run_model_k1.sh <model> int8 rvv_x60 0
#
# and it silently profiles the BASELINE: extract_graph overwrites graph.json
# with a freshly extracted one before anything reads it, the build succeeds, the
# board verifies, and a results.csv describing the unrewritten graph is filed
# under the rewrite's name. That is precisely the gen/vmfb/mlp/.../RVV_fused
# failure -- a negative result recorded as a success -- and it would have been
# reintroduced by a runbook step, not by a mistake at the keyboard.
#
# With MB_IR set, the extraction is skipped and that file becomes the graph.
# weights.npz and io.npz are NOT regenerated and must already be in ${GEN}: a
# fuse/split rewrite changes only the dispatch graph, so the baseline's weights
# and goldens are exactly the ones the rewrite must reproduce -- reusing them is
# what makes max_abs_err a meaningful correctness statement about the rewrite
# rather than about a fresh calibration. If they are missing, run once without
# MB_IR first.
if [[ -n "${MB_IR:-}" ]]; then
    [[ -f "${MB_IR}" ]] || { echo "MB_IR=${MB_IR}: no such file" >&2; exit 2; }
    for _req in weights.npz io.npz; do
        [[ -f "${GEN}/${_req}" ]] || {
            echo "MB_IR set but ${GEN}/${_req} is missing. Run this model once" >&2
            echo "without MB_IR so the weights and goldens exist, then retry." >&2
            exit 2; }
    done
    echo "=== 1/5 SKIPPED -- staging pre-extracted IR ${MB_IR} ==="
    # Only copy when it is not already the same file, so that pointing MB_IR at
    # ${GEN}/graph.json itself is a harmless no-op rather than a truncation.
    if [[ "$(readlink -f "${MB_IR}")" != "$(readlink -f "${GEN}/graph.json")" ]]; then
        cp "${MB_IR}" "${GEN}/graph.json"
    fi
    "${PY}" - "${GEN}/graph.json" <<'PYEOF'
import json, sys
ops = json.load(open(sys.argv[1])).get("ops", [])
n = sum(1 for o in ops if o.get("dispatch_id") is not None)
tiles = sum(1 for o in ops if "split_from" in o)
fused = sum(1 for o in ops if o.get("sub_ops"))
print("  %d dispatches (%d split tiles, %d fused ops)" % (n, tiles, fused))
PYEOF
else

echo "=== 1/5 extract ${MODEL} (${QUANT}) ==="
# NUM_CALIBRATION: how many samples the int8 activation scales are calibrated
# over. It has to be reachable from here, not just from extract_graph's CLI: a
# model whose calibration set is its deployment distribution (fused_full loads
# real gate-course frames from MB_FUSED_CALIB_PKL) silently falls back to ONE
# synthetic get_sample_input() at the default of 1, and the scales -- hence the
# quantization, hence every number measured downstream -- are then not the ones
# the model would ship with. The io.npz golden anchor also becomes the first
# calibration sample, so the board verifies against a real frame.
"${PY}" -m modelblaster.pipeline.extract_graph \
    --model "${MODEL}" --quant "${QUANT}" --out-dir "${GEN}" \
    --num-calibration "${NUM_CALIBRATION:-1}"
fi

echo "=== 2/5 generate skeleton (platform=linux) ==="
# platform=linux matters: the Zephyr flavour emits rdcycle, which raises SIGILL
# from userspace on this board. A zephyr-platform binary does not run slowly
# here, it dies on the first timed dispatch.
"${PY}" -m modelblaster.pipeline.generate_skeleton \
    --ir "${GEN}/graph.json" --weights "${GEN}/weights.npz" --io "${GEN}/io.npz" \
    --out-dir "${GEN}/generated" --backend "${TARGET}" --platform linux

echo "=== 3/5 generate kernels (${TARGET}) ==="
# KEEP_REFERENCE_OPS pins the named op kinds to their portable reference
# kernel instead of swapping in the curated one. It is how XPU-RT's
# `choose_implementation` advice is applied, and it is also how the `unfuse`
# precondition is reconstructed -- a fused op forced onto the reference is
# exactly the measured fallback `unfuse_advice` fires on.
#
# Step 3/5 regenerates unconditionally, so there is no way to pre-stage kernels
# into the build dir and have them survive; the flag has to be threaded here.
KEEPREF_ARGS=()
[[ -n "${KEEP_REFERENCE_OPS:-}" ]] && \
    KEEPREF_ARGS+=(--keep-reference-ops "${KEEP_REFERENCE_OPS}")

"${PY}" -m modelblaster.pipeline.generate_kernels \
    --ir "${GEN}/graph.json" --out-dir "${GEN}/generated" \
    --target "${TARGET}" --backend "${BACKEND:-reference}" --quant "${QUANT}" \
    --global-curated-dir "${REPO_ROOT}/kernels" "${KEEPREF_ARGS[@]}"

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
# MODELBLASTER_KERNEL_CC: compile kernels.c with a different compiler than the
# rest of the harness. Needed only for a model with an fp16 island on this box,
# where the curated Zvfh kernels use RVV fp16 intrinsics that GCC 13.2 -- the
# only riscv64-unknown-linux-gnu compiler installed -- does not have. See the
# KERNEL_CC comment in harness_linux/Makefile. Unset by default.
make -s -C "${REPO_ROOT}/harness_linux" \
    MODEL_DIR="${GEN}/generated" CROSS="${CROSS}" "${POOL_ARGS[@]}" \
    ${MODELBLASTER_KERNEL_CC:+KERNEL_CC="${MODELBLASTER_KERNEL_CC}"} \
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

echo "=== 5/5 deploy + run on ${HOST} (pinned to cpu ${RUN_CPU}) ==="
PROFILE_ARGS=()
# ITERS: repetitions on the board, median per dispatch, iteration 0 dropped as
# warmup. Leave unset for the historical one-sample behaviour. For a stateful
# model the in-binary verify compares the LAST iteration against a golden that
# describes iteration 0, so a run with ITERS>1 is a timing run, not a
# correctness run -- do both, separately.
if [[ -n "${ITERS:-}" ]]; then
    PROFILE_ARGS+=(--iters "${ITERS}")
fi
if [[ -n "${PROFILE_OUT_ROOT:-}" ]]; then
    PROFILE_ARGS+=(--profile-out-root "${PROFILE_OUT_ROOT}"
                   --profile-backend "${TARGET}")
fi
"${PY}" -m modelblaster.validation.k1_runner \
    --elf "${BIN}" --host "${HOST}" --cpu "${RUN_CPU}" \
    "${PROFILE_CORES_ARGS[@]}" \
    --io "${GEN}/io.npz" --quant "${QUANT}" \
    --profile-csv "${GEN}/profile_k1.csv" --model-name "${MODEL}" \
    --gen-dir "${GEN}/generated" \
    --repo-root "${REPO_ROOT}" "${PROFILE_ARGS[@]}"
