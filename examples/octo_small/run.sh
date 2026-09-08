#!/usr/bin/env bash
# End-to-end runner for the Octo octo-small-1.5 policy (27.0M params; the
# 109.6M T5 text encoder is deliberately not part of the port -- the
# language embedding is a (1,16,768) input, which is upstream's own
# LanguageTokenizer contract for a precomputed instruction).
#
# EXTRACTOR=export is mandatory, not a preference: the ViT's head-splitting
# and learned position embeddings reach torch.fx as shape call_methods and
# get_attrs, 132 of which extract_graph rejects.
#
# The coverage knobs below trade three faithful-but-unsupported constructs
# for supported equivalents, each validated against the JAX reference in
# experiments/octo_port/NOTES.md:
#   GN=layernorm  group_norm -> layer_norm            (1.79e-06)
#   TIME=lut      fourier time embedding -> lookup    (bit-exact)
#   NORM=0        drops the /127.5-1 image normalise  (fold into calibration)
#   ATTN=matmul   hand-decomposed attention. REQUIRED: the walker's SDPA
#                 decomposition never reads args[3] and so drops the
#                 attention mask, and the reference sdpa kernel documents
#                 itself as maskless. Octo's attention is masked.
set -euo pipefail
export MB_DRIFT_ATOL="${MB_DRIFT_ATOL:-2}"
export EXTRACTOR="${EXTRACTOR:-export}"
export MODELBLASTER_OCTO_GN="${MODELBLASTER_OCTO_GN:-layernorm}"
export MODELBLASTER_OCTO_TIME="${MODELBLASTER_OCTO_TIME:-lut}"
export MODELBLASTER_OCTO_NORM="${MODELBLASTER_OCTO_NORM:-0}"
export MODELBLASTER_OCTO_ATTN="${MODELBLASTER_OCTO_ATTN:-matmul}"
# Per-output-channel weight scales. 83 linears and 10 convs deep, a single
# scale per weight tensor is not enough: the fp32 reference and the int8
# graph disagree on the whole output range without this.
EXTRACT_EXTRA_ARGS="${EXTRACT_EXTRA_ARGS:---per-channel}"
export EXTRACT_EXTRA_ARGS
MODEL_NAME=octo_small
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MODEL_NAME REPO_ROOT

# The converted checkpoint. get_model() falls back to RANDOM-INIT weights with
# only a warning if it cannot find one, so locate it here rather than relying
# on that default: the port lives in <repo>/experiments/octo_port/, which is
# the RoSE tree when modelblaster is nested as a submodule and the modelblaster
# root in a standalone checkout. Walk up until it turns up.
if [[ -z "${MODELBLASTER_OCTO_CKPT:-}" ]]; then
    _d="${REPO_ROOT}"
    for _ in 1 2 3 4 5 6 7 8; do
        if [[ -f "${_d}/experiments/octo_port/octo_small_torch.pt" ]]; then
            export MODELBLASTER_OCTO_CKPT="${_d}/experiments/octo_port/octo_small_torch.pt"
            break
        fi
        _d="$(dirname "${_d}")"
        [[ "${_d}" == "/" ]] && break
    done
fi
if [[ -n "${MODELBLASTER_OCTO_CKPT:-}" ]]; then
    echo "  octo checkpoint: ${MODELBLASTER_OCTO_CKPT}"
else
    echo "  WARNING: no octo_small_torch.pt found; weights will be RANDOM" >&2
fi
source "${REPO_ROOT}/examples/_run_lib.sh"
