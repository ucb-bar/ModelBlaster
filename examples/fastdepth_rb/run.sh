#!/usr/bin/env bash
# FastDepth, TRAINED configuration (RoSE experiments/fastdepth_rb): MobileNetV2-1.0
# encoder, 512-ch NNConv5 decoder with skips, valid-region NYU crop, int8 with
# per-channel DEPTHWISE weights (dense convs stay per-tensor so Gemmini's
# conv2d_s8 applies), clamp-aware ranges, 8 calibration frames from the NYU
# TRAIN split through mb_datasets/nyu_depth.
#
# Required:
#   MODELBLASTER_FASTDEPTH_CKPT     trained checkpoint (fastdepth_rb ckpt/<size>/best.pt)
#   MODELBLASTER_FASTDEPTH_NYU_DIR  <nyu_h5>/train (FastDepth HDF5 repack)
# Optional:
#   MODELBLASTER_FASTDEPTH_INPUT_HW default 96x128 (RiskyBird); 224x224 for the reference
#   EXAMPLE_NAME                    separate IR/build tree per size (default fastdepth_rb)
# The extract stage needs h5py (NYU HDF5 calibration frames) in the python on PATH;
# the zephyr conda env lacks it, so extract once from an env that has it
# (python -m modelblaster.pipeline.extract_graph with these variables) and rerun:
# the stage is skipped when the IR is present.
# Golden checks are exact by default (MB_DRIFT_ATOL=0); pass MB_DRIFT_ATOL /
# MAX_ACCURACY_CLASS to allow the curated drift-declared kernels.
set -euo pipefail
: "${MODELBLASTER_FASTDEPTH_CKPT:?set MODELBLASTER_FASTDEPTH_CKPT to a fastdepth_rb checkpoint}"
: "${MODELBLASTER_FASTDEPTH_NYU_DIR:?set MODELBLASTER_FASTDEPTH_NYU_DIR to <nyu_h5>/train}"
export MODELBLASTER_FASTDEPTH_INPUT_HW="${MODELBLASTER_FASTDEPTH_INPUT_HW:-96x128}"
export MODELBLASTER_FASTDEPTH_WIDTH_MULT=1.0 MODELBLASTER_FASTDEPTH_DECODER_CH=512
export MODELBLASTER_FASTDEPTH_SKIPS=1 MODELBLASTER_FASTDEPTH_PRETRAINED=1
export MODELBLASTER_FASTDEPTH_NYU_CROP=valid
export MB_INT8_CLAMP_AWARE_RANGES=1 MB_INT8_PC_DEPTHWISE=1
export NUM_CALIBRATION="${NUM_CALIBRATION:-8}"
export QUANT="${QUANT:-int8}"
export MB_DRIFT_ATOL="${MB_DRIFT_ATOL:-0}"
export EXAMPLE_NAME="${EXAMPLE_NAME:-fastdepth_rb}"
MODEL_NAME=fastdepth
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MODEL_NAME REPO_ROOT
source "${REPO_ROOT}/examples/_run_lib.sh"
