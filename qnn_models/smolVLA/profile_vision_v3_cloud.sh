#!/usr/bin/env bash
#
# Cloud QRB5165 reprofile wrapper around profile_vision_v3_correct.sh.
#
# This used to be a ~360-line copy of profile_vision_v3_correct.sh with cloud
# paths hand-edited. Now it's a thin env-var preset.
#
# To retarget to a different cloud session, change QNN_BOARD_HOST (e.g.
# qrb_cloud is the SSH-config alias for the QDC tunnel).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export QNN_BOARD_HOST="${QNN_BOARD_HOST:-qrb_cloud}"
export PROFILE_DIR="${PROFILE_DIR:-$REPO_ROOT/qnn_models/boards/qrb5165_v66_cloud/profiles/smolvlm_vision_v3}"
export REMOTE_BASE="${REMOTE_BASE:-/root/profile_v3}"
# Cloud QRB5165 needs /unsigned in ADSP_LIBRARY_PATH for DSP firmware load.
export ADSP_EXTRA_PATHS="${ADSP_EXTRA_PATHS:-/root/qairt/lib/hexagon-v66/unsigned}"

exec "$SCRIPT_DIR/profile_vision_v3_correct.sh" "$@"
