#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
	echo "[markov_xgboost.sh] This script is config-only. " \
		"Edit top CONFIG and run without args." >&2
	exit 1
fi

# --------------------------
# CONFIG (edit here)
# --------------------------
set -a
SPECIES="Athal, Dmel, Mmus, Hsap"
DONOR_LEN="30"
ACCEPTOR_LEN="50"
TRAIN_POS_PATH=""
TRAIN_NEG_PATH=""
TEST_TSV_PATH=""
CLASS_FILE_PATH=""
MASK_MODE="off"
MASK_TEST_TSV_PATH=""

SEQUENCE_TRANSFORM="none"
MARKOV_ORDER="2"
MARKOV_ALPHA="0.5"
MARKOV_FEATURE_MODE="per_base"
MARKOV_CACHE_MODE="auto"
MARKOV_CACHE_DIR=""
VAL_FRAC="0.1"

XGB_N_ESTIMATORS="300"
XGB_MAX_DEPTH="4"
XGB_LEARNING_RATE="0.05"
XGB_SUBSAMPLE="0.9"
XGB_COLSAMPLE_BYTREE="1.0"
XGB_MIN_CHILD_WEIGHT="1.0"
XGB_REG_LAMBDA="1.0"
XGB_REG_ALPHA="0.0"
XGB_TREE_METHOD="hist"
XGB_N_JOBS="-1"

TRANSCRIPT_SCORE_AGG="min"
SOFTMIN_TAU="1.0"
SEED="1337"
NAME_FIELDS=""
TAG=""
VISUALIZE="true"

SKIP_TRAINING="0"
CONTINUE_TRAINING="0"
TRAIN_ONLY="0"
PRECOMPUTED_SITE_SCORE_TSV=""
CHECKPOINT_TOP_K="3"
CHECKPOINT_PRUNE_DRY_RUN="0"

DEVICE="cpu"
COMPILE_MODE="off"
MPS_MAX_BATCH_SIZE="2048"
set +a

# --------------------------
# Runtime implementation
# --------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"
intronmodel_activate_conda "intronmodel"
intronmodel_init_paths "${BASH_SOURCE[0]}"

# Auto-run inside tmux on SSH so jobs survive disconnects.
# Set INTRONMODEL_AUTO_TMUX=off|on|auto (default: auto).
intronmodel_enable_auto_tmux "${PROJECT_ROOT}" "$0" "${BASH_SOURCE[0]##*/}"

intronmodel_start_timer "markov_xgboost.sh"
trap 'intronmodel_print_timing' EXIT

PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
	python3 "${PROJECT_ROOT}/src/tools/run_wrapper_pipeline.py" \
		--script-name "markov_xgboost.sh"
