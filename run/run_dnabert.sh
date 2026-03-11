#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
	echo "[dnabert.sh] This script is config-only. Edit top CONFIG and run without args." >&2
	exit 1
fi

# --------------------------
# CONFIG (edit here)
# --------------------------
# Frequently edited knobs are intentionally placed first in this block.
# Advanced per-task overrides are kept below.
set -a
DNABERT_VARIANT="6"
SPECIES="Mmus, Dmel, Athal, Hsap"
DONOR_LEN="100"
ACCEPTOR_LEN="100"

PRETRAINED_MODEL_NAME=""
PRETRAINED_MODEL_RELATIVE_PATH_2="pretrained/dnabert2-117m-7bce263b15377fc15361f52cfab88f8b586abda0"
PRETRAINED_MODEL_RELATIVE_PATH_6="pretrained/dnabert6"
PRETRAINED_REVISION=""
TRUST_REMOTE_CODE="1"

INTRON_SCORE_OP="*"
TRANSCRIPT_SCORE_AGG="min"
SOFTMIN_TAU="1.0"
SEED="1337"
NAME_FIELDS=""
VISUALIZE="true"
SKIP_TRAINING="0"
CONTINUE_TRAINING="0"
TRAIN_ONLY="0"
PRECOMPUTED_SITE_SCORE_TSV=""
CHECKPOINT_TOP_K="3"
CHECKPOINT_PRUNE_DRY_RUN="0"

EPOCHS="6"
MAX_EPOCHS="100"
EARLY_STOP_PATIENCE="12"
EARLY_STOP_MIN_DELTA="0.0"
BATCH_SIZE="64"
LR="2e-5"
LOSS="weighted_bce"
MAX_TOKENS="auto"
DROPOUT="0.1"
WEIGHT_DECAY="0.01"
ETA_MIN_RATIO="0.01"
VAL_FRAC="0.1"
GRAD_CLIP="1.0"
POS_WEIGHT_CAP="20.0"
FOCAL_GAMMA="2.0"
FOCAL_ALPHA_POS=""
ASYM_GAMMA_POS="0.0"
ASYM_GAMMA_NEG="4.0"
ASYM_ALPHA_POS=""
TRAIN_TARGET="both"
USE_TUNED_HPARAMS="auto"
DONOR_TUNED_CONFIG_PATH=""
ACCEPTOR_TUNED_CONFIG_PATH=""
SHARED_TUNED_CONFIG_PATH=""

DONOR_BATCH_SIZE=""
ACCEPTOR_BATCH_SIZE=""
DONOR_LR=""
ACCEPTOR_LR=""
DONOR_LOSS=""
ACCEPTOR_LOSS=""
DONOR_MAX_TOKENS=""
ACCEPTOR_MAX_TOKENS=""
DONOR_DROPOUT=""
ACCEPTOR_DROPOUT=""
DONOR_WEIGHT_DECAY=""
ACCEPTOR_WEIGHT_DECAY=""
DONOR_ETA_MIN_RATIO=""
ACCEPTOR_ETA_MIN_RATIO=""
DONOR_VAL_FRAC=""
ACCEPTOR_VAL_FRAC=""
DONOR_GRAD_CLIP=""
ACCEPTOR_GRAD_CLIP=""
DONOR_POS_WEIGHT_CAP=""
ACCEPTOR_POS_WEIGHT_CAP=""
DONOR_FOCAL_GAMMA=""
ACCEPTOR_FOCAL_GAMMA=""
DONOR_FOCAL_ALPHA_POS=""
ACCEPTOR_FOCAL_ALPHA_POS=""
DONOR_ASYM_GAMMA_POS=""
ACCEPTOR_ASYM_GAMMA_POS=""
DONOR_ASYM_GAMMA_NEG=""
ACCEPTOR_ASYM_GAMMA_NEG=""
DONOR_ASYM_ALPHA_POS=""
ACCEPTOR_ASYM_ALPHA_POS=""

DEVICE="auto"
USE_AMP="1"
AMP_DTYPE="auto"
COMPILE_MODE="auto"
ALLOW_TF32="1"
CUDNN_BENCHMARK="1"
DETERMINISTIC="0"
NUM_WORKERS="auto"
PREFETCH_FACTOR="4"
PERSISTENT_WORKERS="1"
PIN_MEMORY="1"
MIN_BATCH_SIZE="64"
MAX_OOM_RETRIES="8"
MPS_MAX_BATCH_SIZE="1024"



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

resolve_dnabert_relative_path() {
	local variant="$1"
	local relative_path_2="$2"
	local relative_path_6="$3"

	if [[ "${variant}" != "2" && "${variant}" != "6" ]]; then
		echo "[dnabert.sh] DNABERT_VARIANT must be 2 or 6." >&2
		return 1
	fi

	local resolved_path
	if [[ "${variant}" == "2" ]]; then
		resolved_path="${relative_path_2}"
	else
		resolved_path="${relative_path_6}"
	fi
	if [[ -z "${resolved_path}" ]]; then
		echo "[dnabert.sh] pretrained relative path is empty for variant=${variant}." >&2
		return 1
	fi
	printf '%s\n' "${resolved_path}"
}

intronmodel_start_timer "dnabert.sh"
trap 'intronmodel_print_timing' EXIT

if [[ -z "${PRETRAINED_MODEL_NAME}" ]]; then
	PRETRAINED_MODEL_RELATIVE_PATH="$(
		resolve_dnabert_relative_path \
			"${DNABERT_VARIANT}" \
			"${PRETRAINED_MODEL_RELATIVE_PATH_2}" \
			"${PRETRAINED_MODEL_RELATIVE_PATH_6}"
	)"
	export PRETRAINED_MODEL_RELATIVE_PATH
fi

PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
	python3 "${PROJECT_ROOT}/src/tools/run_wrapper_pipeline.py" \
		--script-name "dnabert.sh"
