#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
	echo "[dnabert_pair.sh] This script is config-only. Edit top CONFIG and run without args." >&2
	exit 1
fi

# --------------------------
# CONFIG (edit here)
# --------------------------
# Frequently edited knobs are intentionally placed first in this block.
# Advanced runtime controls are kept below.
set -a
DNABERT_VARIANT="2"
SPECIES="Athal, Dmel, Hsap, Mmus"
DONOR_LEN="100"
ACCEPTOR_LEN="100"
TRUNC_MODE="on"
INTRONMODEL_AUTO_TMUX="on"

PRETRAINED_MODEL_NAME=""
PRETRAINED_MODEL_RELATIVE_PATH_2="pretrained/dnabert2-117m-7bce263b15377fc15361f52cfab88f8b586abda0"
PRETRAINED_MODEL_RELATIVE_PATH_6="pretrained/dnabert6"
PRETRAINED_MODEL_RELATIVE_PATH_S="pretrained/dnabert-s"
PRETRAINED_REVISION=""
TRUST_REMOTE_CODE="1"

TRANSCRIPT_SCORE_AGG="min"
SOFTMIN_TAU="1.0"
SEED="1337"
NAME_FIELDS=""
PROCESS_TITLE="ETA"
# Optional output/data overrides for trunc-data runs.
TAG=""
TRAIN_POS_PATH=""
TRAIN_NEG_PATH=""
MASK_TEST_TSV_PATH=""
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
INFER_BATCH_SIZE="256"
LR="2e-5"
LOSS="weighted_bce"
MAX_TOKENS="auto"
DROPOUT="0.1"
HEAD_LAYER_NORM="1"
READOUT_TYPE="cnn"
READOUT_CNN_KERNEL_SIZE="3"
READOUT_MLP_HIDDEN_DIM="256"
READOUT_MLP_LAYERS="1"
WEIGHT_DECAY="0.01"
ETA_MIN_RATIO="0.01"
LR_SCHEDULE="cosine"
WARMUP_RATIO="0.01"
ADAM_BETA1="0.9"
ADAM_BETA2="0.98"
ADAM_EPS="1e-8"
VAL_FRAC="0.1"
GRAD_CLIP="1.0"
POS_WEIGHT_CAP="20.0"
FOCAL_GAMMA="2.0"
FOCAL_ALPHA_POS=""
ASYM_GAMMA_POS="0.0"
ASYM_GAMMA_NEG="4.0"
ASYM_ALPHA_POS=""
USE_TUNED_HPARAMS="auto"
TUNED_HPARAMS_MODE="normal"
PAIR_TUNED_CONFIG_PATH=""
SHARED_TUNED_CONFIG_PATH=""

DEVICE="auto"
GPU_IDS="auto"
MAX_PARALLEL_TRIALS="auto"
USE_AMP="1"
AMP_DTYPE="auto"
COMPILE_MODE="off"
INFER_USE_AMP="1"
INFER_AMP_DTYPE="auto"
INFER_COMPILE="0"
INFER_COMPILE_MODE="auto"
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
	local relative_path_s="$4"

	local normalized_variant="${variant,,}"
	if [[ "${normalized_variant}" != "2" \
		&& "${normalized_variant}" != "6" \
		&& "${normalized_variant}" != "s" ]]; then
		echo "[dnabert_pair.sh] DNABERT_VARIANT must be 2, 6, or s." >&2
		return 1
	fi

	local resolved_path
	if [[ "${normalized_variant}" == "2" ]]; then
		resolved_path="${relative_path_2}"
	elif [[ "${normalized_variant}" == "6" ]]; then
		resolved_path="${relative_path_6}"
	else
		resolved_path="${relative_path_s}"
	fi
	if [[ -z "${resolved_path}" ]]; then
		echo "[dnabert_pair.sh] pretrained relative path is empty for variant=${variant}." >&2
		return 1
	fi
	printf '%s\n' "${resolved_path}"
}

intronmodel_start_timer "dnabert_pair.sh"
trap 'intronmodel_print_timing' EXIT

if [[ -z "${PRETRAINED_MODEL_NAME}" ]]; then
	PRETRAINED_MODEL_RELATIVE_PATH="$(
		resolve_dnabert_relative_path \
			"${DNABERT_VARIANT}" \
			"${PRETRAINED_MODEL_RELATIVE_PATH_2}" \
			"${PRETRAINED_MODEL_RELATIVE_PATH_6}" \
			"${PRETRAINED_MODEL_RELATIVE_PATH_S}"
	)"
	export PRETRAINED_MODEL_RELATIVE_PATH
fi
if [[ "${TRUNC_MODE}" != "off" && "${TRUNC_MODE}" != "on" ]]; then
	echo "[dnabert_pair.sh] TRUNC_MODE must be off|on." >&2
	exit 1
fi
if [[ "${HEAD_LAYER_NORM}" != "0" && "${HEAD_LAYER_NORM}" != "1" ]]; then
	echo "[dnabert_pair.sh] HEAD_LAYER_NORM must be 0 or 1." >&2
	exit 1
fi
if [[ "${READOUT_TYPE}" != "cnn" \
	&& "${READOUT_TYPE}" != "linear" \
	&& "${READOUT_TYPE}" != "mlp" ]]; then
	echo "[dnabert_pair.sh] READOUT_TYPE must be cnn|linear|mlp." >&2
	exit 1
fi
if ! [[ "${READOUT_CNN_KERNEL_SIZE}" =~ ^[0-9]+$ ]] \
	|| [[ "${READOUT_CNN_KERNEL_SIZE}" -le 0 ]] \
	|| (( READOUT_CNN_KERNEL_SIZE % 2 == 0 )); then
	echo "[dnabert_pair.sh] READOUT_CNN_KERNEL_SIZE must be a positive odd integer." >&2
	exit 1
fi
if ! [[ "${READOUT_MLP_HIDDEN_DIM}" =~ ^[0-9]+$ ]] \
	|| [[ "${READOUT_MLP_HIDDEN_DIM}" -le 0 ]]; then
	echo "[dnabert_pair.sh] READOUT_MLP_HIDDEN_DIM must be a positive integer." >&2
	exit 1
fi
if ! [[ "${READOUT_MLP_LAYERS}" =~ ^[0-9]+$ ]] \
	|| [[ "${READOUT_MLP_LAYERS}" -le 0 ]]; then
	echo "[dnabert_pair.sh] READOUT_MLP_LAYERS must be a positive integer." >&2
	exit 1
fi
if [[ "${LR_SCHEDULE}" != "cosine" && "${LR_SCHEDULE}" != "linear" ]]; then
	echo "[dnabert_pair.sh] LR_SCHEDULE must be cosine|linear." >&2
	exit 1
fi
if ! [[ "${WARMUP_RATIO}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
	echo "[dnabert_pair.sh] WARMUP_RATIO must be numeric in [0,1)." >&2
	exit 1
fi
if ! awk -v x="${WARMUP_RATIO}" 'BEGIN{exit !(x>=0 && x<1)}'; then
	echo "[dnabert_pair.sh] WARMUP_RATIO must be in [0,1)." >&2
	exit 1
fi
if ! [[ "${ADAM_BETA1}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
	echo "[dnabert_pair.sh] ADAM_BETA1 must be numeric in (0,1)." >&2
	exit 1
fi
if ! awk -v x="${ADAM_BETA1}" 'BEGIN{exit !(x>0 && x<1)}'; then
	echo "[dnabert_pair.sh] ADAM_BETA1 must be in (0,1)." >&2
	exit 1
fi
if ! [[ "${ADAM_BETA2}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
	echo "[dnabert_pair.sh] ADAM_BETA2 must be numeric in (0,1)." >&2
	exit 1
fi
if ! awk -v x="${ADAM_BETA2}" 'BEGIN{exit !(x>0 && x<1)}'; then
	echo "[dnabert_pair.sh] ADAM_BETA2 must be in (0,1)." >&2
	exit 1
fi
if ! awk -v b1="${ADAM_BETA1}" -v b2="${ADAM_BETA2}" \
	'BEGIN{exit !(b1<b2)}'; then
	echo "[dnabert_pair.sh] ADAM_BETA1 must be smaller than ADAM_BETA2." >&2
	exit 1
fi
if ! [[ "${ADAM_EPS}" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
	echo "[dnabert_pair.sh] ADAM_EPS must be a positive number." >&2
	exit 1
fi
if ! awk -v x="${ADAM_EPS}" 'BEGIN{exit !(x>0)}'; then
	echo "[dnabert_pair.sh] ADAM_EPS must be > 0." >&2
	exit 1
fi
MASK_MODE="${TRUNC_MODE}"
export MASK_MODE

(
	export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
	intronmodel_run_with_process_title \
		"${PROCESS_TITLE}" \
		python3 "${PROJECT_ROOT}/src/tools/run_wrapper_pipeline.py" \
			--script-name "dnabert_pair.sh"
)
