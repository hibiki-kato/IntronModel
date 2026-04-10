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
DNABERT_VARIANT="2"
SPECIES="Dmel"
DONOR_LEN="100"
ACCEPTOR_LEN="100"
TRUNC_MODE="off"
INTRONMODEL_AUTO_TMUX="off"
SKIP_TRAINING="0"
CONTINUE_TRAINING="0"

PRETRAINED_MODEL_NAME=""
PRETRAINED_MODEL_RELATIVE_PATH_2="pretrained/dnabert2-117m-7bce263b15377fc15361f52cfab88f8b586abda0"
PRETRAINED_MODEL_RELATIVE_PATH_6="pretrained/dnabert6"
PRETRAINED_MODEL_RELATIVE_PATH_S="pretrained/dnabert-s"
PRETRAINED_REVISION=""
TRUST_REMOTE_CODE="1"

INTRON_SCORE_OP="*"
TRANSCRIPT_SCORE_AGG="min"
SOFTMIN_TAU="1.0"
SEED="1337"
NAME_FIELDS="tag"
PROCESS_TITLE="use? email me"
# Optional output/data overrides for trunc-data runs.
TAG=""
TRAIN_POS_PATH=""
TRAIN_NEG_PATH=""
MASK_TEST_TSV_PATH=""
VISUALIZE="true"
TRAIN_ONLY="0"
PRECOMPUTED_SITE_SCORE_TSV=""
CHECKPOINT_TOP_K="3"
CHECKPOINT_PRUNE_DRY_RUN="0"

EPOCHS="auto"
MAX_EPOCHS="10"
EARLY_STOP_PATIENCE="2"
EARLY_STOP_MIN_DELTA="0.01"
BATCH_SIZE="64"
INFER_BATCH_SIZE="256"
LR="2e-5"
LOSS="weighted_bce"
MAX_TOKENS="auto"
DROPOUT="0.1"
HEAD_LAYER_NORM="1"
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
TRAIN_TARGET="both"
USE_TUNED_HPARAMS="auto"
TUNED_HPARAMS_MODE="normal"
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
DONOR_HEAD_LAYER_NORM=""
ACCEPTOR_HEAD_LAYER_NORM=""
DONOR_WEIGHT_DECAY=""
ACCEPTOR_WEIGHT_DECAY=""
DONOR_ETA_MIN_RATIO=""
ACCEPTOR_ETA_MIN_RATIO=""
DONOR_LR_SCHEDULE=""
ACCEPTOR_LR_SCHEDULE=""
DONOR_WARMUP_RATIO=""
ACCEPTOR_WARMUP_RATIO=""
DONOR_ADAM_BETA1=""
ACCEPTOR_ADAM_BETA1=""
DONOR_ADAM_BETA2=""
ACCEPTOR_ADAM_BETA2=""
DONOR_ADAM_EPS=""
ACCEPTOR_ADAM_EPS=""
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
GPU_IDS="auto"
MAX_PARALLEL_TRIALS="auto"
USE_AMP="1"
AMP_DTYPE="auto"
COMPILE_MODE="on"
INFER_USE_AMP="1"
INFER_AMP_DTYPE="auto"
INFER_COMPILE="0"
INFER_COMPILE_MODE="off"
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
	intronmodel_resolve_dnabert_relative_path "dnabert.sh" "$@"
}

intronmodel_start_timer "dnabert.sh"
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
	echo "[dnabert.sh] TRUNC_MODE must be off|on." >&2
	exit 1
fi
if [[ "${HEAD_LAYER_NORM}" != "0" && "${HEAD_LAYER_NORM}" != "1" ]]; then
	echo "[dnabert.sh] HEAD_LAYER_NORM must be 0 or 1." >&2
	exit 1
fi
if [[ "${LR_SCHEDULE}" != "cosine" && "${LR_SCHEDULE}" != "linear" ]]; then
	echo "[dnabert.sh] LR_SCHEDULE must be cosine|linear." >&2
	exit 1
fi
if ! [[ "${WARMUP_RATIO}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
	echo "[dnabert.sh] WARMUP_RATIO must be numeric in [0,1)." >&2
	exit 1
fi
if ! awk -v x="${WARMUP_RATIO}" 'BEGIN{exit !(x>=0 && x<1)}'; then
	echo "[dnabert.sh] WARMUP_RATIO must be in [0,1)." >&2
	exit 1
fi
if ! [[ "${ADAM_BETA1}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
	echo "[dnabert.sh] ADAM_BETA1 must be numeric in (0,1)." >&2
	exit 1
fi
if ! awk -v x="${ADAM_BETA1}" 'BEGIN{exit !(x>0 && x<1)}'; then
	echo "[dnabert.sh] ADAM_BETA1 must be in (0,1)." >&2
	exit 1
fi
if ! [[ "${ADAM_BETA2}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
	echo "[dnabert.sh] ADAM_BETA2 must be numeric in (0,1)." >&2
	exit 1
fi
if ! awk -v x="${ADAM_BETA2}" 'BEGIN{exit !(x>0 && x<1)}'; then
	echo "[dnabert.sh] ADAM_BETA2 must be in (0,1)." >&2
	exit 1
fi
if ! awk -v b1="${ADAM_BETA1}" -v b2="${ADAM_BETA2}" \
	'BEGIN{exit !(b1<b2)}'; then
	echo "[dnabert.sh] ADAM_BETA1 must be smaller than ADAM_BETA2." >&2
	exit 1
fi
if ! [[ "${ADAM_EPS}" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
	echo "[dnabert.sh] ADAM_EPS must be a positive number." >&2
	exit 1
fi
if ! awk -v x="${ADAM_EPS}" 'BEGIN{exit !(x>0)}'; then
	echo "[dnabert.sh] ADAM_EPS must be > 0." >&2
	exit 1
fi
MASK_MODE="${TRUNC_MODE}"
export MASK_MODE

# Auto-generate versioning tag from DNABERT variant if TAG is empty
if [[ -z "${TAG}" ]]; then
	DNABERT_VARIANT_LOWER="${DNABERT_VARIANT,,}"
	TAG="dnabert${DNABERT_VARIANT_LOWER}"
	export TAG
fi

(
	export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
	intronmodel_run_with_process_title \
		"${PROCESS_TITLE}" \
		python3 "${PROJECT_ROOT}/src/tools/run_wrapper_pipeline.py" \
			--script-name "dnabert.sh"
)
