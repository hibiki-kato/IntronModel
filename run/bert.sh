#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
	echo "[bert.sh] This script is config-only. Edit top CONFIG and run without args." >&2
	exit 1
fi

# --------------------------
# CONFIG (edit here)
# --------------------------
# Frequently edited knobs are intentionally placed first in this block.
# Advanced per-task overrides are kept below.
set -a
MODEL="bert"
SPECIES="Mmus"
DONOR_LEN="100"
ACCEPTOR_LEN="100"

EPOCHS="auto"
MAX_EPOCHS="100"
EARLY_STOP_PATIENCE="12"
EARLY_STOP_MIN_DELTA="0.0"
BATCH_SIZE="256"
LR="2e-4"
LOSS="weighted_bce"
KMER_K="3"
MAX_TOKENS="auto"
D_MODEL="128"
N_HEADS="4"
N_LAYERS="4"
FF_MULT="4"
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

INTRON_SCORE_OP="*"
TRANSCRIPT_SCORE_AGG="min"
SOFTMIN_TAU="1.0"
SEED="1337"
NAME_FIELDS=""
VISUALIZE="true"
SKIP_TRAINING="0"
CONTINUE_TRAINING="1"
TRAIN_ONLY="0"
PRECOMPUTED_SITE_SCORE_TSV=""
CHECKPOINT_TOP_K="3"
CHECKPOINT_PRUNE_DRY_RUN="0"

DONOR_BATCH_SIZE=""
ACCEPTOR_BATCH_SIZE=""
DONOR_LR=""
ACCEPTOR_LR=""
DONOR_LOSS=""
ACCEPTOR_LOSS=""
DONOR_KMER_K=""
ACCEPTOR_KMER_K=""
DONOR_MAX_TOKENS=""
ACCEPTOR_MAX_TOKENS=""
DONOR_D_MODEL=""
ACCEPTOR_D_MODEL=""
DONOR_N_HEADS=""
ACCEPTOR_N_HEADS=""
DONOR_N_LAYERS=""
ACCEPTOR_N_LAYERS=""
DONOR_FF_MULT=""
ACCEPTOR_FF_MULT=""
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
MPS_MAX_BATCH_SIZE="2048"



set +a

# --------------------------
# Runtime implementation
# --------------------------
# Ensure conda is available in non-interactive shells.
if command -v conda >/dev/null 2>&1; then
	CONDA_BASE="$(conda info --base 2>/dev/null || true)"
	if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
		# shellcheck source=/dev/null
		source "${CONDA_BASE}/etc/profile.d/conda.sh"
	fi
fi

conda activate intronmodel

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Auto-run inside tmux on SSH so jobs survive disconnects.
# Set INTRONMODEL_AUTO_TMUX=off|on|auto (default: auto).
# shellcheck source=/dev/null
source "${PROJECT_ROOT}/run/_auto_tmux.sh"
intronmodel_auto_tmux "$0" "${BASH_SOURCE[0]##*/}"

format_elapsed() {
	local total_seconds="$1"
	local hours=$((total_seconds / 3600))
	local minutes=$(((total_seconds % 3600) / 60))
	local seconds=$((total_seconds % 60))
	printf '%02d:%02d:%02d' "${hours}" "${minutes}" "${seconds}"
}

SCRIPT_START_EPOCH="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SCRIPT_START_SECONDS="${SECONDS}"

print_script_timing() {
	local exit_code="$?"
	local script_end_epoch
	script_end_epoch="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
	local elapsed_seconds=$((SECONDS - SCRIPT_START_SECONDS))
	local elapsed_hms
	elapsed_hms="$(format_elapsed "${elapsed_seconds}")"
	echo "[bert.sh] timing: start=${SCRIPT_START_EPOCH} end=${script_end_epoch} "\
		"elapsed=${elapsed_hms} (${elapsed_seconds}s) exit=${exit_code}"
	return "${exit_code}"
}

trap 'print_script_timing' EXIT

PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
	python3 "${PROJECT_ROOT}/src/tools/run_wrapper_pipeline.py" \
		--script-name "bert.sh"
