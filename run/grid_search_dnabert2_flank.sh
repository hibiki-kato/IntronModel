#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
	echo "[grid_search_dnabert2_flank.sh] This script is config-only." \
		"Edit top CONFIG and run without args." >&2
	exit 1
fi

# --------------------------
# CONFIG (edit here)
# --------------------------

SPECIES="Dmel, Hsap, Mmus"

# "donor", "acceptor", or "both"
TARGET="donor"
INTRONMODEL_AUTO_TMUX="${INTRONMODEL_AUTO_TMUX:-on}"
# Set to a single value to run a 1x1 grid for quick test draws.
INTRONMODEL_GRID_UPSTREAM_VALS="20, 40, 60, 80, 100"
INTRONMODEL_GRID_DOWNSTREAM_VALS="20, 40, 60, 80, 100"
# GPU IDs to use (comma-separated), or "auto"
GPU_IDS="4,5,6,7"
FULL_EPOCHS="3"
BASE_SEED="1337"
BATCH_SIZE="64"
VAL_FRAC="0.2"

DNABERT_VARIANT="2"
PRETRAINED_MODEL_NAME=""
PRETRAINED_MODEL_RELATIVE_PATH_2="pretrained/dnabert2-117m-7bce263b15377fc15361f52cfab88f8b586abda0"
PRETRAINED_MODEL_RELATIVE_PATH_6="pretrained/dnabert6"
PRETRAINED_MODEL_RELATIVE_PATH_S="pretrained/dnabert-s"
PRETRAINED_REVISION=""
TRUST_REMOTE_CODE="1"
MAX_TOKENS="auto"
HEAD_LAYER_NORM="1"

COMPILE_MODE="on"
INFER_COMPILE="0"
INFER_COMPILE_MODE="off"

# Output directory for figures and results JSON.
# Defaults to data/<SPECIES>/grid_search/dnabert2 if left empty.
OUTPUT_DIR=""

# Set to "1" to skip training and only regenerate figures from cached JSON.
FIGURES_ONLY="0"

export INTRONMODEL_AUTO_TMUX
export INTRONMODEL_GRID_UPSTREAM_VALS
export INTRONMODEL_GRID_DOWNSTREAM_VALS

# --------------------------
# Runtime implementation
# --------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"
intronmodel_activate_conda "intronmodel"
intronmodel_init_paths "${BASH_SOURCE[0]}"

intronmodel_enable_auto_tmux "${PROJECT_ROOT}" "$0" "${BASH_SOURCE[0]##*/}"

resolve_python_bin() {
	intronmodel_resolve_python_bin "grid_search_dnabert2_flank.sh"
}

resolve_dnabert_model() {
	local variant="$1"
	local explicit_pretrained="$2"
	local model_root="$3"
	local relative_path_2="$4"
	local relative_path_6="$5"
	local relative_path_s="$6"

	MODEL_NAME="$(
		intronmodel_resolve_dnabert_model_name \
			"grid_search_dnabert2_flank.sh" \
			"${variant}"
	)" || return 1
	PRETRAINED_MODEL_NAME_RESOLVED="$(
		intronmodel_resolve_dnabert_pretrained_name \
			"grid_search_dnabert2_flank.sh" \
			"${variant}" \
			"${explicit_pretrained}" \
			"${model_root}" \
			"${relative_path_2}" \
			"${relative_path_6}" \
			"${relative_path_s}"
	)" || return 1
}

PYTHON_BIN="$(resolve_python_bin)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME=""
PRETRAINED_MODEL_NAME_RESOLVED=""
resolve_dnabert_model \
	"${DNABERT_VARIANT}" \
	"${PRETRAINED_MODEL_NAME}" \
	"${MODEL_ROOT}" \
	"${PRETRAINED_MODEL_RELATIVE_PATH_2}" \
	"${PRETRAINED_MODEL_RELATIVE_PATH_6}" \
	"${PRETRAINED_MODEL_RELATIVE_PATH_S}"

build_species_list() {
	local raw_species="$1"
	local species_list=()
	local entry=""
	IFS=', ' read -ra species_list <<< "${raw_species}"
	for entry in "${species_list[@]}"; do
		[[ -n "${entry}" ]] && printf '%s\n' "${entry}"
	done
}

if [[ -n "${INTRONMODEL_GRID_SPECIES_OVERRIDE:-}" ]]; then
	SPECIES="${INTRONMODEL_GRID_SPECIES_OVERRIDE}"
fi

BASE_ARGS=(
	"--model" "${MODEL_NAME}"
	"--target" "${TARGET}"
	"--gpus" "${GPU_IDS}"
	"--epochs" "${FULL_EPOCHS}"
	"--seed" "${BASE_SEED}"
	"--batch_size" "${BATCH_SIZE}"
	"--val_frac" "${VAL_FRAC}"
	"--pretrained_model_name" "${PRETRAINED_MODEL_NAME_RESOLVED}"
	"--pretrained_revision" "${PRETRAINED_REVISION}"
	"--trust_remote_code" "${TRUST_REMOTE_CODE}"
	"--max_tokens" "${MAX_TOKENS}"
	"--head_layer_norm" "${HEAD_LAYER_NORM}"
	"--compile_mode" "${COMPILE_MODE}"
	"--infer_compile" "${INFER_COMPILE}"
	"--infer_compile_mode" "${INFER_COMPILE_MODE}"
)

if [[ "${FIGURES_ONLY}" == "1" ]]; then
	BASE_ARGS+=("--figures_only")
fi

FILTERED_SPECIES=()
while IFS= read -r sp; do
	FILTERED_SPECIES+=("${sp}")
done < <(build_species_list "${SPECIES}")

# Compute trials per target from explicit grid env vars if provided,
# otherwise fall back to 10x10 default.
if [[ -n "${INTRONMODEL_GRID_UPSTREAM_VALS:-}" ]]; then
	IFS=',' read -ra _UP_ARR <<< "${INTRONMODEL_GRID_UPSTREAM_VALS}"
	UP_COUNT=${#_UP_ARR[@]}
else
	UP_COUNT=10
fi
if [[ -n "${INTRONMODEL_GRID_DOWNSTREAM_VALS:-}" ]]; then
	IFS=',' read -ra _DN_ARR <<< "${INTRONMODEL_GRID_DOWNSTREAM_VALS}"
	DN_COUNT=${#_DN_ARR[@]}
else
	DN_COUNT=10
fi
TRIALS_PER_TARGET="$(( UP_COUNT * DN_COUNT ))"
if [[ "${TARGET}" == "both" ]]; then
	TRIALS_PER_SPECIES="$((TRIALS_PER_TARGET * 2))"
else
	TRIALS_PER_SPECIES="${TRIALS_PER_TARGET}"
fi
TOTAL_GLOBAL_TRIALS="$(( ${#FILTERED_SPECIES[@]} * TRIALS_PER_SPECIES ))"


for idx in "${!FILTERED_SPECIES[@]}"; do
	sp="${FILTERED_SPECIES[$idx]}"
	[[ -z "${sp}" ]] && continue
	GLOBAL_TRIAL_OFFSET="$(( idx * TRIALS_PER_SPECIES ))"

	ARGS=(
		"--species" "${sp}"
		"--global_trial_offset" "${GLOBAL_TRIAL_OFFSET}"
		"--global_trial_total" "${TOTAL_GLOBAL_TRIALS}"
		"${BASE_ARGS[@]}"
	)

	if [[ -n "${OUTPUT_DIR}" ]]; then
		ARGS+=("--output_dir" "${OUTPUT_DIR}/${sp}")
	fi

	echo "[grid_search_dnabert2_flank.sh] species=${sp} target=${TARGET}" \
		"gpus=${GPU_IDS} epochs=${FULL_EPOCHS}" \
		"compile=${COMPILE_MODE}/${INFER_COMPILE_MODE}" \
		"global_trials=$((GLOBAL_TRIAL_OFFSET + 1))-${TOTAL_GLOBAL_TRIALS}"

	"${PYTHON_BIN}" \
		"${PROJECT_ROOT}/src/tools/grid_search_flank.py" \
		"${ARGS[@]}"
done
