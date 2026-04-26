#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
	echo "[grid_search_cnn_v2_flank.sh] This script is config-only." \
		"Edit top CONFIG and run without args." >&2
	exit 1
fi

# --------------------------
# CONFIG (edit here)
# --------------------------

SPECIES="Athal, Dmel, Hsap, Mmus"

# "donor", "acceptor", or "both"
TARGET="both"
INTRONMODEL_AUTO_TMUX="on"
# GPU IDs to use (comma-separated), or "auto"
GPU_IDS="0,2,5,7"
FULL_EPOCHS="15"
BASE_SEED="1337"
BATCH_SIZE="512"
VAL_FRAC="0.2"
COMPILE_MODE="quick"
INFER_COMPILE="1"
INFER_COMPILE_MODE="quick"
GRID_VALUES="10"

# Output directory for figures and results JSON.
# Defaults to data/<SPECIES>/grid_search/cnn_v2 if left empty.
OUTPUT_DIR=""

# Set to "1" to skip training and only regenerate figures from cached JSON.
FIGURES_ONLY="0"

export INTRONMODEL_AUTO_TMUX

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
	intronmodel_resolve_python_bin "grid_search_cnn_v2_flank.sh"
}

PYTHON_BIN="$(resolve_python_bin)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

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

# --------------------------
# Build base argument list (species-independent)
# --------------------------

BASE_ARGS=(
	"--target"     "${TARGET}"
	"--gpus"       "${GPU_IDS}"
	"--epochs"     "${FULL_EPOCHS}"
	"--seed"       "${BASE_SEED}"
	"--batch_size" "${BATCH_SIZE}"
	"--val_frac"   "${VAL_FRAC}"
	"--compile_mode" "${COMPILE_MODE}"
	"--infer_compile" "${INFER_COMPILE}"
	"--infer_compile_mode" "${INFER_COMPILE_MODE}"
)

if [[ "${FIGURES_ONLY}" == "1" ]]; then
	BASE_ARGS+=("--figures_only")
fi

# --------------------------
# Loop over species
# --------------------------

FILTERED_SPECIES=()
while IFS= read -r sp; do
	FILTERED_SPECIES+=("${sp}")
done < <(build_species_list "${SPECIES}")

TRIALS_PER_TARGET="100"
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

	echo "[grid_search_cnn_v2_flank.sh] species=${sp} target=${TARGET}" \
		"gpus=${GPU_IDS} epochs=${FULL_EPOCHS}" \
		"compile=${COMPILE_MODE}/${INFER_COMPILE_MODE}" \
		"global_trials=$((GLOBAL_TRIAL_OFFSET + 1))-${TOTAL_GLOBAL_TRIALS}"

	"${PYTHON_BIN}" \
		"${PROJECT_ROOT}/src/tools/grid_search_flank.py" \
		"${ARGS[@]}"
done
