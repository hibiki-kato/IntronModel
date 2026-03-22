#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
	echo "[tune_cnn_v2_time.sh] This script is config-only." \
		"Edit top CONFIG and run without args." >&2
	exit 1
fi

# --------------------------
# CONFIG (edit here)
# --------------------------
# Frequently edited knobs are intentionally placed first in this block.
# Advanced fallback defaults are kept below.
TIME_BUDGET_MINUTES="200"

INTRONMODEL_AUTO_TMUX=on
# Optional output/data overrides for tagged or mask-data tuning runs.
TAG=""
TRAIN_POS_PATH=""
TRAIN_NEG_PATH=""
MASK_MODE="on"
CHEAT_MODE="off"
DONOR_LEN="100"
ACCEPTOR_LEN="100"
VAL_FRAC="0.2"
BASE_SEED="1337"
# Deprecated: SEED_LIST is ignored. Only BASE_SEED is used.
SEED_LIST=""
PROCESS_TITLE="ETA"

QUICK_TRIALS="12"
QUICK_EPOCHS="2"
TOP_K="3"
FULL_EPOCHS="15"
QUICK_COMPILE_MODE="off"
FULL_COMPILE_MODE="off"

GPU_IDS="5,6,7"
# auto: use one concurrent trial per configured GPU_IDS entry.
MAX_PARALLEL_TRIALS="auto"

DEVICE="auto"
USE_AMP="1"
AMP_DTYPE="auto"
ALLOW_TF32="1"
CUDNN_BENCHMARK="1"
DETERMINISTIC="0"
NUM_WORKERS="auto"
PREFETCH_FACTOR="4"
PERSISTENT_WORKERS="1"
PIN_MEMORY="1"
MIN_BATCH_SIZE="64"
MAX_OOM_RETRIES="5"
MAX_MODEL_PARAMS="auto"
MAX_MODEL_PARAMS_FALLBACK="30000000"
MAX_MODEL_PARAMS_MEM_FRACTION="0.80"
MAX_MODEL_PARAMS_RESERVE_MIB="2048"
MAX_MODEL_PARAMS_BYTES_PER_PARAM="32"
MAX_MODEL_PARAMS_MODEL_FACTOR="0.75"

VISUALIZE="none"
NAME_FIELDS="none"
SEQUENCE_TRANSFORM="none"
UPDATE_DOUBLE_DESCENT_PLOT="0"

SEARCH_ALGO="history_guided"
HISTORY_TOP_N="512"
GUIDED_RANDOM_FRACTION="0.20"
GUIDED_MUTATION_RATE="0.35"
SEARCH_SPACE_FILE="auto"
MAX_POOL_SIZE="2"
CONV_STRIDE="1"
HEAD_TYPE="gap"

CROSS_SPECIES_BEST_MODE="auto"
CROSS_SPECIES_BEST_OVERRIDE=""
CROSS_SPECIES_BEST_PREFERRED_SPECIES=""

# Species scheduling order for repeated short cycles.
JOB_ORDER=(
	"Athal"
	"Dmel"
	"Hsap"
	"Mmus"
)

# Tune site tasks independently.
TARGET_ORDER=(
	"donor"
	"acceptor"
)

DEFAULT_SEARCH_SPACE_JSON_SITE="$(cat <<'JSON'
{
  "donor_len": {"type": "int", "min": 40, "max": 100, "step": 10},
  "acceptor_len": {"type": "int", "min": 40, "max": 100, "step": 10},
  "lr": {"type": "float", "min": 8e-5, "max": 3e-3, "scale": "log"},
	"batch_size": {
		"type": "categorical",
		"values": [128, 256, 512, 1024, 2048]
	},
  "dropout": {"type": "float", "min": 0.0, "max": 0.55, "scale": "linear"},
  "weight_decay": {"type": "float", "min": 1e-8, "max": 2e-2, "scale": "log"},
	"conv_depth": {"type": "int", "min": 3, "max": 5, "step": 1},
	"channel_candidates": {
		"type": "categorical",
		"values": ["64,96,128,192,256,384,512,768"]
	},
	"kernel_candidates": {
		"type": "categorical",
		"values": ["3,5,7,9,11,13,15,17,19"]
	},
	"channel_order": {"type": "categorical", "values": ["nondecreasing"]},
	"kernel_order": {"type": "categorical", "values": ["nonincreasing"]},
	"max_pool_candidates": {
		"type": "categorical",
		"values": ["1,2,3"]
	},
	"conv_stride_candidates": {
		"type": "categorical",
		"values": ["1,2"]
	},
	"head_type": {"type": "categorical", "values": ["gap", "center"]},
	"sequence_transform": {
		"type": "categorical",
		"values": ["none", "mask_outside_intron_n", "truncate_outside_intron"]
	},
  "loss": {
    "type": "categorical",
    "values": ["weighted_bce", "focal", "asymmetric_focal", "f1", "weighted_bce_f1", "focal_f1"]
  },
  "f1_lambda": {"type": "float", "min": 0.02, "max": 0.5, "scale": "log"}
}
JSON
)"


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

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/tuning_cross_species_best.sh"

# Keep process title fixed during tune_time runs.
export INTRONMODEL_DISABLE_ETA_PROCESS_TITLE="1"

format_elapsed() {
	intronmodel_format_elapsed "$1"
}

format_eta() {
	intronmodel_format_eta_epoch "$1"
}

build_eta_process_title() {
	intronmodel_build_eta_process_title "$1"
}

resolve_species_case() {
	intronmodel_resolve_species_case "$1" "$2" ""
}

resolve_python_bin() {
	intronmodel_resolve_python_bin "tune_cnn_v2_time.sh"
}

resolve_seed_list() {
	intronmodel_resolve_seed_list \
		"tune_cnn_v2_time.sh" \
		"${BASE_SEED}" \
		"${SEED_LIST}" \
		"${PYTHON_BIN}"
}

resolve_search_space_file() {
	local explicit_file="$1"
	local species="$2"
	local tuning_model_name="$3"
	local target_name="$4"

	if [[ -n "${explicit_file}" && "${explicit_file}" != "auto" ]]; then
		if [[ -f "${explicit_file}" ]]; then
			printf '%s\n' "${explicit_file}"
			return 0
		fi
		echo "[tune_cnn_v2_time.sh] SEARCH_SPACE_FILE not found: ${explicit_file}" >&2
		return 2
	fi

	local target_file="${DATA_ROOT}/${species}/tuning/${tuning_model_name}/${target_name}/search_space.json"
	if [[ -f "${target_file}" ]]; then
		printf '%s\n' "${target_file}"
		return 0
	fi
	local both_target_file="${DATA_ROOT}/${species}/tuning/${tuning_model_name}/both/search_space.json"
	if [[ -f "${both_target_file}" ]]; then
		printf '%s\n' "${both_target_file}"
		return 0
	fi

	local species_file="${DATA_ROOT}/${species}/tuning/${tuning_model_name}/search_space.json"
	if [[ -f "${species_file}" ]]; then
		printf '%s\n' "${species_file}"
		return 0
	fi
	local base_species_file="${DATA_ROOT}/${species}/tuning/cnn/search_space.json"
	if [[ -f "${base_species_file}" ]]; then
		printf '%s\n' "${base_species_file}"
		return 0
	fi

	return 1
}

normalize_json_object_file() {
	local python_bin="$1"
	local json_path="$2"

	"${python_bin}" - "${json_path}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if not isinstance(payload, dict):
    raise ValueError("Search-space file must contain a JSON object.")
print(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
PY
}

run_double_descent_plot() {
	local python_bin="$1"
	local project_root="$2"
	local species_name="$3"
	local target_name="$4"
	local model_name="$5"

	"${python_bin}" "${project_root}/src/tools/plot_tuning_double_descent.py" \
		--project_root "${project_root}" \
		--species "${species_name}" \
		--target "${target_name}" \
		--model "${model_name}" || true
}

if ! [[ "${TIME_BUDGET_MINUTES}" =~ ^[0-9]+$ ]] \
	|| [[ "${TIME_BUDGET_MINUTES}" -le 0 ]]; then
	echo "[tune_cnn_v2_time.sh] TIME_BUDGET_MINUTES must be a positive integer." >&2
	exit 1
fi
if ! [[ "${QUICK_TRIALS}" =~ ^[0-9]+$ ]] || [[ "${QUICK_TRIALS}" -le 0 ]]; then
	echo "[tune_cnn_v2_time.sh] QUICK_TRIALS must be a positive integer." >&2
	exit 1
fi
if ! [[ "${QUICK_EPOCHS}" =~ ^[0-9]+$ ]] || [[ "${QUICK_EPOCHS}" -le 0 ]]; then
	echo "[tune_cnn_v2_time.sh] QUICK_EPOCHS must be a positive integer." >&2
	exit 1
fi
if ! [[ "${TOP_K}" =~ ^[0-9]+$ ]] || [[ "${TOP_K}" -le 0 ]]; then
	echo "[tune_cnn_v2_time.sh] TOP_K must be a positive integer." >&2
	exit 1
fi
if ! [[ "${FULL_EPOCHS}" =~ ^[0-9]+$ ]] || [[ "${FULL_EPOCHS}" -le 0 ]]; then
	echo "[tune_cnn_v2_time.sh] FULL_EPOCHS must be a positive integer." >&2
	exit 1
fi
if [[ "${QUICK_COMPILE_MODE}" != "off" \
	&& "${QUICK_COMPILE_MODE}" != "on" \
	&& "${QUICK_COMPILE_MODE}" != "auto" ]]; then
	echo "[tune_cnn_v2_time.sh] QUICK_COMPILE_MODE must be off|on|auto." >&2
	exit 1
fi
if [[ "${FULL_COMPILE_MODE}" != "off" \
	&& "${FULL_COMPILE_MODE}" != "on" \
	&& "${FULL_COMPILE_MODE}" != "auto" ]]; then
	echo "[tune_cnn_v2_time.sh] FULL_COMPILE_MODE must be off|on|auto." >&2
	exit 1
fi
if [[ "${SEARCH_ALGO}" != "random" && "${SEARCH_ALGO}" != "history_guided" ]]; then
	echo "[tune_cnn_v2_time.sh] SEARCH_ALGO must be random|history_guided." >&2
	exit 1
fi
if ! [[ "${HISTORY_TOP_N}" =~ ^[0-9]+$ ]] || [[ "${HISTORY_TOP_N}" -le 0 ]]; then
	echo "[tune_cnn_v2_time.sh] HISTORY_TOP_N must be a positive integer." >&2
	exit 1
fi
if ! [[ "${GUIDED_RANDOM_FRACTION}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
	echo "[tune_cnn_v2_time.sh] GUIDED_RANDOM_FRACTION must be numeric in [0,1]." >&2
	exit 1
fi
if ! awk -v x="${GUIDED_RANDOM_FRACTION}" 'BEGIN{exit !(x>=0 && x<=1)}'; then
	echo "[tune_cnn_v2_time.sh] GUIDED_RANDOM_FRACTION must be in [0,1]." >&2
	exit 1
fi
if ! [[ "${GUIDED_MUTATION_RATE}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
	echo "[tune_cnn_v2_time.sh] GUIDED_MUTATION_RATE must be numeric in [0,1]." >&2
	exit 1
fi
if ! awk -v x="${GUIDED_MUTATION_RATE}" 'BEGIN{exit !(x>=0 && x<=1)}'; then
	echo "[tune_cnn_v2_time.sh] GUIDED_MUTATION_RATE must be in [0,1]." >&2
	exit 1
fi
if [[ ${#JOB_ORDER[@]} -eq 0 ]]; then
	echo "[tune_cnn_v2_time.sh] JOB_ORDER must contain at least one species." >&2
	exit 1
fi
if [[ ${#TARGET_ORDER[@]} -eq 0 ]]; then
	echo "[tune_cnn_v2_time.sh] TARGET_ORDER must contain at least one task." >&2
	exit 1
fi
for target_name in "${TARGET_ORDER[@]}"; do
	if [[ "${target_name}" != "donor" && "${target_name}" != "acceptor" ]]; then
		echo "[tune_cnn_v2_time.sh] TARGET_ORDER values must be donor|acceptor." >&2
		exit 1
	fi
done
if [[ "${UPDATE_DOUBLE_DESCENT_PLOT}" != "0" \
	&& "${UPDATE_DOUBLE_DESCENT_PLOT}" != "1" ]]; then
	echo "[tune_cnn_v2_time.sh] UPDATE_DOUBLE_DESCENT_PLOT must be 0 or 1." >&2
	exit 1
fi
if [[ "${MASK_MODE}" != "off" && "${MASK_MODE}" != "on" ]]; then
	echo "[tune_cnn_v2_time.sh] MASK_MODE must be off|on." >&2
	exit 1
fi
if [[ "${CHEAT_MODE}" != "off" && "${CHEAT_MODE}" != "on" ]]; then
	echo "[tune_cnn_v2_time.sh] CHEAT_MODE must be off|on." >&2
	exit 1
fi
TUNING_MODEL_NAME="cnn_v2"

PYTHON_BIN="$(resolve_python_bin)"
mapfile -t SEED_VALUES < <(resolve_seed_list)
RESOLVED_MAX_MODEL_PARAMS="$(
	intronmodel_resolve_max_model_params \
		"tune_cnn_v2_time.sh" \
		"${MAX_MODEL_PARAMS}" \
		"${GPU_IDS}" \
		"${MAX_MODEL_PARAMS_FALLBACK}" \
		"${MAX_MODEL_PARAMS_MEM_FRACTION}" \
		"${MAX_MODEL_PARAMS_RESERVE_MIB}" \
		"${MAX_MODEL_PARAMS_BYTES_PER_PARAM}" \
		"${MAX_MODEL_PARAMS_MODEL_FACTOR}" \
		"${PYTHON_BIN}"
)"
START_SECONDS="${SECONDS}"
START_UNIX_SECONDS="$(date +%s)"
BUDGET_SECONDS=$((TIME_BUDGET_MINUTES * 60))
ETA_DEADLINE_EPOCH=$((START_UNIX_SECONDS + BUDGET_SECONDS))
ETA_DEADLINE_LABEL="$(format_eta "${ETA_DEADLINE_EPOCH}")"
RUNTIME_PROCESS_TITLE="$(build_eta_process_title "${ETA_DEADLINE_LABEL}")"
START_EPOCH="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TOTAL_CYCLE_SECONDS=0
COMPLETED_CYCLES=0

echo "[tune_cnn_v2_time.sh] start=${START_EPOCH} budget=${TIME_BUDGET_MINUTES}min"
echo "[tune_cnn_v2_time.sh] quick+full cycles: "\
	"quick_trials=${QUICK_TRIALS} quick_epochs=${QUICK_EPOCHS} "\
	"top_k=${TOP_K} full_epochs=${FULL_EPOCHS}"
echo "[tune_cnn_v2_time.sh] schedule=${JOB_ORDER[*]}"
echo "[tune_cnn_v2_time.sh] targets=${TARGET_ORDER[*]}"
echo "[tune_cnn_v2_time.sh] seeds=${SEED_VALUES[*]}"

job_index=0
while [[ $((SECONDS - START_SECONDS)) -lt "${BUDGET_SECONDS}" ]]; do
	elapsed_seconds=$((SECONDS - START_SECONDS))
	remaining_seconds=$((BUDGET_SECONDS - elapsed_seconds))
	if [[ "${COMPLETED_CYCLES}" -gt 0 ]]; then
		avg_cycle_seconds_guard=$((TOTAL_CYCLE_SECONDS / COMPLETED_CYCLES))
		if [[ "${avg_cycle_seconds_guard}" -gt 0 ]] \
			&& [[ "${remaining_seconds}" -lt "${avg_cycle_seconds_guard}" ]]; then
			echo "[tune_cnn_v2_time.sh] stop before next cycle: "\
				"remaining=$(format_elapsed "${remaining_seconds}") "\
				"< avg_cycle=$(format_elapsed "${avg_cycle_seconds_guard}")"
			break
		fi
	fi
	remaining_hms="$(format_elapsed "${remaining_seconds}")"

	schedule_index=$((job_index % (${#JOB_ORDER[@]} * ${#TARGET_ORDER[@]} * ${#SEED_VALUES[@]})))
	species_index=$((schedule_index % ${#JOB_ORDER[@]}))
	target_index=$(((schedule_index / ${#JOB_ORDER[@]}) % ${#TARGET_ORDER[@]}))
	seed_index=$((schedule_index / (${#JOB_ORDER[@]} * ${#TARGET_ORDER[@]})))
	raw_species="${JOB_ORDER[${species_index}]}"
	species="$(resolve_species_case "${raw_species}" "${DATA_ROOT}")"
	target_name="${TARGET_ORDER[${target_index}]}"
	base_seed="${SEED_VALUES[${seed_index}]}"
	run_stamp="$(date +%Y%m%d_%H%M%S)"
	run_id="${run_stamp}_seed${base_seed}_c$(printf '%03d' "${job_index}")"
	output_dir="${DATA_ROOT}/${species}/tuning/${TUNING_MODEL_NAME}/${target_name}/${run_id}"
	global_best_path="${DATA_ROOT}/${species}/tuning/${TUNING_MODEL_NAME}/${target_name}/best_config.json"
	SEED_BEST_CONFIG_PATH=""
	if ! SEED_BEST_CONFIG_PATH="$(
		resolve_cross_species_best_seed \
			"tune_cnn_v2_time.sh" \
			"${PYTHON_BIN}" \
			"${DATA_ROOT}" \
			"${TUNING_MODEL_NAME}" \
			"${species}" \
			"${target_name}" \
			"${global_best_path}" \
			"${CROSS_SPECIES_BEST_MODE}" \
			"${CROSS_SPECIES_BEST_OVERRIDE}" \
			"${CROSS_SPECIES_BEST_PREFERRED_SPECIES}"
	)"; then
		exit 1
	fi
	SEED_BEST_CONFIG_JSON="null"
	if [[ -n "${SEED_BEST_CONFIG_PATH}" ]]; then
		SEED_BEST_CONFIG_JSON="\"${SEED_BEST_CONFIG_PATH}\""
	fi

	objective_metric="${target_name}_pr_auc"
	if [[ "${CHEAT_MODE}" == "on" ]]; then
		objective_metric="test_pr_auc"
	fi
	config_path="${output_dir}/hparam_search_config.json"
	mkdir -p "${output_dir}"
	TAG_JSON="$(intronmodel_json_string_or_null "${PYTHON_BIN}" "${TAG}")"
	resolved_train_paths="$(
		intronmodel_resolve_and_validate_train_paths \
			"tune_cnn_v2_time.sh" \
			"${species}" \
			"${TRAIN_POS_PATH}" \
			"${TRAIN_NEG_PATH}"
	)" || exit 1
	IFS=$'\t' read -r TRAIN_POS_PATH_RESOLVED TRAIN_NEG_PATH_RESOLVED <<< \
		"${resolved_train_paths}"
	TRAIN_POS_PATH_JSON="$(
		intronmodel_json_string_or_null \
			"${PYTHON_BIN}" \
			"${TRAIN_POS_PATH_RESOLVED}"
	)"
	TRAIN_NEG_PATH_JSON="$(
		intronmodel_json_string_or_null \
			"${PYTHON_BIN}" \
			"${TRAIN_NEG_PATH_RESOLVED}"
	)"
	target_search_space_json="${DEFAULT_SEARCH_SPACE_JSON_SITE}"
	search_space_path=""
	if search_space_resolved="$(
		resolve_search_space_file \
			"${SEARCH_SPACE_FILE}" \
			"${species}" \
			"${TUNING_MODEL_NAME}" \
			"${target_name}"
	)"; then
		search_space_path="${search_space_resolved}"
		if ! target_space_json="$(
			normalize_json_object_file \
				"${PYTHON_BIN}" \
				"${search_space_path}" 2>&1
		)"; then
			echo "[tune_cnn_v2_time.sh] failed to parse search-space file: "\
				"${search_space_path}" >&2
			echo "[tune_cnn_v2_time.sh] parse detail: ${target_space_json}" >&2
			exit 1
		fi
		target_search_space_json="${target_space_json}"
		echo "[tune_cnn_v2_time.sh] using search space: ${search_space_path}"
	else
		search_space_status=$?
		if [[ "${search_space_status}" -eq 2 ]]; then
			exit 1
		fi
		echo "[tune_cnn_v2_time.sh] using embedded site search space."
	fi

	cat > "${config_path}" <<JSON
{
  "project_root": "${PROJECT_ROOT}",
  "species": "${species}",
  "output_dir": "${output_dir}",
  "quick_trials": ${QUICK_TRIALS},
  "quick_epochs": ${QUICK_EPOCHS},
  "top_k": ${TOP_K},
  "full_epochs": ${FULL_EPOCHS},
  "base_seed": ${base_seed},
  "gpu_ids": "${GPU_IDS}",
  "max_parallel_trials": "${MAX_PARALLEL_TRIALS}",
  "objective_metric": "${objective_metric}",
  "global_best_config_path": "${global_best_path}",
  "seed_best_config_path": ${SEED_BEST_CONFIG_JSON},
  "search_algo": "${SEARCH_ALGO}",
  "history_top_n": ${HISTORY_TOP_N},
  "guided_random_fraction": ${GUIDED_RANDOM_FRACTION},
  "guided_mutation_rate": ${GUIDED_MUTATION_RATE},
  "min_batch_size": ${MIN_BATCH_SIZE},
  "max_oom_retries": ${MAX_OOM_RETRIES},
  "max_model_params": ${RESOLVED_MAX_MODEL_PARAMS},
	"base_args": {
	"model": "cnn_v2",
    "species": "${species}",
	"train_target": "${target_name}",
    "seed": ${base_seed},
    "donor_len": ${DONOR_LEN},
    "acceptor_len": ${ACCEPTOR_LEN},
    "val_frac": ${VAL_FRAC},
	"input_mode": "onehot",
	"pair_mode": "independent",
	"embedding_dim": 32,
	"bpe_pretrained_model_name": "zhihan1996/DNABERT-2-117M",
	"bpe_trust_remote_code": 0,
    "device": "${DEVICE}",
    "visualize": "${VISUALIZE}",
    "name_fields": "${NAME_FIELDS}",
    "tag": ${TAG_JSON},
    "sequence_transform": "${SEQUENCE_TRANSFORM}",
    "use_amp": ${USE_AMP},
    "amp_dtype": "${AMP_DTYPE}",
    "allow_tf32": ${ALLOW_TF32},
    "cudnn_benchmark": ${CUDNN_BENCHMARK},
    "deterministic": ${DETERMINISTIC},
    "num_workers": "${NUM_WORKERS}",
    "prefetch_factor": ${PREFETCH_FACTOR},
    "persistent_workers": ${PERSISTENT_WORKERS},
    "pin_memory": ${PIN_MEMORY},
    "min_batch_size": ${MIN_BATCH_SIZE},
    "max_oom_retries": ${MAX_OOM_RETRIES},
    "train_pos_path": ${TRAIN_POS_PATH_JSON},
    "train_neg_path": ${TRAIN_NEG_PATH_JSON}
  },
  "quick_overrides": {
    "epochs": ${QUICK_EPOCHS},
    "compile_mode": "${QUICK_COMPILE_MODE}"
  },
  "full_overrides": {
    "epochs": ${FULL_EPOCHS},
    "compile_mode": "${FULL_COMPILE_MODE}"
  },
  "search_space": ${target_search_space_json}
}
JSON

	job_start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
	job_start_seconds="${SECONDS}"
	job_elapsed_hms="$(format_elapsed "${elapsed_seconds}")"
	printf '[tune_cnn_v2_time.sh] cycle=%s elapsed=%s start=%s ' \
		"${job_index}" "${job_elapsed_hms}" "${job_start}"
	printf 'ETA_remaining=%s species=%s target=%s seed=%s\n' \
		"${remaining_hms}" "${species}" "${target_name}" "${base_seed}"
	run_status=0
	intronmodel_run_with_process_title \
		"${RUNTIME_PROCESS_TITLE}" \
		"${PYTHON_BIN}" \
		"${PROJECT_ROOT}/src/tools/hparam_search.py" \
		--config "${config_path}" || run_status=$?
	if [[ "${run_status}" -eq 130 ]]; then
		echo "[tune_cnn_v2_time.sh] interrupted by user; stopping." >&2
		exit 130
	fi
	if [[ "${run_status}" -ne 0 ]]; then
		echo "[tune_cnn_v2_time.sh] cycle=${job_index} failed "\
			"species=${species} target=${target_name} seed=${base_seed} "\
			"(exit=${run_status})" >&2
	fi
	if [[ "${UPDATE_DOUBLE_DESCENT_PLOT}" == "1" ]]; then
		run_double_descent_plot \
			"${PYTHON_BIN}" \
			"${PROJECT_ROOT}" \
			"${species}" \
			"${target_name}" \
			"${TUNING_MODEL_NAME}"
	fi
	cycle_duration_seconds=$((SECONDS - job_start_seconds))
	TOTAL_CYCLE_SECONDS=$((TOTAL_CYCLE_SECONDS + cycle_duration_seconds))
	COMPLETED_CYCLES=$((COMPLETED_CYCLES + 1))
	avg_cycle_seconds=$((TOTAL_CYCLE_SECONDS / COMPLETED_CYCLES))
	remaining_seconds=$((BUDGET_SECONDS - (SECONDS - START_SECONDS)))
	if [[ "${remaining_seconds}" -lt 0 ]]; then
		remaining_seconds=0
	fi
	estimated_cycles_left=0
	if [[ "${avg_cycle_seconds}" -gt 0 ]]; then
		estimated_cycles_left=$((remaining_seconds / avg_cycle_seconds))
	fi
	printf '[tune_cnn_v2_time.sh] cycle_done=%s cycle_time=%s avg_cycle=%s ' \
		"${job_index}" \
		"$(format_elapsed "${cycle_duration_seconds}")" \
		"$(format_elapsed "${avg_cycle_seconds}")"
	printf 'ETA_cycles_left=%s\n' "${estimated_cycles_left}"

	job_index=$((job_index + 1))
done

if [[ "${UPDATE_DOUBLE_DESCENT_PLOT}" == "1" ]]; then
	final_plot_species=("Hsap" "Dmel")
	for final_species in "${final_plot_species[@]}"; do
		for final_target in "${TARGET_ORDER[@]}"; do
			run_double_descent_plot \
				"${PYTHON_BIN}" \
				"${PROJECT_ROOT}" \
				"${final_species}" \
				"${final_target}" \
				"${TUNING_MODEL_NAME}"
		done
	done
fi

END_EPOCH="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TOTAL_SECONDS=$((SECONDS - START_SECONDS))
TOTAL_HMS="$(format_elapsed "${TOTAL_SECONDS}")"
echo "[tune_cnn_v2_time.sh] done start=${START_EPOCH} end=${END_EPOCH} "\
	"elapsed=${TOTAL_HMS} cycles=${job_index}"
