#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
	echo "[tune_cnn_pair_time.sh] This script is config-only." \
		"Edit top CONFIG and run without args." >&2
	exit 1
fi

# --------------------------
# CONFIG (edit here)
# --------------------------
# Frequently edited knobs are intentionally placed first in this block.
# Advanced fallback defaults are kept below.
TIME_BUDGET_MINUTES="60"

DONOR_LEN="100"
ACCEPTOR_LEN="100"
BASE_SEED="1337"

QUICK_TRIALS="32"
QUICK_EPOCHS="2"
TOP_K="8"
FULL_EPOCHS="8"
QUICK_COMPILE_MODE="off"
FULL_COMPILE_MODE="off"

GPU_IDS="auto"
# Keep default conservative for single-GPU runs.
MAX_PARALLEL_TRIALS="1"

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
MAX_MODEL_PARAMS_FALLBACK="50000000"
MAX_MODEL_PARAMS_MEM_FRACTION="0.80"
MAX_MODEL_PARAMS_RESERVE_MIB="2048"
MAX_MODEL_PARAMS_BYTES_PER_PARAM="32"
MAX_MODEL_PARAMS_MODEL_FACTOR="0.75"

VISUALIZE="none"
NAME_FIELDS="none"
SEQUENCE_TRANSFORM="none"
UPDATE_DOUBLE_DESCENT_PLOT="1"

SEARCH_ALGO="history_guided"
HISTORY_TOP_N="512"
GUIDED_RANDOM_FRACTION="0.20"
GUIDED_MUTATION_RATE="0.35"
SEARCH_SPACE_FILE="auto"

CROSS_SPECIES_BEST_MODE="auto"
CROSS_SPECIES_BEST_OVERRIDE=""
CROSS_SPECIES_BEST_PREFERRED_SPECIES=""

# Species scheduling order for repeated short cycles.
JOB_ORDER=(
	"Dmel"
	"Mmus"
	"Athal"
)

DEFAULT_SEARCH_SPACE_JSON_PAIR="$(cat <<'JSON'
{
  "lr": {"type": "float", "min": 8e-5, "max": 3e-3, "scale": "log"},
  "batch_size": {
    "type": "categorical",
    "values": [128, 256, 512, 1024, 2048, 4096]
  },
  "dropout": {"type": "float", "min": 0.0, "max": 0.55, "scale": "linear"},
  "weight_decay": {"type": "float", "min": 1e-8, "max": 2e-2, "scale": "log"},
  "loss": {
    "type": "categorical",
    "values": ["weighted_bce", "focal", "asymmetric_focal", "f1", "weighted_bce_f1", "focal_f1"]
  },
  "f1_lambda": {"type": "float", "min": 0.02, "max": 0.5, "scale": "log"},
  "donor_conv_depth": {
    "type": "categorical",
    "values": [2, 3, 4, 5]
  },
  "acceptor_conv_depth": {
    "type": "categorical",
    "values": [2, 3, 4, 5]
  },
  "donor_channel_candidates": {
    "type": "categorical",
    "values": ["64,96,128,192,256,384,512"]
  },
  "acceptor_channel_candidates": {
    "type": "categorical",
    "values": ["64,96,128,192,256,384,512"]
  },
  "donor_kernel_candidates": {
    "type": "categorical",
    "values": [
      "3,5,7,9,11,13,15",
      "5,7,9,11,13,15,17",
      "7,9,11,13,15,17,19"
    ]
  },
  "acceptor_kernel_candidates": {
    "type": "categorical",
    "values": [
      "3,5,7,9,11,13,15",
      "5,7,9,11,13,15,17",
      "7,9,11,13,15,17,19"
    ]
  },
  "fusion_mode": {
    "type": "categorical",
    "values": ["late", "early_channel"]
  },
  "fc_hidden": {
    "type": "categorical",
    "values": [128, 256, 512, 1024, 1536, 2048]
  }
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

format_elapsed() {
	intronmodel_format_elapsed "$1"
}

resolve_species_case() {
	intronmodel_resolve_species_case "$1" "$2" ""
}

resolve_python_bin() {
	intronmodel_resolve_python_bin "tune_cnn_pair_time.sh"
}

resolve_search_space_file() {
	local explicit_file="$1"
	local species="$2"

	if [[ -n "${explicit_file}" && "${explicit_file}" != "auto" ]]; then
		if [[ -f "${explicit_file}" ]]; then
			printf '%s\n' "${explicit_file}"
			return 0
		fi
		echo "[tune_cnn_pair_time.sh] SEARCH_SPACE_FILE not found: ${explicit_file}" >&2
		return 2
	fi

	local target_file="${DATA_ROOT}/${species}/tuning/cnn_pair/pair/search_space.json"
	if [[ -f "${target_file}" ]]; then
		printf '%s\n' "${target_file}"
		return 0
	fi

	local species_file="${DATA_ROOT}/${species}/tuning/cnn_pair/search_space.json"
	if [[ -f "${species_file}" ]]; then
		printf '%s\n' "${species_file}"
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

	"${python_bin}" "${project_root}/src/tools/plot_tuning_double_descent.py" \
		--project_root "${project_root}" \
		--species "${species_name}" \
		--target "pair" \
		--model "cnn_pair" || true
}

if ! [[ "${TIME_BUDGET_MINUTES}" =~ ^[0-9]+$ ]] \
	|| [[ "${TIME_BUDGET_MINUTES}" -le 0 ]]; then
	echo "[tune_cnn_pair_time.sh] TIME_BUDGET_MINUTES must be a positive integer." >&2
	exit 1
fi
if ! [[ "${QUICK_TRIALS}" =~ ^[0-9]+$ ]] || [[ "${QUICK_TRIALS}" -le 0 ]]; then
	echo "[tune_cnn_pair_time.sh] QUICK_TRIALS must be a positive integer." >&2
	exit 1
fi
if ! [[ "${QUICK_EPOCHS}" =~ ^[0-9]+$ ]] || [[ "${QUICK_EPOCHS}" -le 0 ]]; then
	echo "[tune_cnn_pair_time.sh] QUICK_EPOCHS must be a positive integer." >&2
	exit 1
fi
if ! [[ "${TOP_K}" =~ ^[0-9]+$ ]] || [[ "${TOP_K}" -le 0 ]]; then
	echo "[tune_cnn_pair_time.sh] TOP_K must be a positive integer." >&2
	exit 1
fi
if ! [[ "${FULL_EPOCHS}" =~ ^[0-9]+$ ]] || [[ "${FULL_EPOCHS}" -le 0 ]]; then
	echo "[tune_cnn_pair_time.sh] FULL_EPOCHS must be a positive integer." >&2
	exit 1
fi
if [[ "${QUICK_COMPILE_MODE}" != "off" \
	&& "${QUICK_COMPILE_MODE}" != "on" \
	&& "${QUICK_COMPILE_MODE}" != "auto" ]]; then
	echo "[tune_cnn_pair_time.sh] QUICK_COMPILE_MODE must be off|on|auto." >&2
	exit 1
fi
if [[ "${FULL_COMPILE_MODE}" != "off" \
	&& "${FULL_COMPILE_MODE}" != "on" \
	&& "${FULL_COMPILE_MODE}" != "auto" ]]; then
	echo "[tune_cnn_pair_time.sh] FULL_COMPILE_MODE must be off|on|auto." >&2
	exit 1
fi
if [[ "${SEARCH_ALGO}" != "random" && "${SEARCH_ALGO}" != "history_guided" ]]; then
	echo "[tune_cnn_pair_time.sh] SEARCH_ALGO must be random|history_guided." >&2
	exit 1
fi
if ! [[ "${HISTORY_TOP_N}" =~ ^[0-9]+$ ]] || [[ "${HISTORY_TOP_N}" -le 0 ]]; then
	echo "[tune_cnn_pair_time.sh] HISTORY_TOP_N must be a positive integer." >&2
	exit 1
fi
if ! [[ "${GUIDED_RANDOM_FRACTION}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
	echo "[tune_cnn_pair_time.sh] GUIDED_RANDOM_FRACTION must be numeric in [0,1]." >&2
	exit 1
fi
if ! awk -v x="${GUIDED_RANDOM_FRACTION}" 'BEGIN{exit !(x>=0 && x<=1)}'; then
	echo "[tune_cnn_pair_time.sh] GUIDED_RANDOM_FRACTION must be in [0,1]." >&2
	exit 1
fi
if ! [[ "${GUIDED_MUTATION_RATE}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
	echo "[tune_cnn_pair_time.sh] GUIDED_MUTATION_RATE must be numeric in [0,1]." >&2
	exit 1
fi
if ! awk -v x="${GUIDED_MUTATION_RATE}" 'BEGIN{exit !(x>=0 && x<=1)}'; then
	echo "[tune_cnn_pair_time.sh] GUIDED_MUTATION_RATE must be in [0,1]." >&2
	exit 1
fi
if [[ ${#JOB_ORDER[@]} -eq 0 ]]; then
	echo "[tune_cnn_pair_time.sh] JOB_ORDER must contain at least one species." >&2
	exit 1
fi
if [[ "${UPDATE_DOUBLE_DESCENT_PLOT}" != "0" \
	&& "${UPDATE_DOUBLE_DESCENT_PLOT}" != "1" ]]; then
	echo "[tune_cnn_pair_time.sh] UPDATE_DOUBLE_DESCENT_PLOT must be 0 or 1." >&2
	exit 1
fi

PYTHON_BIN="$(resolve_python_bin)"
RESOLVED_MAX_MODEL_PARAMS="$(
	intronmodel_resolve_max_model_params \
		"tune_cnn_pair_time.sh" \
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
BUDGET_SECONDS=$((TIME_BUDGET_MINUTES * 60))
START_EPOCH="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TOTAL_CYCLE_SECONDS=0
COMPLETED_CYCLES=0

echo "[tune_cnn_pair_time.sh] start=${START_EPOCH} budget=${TIME_BUDGET_MINUTES}min"
echo "[tune_cnn_pair_time.sh] quick+full cycles: "\
	"quick_trials=${QUICK_TRIALS} quick_epochs=${QUICK_EPOCHS} "\
	"top_k=${TOP_K} full_epochs=${FULL_EPOCHS}"
echo "[tune_cnn_pair_time.sh] schedule=${JOB_ORDER[*]}"

job_index=0
while true; do
	elapsed_seconds=$((SECONDS - START_SECONDS))
	if [[ "${elapsed_seconds}" -ge "${BUDGET_SECONDS}" ]]; then
		break
	fi
	remaining_seconds=$((BUDGET_SECONDS - elapsed_seconds))
	if [[ "${COMPLETED_CYCLES}" -gt 0 ]]; then
		avg_cycle_seconds_guard=$((TOTAL_CYCLE_SECONDS / COMPLETED_CYCLES))
		if [[ "${avg_cycle_seconds_guard}" -gt 0 ]] \
			&& [[ "${remaining_seconds}" -lt "${avg_cycle_seconds_guard}" ]]; then
			echo "[tune_cnn_pair_time.sh] stop before next cycle: "\
				"remaining=$(format_elapsed "${remaining_seconds}") "\
				"< avg_cycle=$(format_elapsed "${avg_cycle_seconds_guard}")"
			break
		fi
	fi
	remaining_hms="$(format_elapsed "${remaining_seconds}")"

	raw_species="${JOB_ORDER[$((job_index % ${#JOB_ORDER[@]}))]}"
	species="$(resolve_species_case "${raw_species}" "${DATA_ROOT}")"
	run_stamp="$(date +%Y%m%d_%H%M%S)"
	run_id="${run_stamp}_c$(printf '%03d' "${job_index}")"
	output_dir="${DATA_ROOT}/${species}/tuning/cnn_pair/pair/${run_id}"
	global_best_path="${DATA_ROOT}/${species}/tuning/cnn_pair/pair/best_config.json"
	SEED_BEST_CONFIG_PATH=""
	if ! SEED_BEST_CONFIG_PATH="$(
		resolve_cross_species_best_seed \
			"tune_cnn_pair_time.sh" \
			"${PYTHON_BIN}" \
			"${DATA_ROOT}" \
			"cnn_pair" \
			"${species}" \
			"pair" \
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

	objective_metric="pair_pr_auc"
	config_path="${output_dir}/hparam_search_config.json"
	mkdir -p "${output_dir}"
	target_search_space_json="${DEFAULT_SEARCH_SPACE_JSON_PAIR}"
	search_space_path=""
	if search_space_resolved="$(
		resolve_search_space_file \
			"${SEARCH_SPACE_FILE}" \
			"${species}"
	)"; then
		search_space_path="${search_space_resolved}"
		if ! target_space_json="$(
			normalize_json_object_file \
				"${PYTHON_BIN}" \
				"${search_space_path}" 2>&1
		)"; then
			echo "[tune_cnn_pair_time.sh] failed to parse search-space file: "\
				"${search_space_path}" >&2
			echo "[tune_cnn_pair_time.sh] parse detail: ${target_space_json}" >&2
			exit 1
		fi
		target_search_space_json="${target_space_json}"
		echo "[tune_cnn_pair_time.sh] using search space: ${search_space_path}"
	else
		search_space_status=$?
		if [[ "${search_space_status}" -eq 2 ]]; then
			exit 1
		fi
		echo "[tune_cnn_pair_time.sh] using embedded pair search space."
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
  "base_seed": ${BASE_SEED},
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
    "model": "cnn_pair",
    "species": "${species}",
    "train_target": "pair",
    "donor_len": ${DONOR_LEN},
    "acceptor_len": ${ACCEPTOR_LEN},
    "donor_conv_depth": 3,
    "acceptor_conv_depth": 3,
    "donor_channel_candidates": "64,96,128,192,256,384,512",
    "acceptor_channel_candidates": "64,96,128,192,256,384,512",
    "donor_kernel_candidates": "3,5,7,9,11,13,15",
    "acceptor_kernel_candidates": "3,5,7,9,11,13,15",
    "fusion_mode": "late",
    "device": "${DEVICE}",
    "visualize": "${VISUALIZE}",
    "name_fields": "${NAME_FIELDS}",
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
    "train_pos_path": null,
    "train_neg_path": null
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
	printf '[tune_cnn_pair_time.sh] cycle=%s elapsed=%s start=%s ' \
		"${job_index}" "${job_elapsed_hms}" "${job_start}"
	printf 'ETA_remaining=%s species=%s target=pair\n' \
		"${remaining_hms}" "${species}"
	if ! "${PYTHON_BIN}" "${PROJECT_ROOT}/src/tools/hparam_search.py" \
		--config "${config_path}"; then
		echo "[tune_cnn_pair_time.sh] cycle=${job_index} failed "\
			"species=${species} target=pair" >&2
	fi
	if [[ "${UPDATE_DOUBLE_DESCENT_PLOT}" == "1" ]]; then
		run_double_descent_plot \
			"${PYTHON_BIN}" \
			"${PROJECT_ROOT}" \
			"${species}"
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
	printf '[tune_cnn_pair_time.sh] cycle_done=%s cycle_time=%s avg_cycle=%s ' \
		"${job_index}" \
		"$(format_elapsed "${cycle_duration_seconds}")" \
		"$(format_elapsed "${avg_cycle_seconds}")"
	printf 'ETA_cycles_left=%s\n' "${estimated_cycles_left}"

	job_index=$((job_index + 1))
done

if [[ "${UPDATE_DOUBLE_DESCENT_PLOT}" == "1" ]]; then
	final_plot_species=("Athal" "Dmel" "Mmus")
	for final_species in "${final_plot_species[@]}"; do
		run_double_descent_plot \
			"${PYTHON_BIN}" \
			"${PROJECT_ROOT}" \
			"${final_species}"
	done
fi

END_EPOCH="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TOTAL_SECONDS=$((SECONDS - START_SECONDS))
TOTAL_HMS="$(format_elapsed "${TOTAL_SECONDS}")"
echo "[tune_cnn_pair_time.sh] done start=${START_EPOCH} end=${END_EPOCH} "\
	"elapsed=${TOTAL_HMS} cycles=${job_index}"
