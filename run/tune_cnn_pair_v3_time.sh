#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
	echo "[tune_cnn_pair_v3_time.sh] This script is config-only." \
		"Edit top CONFIG and run without args." >&2
	exit 1
fi

# --------------------------
# CONFIG (edit here)
# --------------------------
# Frequently edited knobs are intentionally placed first in this block.
# Advanced fallback defaults are kept below.
TIME_BUDGET_MINUTES="300"
TIMEOUT_GRACE_SECONDS="30"

INTRONMODEL_AUTO_TMUX="on"
# Validation / objective controls.
VAL_FRAC="0.2"
OBJECTIVE_METRIC="max_f1"
# Optional explicit training-data overrides.
TRAIN_POS_PATH=""
TRAIN_NEG_PATH=""
CHEAT_MODE="off"
TAG=""
DONOR_LEN="100"
ACCEPTOR_LEN="100"
BASE_SEED="0"
SEED_LIST=""
PROCESS_TITLE="ETA"

QUICK_TRIALS="8"
QUICK_EPOCHS="2"
TOP_K="2"
FULL_EPOCHS="5"
QUICK_COMPILE_MODE="off"
FULL_COMPILE_MODE="on"
TRIAL_STREAM_MODE="errors"

GPU_IDS="auto"
# Keep default conservative for single-GPU runs.
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

VISUALIZE="none"
NAME_FIELDS="none"
UPDATE_DOUBLE_DESCENT_PLOT="0"

SEARCH_ALGO="reinforce"
HISTORY_TOP_N="512"
GUIDED_RANDOM_FRACTION="0.20"
GUIDED_MUTATION_RATE="0.35"
REINFORCE_TEMPERATURE="0.75"
SEARCH_SPACE_FILE="auto"
HEAD_TYPE="gap"

# Species scheduling order for repeated short cycles.
JOB_ORDER=(
	"Hsap"
	"Dmel"
	"Athal"
	"Mmus"
)

DEFAULT_SEARCH_SPACE_JSON_PAIR="$(cat <<'JSON'
{
  "donor_len": {"type": "int", "min": 40, "max": 100, "step": 10},
  "acceptor_len": {"type": "int", "min": 40, "max": 100, "step": 10},
  "lr": {"type": "float", "min": 8e-5, "max": 3e-3, "scale": "log"},
	"batch_size": {
		"type": "categorical",
		"values": [64, 128, 256, 512, 1024, 2048, 4096, 8192]
	},
  "dropout": {"type": "float", "min": 0.0, "max": 0.55, "scale": "linear"},
  "weight_decay": {"type": "float", "min": 1e-8, "max": 2e-2, "scale": "log"},
	"input_mode": {
		"type": "categorical",
		"values": ["onehot", "kmer3", "bpe"]
	},
	"fusion_mode": {
		"type": "categorical",
		"values": ["late", "mid", "early"]
	},
	"mask": {
		"type": "categorical",
		"values": ["off", "on"]
	},
	"embedding_dim": {
		"type": "categorical",
		"values": [16, 24, 32, 48, 64, 96, 128, 192]
	},
	"fc_hidden": {
		"type": "categorical",
		"values": [64, 96, 128, 192, 256, 384, 512, 768]
	},
	"arch_init_depth": {"type": "int", "min": 2, "max": 4, "step": 1},
	"arch_max_depth": {"type": "int", "min": 5, "max": 9, "step": 1},
	"arch_init_channels": {
		"type": "categorical",
		"values": [32, 48, 64, 96, 128]
	},
	"arch_channel_step": {
		"type": "categorical",
		"values": [16, 24, 32, 48, 64, 96]
	},
	"arch_init_kernel_size": {
		"type": "categorical",
		"values": [7, 9, 11, 13, 15]
	},
	"arch_min_kernel_size": {
		"type": "categorical",
		"values": [3, 5, 7]
	},
	"arch_mutation_steps": {
		"type": "int",
		"min": 2,
		"max": 8,
		"step": 1
	},
	"arch_max_dilation": {
		"type": "categorical",
		"values": [8, 16, 32, 64]
	},
	"arch_add_block_prob": {
		"type": "float",
		"min": 0.10,
		"max": 0.45,
		"scale": "linear"
	},
	"arch_widen_prob": {
		"type": "float",
		"min": 0.10,
		"max": 0.55,
		"scale": "linear"
	},
	"arch_dilation_prob": {
		"type": "float",
		"min": 0.05,
		"max": 0.35,
		"scale": "linear"
	},
	"arch_residual_prob": {
		"type": "float",
		"min": 0.05,
		"max": 0.25,
		"scale": "linear"
	},
	"head_type": {"type": "categorical", "values": ["gap", "center"]},
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

# Keep process title fixed during tune_time runs.
export INTRONMODEL_DISABLE_ETA_PROCESS_TITLE="1"

resolve_species_case() {
	intronmodel_resolve_species_case "$1" "$2" ""
}

resolve_python_bin() {
	intronmodel_resolve_python_bin "tune_cnn_pair_v3_time.sh"
}

resolve_seed_list() {
	intronmodel_resolve_seed_list \
		"tune_cnn_pair_v3_time.sh" \
		"${BASE_SEED}" \
		"${SEED_LIST}" \
		"${PYTHON_BIN}"
}

resolve_search_space_file() {
	local explicit_file="$1"
	local species="$2"
	local tuning_model_name="$3"
	local -a candidates=(
		"${DATA_ROOT}/${species}/tuning/${tuning_model_name}/pair/search_space.json"
		"${DATA_ROOT}/${species}/tuning/${tuning_model_name}/search_space.json"
	)

	if [[ "${tuning_model_name}" != "cnn_pair" ]]; then
		candidates+=(
			"${DATA_ROOT}/${species}/tuning/cnn_pair/pair/search_space.json"
			"${DATA_ROOT}/${species}/tuning/cnn_pair/search_space.json"
		)
	fi

	intronmodel_resolve_search_space_file \
		"tune_cnn_pair_v3_time.sh" \
		"${explicit_file}" \
		"${candidates[@]}"
}




should_dispatch_next_cycle() {
	local remaining_seconds="$1"

	if [[ "${COMPLETED_CYCLES}" -gt 0 ]]; then
		local avg_cycle_seconds_guard=$((TOTAL_CYCLE_SECONDS / COMPLETED_CYCLES))
		if [[ "${avg_cycle_seconds_guard}" -gt 0 ]] \
			&& [[ "${remaining_seconds}" -lt "${avg_cycle_seconds_guard}" ]]; then
			echo "[tune_cnn_pair_v3_time.sh] stop before next cycle: "\
				"remaining=$(format_elapsed "${remaining_seconds}") "\
				"< avg_cycle=$(format_elapsed "${avg_cycle_seconds_guard}")"
			return 1
		fi
	fi
	return 0
}

run_cycle_process() {
	local cycle_index="$1"
	local species="$2"
	local base_seed="$3"
	local output_dir="$4"
	local config_path="$5"
	local stdout_log="$6"
	local run_status=0
	local cycle_prefix="[tune_cnn_pair_v3_time.sh][cycle=${cycle_index}][species=${species}][target=pair][seed=${base_seed}]"

	mkdir -p "$(dirname "${stdout_log}")"
	: > "${stdout_log}"
	if intronmodel_run_with_deadline \
		"${ETA_DEADLINE_EPOCH}" \
		"${TIMEOUT_GRACE_SECONDS}" \
		"${RUNTIME_PROCESS_TITLE}" \
		"${PYTHON_BIN}" \
		"${PROJECT_ROOT}/src/tools/hparam_search.py" \
		--config "${config_path}" \
		> >(
			while IFS= read -r line; do
				printf '%s %s\n' "${cycle_prefix}" "${line}"
			done | tee -a "${stdout_log}"
		) \
		2> >(
			while IFS= read -r line; do
				printf '%s %s\n' "${cycle_prefix}" "${line}" >&2
			done | tee -a "${stdout_log}" >&2
		); then
		run_status=0
	else
		run_status=$?
	fi

	if [[ "${run_status}" -eq 124 ]]; then
		{
			echo "[tune_cnn_pair_v3_time.sh] time budget reached; "\
				"stopping current cycle and cleaning up."
			intronmodel_prune_timeout_artifacts \
				"tune_cnn_pair_v3_time.sh" \
				"${PYTHON_BIN}" \
				"${PROJECT_ROOT}" \
				"${DATA_ROOT}" \
				"${MODEL_ROOT}" \
				"${species}" \
				"${TUNING_MODEL_NAME}" \
				"${output_dir}" || true
		} >>"${stdout_log}" 2>&1
		return 124
	fi

	if [[ "${run_status}" -ne 0 ]]; then
		echo "[tune_cnn_pair_v3_time.sh] cycle=${cycle_index} failed "\
			"species=${species} target=pair seed=${base_seed} "\
			"(exit=${run_status})" >>"${stdout_log}"
	fi
	if [[ "${UPDATE_DOUBLE_DESCENT_PLOT}" == "1" ]]; then
		intronmodel_run_double_descent_plot \
			"${PYTHON_BIN}" \
			"${PROJECT_ROOT}" \
			"${species}" \
			"${TUNING_MODEL_NAME}" >>"${stdout_log}" 2>&1
	fi
	return "${run_status}"
}

dispatch_cycle() {
	local cycle_index="$1"
	local assigned_gpu_csv="$2"
	local assigned_parallel_slots="$3"
	local elapsed_seconds="$4"
	local remaining_seconds="$5"
	local remaining_hms="$6"

	local schedule_index=$((cycle_index % (${#JOB_ORDER[@]} * ${#SEED_VALUES[@]})))
	local species_index=$((schedule_index % ${#JOB_ORDER[@]}))
	local seed_index=$((schedule_index / ${#JOB_ORDER[@]}))
	local raw_species="${JOB_ORDER[${species_index}]}"
	local species
	species="$(resolve_species_case "${raw_species}" "${DATA_ROOT}")"
	local base_seed="${SEED_VALUES[${seed_index}]}"
	local resolved_tag="${TAG}"
	local resolved_train_pos_path="${TRAIN_POS_PATH}"
	local resolved_train_neg_path="${TRAIN_NEG_PATH}"
	local resolved_train_paths
	resolved_train_paths="$(
		intronmodel_resolve_and_validate_train_paths \
			"tune_cnn_pair_v3_time.sh" \
			"${species}" \
			"${resolved_train_pos_path}" \
			"${resolved_train_neg_path}"
	)" || return 1
	local TRAIN_POS_PATH_RESOLVED=""
	local TRAIN_NEG_PATH_RESOLVED=""
	IFS=$'\t' read -r TRAIN_POS_PATH_RESOLVED TRAIN_NEG_PATH_RESOLVED <<< \
		"${resolved_train_paths}"
	local run_stamp
	run_stamp="$(date +%Y%m%d_%H%M%S)"
	local run_id="${run_stamp}_seed${base_seed}_c$(printf '%03d' "${cycle_index}")"
	local output_dir="${DATA_ROOT}/${species}/tuning/${TUNING_MODEL_NAME}/pair/${run_id}"
	local global_best_path
	global_best_path="$(
		intronmodel_resolve_pair_best_config_path \
			"${DATA_ROOT}" \
			"${species}" \
			"${TUNING_MODEL_NAME}"
	)"
	local objective_metric="pair_${OBJECTIVE_METRIC}"
	if [[ "${CHEAT_MODE}" == "on" ]]; then
		objective_metric="test_${OBJECTIVE_METRIC}"
	fi
	local config_path="${output_dir}/hparam_search_config.json"
	local gpu_release_events_path="${output_dir}/gpu_release_events.jsonl"
	local stdout_log="${output_dir}/cycle_stdout.log"
	mkdir -p "${output_dir}"

	local TRAIN_POS_PATH_JSON
	local TRAIN_POS_PATH_CONFIG=""
	if [[ -n "${TRAIN_POS_PATH_RESOLVED}" ]]; then
		TRAIN_POS_PATH_CONFIG="$(
			intronmodel_relpath_from_project_root "${TRAIN_POS_PATH_RESOLVED}"
		)"
	fi
	TRAIN_POS_PATH_JSON="$(
		intronmodel_json_string_or_null \
			"${PYTHON_BIN}" \
			"${TRAIN_POS_PATH_CONFIG}"
	)"
	local TRAIN_NEG_PATH_JSON
	local TRAIN_NEG_PATH_CONFIG=""
	if [[ -n "${TRAIN_NEG_PATH_RESOLVED}" ]]; then
		TRAIN_NEG_PATH_CONFIG="$(
			intronmodel_relpath_from_project_root "${TRAIN_NEG_PATH_RESOLVED}"
		)"
	fi
	TRAIN_NEG_PATH_JSON="$(
		intronmodel_json_string_or_null \
			"${PYTHON_BIN}" \
			"${TRAIN_NEG_PATH_CONFIG}"
	)"
	local output_dir_rel
	output_dir_rel="$(intronmodel_relpath_from_project_root "${output_dir}")"
	local gpu_release_events_path_rel
	gpu_release_events_path_rel="$(
		intronmodel_relpath_from_project_root "${gpu_release_events_path}"
	)"
	local global_best_path_rel
	global_best_path_rel="$(
		intronmodel_relpath_from_project_root "${global_best_path}"
	)"
	local target_search_space_json="${DEFAULT_SEARCH_SPACE_JSON_PAIR}"
	local search_space_path=""
	if search_space_resolved="$(
		resolve_search_space_file \
			"${SEARCH_SPACE_FILE}" \
			"${species}" \
			"${TUNING_MODEL_NAME}"
	)"; then
		search_space_path="${search_space_resolved}"
		if ! target_space_json="$(
			intronmodel_normalize_json_object_file \
				"${PYTHON_BIN}" \
				"${search_space_path}" 2>&1
		)"; then
			echo "[tune_cnn_pair_v3_time.sh] failed to parse search-space file: "\
				"${search_space_path}" >&2
			echo "[tune_cnn_pair_v3_time.sh] parse detail: ${target_space_json}" >&2
			return 1
		fi
		target_search_space_json="${target_space_json}"
	else
		local search_space_status=$?
		if [[ "${search_space_status}" -eq 2 ]]; then
			return 1
		fi
	fi

	cat > "${config_path}" <<JSON
{
  "project_root": ".",
  "species": "${species}",
  "output_dir": "${output_dir_rel}",
  "quick_trials": ${QUICK_TRIALS},
  "quick_epochs": ${QUICK_EPOCHS},
  "top_k": ${TOP_K},
  "full_epochs": ${FULL_EPOCHS},
  "base_seed": ${base_seed},
  "gpu_ids": "${assigned_gpu_csv}",
  "max_parallel_trials": "${assigned_parallel_slots}",
  "trial_stream_mode": "${TRIAL_STREAM_MODE}",
  "enable_phase_overlap": true,
  "gpu_release_events_path": "${gpu_release_events_path_rel}",
  "objective_metric": "${objective_metric}",
  "global_best_config_path": "${global_best_path_rel}",
  "seed_best_config_path": null,
  "search_algo": "${SEARCH_ALGO}",
  "history_top_n": ${HISTORY_TOP_N},
  "guided_random_fraction": ${GUIDED_RANDOM_FRACTION},
  "guided_mutation_rate": ${GUIDED_MUTATION_RATE},
  "reinforce_temperature": ${REINFORCE_TEMPERATURE},
  "min_batch_size": ${MIN_BATCH_SIZE},
  "max_oom_retries": ${MAX_OOM_RETRIES},
	"base_args": {
	"model": "cnn_pair_v3",
    "species": "${species}",
    "train_target": "pair",
    "seed": ${base_seed},
    "donor_len": ${DONOR_LEN},
    "acceptor_len": ${ACCEPTOR_LEN},
    "val_frac": ${VAL_FRAC},
	"input_mode": "onehot",
	"pair_mode": "pair",
	"head_type": "${HEAD_TYPE}",
	"embedding_dim": 32,
    "bpe_pretrained_model_name": "zhihan1996/DNABERT-2-117M",
    "bpe_trust_remote_code": 0,
    "device": "${DEVICE}",
    "visualize": "${VISUALIZE}",
    "tag": "${resolved_tag}",
    "name_fields": "${NAME_FIELDS}",
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

	printf '[tune_cnn_pair_v3_time.sh] cycle=%s elapsed=%s start=%s ' \
		"${cycle_index}" \
		"$(format_elapsed "${elapsed_seconds}")" \
		"$(date -u +%Y-%m-%dT%H:%M:%SZ)"
	printf 'ETA:%s species=%s target=pair seed=%s gpus=%s log=%s\n' \
		"${remaining_hms}" \
		"${species}" \
		"${base_seed}" \
		"${assigned_gpu_csv}" \
		"${stdout_log}"

	run_cycle_process \
		"${cycle_index}" \
		"${species}" \
		"${base_seed}" \
		"${output_dir}" \
		"${config_path}" \
		"${stdout_log}" &

	LAST_DISPATCH_PID="$!"
	LAST_DISPATCH_RELEASE_FILE="${gpu_release_events_path}"
	LAST_DISPATCH_CURSOR_FILE="${output_dir}/gpu_release_events.cursor"
	LAST_DISPATCH_STDOUT_LOG="${stdout_log}"
	LAST_DISPATCH_GPU_CSV="${assigned_gpu_csv}"
	return 0
}

write_cycle_template_config() {
	local template_path="$1"
	local species="$2"
	local base_seed="$3"
	local resolved_tag="$4"
	local train_pos_path_json="$5"
	local train_neg_path_json="$6"
	local target_search_space_json="$7"
	local objective_metric="$8"
	local global_best_path="$9"
	local global_best_path_rel
	global_best_path_rel="$(
		intronmodel_relpath_from_project_root "${global_best_path}"
	)"

	cat > "${template_path}" <<JSON
{
  "project_root": ".",
  "species": "${species}",
  "quick_trials": ${QUICK_TRIALS},
  "quick_epochs": ${QUICK_EPOCHS},
  "top_k": ${TOP_K},
  "full_epochs": ${FULL_EPOCHS},
  "base_seed": ${base_seed},
  "trial_stream_mode": "${TRIAL_STREAM_MODE}",
  "enable_phase_overlap": true,
  "objective_metric": "${objective_metric}",
  "global_best_config_path": "${global_best_path_rel}",
  "seed_best_config_path": null,
  "search_algo": "${SEARCH_ALGO}",
  "history_top_n": ${HISTORY_TOP_N},
  "guided_random_fraction": ${GUIDED_RANDOM_FRACTION},
  "guided_mutation_rate": ${GUIDED_MUTATION_RATE},
  "reinforce_temperature": ${REINFORCE_TEMPERATURE},
  "min_batch_size": ${MIN_BATCH_SIZE},
  "max_oom_retries": ${MAX_OOM_RETRIES},
  "base_args": {
    "model": "cnn_pair_v3",
    "species": "${species}",
    "train_target": "pair",
    "seed": ${base_seed},
    "donor_len": ${DONOR_LEN},
    "acceptor_len": ${ACCEPTOR_LEN},
    "val_frac": ${VAL_FRAC},
    "input_mode": "onehot",
    "pair_mode": "pair",
    "head_type": "${HEAD_TYPE}",
    "embedding_dim": 32,
    "bpe_pretrained_model_name": "zhihan1996/DNABERT-2-117M",
    "bpe_trust_remote_code": 0,
    "device": "${DEVICE}",
    "visualize": "${VISUALIZE}",
    "tag": "${resolved_tag}",
    "name_fields": "${NAME_FIELDS}",
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
    "train_pos_path": ${train_pos_path_json},
    "train_neg_path": ${train_neg_path_json}
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
}

append_scheduler_job_entry() {
	local jobs_file="$1"
	local species="$2"
	local base_seed="$3"
	local template_path="$4"
	local output_parent_dir="$5"
	local template_path_rel
	template_path_rel="$(intronmodel_relpath_from_project_root "${template_path}")"
	local output_parent_dir_rel
	output_parent_dir_rel="$(
		intronmodel_relpath_from_project_root "${output_parent_dir}"
	)"

	"${PYTHON_BIN}" - \
		"${jobs_file}" \
		"${species}" \
		"${base_seed}" \
		"${TUNING_MODEL_NAME}" \
		"${template_path_rel}" \
		"${output_parent_dir_rel}" <<'PY'
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

jobs_file = Path(sys.argv[1])
payload = {
    "species": sys.argv[2],
    "target_name": "pair",
    "seed": int(sys.argv[3]),
    "tuning_model_name": sys.argv[4],
    "template_config_path": sys.argv[5],
    "output_parent_dir": sys.argv[6],
    "plot_target_name": None,
}
with jobs_file.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, ensure_ascii=False))
    handle.write("\n")
PY
}

if ! [[ "${TIME_BUDGET_MINUTES}" =~ ^[0-9]+$ ]] \
	|| [[ "${TIME_BUDGET_MINUTES}" -le 0 ]]; then
	echo "[tune_cnn_pair_v3_time.sh] TIME_BUDGET_MINUTES must be a positive integer." >&2
	exit 1
fi
if ! [[ "${QUICK_TRIALS}" =~ ^[0-9]+$ ]] || [[ "${QUICK_TRIALS}" -le 0 ]]; then
	echo "[tune_cnn_pair_v3_time.sh] QUICK_TRIALS must be a positive integer." >&2
	exit 1
fi
if ! [[ "${QUICK_EPOCHS}" =~ ^[0-9]+$ ]] || [[ "${QUICK_EPOCHS}" -le 0 ]]; then
	echo "[tune_cnn_pair_v3_time.sh] QUICK_EPOCHS must be a positive integer." >&2
	exit 1
fi
if ! [[ "${TOP_K}" =~ ^[0-9]+$ ]] || [[ "${TOP_K}" -le 0 ]]; then
	echo "[tune_cnn_pair_v3_time.sh] TOP_K must be a positive integer." >&2
	exit 1
fi
if ! [[ "${FULL_EPOCHS}" =~ ^[0-9]+$ ]] || [[ "${FULL_EPOCHS}" -le 0 ]]; then
	echo "[tune_cnn_pair_v3_time.sh] FULL_EPOCHS must be a positive integer." >&2
	exit 1
fi
if [[ "${QUICK_COMPILE_MODE}" != "off" \
	&& "${QUICK_COMPILE_MODE}" != "on" \
	&& "${QUICK_COMPILE_MODE}" != "auto" ]]; then
	echo "[tune_cnn_pair_v3_time.sh] QUICK_COMPILE_MODE must be off|on|auto." >&2
	exit 1
fi
if [[ "${FULL_COMPILE_MODE}" != "off" \
	&& "${FULL_COMPILE_MODE}" != "on" \
	&& "${FULL_COMPILE_MODE}" != "auto" ]]; then
	echo "[tune_cnn_pair_v3_time.sh] FULL_COMPILE_MODE must be off|on|auto." >&2
	exit 1
fi
if [[ "${TRIAL_STREAM_MODE}" != "auto" \
	&& "${TRIAL_STREAM_MODE}" != "full" \
	&& "${TRIAL_STREAM_MODE}" != "errors" \
	&& "${TRIAL_STREAM_MODE}" != "silent" ]]; then
	echo "[tune_cnn_pair_v3_time.sh] TRIAL_STREAM_MODE must be auto|full|errors|silent." >&2
	exit 1
fi
if [[ "${SEARCH_ALGO}" != "random" \
	&& "${SEARCH_ALGO}" != "history_guided" \
	&& "${SEARCH_ALGO}" != "reinforce" ]]; then
	echo "[tune_cnn_pair_v3_time.sh] SEARCH_ALGO must be random|history_guided|reinforce." >&2
	exit 1
fi
if ! [[ "${HISTORY_TOP_N}" =~ ^[0-9]+$ ]] || [[ "${HISTORY_TOP_N}" -le 0 ]]; then
	echo "[tune_cnn_pair_v3_time.sh] HISTORY_TOP_N must be a positive integer." >&2
	exit 1
fi
if ! [[ "${GUIDED_RANDOM_FRACTION}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
	echo "[tune_cnn_pair_v3_time.sh] GUIDED_RANDOM_FRACTION must be numeric in [0,1]." >&2
	exit 1
fi
if ! awk -v x="${GUIDED_RANDOM_FRACTION}" 'BEGIN{exit !(x>=0 && x<=1)}'; then
	echo "[tune_cnn_pair_v3_time.sh] GUIDED_RANDOM_FRACTION must be in [0,1]." >&2
	exit 1
fi
if ! [[ "${GUIDED_MUTATION_RATE}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
	echo "[tune_cnn_pair_v3_time.sh] GUIDED_MUTATION_RATE must be numeric in [0,1]." >&2
	exit 1
fi
if ! awk -v x="${GUIDED_MUTATION_RATE}" 'BEGIN{exit !(x>=0 && x<=1)}'; then
	echo "[tune_cnn_pair_v3_time.sh] GUIDED_MUTATION_RATE must be in [0,1]." >&2
	exit 1
fi
if ! [[ "${REINFORCE_TEMPERATURE}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
	echo "[tune_cnn_pair_v3_time.sh] REINFORCE_TEMPERATURE must be numeric > 0." >&2
	exit 1
fi
if ! awk -v x="${REINFORCE_TEMPERATURE}" 'BEGIN{exit !(x>0)}'; then
	echo "[tune_cnn_pair_v3_time.sh] REINFORCE_TEMPERATURE must be > 0." >&2
	exit 1
fi
if [[ ${#JOB_ORDER[@]} -eq 0 ]]; then
	echo "[tune_cnn_pair_v3_time.sh] JOB_ORDER must contain at least one species." >&2
	exit 1
fi
if [[ "${UPDATE_DOUBLE_DESCENT_PLOT}" != "0" \
	&& "${UPDATE_DOUBLE_DESCENT_PLOT}" != "1" ]]; then
	echo "[tune_cnn_pair_v3_time.sh] UPDATE_DOUBLE_DESCENT_PLOT must be 0 or 1." >&2
	exit 1
fi
if [[ "${CHEAT_MODE}" != "off" && "${CHEAT_MODE}" != "on" ]]; then
	echo "[tune_cnn_pair_v3_time.sh] CHEAT_MODE must be off|on." >&2
	exit 1
fi
if [[ "${OBJECTIVE_METRIC}" != "pr_auc" \
	&& "${OBJECTIVE_METRIC}" != "max_f1" ]]; then
	echo "[tune_cnn_pair_v3_time.sh] OBJECTIVE_METRIC must be pr_auc|max_f1." >&2
	exit 1
fi
TUNING_MODEL_NAME="$(
	intronmodel_resolve_pair_tuning_model_name "cnn_pair_v3"
)"

PYTHON_BIN="$(resolve_python_bin)"
mapfile -t SEED_VALUES < <(resolve_seed_list)
START_SECONDS="${SECONDS}"
START_UNIX_SECONDS="$(date +%s)"
BUDGET_SECONDS=$((TIME_BUDGET_MINUTES * 60))
ETA_DEADLINE_EPOCH=$((START_UNIX_SECONDS + BUDGET_SECONDS))
ETA_DEADLINE_LABEL="$(intronmodel_format_eta_epoch "${ETA_DEADLINE_EPOCH}")"
RUNTIME_PROCESS_TITLE="$(
	intronmodel_build_eta_process_title "${ETA_DEADLINE_LABEL}"
)"
ETA_SCOPE="$(intronmodel_resolve_eta_scope \
	"tune_cnn_pair_v3_time.sh" \
	"${GPU_IDS}" \
	"${MAX_PARALLEL_TRIALS}" \
	"${DEVICE}" \
	"${#JOB_ORDER[@]}" \
	"${PYTHON_BIN}")"
ETA_PREFIX="$(intronmodel_eta_prefix "${ETA_SCOPE}")"
START_EPOCH="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TOTAL_CYCLE_SECONDS=0
COMPLETED_CYCLES=0

echo "[tune_cnn_pair_v3_time.sh] start=${START_EPOCH} budget=${TIME_BUDGET_MINUTES}min"
echo "[tune_cnn_pair_v3_time.sh] quick+full cycles: "\
	"quick_trials=${QUICK_TRIALS} quick_epochs=${QUICK_EPOCHS} "\
	"top_k=${TOP_K} full_epochs=${FULL_EPOCHS}"
echo "[tune_cnn_pair_v3_time.sh] schedule=${JOB_ORDER[*]}"
echo "[tune_cnn_pair_v3_time.sh] seeds=${SEED_VALUES[*]}"
mapfile -t GPU_ID_LIST < <(
	intronmodel_resolve_gpu_ids \
		"tune_cnn_pair_v3_time.sh" \
		"${GPU_IDS}" \
		"${DEVICE}" \
		"${PYTHON_BIN}"
)
PARALLEL_SLOT_COUNT="$(
	intronmodel_resolve_parallel_slots \
		"tune_cnn_pair_v3_time.sh" \
		"${MAX_PARALLEL_TRIALS}" \
		"${#GPU_ID_LIST[@]}"
)"
mkdir -p "${PROJECT_ROOT}/temp"
scheduler_root="$(mktemp -d "${PROJECT_ROOT}/temp/${TUNING_MODEL_NAME}_scheduler_XXXXXX")"
cleanup_scheduler_root() {
	if [[ -n "${scheduler_root:-}" && -d "${scheduler_root}" ]]; then
		rm -rf "${scheduler_root}"
	fi
}
trap cleanup_scheduler_root EXIT INT TERM HUP
jobs_file="${scheduler_root}/jobs.jsonl"
: > "${jobs_file}"
selected_gpu_ids=("${GPU_ID_LIST[@]:0:${PARALLEL_SLOT_COUNT}}")

for base_seed in "${SEED_VALUES[@]}"; do
	for raw_species in "${JOB_ORDER[@]}"; do
		species="$(resolve_species_case "${raw_species}" "${DATA_ROOT}")"
		resolved_tag="${TAG}"
		resolved_train_paths="$(
			intronmodel_resolve_and_validate_train_paths \
				"tune_cnn_pair_v3_time.sh" \
				"${species}" \
				"${TRAIN_POS_PATH}" \
				"${TRAIN_NEG_PATH}"
		)" || exit 1
		TRAIN_POS_PATH_RESOLVED=""
		TRAIN_NEG_PATH_RESOLVED=""
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
		target_search_space_json="${DEFAULT_SEARCH_SPACE_JSON_PAIR}"
		search_space_path=""
		if search_space_resolved="$(
			resolve_search_space_file \
				"${SEARCH_SPACE_FILE}" \
				"${species}" \
				"${TUNING_MODEL_NAME}"
		)"; then
			search_space_path="${search_space_resolved}"
			if ! target_space_json="$(
				intronmodel_normalize_json_object_file \
					"${PYTHON_BIN}" \
					"${search_space_path}" 2>&1
			)"; then
				echo "[tune_cnn_pair_v3_time.sh] failed to parse search-space file: "\
					"${search_space_path}" >&2
				echo "[tune_cnn_pair_v3_time.sh] parse detail: ${target_space_json}" >&2
				exit 1
			fi
			target_search_space_json="${target_space_json}"
		else
			search_space_status=$?
			if [[ "${search_space_status}" -eq 2 ]]; then
				exit 1
			fi
		fi
		objective_metric="pair_${OBJECTIVE_METRIC}"
		if [[ "${CHEAT_MODE}" == "on" ]]; then
			objective_metric="test_${OBJECTIVE_METRIC}"
		fi
		global_best_path="$(
			intronmodel_resolve_pair_best_config_path \
				"${DATA_ROOT}" \
				"${species}" \
				"${TUNING_MODEL_NAME}"
		)"
		template_path="${scheduler_root}/${species}_pair_seed${base_seed}.template.json"
		output_parent_dir="${DATA_ROOT}/${species}/tuning/${TUNING_MODEL_NAME}/pair"
		write_cycle_template_config \
			"${template_path}" \
			"${species}" \
			"${base_seed}" \
			"${resolved_tag}" \
			"${TRAIN_POS_PATH_JSON}" \
			"${TRAIN_NEG_PATH_JSON}" \
			"${target_search_space_json}" \
			"${objective_metric}" \
			"${global_best_path}"
		append_scheduler_job_entry \
			"${jobs_file}" \
			"${species}" \
			"${base_seed}" \
			"${template_path}" \
			"${output_parent_dir}"
	done
done

scheduler_config_path="${scheduler_root}/scheduler_config.json"
"${PYTHON_BIN}" - \
	"${scheduler_config_path}" \
	"${jobs_file}" \
	"${PROJECT_ROOT}" \
	"${DATA_ROOT}" \
	"${MODEL_ROOT}" \
	"${PYTHON_BIN}" \
	"${PROJECT_ROOT}/src/tools/hparam_search.py" \
	"${TIME_BUDGET_MINUTES}" \
	"${TIMEOUT_GRACE_SECONDS}" \
	"${START_EPOCH}" \
	"${PARALLEL_SLOT_COUNT}" \
	"${selected_gpu_ids[@]}" <<'PY'
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
jobs_file = sys.argv[2]
project_root = sys.argv[3]
data_root = sys.argv[4]
model_root = sys.argv[5]
python_bin = sys.argv[6]
hparam_search_path = sys.argv[7]
time_budget_minutes = int(sys.argv[8])
timeout_grace_seconds = int(sys.argv[9])
start_epoch = sys.argv[10]
parallel_slot_count = int(sys.argv[11])
selected_gpu_ids = sys.argv[12:]

def _relpath(raw_path: str) -> str:
    return os.path.relpath(Path(raw_path).resolve(), Path(project_root).resolve())


def _serialize_command_or_path(raw_value: str) -> str:
    if "/" not in raw_value and raw_value not in {".", ".."}:
        return raw_value
    return _relpath(raw_value)


payload = {
    "script_name": "tune_cnn_pair_v3_time.sh",
    "project_root": ".",
    "data_root": _relpath(data_root),
    "model_root": _relpath(model_root),
    "python_bin": _serialize_command_or_path(python_bin),
    "hparam_search_path": _relpath(hparam_search_path),
    "time_budget_minutes": time_budget_minutes,
    "timeout_grace_seconds": timeout_grace_seconds,
    "selected_gpu_ids": selected_gpu_ids,
    "parallel_slot_count": max(1, parallel_slot_count),
    "start_epoch": start_epoch,
    "jobs_file": _relpath(jobs_file),
}
config_path.write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
PY

if PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
	intronmodel_run_with_process_title \
	"${RUNTIME_PROCESS_TITLE}" \
	"${PYTHON_BIN}" \
	"${PROJECT_ROOT}/src/tools/tune_time_scheduler.py" \
	--config "${scheduler_config_path}"; then
	scheduler_exit_code=0
else
	scheduler_exit_code=$?
fi
if [[ "${scheduler_exit_code}" -ne 0 ]]; then
	exit "${scheduler_exit_code}"
fi

if [[ "${UPDATE_DOUBLE_DESCENT_PLOT}" == "1" ]]; then
	final_plot_species=("Hsap" "Dmel")
	for final_species in "${final_plot_species[@]}"; do
		intronmodel_run_double_descent_plot \
			"${PYTHON_BIN}" \
			"${PROJECT_ROOT}" \
			"${final_species}" \
			"${TUNING_MODEL_NAME}"
	done
fi
