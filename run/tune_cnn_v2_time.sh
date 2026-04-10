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
TIME_BUDGET_MINUTES="30"
TIMEOUT_GRACE_SECONDS="30"

INTRONMODEL_AUTO_TMUX="on"
# Validation / objective controls.
VAL_FRAC="0.2"
OBJECTIVE_METRIC="max_f1"
CHEAT_MODE="off"
BASE_SEED="1337"
PROCESS_TITLE="ETA"
TRAIN_POS_PATH=""
TRAIN_NEG_PATH=""
DONOR_LEN="100"
ACCEPTOR_LEN="100"

QUICK_TRIALS="16"
QUICK_EPOCHS="2"
TOP_K="4"
FULL_EPOCHS="15"
QUICK_COMPILE_MODE="off"
FULL_COMPILE_MODE="on"
TRIAL_STREAM_MODE="errors"
ENABLE_PHASE_OVERLAP="1"

GPU_IDS="auto"
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

VISUALIZE="none"
NAME_FIELDS="none"
UPDATE_DOUBLE_DESCENT_PLOT="0"

SEARCH_ALGO="history_guided"
HISTORY_TOP_N="512"
GUIDED_RANDOM_FRACTION="0.20"
GUIDED_MUTATION_RATE="0.35"
SEARCH_SPACE_FILE="auto"
MAX_POOL_SIZE="2"
CONV_STRIDE="1"
HEAD_TYPE="gap"

# Species scheduling order for repeated short cycles.
JOB_ORDER=(
	"Mmus"
	"Hsap"
	"Dmel"
	"Athal"
)

# Tune site tasks independently.
TARGET_ORDER=(
	"acceptor"
)

DEFAULT_SEARCH_SPACE_JSON_SITE="$(cat <<'JSON'
{
  "donor_len": {"type": "int", "min": 40, "max": 100, "step": 10},
  "acceptor_len": {"type": "int", "min": 60, "max": 100, "step": 1},
  "lr": {"type": "float", "min": 8e-5, "max": 3e-3, "scale": "log"},
	"batch_size": {
		"type": "categorical",
		"values": [64, 128, 256, 512, 1024, 2048, 4096]
	},
  "dropout": {"type": "float", "min": 0.0, "max": 0.55, "scale": "linear"},
  "weight_decay": {"type": "float", "min": 1e-8, "max": 2e-2, "scale": "log"},
	"input_mode": {
		"type": "categorical",
		"values": ["onehot", "kmer3", "bpe"]
	},
	"conv_depth": {"type": "int", "min": 2, "max": 7, "step": 1},
	"channel_candidates": {
		"type": "categorical",
		"values": [
			"32,48,64,96,128,192,256,384",
			"48,64,96,128,192,256,320,384,512",
			"64,96,128,160,192,256,320,384,512,768",
			"96,128,192,256,384,512,768,1024"
		]
	},
	"kernel_candidates": {
		"type": "categorical",
		"values": [
			"3,5,7,9,11,13,15",
			"5,7,9,11,13,15,17,19",
			"7,9,11,13,15,17,19,21"
		]
	},
	"channel_order": {"type": "categorical", "values": ["nondecreasing"]},
	"kernel_order": {"type": "categorical", "values": ["nonincreasing"]},
	"max_pool_candidates": {
		"type": "categorical",
		"values": ["1,2,3,4"]
	},
	"conv_stride_candidates": {
		"type": "categorical",
		"values": ["1,2,3"]
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
	intronmodel_resolve_python_bin "tune_cnn_v2_time.sh"
}

resolve_seed_list() {
	intronmodel_resolve_seed_list \
		"tune_cnn_v2_time.sh" \
		"${BASE_SEED}" \
		"" \
		"${PYTHON_BIN}"
}

resolve_search_space_file() {
	local explicit_file="$1"
	local species="$2"
	local tuning_model_name="$3"
	local target_name="$4"

	intronmodel_resolve_search_space_file \
		"tune_cnn_v2_time.sh" \
		"${explicit_file}" \
		"${DATA_ROOT}/${species}/tuning/${tuning_model_name}/${target_name}/search_space.json" \
		"${DATA_ROOT}/${species}/tuning/${tuning_model_name}/search_space.json" \
		"${DATA_ROOT}/${species}/tuning/cnn/search_space.json"
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

append_unique_gpu_ids() {
	local array_name="$1"
	shift || true
	local -n target_ref="${array_name}"
	local candidate=""
	local existing=""
	local found=0

	for candidate in "$@"; do
		if [[ -z "${candidate}" ]]; then
			continue
		fi
		found=0
		for existing in "${target_ref[@]}"; do
			if [[ "${existing}" == "${candidate}" ]]; then
				found=1
				break
			fi
		done
		if [[ "${found}" -eq 0 ]]; then
			target_ref+=("${candidate}")
		fi
	done
}

remove_gpu_from_csv() {
	local gpu_csv="$1"
	local remove_gpu="$2"
	local parts=()
	local kept=()
	local value=""

	IFS=',' read -r -a parts <<< "${gpu_csv}"
	for value in "${parts[@]}"; do
		if [[ -z "${value}" || "${value}" == "${remove_gpu}" ]]; then
			continue
		fi
		kept+=("${value}")
	done
	(
		IFS=,
		printf '%s\n' "${kept[*]}"
	)
}

should_dispatch_next_cycle() {
	local remaining_seconds="$1"

	if [[ "${COMPLETED_CYCLES}" -gt 0 ]]; then
		local avg_cycle_seconds_guard=$((TOTAL_CYCLE_SECONDS / COMPLETED_CYCLES))
		if [[ "${avg_cycle_seconds_guard}" -gt 0 ]] \
			&& [[ "${remaining_seconds}" -lt "${avg_cycle_seconds_guard}" ]]; then
			echo "[tune_cnn_v2_time.sh] stop before next cycle: "\
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
	local target_name="$3"
	local base_seed="$4"
	local output_dir="$5"
	local config_path="$6"
	local stdout_log="$7"
	local run_status=0
	local cycle_prefix="[tune_cnn_v2_time.sh][cycle=${cycle_index}][species=${species}][target=${target_name}][seed=${base_seed}]"

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
			echo "[tune_cnn_v2_time.sh] time budget reached; "\
				"stopping current cycle and cleaning up."
			intronmodel_prune_timeout_artifacts \
				"tune_cnn_v2_time.sh" \
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
		echo "[tune_cnn_v2_time.sh] cycle=${cycle_index} failed "\
			"species=${species} target=${target_name} seed=${base_seed} "\
			"(exit=${run_status})" >>"${stdout_log}"
	fi
	if [[ "${UPDATE_DOUBLE_DESCENT_PLOT}" == "1" ]]; then
		run_double_descent_plot \
			"${PYTHON_BIN}" \
			"${PROJECT_ROOT}" \
			"${species}" \
			"${target_name}" \
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

	local schedule_index=$((cycle_index % (${#JOB_ORDER[@]} * ${#TARGET_ORDER[@]} * ${#SEED_VALUES[@]})))
	local species_index=$((schedule_index % ${#JOB_ORDER[@]}))
	local target_index=$(((schedule_index / ${#JOB_ORDER[@]}) % ${#TARGET_ORDER[@]}))
	local seed_index=$((schedule_index / (${#JOB_ORDER[@]} * ${#TARGET_ORDER[@]})))
	local raw_species="${JOB_ORDER[${species_index}]}"
	local species
	species="$(resolve_species_case "${raw_species}" "${DATA_ROOT}")"
	local target_name="${TARGET_ORDER[${target_index}]}"
	local base_seed="${SEED_VALUES[${seed_index}]}"
	local run_stamp
	run_stamp="$(date +%Y%m%d_%H%M%S)"
	local run_id="${run_stamp}_seed${base_seed}_c$(printf '%03d' "${cycle_index}")"
	local output_dir="${DATA_ROOT}/${species}/tuning/${TUNING_MODEL_NAME}/${target_name}/${run_id}"
	local global_best_path="${DATA_ROOT}/${species}/tuning/${TUNING_MODEL_NAME}/${target_name}/best_config.json"
	local objective_metric="${target_name}_${OBJECTIVE_METRIC}"
	if [[ "${CHEAT_MODE}" == "on" ]]; then
		objective_metric="test_${OBJECTIVE_METRIC}"
	fi
	local config_path="${output_dir}/hparam_search_config.json"
	local gpu_release_events_path="${output_dir}/gpu_release_events.jsonl"
	local stdout_log="${output_dir}/cycle_stdout.log"
	mkdir -p "${output_dir}"

	local resolved_train_paths
	resolved_train_paths="$(
		intronmodel_resolve_and_validate_train_paths \
			"tune_cnn_v2_time.sh" \
			"${species}" \
			"${TRAIN_POS_PATH}" \
			"${TRAIN_NEG_PATH}"
	)" || return 1
	local TRAIN_POS_PATH_RESOLVED=""
	local TRAIN_NEG_PATH_RESOLVED=""
	IFS=$'\t' read -r TRAIN_POS_PATH_RESOLVED TRAIN_NEG_PATH_RESOLVED <<< \
		"${resolved_train_paths}"
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
	local target_search_space_json="${DEFAULT_SEARCH_SPACE_JSON_SITE}"
	local search_space_path=""
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
  "enable_phase_overlap": ${ENABLE_PHASE_OVERLAP_JSON},
  "gpu_release_events_path": "${gpu_release_events_path_rel}",
  "objective_metric": "${objective_metric}",
  "global_best_config_path": "${global_best_path_rel}",
  "seed_best_config_path": null,
  "search_algo": "${SEARCH_ALGO}",
  "history_top_n": ${HISTORY_TOP_N},
  "guided_random_fraction": ${GUIDED_RANDOM_FRACTION},
  "guided_mutation_rate": ${GUIDED_MUTATION_RATE},
  "min_batch_size": ${MIN_BATCH_SIZE},
  "max_oom_retries": ${MAX_OOM_RETRIES},
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

	printf '[tune_cnn_v2_time.sh] cycle=%s elapsed=%s start=%s ' \
		"${cycle_index}" \
		"$(format_elapsed "${elapsed_seconds}")" \
		"$(date -u +%Y-%m-%dT%H:%M:%SZ)"
	printf 'ETA:%s species=%s target=%s seed=%s gpus=%s log=%s\n' \
		"${remaining_hms}" \
		"${species}" \
		"${target_name}" \
		"${base_seed}" \
		"${assigned_gpu_csv}" \
		"${stdout_log}"

	run_cycle_process \
		"${cycle_index}" \
		"${species}" \
		"${target_name}" \
		"${base_seed}" \
		"${output_dir}" \
		"${config_path}" \
		"${stdout_log}" &

	LAST_DISPATCH_PID="$!"
	LAST_DISPATCH_OUTPUT_DIR="${output_dir}"
	LAST_DISPATCH_RELEASE_FILE="${gpu_release_events_path}"
	LAST_DISPATCH_CURSOR_FILE="${output_dir}/gpu_release_events.cursor"
	LAST_DISPATCH_SPECIES="${species}"
	LAST_DISPATCH_TARGET="${target_name}"
	LAST_DISPATCH_SEED="${base_seed}"
	LAST_DISPATCH_STDOUT_LOG="${stdout_log}"
	LAST_DISPATCH_GPU_CSV="${assigned_gpu_csv}"
	return 0
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
if [[ "${TRIAL_STREAM_MODE}" != "auto" \
	&& "${TRIAL_STREAM_MODE}" != "full" \
	&& "${TRIAL_STREAM_MODE}" != "errors" \
	&& "${TRIAL_STREAM_MODE}" != "silent" ]]; then
	echo "[tune_cnn_v2_time.sh] TRIAL_STREAM_MODE must be auto|full|errors|silent." >&2
	exit 1
fi
if [[ "${ENABLE_PHASE_OVERLAP}" != "0" \
	&& "${ENABLE_PHASE_OVERLAP}" != "1" ]]; then
	echo "[tune_cnn_v2_time.sh] ENABLE_PHASE_OVERLAP must be 0 or 1." >&2
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
if [[ "${CHEAT_MODE}" != "off" && "${CHEAT_MODE}" != "on" ]]; then
	echo "[tune_cnn_v2_time.sh] CHEAT_MODE must be off|on." >&2
	exit 1
fi
if [[ "${OBJECTIVE_METRIC}" != "pr_auc" \
	&& "${OBJECTIVE_METRIC}" != "max_f1" ]]; then
	echo "[tune_cnn_v2_time.sh] OBJECTIVE_METRIC must be pr_auc|max_f1." >&2
	exit 1
fi
TUNING_MODEL_NAME="cnn_v2"

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
	"tune_cnn_v2_time.sh" \
	"${GPU_IDS}" \
	"${MAX_PARALLEL_TRIALS}" \
	"${DEVICE}" \
	"${#JOB_ORDER[@]}" \
	"${PYTHON_BIN}")"
ETA_PREFIX="$(intronmodel_eta_prefix "${ETA_SCOPE}")"
START_EPOCH="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TOTAL_CYCLE_SECONDS=0
COMPLETED_CYCLES=0
ENABLE_PHASE_OVERLAP_JSON="false"
if [[ "${ENABLE_PHASE_OVERLAP}" == "1" ]]; then
	ENABLE_PHASE_OVERLAP_JSON="true"
fi

echo "[tune_cnn_v2_time.sh] start=${START_EPOCH} budget=${TIME_BUDGET_MINUTES}min"
echo "[tune_cnn_v2_time.sh] quick+full cycles: "\
	"quick_trials=${QUICK_TRIALS} quick_epochs=${QUICK_EPOCHS} "\
	"top_k=${TOP_K} full_epochs=${FULL_EPOCHS}"
echo "[tune_cnn_v2_time.sh] schedule=${JOB_ORDER[*]}"
echo "[tune_cnn_v2_time.sh] targets=${TARGET_ORDER[*]}"
echo "[tune_cnn_v2_time.sh] seeds=${SEED_VALUES[*]}"
mapfile -t GPU_ID_LIST < <(
	intronmodel_resolve_gpu_ids \
		"tune_cnn_v2_time.sh" \
		"${GPU_IDS}" \
		"${DEVICE}" \
		"${PYTHON_BIN}"
)
PARALLEL_SLOT_COUNT="$(
	intronmodel_resolve_parallel_slots \
		"tune_cnn_v2_time.sh" \
		"${MAX_PARALLEL_TRIALS}" \
		"${#GPU_ID_LIST[@]}"
)"

job_index=0
if [[ ${#GPU_ID_LIST[@]} -le 1 || "${PARALLEL_SLOT_COUNT}" -le 1 ]]; then
	while [[ $((SECONDS - START_SECONDS)) -lt "${BUDGET_SECONDS}" ]]; do
		elapsed_seconds=$((SECONDS - START_SECONDS))
		remaining_seconds=$((BUDGET_SECONDS - elapsed_seconds))
		if ! should_dispatch_next_cycle "${remaining_seconds}"; then
			break
		fi
		remaining_hms="$(format_elapsed "${remaining_seconds}")"
		serial_gpu_csv="${GPU_IDS}"
		if [[ ${#GPU_ID_LIST[@]} -gt 0 ]]; then
			serial_gpu_csv="${GPU_ID_LIST[0]}"
		fi
		if ! dispatch_cycle \
			"${job_index}" \
			"${serial_gpu_csv}" \
			"1" \
			"${elapsed_seconds}" \
			"${remaining_seconds}" \
			"${remaining_hms}"; then
			exit 1
		fi
		if wait "${LAST_DISPATCH_PID}"; then
			completed_code=0
		else
			completed_code=$?
		fi
		if [[ "${completed_code}" -eq 124 ]]; then
			exit 124
		fi
		if [[ "${completed_code}" -eq 130 ]]; then
			exit 130
		fi
		cycle_duration_seconds=$((SECONDS - START_SECONDS - elapsed_seconds))
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
		printf 'ETA_cycles_left=%s log=%s exit=%s\n' \
			"${estimated_cycles_left}" \
			"${LAST_DISPATCH_STDOUT_LOG}" \
			"${completed_code}"
		job_index=$((job_index + 1))
	done
else
	selected_gpu_ids=("${GPU_ID_LIST[@]:0:${PARALLEL_SLOT_COUNT}}")
	gpu_csv="$(IFS=,; echo "${selected_gpu_ids[*]}")"
	echo "[tune_cnn_v2_time.sh] cycle-parallel scheduler across GPUs: ${gpu_csv}"
	declare -A pid_to_cycle=()
	declare -A pid_to_start_seconds=()
	declare -A pid_to_owned_gpu_csv=()
	declare -A pid_to_release_file=()
	declare -A pid_to_cursor_file=()
	declare -A pid_to_stdout_log=()
	running_pids=()
	available_gpu_ids=("${selected_gpu_ids[@]}")
	stop_submitting=0
	first_error_code=0
	while [[ ${#running_pids[@]} -gt 0 || ${stop_submitting} -eq 0 ]]; do
		progress=0
		elapsed_seconds=$((SECONDS - START_SECONDS))
		remaining_seconds=$((BUDGET_SECONDS - elapsed_seconds))
		if [[ "${remaining_seconds}" -lt 0 ]]; then
			remaining_seconds=0
		fi

		for pid in "${running_pids[@]}"; do
			release_file="${pid_to_release_file[$pid]:-}"
			cursor_file="${pid_to_cursor_file[$pid]:-}"
			if [[ -z "${release_file}" || -z "${cursor_file}" ]]; then
				continue
			fi
			mapfile -t released_gpu_ids < <(
				intronmodel_collect_gpu_release_ids \
					"${PYTHON_BIN}" \
					"${release_file}" \
					"${cursor_file}"
			)
			if [[ ${#released_gpu_ids[@]} -eq 0 ]]; then
				continue
			fi
			append_unique_gpu_ids available_gpu_ids "${released_gpu_ids[@]}"
			for released_gpu_id in "${released_gpu_ids[@]}"; do
				pid_to_owned_gpu_csv["${pid}"]="$(remove_gpu_from_csv \
					"${pid_to_owned_gpu_csv[$pid]:-}" \
					"${released_gpu_id}")"
			done
			progress=1
		done

		if [[ ${stop_submitting} -eq 0 ]] \
			&& ! should_dispatch_next_cycle "${remaining_seconds}"; then
			stop_submitting=1
		fi

		while [[ ${stop_submitting} -eq 0 && ${#available_gpu_ids[@]} -gt 0 ]]; do
			assigned_parallel_slots="$(
				intronmodel_resolve_parallel_slots \
					"tune_cnn_v2_time.sh" \
					"${MAX_PARALLEL_TRIALS}" \
					"${#available_gpu_ids[@]}"
			)" || exit 1
			if [[ "${assigned_parallel_slots}" -le 0 ]]; then
				break
			fi
			assigned_gpu_ids=("${available_gpu_ids[@]:0:${assigned_parallel_slots}}")
			available_gpu_ids=("${available_gpu_ids[@]:${assigned_parallel_slots}}")
			assigned_gpu_csv="$(IFS=,; echo "${assigned_gpu_ids[*]}")"
			remaining_hms="$(format_elapsed "${remaining_seconds}")"
			if ! dispatch_cycle \
				"${job_index}" \
				"${assigned_gpu_csv}" \
				"${assigned_parallel_slots}" \
				"${elapsed_seconds}" \
				"${remaining_seconds}" \
				"${remaining_hms}"; then
				exit 1
			fi
			pid="${LAST_DISPATCH_PID}"
			running_pids+=("${pid}")
			pid_to_cycle["${pid}"]="${job_index}"
			pid_to_start_seconds["${pid}"]="${SECONDS}"
			pid_to_owned_gpu_csv["${pid}"]="${LAST_DISPATCH_GPU_CSV}"
			pid_to_release_file["${pid}"]="${LAST_DISPATCH_RELEASE_FILE}"
			pid_to_cursor_file["${pid}"]="${LAST_DISPATCH_CURSOR_FILE}"
			pid_to_stdout_log["${pid}"]="${LAST_DISPATCH_STDOUT_LOG}"
			job_index=$((job_index + 1))
			progress=1
		done

		next_running_pids=()
		for pid in "${running_pids[@]}"; do
			if kill -0 "${pid}" 2>/dev/null; then
				next_running_pids+=("${pid}")
				continue
			fi
			if wait "${pid}"; then
				completed_code=0
			else
				completed_code=$?
			fi
			owned_gpu_csv="${pid_to_owned_gpu_csv[$pid]:-}"
			if [[ -n "${owned_gpu_csv}" ]]; then
				IFS=',' read -r -a owned_gpu_ids <<< "${owned_gpu_csv}"
				append_unique_gpu_ids available_gpu_ids "${owned_gpu_ids[@]}"
			fi
			cycle_index="${pid_to_cycle[$pid]:-0}"
			cycle_duration_seconds=$((SECONDS - ${pid_to_start_seconds[$pid]:-${SECONDS}}))
			if [[ "${completed_code}" -eq 124 || "${completed_code}" -eq 130 ]]; then
				if [[ "${first_error_code}" -eq 0 ]]; then
					first_error_code="${completed_code}"
				fi
				stop_submitting=1
			else
				TOTAL_CYCLE_SECONDS=$((TOTAL_CYCLE_SECONDS + cycle_duration_seconds))
				COMPLETED_CYCLES=$((COMPLETED_CYCLES + 1))
			fi
			avg_cycle_seconds=0
			if [[ "${COMPLETED_CYCLES}" -gt 0 ]]; then
				avg_cycle_seconds=$((TOTAL_CYCLE_SECONDS / COMPLETED_CYCLES))
			fi
			remaining_seconds=$((BUDGET_SECONDS - (SECONDS - START_SECONDS)))
			if [[ "${remaining_seconds}" -lt 0 ]]; then
				remaining_seconds=0
			fi
			estimated_cycles_left=0
			if [[ "${avg_cycle_seconds}" -gt 0 ]]; then
				estimated_cycles_left=$((remaining_seconds / avg_cycle_seconds))
			fi
			printf '[tune_cnn_v2_time.sh] cycle_done=%s cycle_time=%s avg_cycle=%s ' \
				"${cycle_index}" \
				"$(format_elapsed "${cycle_duration_seconds}")" \
				"$(format_elapsed "${avg_cycle_seconds}")"
			printf 'ETA_cycles_left=%s log=%s exit=%s\n' \
				"${estimated_cycles_left}" \
				"${pid_to_stdout_log[$pid]:-}" \
				"${completed_code}"
			unset "pid_to_cycle[${pid}]"
			unset "pid_to_start_seconds[${pid}]"
			unset "pid_to_owned_gpu_csv[${pid}]"
			unset "pid_to_release_file[${pid}]"
			unset "pid_to_cursor_file[${pid}]"
			unset "pid_to_stdout_log[${pid}]"
			progress=1
		done
		running_pids=("${next_running_pids[@]}")

		if [[ ${#running_pids[@]} -eq 0 && ${stop_submitting} -ne 0 ]]; then
			break
		fi
		if [[ "${progress}" -eq 0 ]]; then
			sleep 0.1
		fi
	done
	if [[ "${first_error_code}" -ne 0 ]]; then
		exit "${first_error_code}"
	fi
fi

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
TOTAL_HMS="$(intronmodel_format_elapsed "${TOTAL_SECONDS}")"
echo "[tune_cnn_v2_time.sh] done start=${START_EPOCH} end=${END_EPOCH} "\
	"elapsed=${TOTAL_HMS} cycles=${job_index}"
