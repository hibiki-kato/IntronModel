#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
	echo "[tune_dnabert_pair_time.sh] This script is config-only." \
		"Edit top CONFIG and run without args." >&2
	exit 1
fi

# --------------------------
# CONFIG (edit here)
# --------------------------
# Frequently edited knobs are intentionally placed first in this block.
# Advanced fallback defaults are kept below.
TIME_BUDGET_MINUTES="60"

INTRONMODEL_AUTO_TMUX="on"
CHEAT_MODE="on"
OBJECTIVE_METRIC="max_f1"
TRUNC_MODE="on"
DONOR_LEN="100"
ACCEPTOR_LEN="100"
VAL_FRAC="0.25"
HEAD_LAYER_NORM="1"
READOUT_TYPE="mlp"
READOUT_CNN_KERNEL_SIZE="3"
READOUT_MLP_HIDDEN_DIM="256"
READOUT_MLP_LAYERS="1"
BASE_SEED="1337"
DNABERT_VARIANT="2"
PRETRAINED_MODEL_NAME=""
PRETRAINED_MODEL_RELATIVE_PATH_2="pretrained/dnabert2-117m-7bce263b15377fc15361f52cfab88f8b586abda0"
PRETRAINED_MODEL_RELATIVE_PATH_6="pretrained/dnabert6"
PRETRAINED_MODEL_RELATIVE_PATH_S="pretrained/dnabert-s"
PRETRAINED_REVISION=""
TRUST_REMOTE_CODE="1"

QUICK_TRIALS="12"
QUICK_EPOCHS="2"
TOP_K="2"
FULL_EPOCHS="3"
QUICK_COMPILE_MODE="off"
FULL_COMPILE_MODE="auto"
LR_SCHEDULE="cosine"
WARMUP_RATIO="0.01"
ADAM_BETA1="0.9"
ADAM_BETA2="0.98"
ADAM_EPS="1e-8"

GPU_IDS="0,1,5"
# Keep the default to one concurrent trial for stable single-GPU throughput.
# Increase manually when you intentionally run multi-GPU parallel tuning.
MAX_PARALLEL_TRIALS="auto"
TRIAL_PROCESS_MODE="persistent_all"

DEVICE="auto"
USE_AMP="1"
AMP_DTYPE="auto"
INFER_BATCH_SIZE="256"
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
MIN_BATCH_SIZE="8"
MAX_OOM_RETRIES="5"

VISUALIZE="none"
NAME_FIELDS="none"
# Optional output/data overrides for trunc/cheat-data tuning runs.
TAG=""
TRAIN_POS_PATH=""
TRAIN_NEG_PATH=""
UPDATE_DOUBLE_DESCENT_PLOT="0"
# Optional override. Leave empty to use static ETA title from TIME_BUDGET_MINUTES.
PROCESS_TITLE=""

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
	"Hsap"
	"Mmus"
)

DEFAULT_SEARCH_SPACE_JSON_PAIR="$(cat <<'JSON'
{
  "donor_len": {"type": "int", "min": 40, "max": 100, "step": 10},
  "acceptor_len": {"type": "int", "min": 40, "max": 100, "step": 10},
  "lr": {"type": "float", "min": 8e-6, "max": 8e-5, "scale": "log"},
  "batch_size": {
    "type": "categorical",
    "values": [8, 12, 16, 24, 32, 48, 64, 96, 128]
  },
  "loss": {
    "type": "categorical",
    "values": ["weighted_bce", "focal", "asymmetric_focal"]
  },
  "max_tokens": {
    "type": "categorical",
    "values": ["auto", 160, 192, 224]
  },
  "dropout": {
    "type": "float",
    "min": 0.10,
    "max": 0.45,
    "scale": "linear"
  },
  "readout_type": {
    "type": "categorical",
    "values": ["cnn", "linear", "mlp"]
  },
  "readout_cnn_kernel_size": {
    "type": "categorical",
    "values": [3, 5, 7]
  },
  "readout_mlp_hidden_dim": {
    "type": "categorical",
    "values": [128, 256, 384, 512]
  },
  "readout_mlp_layers": {"type": "int", "min": 1, "max": 3, "step": 1},
  "head_layer_norm": {
    "type": "categorical",
    "values": [0, 1]
  },
  "lr_schedule": {
    "type": "categorical",
    "values": ["cosine", "linear"]
  },
  "warmup_ratio": {"type": "float", "min": 0.005, "max": 0.02, "scale": "linear"},
  "weight_decay": {"type": "float", "min": 1e-5, "max": 6e-2, "scale": "log"},
  "eta_min_ratio": {"type": "float", "min": 5e-4, "max": 7e-2, "scale": "log"},
  "grad_clip": {"type": "float", "min": 0.5, "max": 2.5, "scale": "linear"},
  "pos_weight_cap": {"type": "float", "min": 12.0, "max": 36.0, "scale": "linear"},
  "focal_gamma": {"type": "float", "min": 1.0, "max": 3.5, "scale": "linear"},
  "asym_gamma_pos": {"type": "float", "min": 0.0, "max": 2.5, "scale": "linear"},
  "asym_gamma_neg": {"type": "float", "min": 2.0, "max": 7.0, "scale": "linear"}
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
	intronmodel_resolve_python_bin "tune_dnabert_pair_time.sh"
}

resolve_dnabert_model() {
	local variant="$1"
	local explicit_pretrained="$2"
	local model_root="$3"
	local relative_path_2="$4"
	local relative_path_6="$5"
	local relative_path_s="$6"

	local normalized_variant="${variant,,}"
	if [[ "${normalized_variant}" != "2" \
		&& "${normalized_variant}" != "6" \
		&& "${normalized_variant}" != "s" ]]; then
		echo "[tune_dnabert_pair_time.sh] DNABERT_VARIANT must be 2, 6, or s." >&2
		return 1
	fi

	if [[ "${normalized_variant}" == "s" ]]; then
		MODEL_NAME="dnaberts"
	else
		MODEL_NAME="dnabert${normalized_variant}"
	fi
	if [[ -n "${explicit_pretrained}" ]]; then
		PRETRAINED_MODEL_NAME_RESOLVED="${explicit_pretrained}"
		return 0
	fi

	local relative_path
	if [[ "${normalized_variant}" == "2" ]]; then
		relative_path="${relative_path_2}"
	elif [[ "${normalized_variant}" == "6" ]]; then
		relative_path="${relative_path_6}"
	else
		relative_path="${relative_path_s}"
	fi
	if [[ -z "${relative_path}" ]]; then
		echo "[tune_dnabert_pair_time.sh] pretrained relative path is empty for variant=${variant}." >&2
		return 1
	fi
	PRETRAINED_MODEL_NAME_RESOLVED="${model_root}/${relative_path}"
}

resolve_search_space_file() {
	local explicit_file="$1"
	local project_root="$2"
	local species="$3"
	local target="$4"
	local model_name="$5"

	if [[ -n "${explicit_file}" && "${explicit_file}" != "auto" ]]; then
		if [[ -f "${explicit_file}" ]]; then
			printf '%s\n' "${explicit_file}"
			return 0
		fi
		echo "[tune_dnabert_pair_time.sh] SEARCH_SPACE_FILE not found: ${explicit_file}" >&2
		return 2
	fi

	local target_file="${DATA_ROOT}/${species}/tuning/${model_name}/${target}/search_space.json"
	if [[ -f "${target_file}" ]]; then
		printf '%s\n' "${target_file}"
		return 0
	fi

	local species_file="${DATA_ROOT}/${species}/tuning/${model_name}/search_space.json"
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
	echo "[tune_dnabert_pair_time.sh] TIME_BUDGET_MINUTES must be a positive integer." >&2
	exit 1
fi
if ! [[ "${QUICK_TRIALS}" =~ ^[0-9]+$ ]] || [[ "${QUICK_TRIALS}" -le 0 ]]; then
	echo "[tune_dnabert_pair_time.sh] QUICK_TRIALS must be a positive integer." >&2
	exit 1
fi
if ! [[ "${QUICK_EPOCHS}" =~ ^[0-9]+$ ]] || [[ "${QUICK_EPOCHS}" -le 0 ]]; then
	echo "[tune_dnabert_pair_time.sh] QUICK_EPOCHS must be a positive integer." >&2
	exit 1
fi
if ! [[ "${TOP_K}" =~ ^[0-9]+$ ]] || [[ "${TOP_K}" -le 0 ]]; then
	echo "[tune_dnabert_pair_time.sh] TOP_K must be a positive integer." >&2
	exit 1
fi
if ! [[ "${FULL_EPOCHS}" =~ ^[0-9]+$ ]] || [[ "${FULL_EPOCHS}" -le 0 ]]; then
	echo "[tune_dnabert_pair_time.sh] FULL_EPOCHS must be a positive integer." >&2
	exit 1
fi
if [[ "${QUICK_COMPILE_MODE}" != "off" \
	&& "${QUICK_COMPILE_MODE}" != "on" \
	&& "${QUICK_COMPILE_MODE}" != "auto" ]]; then
	echo "[tune_dnabert_pair_time.sh] QUICK_COMPILE_MODE must be off|on|auto." >&2
	exit 1
fi
if [[ "${FULL_COMPILE_MODE}" != "off" \
	&& "${FULL_COMPILE_MODE}" != "on" \
	&& "${FULL_COMPILE_MODE}" != "auto" ]]; then
	echo "[tune_dnabert_pair_time.sh] FULL_COMPILE_MODE must be off|on|auto." >&2
	exit 1
fi
if [[ "${TRIAL_PROCESS_MODE}" != "subprocess" \
	&& "${TRIAL_PROCESS_MODE}" != "persistent_quick" \
	&& "${TRIAL_PROCESS_MODE}" != "persistent_all" ]]; then
	echo "[tune_dnabert_pair_time.sh] TRIAL_PROCESS_MODE must be "\
		"subprocess|persistent_quick|persistent_all." >&2
	exit 1
fi
if [[ "${SEARCH_ALGO}" != "random" && "${SEARCH_ALGO}" != "history_guided" ]]; then
	echo "[tune_dnabert_pair_time.sh] SEARCH_ALGO must be random|history_guided." >&2
	exit 1
fi
if ! [[ "${HISTORY_TOP_N}" =~ ^[0-9]+$ ]] || [[ "${HISTORY_TOP_N}" -le 0 ]]; then
	echo "[tune_dnabert_pair_time.sh] HISTORY_TOP_N must be a positive integer." >&2
	exit 1
fi
if ! [[ "${GUIDED_RANDOM_FRACTION}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
	echo "[tune_dnabert_pair_time.sh] GUIDED_RANDOM_FRACTION must be numeric in [0,1]." >&2
	exit 1
fi
if ! awk -v x="${GUIDED_RANDOM_FRACTION}" 'BEGIN{exit !(x>=0 && x<=1)}'; then
	echo "[tune_dnabert_pair_time.sh] GUIDED_RANDOM_FRACTION must be in [0,1]." >&2
	exit 1
fi
if ! [[ "${GUIDED_MUTATION_RATE}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
	echo "[tune_dnabert_pair_time.sh] GUIDED_MUTATION_RATE must be numeric in [0,1]." >&2
	exit 1
fi
if ! awk -v x="${GUIDED_MUTATION_RATE}" 'BEGIN{exit !(x>=0 && x<=1)}'; then
	echo "[tune_dnabert_pair_time.sh] GUIDED_MUTATION_RATE must be in [0,1]." >&2
	exit 1
fi
if [[ ${#JOB_ORDER[@]} -eq 0 ]]; then
	echo "[tune_dnabert_pair_time.sh] JOB_ORDER must contain at least one" \
		"species entry." >&2
	exit 1
fi
if [[ "${UPDATE_DOUBLE_DESCENT_PLOT}" != "0" \
	&& "${UPDATE_DOUBLE_DESCENT_PLOT}" != "1" ]]; then
	echo "[tune_dnabert_pair_time.sh] UPDATE_DOUBLE_DESCENT_PLOT must be 0 or 1." >&2
	exit 1
fi
if [[ "${TRUNC_MODE}" != "off" && "${TRUNC_MODE}" != "on" ]]; then
	echo "[tune_dnabert_pair_time.sh] TRUNC_MODE must be off|on." >&2
	exit 1
fi
if [[ "${CHEAT_MODE}" != "off" && "${CHEAT_MODE}" != "on" ]]; then
	echo "[tune_dnabert_pair_time.sh] CHEAT_MODE must be off|on." >&2
	exit 1
fi
if [[ "${OBJECTIVE_METRIC}" != "pr_auc" \
	&& "${OBJECTIVE_METRIC}" != "max_f1" ]]; then
	echo "[tune_dnabert_pair_time.sh] OBJECTIVE_METRIC must be pr_auc|max_f1." >&2
	exit 1
fi
if ! [[ "${INFER_BATCH_SIZE}" =~ ^[0-9]+$ ]] \
	|| [[ "${INFER_BATCH_SIZE}" -le 0 ]]; then
	echo "[tune_dnabert_pair_time.sh] INFER_BATCH_SIZE must be a positive integer." >&2
	exit 1
fi
if [[ "${INFER_USE_AMP}" != "0" && "${INFER_USE_AMP}" != "1" ]]; then
	echo "[tune_dnabert_pair_time.sh] INFER_USE_AMP must be 0 or 1." >&2
	exit 1
fi
if [[ "${INFER_AMP_DTYPE}" != "auto" \
	&& "${INFER_AMP_DTYPE}" != "bf16" \
	&& "${INFER_AMP_DTYPE}" != "fp16" ]]; then
	echo "[tune_dnabert_pair_time.sh] INFER_AMP_DTYPE must be auto|bf16|fp16." >&2
	exit 1
fi
if [[ "${INFER_COMPILE}" != "0" && "${INFER_COMPILE}" != "1" ]]; then
	echo "[tune_dnabert_pair_time.sh] INFER_COMPILE must be 0 or 1." >&2
	exit 1
fi
if [[ "${INFER_COMPILE_MODE}" != "off" \
	&& "${INFER_COMPILE_MODE}" != "on" \
	&& "${INFER_COMPILE_MODE}" != "auto" ]]; then
	echo "[tune_dnabert_pair_time.sh] INFER_COMPILE_MODE must be off|on|auto." >&2
	exit 1
fi
if [[ "${HEAD_LAYER_NORM}" != "0" && "${HEAD_LAYER_NORM}" != "1" ]]; then
	echo "[tune_dnabert_pair_time.sh] HEAD_LAYER_NORM must be 0 or 1." >&2
	exit 1
fi
if [[ "${READOUT_TYPE}" != "cnn" \
	&& "${READOUT_TYPE}" != "linear" \
	&& "${READOUT_TYPE}" != "mlp" ]]; then
	echo "[tune_dnabert_pair_time.sh] READOUT_TYPE must be cnn|linear|mlp." >&2
	exit 1
fi
if ! [[ "${READOUT_CNN_KERNEL_SIZE}" =~ ^[0-9]+$ ]] \
	|| [[ "${READOUT_CNN_KERNEL_SIZE}" -le 0 ]] \
	|| (( READOUT_CNN_KERNEL_SIZE % 2 == 0 )); then
	echo "[tune_dnabert_pair_time.sh] READOUT_CNN_KERNEL_SIZE must be a positive odd integer." >&2
	exit 1
fi
if ! [[ "${READOUT_MLP_HIDDEN_DIM}" =~ ^[0-9]+$ ]] \
	|| [[ "${READOUT_MLP_HIDDEN_DIM}" -le 0 ]]; then
	echo "[tune_dnabert_pair_time.sh] READOUT_MLP_HIDDEN_DIM must be a positive integer." >&2
	exit 1
fi
if ! [[ "${READOUT_MLP_LAYERS}" =~ ^[0-9]+$ ]] \
	|| [[ "${READOUT_MLP_LAYERS}" -le 0 ]]; then
	echo "[tune_dnabert_pair_time.sh] READOUT_MLP_LAYERS must be a positive integer." >&2
	exit 1
fi
if [[ "${TRUNC_MODE}" == "on" ]]; then
	trunc_bp="${DONOR_LEN}"
	if (( ACCEPTOR_LEN > DONOR_LEN )); then
		trunc_bp="${ACCEPTOR_LEN}"
	fi
	if [[ -z "${TRAIN_POS_PATH}" ]]; then
		TRAIN_POS_PATH="data/{species}/processed/${trunc_bp}bp_trimmed_npad.err"
	fi
	if [[ -z "${TRAIN_NEG_PATH}" ]]; then
		TRAIN_NEG_PATH="data/{species}/processed/${trunc_bp}bp_trimmed_npad.neg.err"
	fi
	if [[ -z "${TAG}" ]]; then
		TAG="trunc"
	fi
	if [[ "${NAME_FIELDS}" == "none" || -z "${NAME_FIELDS}" ]]; then
		NAME_FIELDS="tag"
	elif [[ ",${NAME_FIELDS}," != *",tag,"* ]]; then
		NAME_FIELDS="${NAME_FIELDS},tag"
	fi
fi
if [[ "${CHEAT_MODE}" == "on" ]]; then
	if [[ -z "${TAG}" ]]; then
		TAG="cheat"
	elif [[ "${TAG}" != *"cheat"* ]]; then
		TAG="${TAG}_cheat"
	fi
	if [[ "${NAME_FIELDS}" == "none" || -z "${NAME_FIELDS}" ]]; then
		NAME_FIELDS="tag"
	elif [[ ",${NAME_FIELDS}," != *",tag,"* ]]; then
		NAME_FIELDS="${NAME_FIELDS},tag"
	fi
fi

PYTHON_BIN="$(resolve_python_bin)"
MODEL_NAME=""
PRETRAINED_MODEL_NAME_RESOLVED=""
resolve_dnabert_model \
	"${DNABERT_VARIANT}" \
	"${PRETRAINED_MODEL_NAME}" \
	"${MODEL_ROOT}" \
	"${PRETRAINED_MODEL_RELATIVE_PATH_2}" \
	"${PRETRAINED_MODEL_RELATIVE_PATH_6}" \
	"${PRETRAINED_MODEL_RELATIVE_PATH_S}"
TUNING_MODEL_NAME="${MODEL_NAME}_pair"
if [[ "${TRUNC_MODE}" == "on" ]]; then
	TUNING_MODEL_NAME="${TUNING_MODEL_NAME}_trunc"
fi
if [[ "${CHEAT_MODE}" == "on" ]]; then
	TUNING_MODEL_NAME="${TUNING_MODEL_NAME}_cheat"
fi
if [[ "${TRUST_REMOTE_CODE}" != "0" && "${TRUST_REMOTE_CODE}" != "1" ]]; then
	echo "[tune_dnabert_pair_time.sh] TRUST_REMOTE_CODE must be 0 or 1." >&2
	exit 1
fi
START_SECONDS="${SECONDS}"
START_UNIX_SECONDS="$(date +%s)"
BUDGET_SECONDS=$((TIME_BUDGET_MINUTES * 60))
ETA_DEADLINE_EPOCH=$((START_UNIX_SECONDS + BUDGET_SECONDS))
ETA_DEADLINE_LABEL="$(format_eta "${ETA_DEADLINE_EPOCH}")"
RUNTIME_PROCESS_TITLE="$(build_eta_process_title "${ETA_DEADLINE_LABEL}")"
if [[ -n "${PROCESS_TITLE}" ]]; then
	RUNTIME_PROCESS_TITLE="${PROCESS_TITLE}"
fi
START_EPOCH="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TOTAL_CYCLE_SECONDS=0
COMPLETED_CYCLES=0

echo "[tune_dnabert_pair_time.sh] start=${START_EPOCH} budget=${TIME_BUDGET_MINUTES}min"
echo "[tune_dnabert_pair_time.sh] quick+full cycles: "\
	"quick_trials=${QUICK_TRIALS} quick_epochs=${QUICK_EPOCHS} "\
	"top_k=${TOP_K} full_epochs=${FULL_EPOCHS}"
echo "[tune_dnabert_pair_time.sh] schedule=${JOB_ORDER[*]}"

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
			echo "[tune_dnabert_pair_time.sh] stop before next cycle: "\
				"remaining=$(format_elapsed "${remaining_seconds}") "\
				"< avg_cycle=$(format_elapsed "${avg_cycle_seconds_guard}")"
			break
		fi
	fi
	remaining_hms="$(format_elapsed "${remaining_seconds}")"

	scheduled_entry="${JOB_ORDER[$((job_index % ${#JOB_ORDER[@]}))]}"
	raw_species="${scheduled_entry%%:*}"
	target="pair"
	species="$(resolve_species_case "${raw_species}" "${DATA_ROOT}")"
	if [[ "${scheduled_entry}" == *:* ]]; then
		scheduled_target="${scheduled_entry##*:}"
		if [[ "${scheduled_target}" != "pair" ]]; then
			echo "[tune_dnabert_pair_time.sh] invalid target in JOB_ORDER:" \
				"${scheduled_entry}" >&2
			echo "[tune_dnabert_pair_time.sh] Use species or species:pair only." >&2
			exit 1
		fi
	fi

	run_stamp="$(date +%Y%m%d_%H%M%S)"
	run_id="${run_stamp}_c$(printf '%03d' "${job_index}")"
	output_dir="${DATA_ROOT}/${species}/tuning/${TUNING_MODEL_NAME}/pair/${run_id}"
	global_best_path="${DATA_ROOT}/${species}/tuning/${TUNING_MODEL_NAME}/pair/best_config.json"
	SEED_BEST_CONFIG_PATH=""
	if ! SEED_BEST_CONFIG_PATH="$(
		resolve_cross_species_best_seed \
			"tune_dnabert_pair_time.sh" \
			"${PYTHON_BIN}" \
			"${DATA_ROOT}" \
			"${TUNING_MODEL_NAME}" \
			"${species}" \
			"${target}" \
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
	resolved_objective_metric="pair_${OBJECTIVE_METRIC}"
	if [[ "${CHEAT_MODE}" == "on" ]]; then
		resolved_objective_metric="test_${OBJECTIVE_METRIC}"
	fi
	config_path="${output_dir}/hparam_search_config.json"
	mkdir -p "${output_dir}"
	TAG_JSON="$(intronmodel_json_string_or_null "${PYTHON_BIN}" "${TAG}")"
	resolved_train_paths="$(
		intronmodel_resolve_and_validate_train_paths \
			"tune_dnabert_pair_time.sh" \
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
	target_search_space_json="${DEFAULT_SEARCH_SPACE_JSON_PAIR}"
	search_space_path=""
	if search_space_resolved="$(
		resolve_search_space_file \
			"${SEARCH_SPACE_FILE}" \
			"${PROJECT_ROOT}" \
			"${species}" \
			"pair" \
			"${TUNING_MODEL_NAME}"
	)"; then
		search_space_path="${search_space_resolved}"
		if ! target_space_json="$(
			normalize_json_object_file \
				"${PYTHON_BIN}" \
				"${search_space_path}" 2>&1
		)"; then
			echo "[tune_dnabert_pair_time.sh] failed to parse search-space file: "\
				"${search_space_path}" >&2
			echo "[tune_dnabert_pair_time.sh] parse detail: ${target_space_json}" >&2
			exit 1
		fi
		target_search_space_json="${target_space_json}"
		echo "[tune_dnabert_pair_time.sh] using search space: ${search_space_path}"
	else
		search_space_status=$?
		if [[ "${search_space_status}" -eq 2 ]]; then
			exit 1
		fi
		echo "[tune_dnabert_pair_time.sh] using embedded pair search space."
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
  "trial_process_mode": "${TRIAL_PROCESS_MODE}",
  "objective_metric": "${resolved_objective_metric}",
  "global_best_config_path": "${global_best_path}",
  "seed_best_config_path": ${SEED_BEST_CONFIG_JSON},
  "search_algo": "${SEARCH_ALGO}",
  "history_top_n": ${HISTORY_TOP_N},
  "guided_random_fraction": ${GUIDED_RANDOM_FRACTION},
  "guided_mutation_rate": ${GUIDED_MUTATION_RATE},
  "min_batch_size": ${MIN_BATCH_SIZE},
  "max_oom_retries": ${MAX_OOM_RETRIES},
  "base_args": {
    "model": "${MODEL_NAME}_pair",
    "species": "${species}",
    "train_target": "pair",
    "tag": ${TAG_JSON},
    "seed": ${BASE_SEED},
    "donor_len": ${DONOR_LEN},
    "acceptor_len": ${ACCEPTOR_LEN},
    "val_frac": ${VAL_FRAC},
    "lr_schedule": "${LR_SCHEDULE}",
    "warmup_ratio": ${WARMUP_RATIO},
    "adam_beta1": ${ADAM_BETA1},
    "adam_beta2": ${ADAM_BETA2},
    "adam_eps": ${ADAM_EPS},
    "head_layer_norm": ${HEAD_LAYER_NORM},
    "readout_type": "${READOUT_TYPE}",
    "readout_cnn_kernel_size": ${READOUT_CNN_KERNEL_SIZE},
    "readout_mlp_hidden_dim": ${READOUT_MLP_HIDDEN_DIM},
    "readout_mlp_layers": ${READOUT_MLP_LAYERS},
    "pretrained_model_name": "${PRETRAINED_MODEL_NAME_RESOLVED}",
    "pretrained_revision": "${PRETRAINED_REVISION}",
    "trust_remote_code": ${TRUST_REMOTE_CODE},
    "device": "${DEVICE}",
    "visualize": "${VISUALIZE}",
    "name_fields": "${NAME_FIELDS}",
    "use_amp": ${USE_AMP},
    "amp_dtype": "${AMP_DTYPE}",
    "infer_batch_size": ${INFER_BATCH_SIZE},
    "infer_use_amp": ${INFER_USE_AMP},
    "infer_amp_dtype": "${INFER_AMP_DTYPE}",
    "infer_compile": ${INFER_COMPILE},
    "infer_compile_mode": "${INFER_COMPILE_MODE}",
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
	printf '[tune_dnabert_pair_time.sh] cycle=%s elapsed=%s start=%s ' \
		"${job_index}" "${job_elapsed_hms}" "${job_start}"
	printf 'ETA_remaining=%s species=%s target=%s\n' \
		"${remaining_hms}" "${species}" "${target}"
	if ! intronmodel_run_with_process_title \
		"${RUNTIME_PROCESS_TITLE}" \
		"${PYTHON_BIN}" \
		"${PROJECT_ROOT}/src/tools/hparam_search.py" \
		--config "${config_path}"; then
		echo "[tune_dnabert_pair_time.sh] cycle=${job_index} failed "\
			"species=${species} target=${target}" >&2
	fi
	if [[ "${UPDATE_DOUBLE_DESCENT_PLOT}" == "1" ]]; then
		run_double_descent_plot \
			"${PYTHON_BIN}" \
			"${PROJECT_ROOT}" \
			"${species}" \
			"${target}" \
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
	printf '[tune_dnabert_pair_time.sh] cycle_done=%s cycle_time=%s avg_cycle=%s ' \
		"${job_index}" \
		"$(format_elapsed "${cycle_duration_seconds}")" \
		"$(format_elapsed "${avg_cycle_seconds}")"
	printf 'ETA_cycles_left=%s\n' "${estimated_cycles_left}"

	job_index=$((job_index + 1))
done

if [[ "${UPDATE_DOUBLE_DESCENT_PLOT}" == "1" ]]; then
	final_plot_species=()
	for scheduled_entry in "${JOB_ORDER[@]}"; do
		species_name="${scheduled_entry%%:*}"
		if [[ " ${final_plot_species[*]} " != *" ${species_name} "* ]]; then
			final_plot_species+=("${species_name}")
		fi
	done
	for final_species in "${final_plot_species[@]}"; do
		run_double_descent_plot \
			"${PYTHON_BIN}" \
			"${PROJECT_ROOT}" \
			"${final_species}" \
			"pair" \
			"${TUNING_MODEL_NAME}"
	done
fi

END_EPOCH="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TOTAL_SECONDS=$((SECONDS - START_SECONDS))
TOTAL_HMS="$(format_elapsed "${TOTAL_SECONDS}")"
echo "[tune_dnabert_pair_time.sh] done start=${START_EPOCH} end=${END_EPOCH} "\
	"elapsed=${TOTAL_HMS} cycles=${job_index}"
