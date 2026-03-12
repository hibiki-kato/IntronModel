#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
	echo "[tune_cnn.sh] This script is config-only. Edit top CONFIG and run without args." >&2
	exit 1
fi

# --------------------------
# CONFIG (edit here)
# --------------------------
# Frequently edited knobs are intentionally placed first in this block.
# Advanced fallback defaults are kept below.
SPECIES="Hsap"
DONOR_LEN="100"
ACCEPTOR_LEN="100"
VAL_FRAC="0.1"
BASE_SEED="1337"
# Deprecated: SEED_LIST is ignored. Only BASE_SEED is used.
SEED_LIST=""
PROCESS_TITLE="use? email me"

QUICK_TRIALS="32"
QUICK_EPOCHS="3"
TOP_K="8"
FULL_EPOCHS="10"
TUNE_TARGETS="donor,acceptor"
QUICK_TRIALS_MODE="fixed"
TARGET_TIME_BUDGET_MINUTES="0"
TOTAL_TIME_BUDGET_MINUTES="0"
MIN_QUICK_TRIALS="8"
QUICK_TRIAL_SEC_FALLBACK="12.0"
SEARCH_ALGO="history_guided"
HISTORY_TOP_N="128"
GUIDED_RANDOM_FRACTION="0.35"
GUIDED_MUTATION_RATE="0.25"

GPU_IDS="auto"
MAX_PARALLEL_TRIALS="auto"
TRIAL_PROCESS_MODE="persistent_all"

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
MAX_OOM_RETRIES="8"
MAX_MODEL_PARAMS="auto"
MAX_MODEL_PARAMS_FALLBACK="30000000"
MAX_MODEL_PARAMS_MEM_FRACTION="0.80"
MAX_MODEL_PARAMS_RESERVE_MIB="2048"
MAX_MODEL_PARAMS_BYTES_PER_PARAM="32"
MAX_MODEL_PARAMS_MODEL_FACTOR="1.00"
SEARCH_SPACE_FILE="auto"

CROSS_SPECIES_BEST_MODE="auto"
CROSS_SPECIES_BEST_OVERRIDE=""
CROSS_SPECIES_BEST_PREFERRED_SPECIES=""

VISUALIZE="none"
NAME_FIELDS="none"
# Optional output/data overrides for tagged or mask-data tuning runs.
TAG=""
TRAIN_POS_PATH=""
TRAIN_NEG_PATH=""
MASK_MODE="off"
MAX_POOL_SIZE="2"
CONV_STRIDE="1"
HEAD_TYPE="gap"

DEFAULT_SEARCH_SPACE_JSON_DONOR="$(cat <<'JSON'
{
  "donor_len": {"type": "int", "min": 40, "max": 100, "step": 10},
  "lr": {"type": "float", "min": 1e-4, "max": 3e-3, "scale": "log"},
  "batch_size": {
    "type": "categorical",
    "values": [128, 256, 512, 1024, 2048, 4096]
  },
  "dropout": {"type": "float", "min": 0.0, "max": 0.7, "scale": "linear"},
  "weight_decay": {"type": "float", "min": 1e-6, "max": 1e-2, "scale": "log"},
  "loss": {
    "type": "categorical",
    "values": ["weighted_bce", "focal", "asymmetric_focal", "f1", "weighted_bce_f1", "focal_f1"]
  },
  "f1_lambda": {"type": "float", "min": 0.02, "max": 0.5, "scale": "log"},
  "conv_depth": {
    "type": "categorical",
    "values": [2, 3, 4, 5]
  },
  "channel_candidates": {
    "type": "categorical",
    "values": [
      "64,96,128,192,256,384,512,768,1024"
    ]
  },
  "kernel_candidates": {
    "type": "categorical",
    "values": [
      "3,5,7,9,11,13,15,17,19"
    ]
  },
  "max_pool_size": {
    "type": "categorical",
    "values": [1, 2, 3, 4]
  },
  "conv_stride": {
    "type": "categorical",
    "values": [1, 2]
  },
  "head_type": {
    "type": "categorical",
    "values": ["gap", "center"]
  },
  "fc_hidden": {
    "type": "categorical",
    "values": [64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096]
  }
}
JSON
)"

DEFAULT_SEARCH_SPACE_JSON_ACCEPTOR="$(cat <<'JSON'
{
  "acceptor_len": {"type": "int", "min": 40, "max": 100, "step": 10},
  "lr": {"type": "float", "min": 8e-5, "max": 3e-3, "scale": "log"},
  "batch_size": {
    "type": "categorical",
    "values": [128, 256, 512, 1024, 2048, 4096, 8192]
  },
  "dropout": {"type": "float", "min": 0.0, "max": 0.55, "scale": "linear"},
  "weight_decay": {"type": "float", "min": 1e-8, "max": 2e-2, "scale": "log"},
  "loss": {
    "type": "categorical",
    "values": ["weighted_bce", "focal", "asymmetric_focal", "f1", "weighted_bce_f1", "focal_f1"]
  },
  "f1_lambda": {"type": "float", "min": 0.02, "max": 0.5, "scale": "log"},
  "conv_depth": {
    "type": "categorical",
    "values": [2, 3, 4, 5]
  },
  "channel_candidates": {
    "type": "categorical",
    "values": [
      "64,96,128,192,256,384,512,768,1024"
    ]
  },
  "kernel_candidates": {
    "type": "categorical",
    "values": [
      "3,5,7,9,11,13,15,17,19"
    ]
  },
  "max_pool_size": {
    "type": "categorical",
    "values": [1, 2, 3, 4]
  },
  "conv_stride": {
    "type": "categorical",
    "values": [1, 2]
  },
  "head_type": {
    "type": "categorical",
    "values": ["gap", "center"]
  },
  "fc_hidden": {
    "type": "categorical",
    "values": [128, 256, 512, 1024, 1536, 2048, 3072, 4096, 6144]
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
INTRONMODEL_AUTO_TMUX="off"
intronmodel_enable_auto_tmux "${PROJECT_ROOT}" "$0" "${BASH_SOURCE[0]##*/}"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/tuning_cross_species_best.sh"

format_elapsed() {
	intronmodel_format_elapsed "$1"
}

intronmodel_start_timer "tune_cnn.sh"
trap 'intronmodel_print_timing' EXIT

resolve_species_case() {
	intronmodel_resolve_species_case "$1" "$2" "tune_cnn.sh"
}

resolve_tune_targets() {
	intronmodel_resolve_tune_targets "$1" "tune_cnn.sh"
}

resolve_python_bin() {
	intronmodel_resolve_python_bin "tune_cnn.sh"
}

resolve_seed_list() {
	intronmodel_resolve_seed_list \
		"tune_cnn.sh" \
		"${BASE_SEED}" \
		"${SEED_LIST}" \
		"${PYTHON_BIN}"
}

resolve_search_space_file() {
	local explicit_file="$1"
	local project_root="$2"
	local species="$3"
	local target="$4"
	local tuning_model_name="$5"

	if [[ -n "${explicit_file}" && "${explicit_file}" != "auto" ]]; then
		if [[ -f "${explicit_file}" ]]; then
			printf '%s\n' "${explicit_file}"
			return 0
		fi
		echo "[tune_cnn.sh] SEARCH_SPACE_FILE not found: ${explicit_file}" >&2
		return 2
	fi

	local target_file="${DATA_ROOT}/${species}/tuning/${tuning_model_name}/${target}/search_space.json"
	if [[ -f "${target_file}" ]]; then
		printf '%s\n' "${target_file}"
		return 0
	fi

	local species_file="${DATA_ROOT}/${species}/tuning/${tuning_model_name}/search_space.json"
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

estimate_trials_for_budget() {
	local python_bin="$1"
	local data_root="$2"
	local species="$3"
	local target="$4"
	local budget_minutes="$5"
	local top_k="$6"
	local quick_epochs="$7"
	local full_epochs="$8"
	local min_quick_trials="$9"
	local fallback_quick_sec="${10}"

	"${python_bin}" - \
		"${data_root}" \
		"${species}" \
		"${target}" \
		"${budget_minutes}" \
		"${top_k}" \
		"${quick_epochs}" \
		"${full_epochs}" \
		"${min_quick_trials}" \
		"${fallback_quick_sec}" <<'PY'
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path


def mean_duration(path_pattern: Path) -> tuple[float | None, int]:
    durations: list[float] = []
    for tsv_path in sorted(path_pattern.glob("*/quick_trials.tsv")):
        with tsv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                if row.get("status") != "success":
                    continue
                raw_value = row.get("duration_sec", "")
                try:
                    value = float(raw_value)
                except ValueError:
                    continue
                if math.isfinite(value) and value > 0.0:
                    durations.append(value)
    if not durations:
        return None, 0
    return sum(durations) / len(durations), len(durations)


def mean_duration_full(path_pattern: Path) -> tuple[float | None, int]:
    durations: list[float] = []
    for tsv_path in sorted(path_pattern.glob("*/full_trials.tsv")):
        with tsv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                if row.get("status") != "success":
                    continue
                raw_value = row.get("duration_sec", "")
                try:
                    value = float(raw_value)
                except ValueError:
                    continue
                if math.isfinite(value) and value > 0.0:
                    durations.append(value)
    if not durations:
        return None, 0
    return sum(durations) / len(durations), len(durations)


data_root = Path(sys.argv[1])
species = sys.argv[2]
target = sys.argv[3]
budget_minutes = int(sys.argv[4])
top_k = int(sys.argv[5])
quick_epochs = int(sys.argv[6])
full_epochs = int(sys.argv[7])
min_quick_trials = int(sys.argv[8])
fallback_quick_sec = float(sys.argv[9])

if budget_minutes <= 0:
    raise ValueError("budget_minutes must be > 0")
if top_k <= 0:
    raise ValueError("top_k must be > 0")
if quick_epochs <= 0:
    raise ValueError("quick_epochs must be > 0")
if full_epochs <= 0:
    raise ValueError("full_epochs must be > 0")
if min_quick_trials <= 0:
    raise ValueError("min_quick_trials must be > 0")
if not math.isfinite(fallback_quick_sec) or fallback_quick_sec <= 0.0:
    raise ValueError("fallback_quick_sec must be > 0")

target_root = data_root / species / "tuning" / "cnn" / target
quick_mean, quick_hist_n = mean_duration(target_root)
if quick_mean is None:
    quick_mean = fallback_quick_sec

full_mean, full_hist_n = mean_duration_full(target_root)
if full_mean is None:
    full_scale = float(full_epochs) / float(quick_epochs)
    full_mean = quick_mean * full_scale

budget_seconds = float(budget_minutes * 60)
reserved_full_seconds = float(top_k) * float(full_mean)
usable_quick_seconds = budget_seconds - reserved_full_seconds
if usable_quick_seconds <= 0.0:
    quick_trials = min_quick_trials
else:
    quick_trials = int(usable_quick_seconds // quick_mean)
    quick_trials = max(min_quick_trials, quick_trials)

projected_total_seconds = (quick_trials * quick_mean) + reserved_full_seconds
print(
    f"{quick_trials}\t{quick_mean:.4f}\t{full_mean:.4f}\t"
    f"{quick_hist_n}\t{full_hist_n}\t{projected_total_seconds:.4f}"
)
PY
}

SPECIES="$(resolve_species_case "${SPECIES}" "${DATA_ROOT}")"
mapfile -t TARGET_LIST < <(resolve_tune_targets "${TUNE_TARGETS}")
PYTHON_BIN="$(resolve_python_bin)"
mapfile -t SEED_VALUES < <(resolve_seed_list)
RESOLVED_MAX_MODEL_PARAMS="$(
	intronmodel_resolve_max_model_params \
		"tune_cnn.sh" \
		"${MAX_MODEL_PARAMS}" \
		"${GPU_IDS}" \
		"${MAX_MODEL_PARAMS_FALLBACK}" \
		"${MAX_MODEL_PARAMS_MEM_FRACTION}" \
		"${MAX_MODEL_PARAMS_RESERVE_MIB}" \
		"${MAX_MODEL_PARAMS_BYTES_PER_PARAM}" \
		"${MAX_MODEL_PARAMS_MODEL_FACTOR}" \
		"${PYTHON_BIN}"
)"

if [[ "${QUICK_TRIALS_MODE}" != "fixed" && "${QUICK_TRIALS_MODE}" != "budget" ]]; then
	echo "[tune_cnn.sh] QUICK_TRIALS_MODE must be fixed|budget." >&2
	exit 1
fi
if [[ "${SEARCH_ALGO}" != "random" && "${SEARCH_ALGO}" != "history_guided" ]]; then
	echo "[tune_cnn.sh] SEARCH_ALGO must be random|history_guided." >&2
	exit 1
fi
if [[ "${TRIAL_PROCESS_MODE}" != "subprocess" && \
	"${TRIAL_PROCESS_MODE}" != "persistent_quick" && \
	"${TRIAL_PROCESS_MODE}" != "persistent_all" ]]; then
	echo "[tune_cnn.sh] TRIAL_PROCESS_MODE must be "\
		"subprocess|persistent_quick|persistent_all." >&2
	exit 1
fi
if ! [[ "${QUICK_TRIALS}" =~ ^[0-9]+$ ]] || [[ "${QUICK_TRIALS}" -le 0 ]]; then
	echo "[tune_cnn.sh] QUICK_TRIALS must be a positive integer." >&2
	exit 1
fi
if ! [[ "${HISTORY_TOP_N}" =~ ^[0-9]+$ ]] || [[ "${HISTORY_TOP_N}" -le 0 ]]; then
	echo "[tune_cnn.sh] HISTORY_TOP_N must be a positive integer." >&2
	exit 1
fi
if ! [[ "${MIN_QUICK_TRIALS}" =~ ^[0-9]+$ ]] || [[ "${MIN_QUICK_TRIALS}" -le 0 ]]; then
	echo "[tune_cnn.sh] MIN_QUICK_TRIALS must be a positive integer." >&2
	exit 1
fi
if ! [[ "${TARGET_TIME_BUDGET_MINUTES}" =~ ^[0-9]+$ ]]; then
	echo "[tune_cnn.sh] TARGET_TIME_BUDGET_MINUTES must be an integer." >&2
	exit 1
fi
if ! [[ "${TOTAL_TIME_BUDGET_MINUTES}" =~ ^[0-9]+$ ]]; then
	echo "[tune_cnn.sh] TOTAL_TIME_BUDGET_MINUTES must be an integer." >&2
	exit 1
fi
if ! [[ "${QUICK_TRIAL_SEC_FALLBACK}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
	echo "[tune_cnn.sh] QUICK_TRIAL_SEC_FALLBACK must be a positive number." >&2
	exit 1
fi
if ! [[ "${GUIDED_RANDOM_FRACTION}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
	echo "[tune_cnn.sh] GUIDED_RANDOM_FRACTION must be numeric in [0,1]." >&2
	exit 1
fi
if ! awk -v x="${GUIDED_RANDOM_FRACTION}" 'BEGIN{exit !(x>=0 && x<=1)}'; then
	echo "[tune_cnn.sh] GUIDED_RANDOM_FRACTION must be in [0,1]." >&2
	exit 1
fi
if ! [[ "${GUIDED_MUTATION_RATE}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
	echo "[tune_cnn.sh] GUIDED_MUTATION_RATE must be numeric in [0,1]." >&2
	exit 1
fi
if ! awk -v x="${GUIDED_MUTATION_RATE}" 'BEGIN{exit !(x>=0 && x<=1)}'; then
	echo "[tune_cnn.sh] GUIDED_MUTATION_RATE must be in [0,1]." >&2
	exit 1
fi
if [[ "${QUICK_TRIALS_MODE}" == "budget" ]]; then
	if [[ "${TARGET_TIME_BUDGET_MINUTES}" -le 0 && "${TOTAL_TIME_BUDGET_MINUTES}" -le 0 ]]; then
		echo "[tune_cnn.sh] budget mode requires TARGET_TIME_BUDGET_MINUTES>0 "\
			"or TOTAL_TIME_BUDGET_MINUTES>0." >&2
		exit 1
	fi
fi

if [[ "${MASK_MODE}" != "off" && "${MASK_MODE}" != "on" ]]; then
	echo "[tune_cnn.sh] MASK_MODE must be off|on." >&2
	exit 1
fi
if [[ "${MASK_MODE}" == "on" ]]; then
	mask_bp="${DONOR_LEN}"
	if (( ACCEPTOR_LEN > DONOR_LEN )); then
		mask_bp="${ACCEPTOR_LEN}"
	fi
	if [[ -z "${TRAIN_POS_PATH}" ]]; then
		TRAIN_POS_PATH="data/{species}/processed/${mask_bp}bp_trimmed_npad.err"
	fi
	if [[ -z "${TRAIN_NEG_PATH}" ]]; then
		TRAIN_NEG_PATH="data/{species}/processed/${mask_bp}bp_trimmed_npad.neg.err"
	fi
	if [[ -z "${TAG}" ]]; then
		TAG="mask"
	fi
	if [[ "${NAME_FIELDS}" == "none" || -z "${NAME_FIELDS}" ]]; then
		NAME_FIELDS="tag"
	elif [[ ",${NAME_FIELDS}," != *",tag,"* ]]; then
		NAME_FIELDS="${NAME_FIELDS},tag"
	fi
fi

TUNING_MODEL_NAME="cnn"
if [[ "${MASK_MODE}" == "on" ]]; then
	TUNING_MODEL_NAME="cnn_mask"
fi

RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
echo "[tune_cnn.sh] species=${SPECIES}"
echo "[tune_cnn.sh] targets=${TARGET_LIST[*]}"
echo "[tune_cnn.sh] seeds=${SEED_VALUES[*]}"

for TARGET in "${TARGET_LIST[@]}"; do
	OBJECTIVE_METRIC="${TARGET}_pr_auc"
	GLOBAL_BEST_CONFIG_PATH="${DATA_ROOT}/${SPECIES}/tuning/${TUNING_MODEL_NAME}/${TARGET}/best_config.json"
	SEED_BEST_CONFIG_PATH=""
	if ! SEED_BEST_CONFIG_PATH="$(
		resolve_cross_species_best_seed \
			"tune_cnn.sh" \
			"${PYTHON_BIN}" \
			"${DATA_ROOT}" \
			"${TUNING_MODEL_NAME}" \
			"${SPECIES}" \
			"${TARGET}" \
			"${GLOBAL_BEST_CONFIG_PATH}" \
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
	QUICK_TRIALS_TARGET="${QUICK_TRIALS}"
	TARGET_SEARCH_SPACE_JSON="${DEFAULT_SEARCH_SPACE_JSON_DONOR}"
	if [[ "${TARGET}" == "acceptor" ]]; then
		TARGET_SEARCH_SPACE_JSON="${DEFAULT_SEARCH_SPACE_JSON_ACCEPTOR}"
	fi

	if [[ "${QUICK_TRIALS_MODE}" == "budget" ]]; then
		TARGET_BUDGET_MINUTES="${TARGET_TIME_BUDGET_MINUTES}"
		if [[ "${TARGET_BUDGET_MINUTES}" -le 0 ]]; then
			TARGET_COUNT="${#TARGET_LIST[@]}"
			TARGET_BUDGET_MINUTES=$((TOTAL_TIME_BUDGET_MINUTES / TARGET_COUNT))
		fi
		if [[ "${TARGET_BUDGET_MINUTES}" -le 0 ]]; then
			echo "[tune_cnn.sh] target budget computed as 0 minutes. "\
				"Increase TOTAL_TIME_BUDGET_MINUTES." >&2
			exit 1
		fi
		if ! estimate_result="$(
			estimate_trials_for_budget \
				"${PYTHON_BIN}" \
				"${DATA_ROOT}" \
				"${SPECIES}" \
				"${TARGET}" \
				"${TARGET_BUDGET_MINUTES}" \
				"${TOP_K}" \
				"${QUICK_EPOCHS}" \
				"${FULL_EPOCHS}" \
				"${MIN_QUICK_TRIALS}" \
				"${QUICK_TRIAL_SEC_FALLBACK}" 2>&1
		)"; then
			echo "[tune_cnn.sh] failed to estimate quick trials from budget." >&2
			echo "[tune_cnn.sh] estimator detail: ${estimate_result}" >&2
			exit 1
		fi
		IFS=$'\t' read -r \
			QUICK_TRIALS_TARGET \
			EST_QUICK_SEC \
			EST_FULL_SEC \
			EST_QUICK_N \
			EST_FULL_N \
			EST_TOTAL_SEC <<< "${estimate_result}"
		EST_TOTAL_HMS="$(format_elapsed "${EST_TOTAL_SEC%.*}")"
		echo "[tune_cnn.sh] target=${TARGET} budget=${TARGET_BUDGET_MINUTES}min "\
			"quick_trials=${QUICK_TRIALS_TARGET} "\
			"est_quick=${EST_QUICK_SEC}s est_full=${EST_FULL_SEC}s "\
			"hist_quick_n=${EST_QUICK_N} hist_full_n=${EST_FULL_N} "\
			"projected_total=${EST_TOTAL_HMS}"
	fi

	search_space_path=""
	if search_space_resolved="$(
		resolve_search_space_file \
			"${SEARCH_SPACE_FILE}" \
			"${PROJECT_ROOT}" \
			"${SPECIES}" \
			"${TARGET}" \
			"${TUNING_MODEL_NAME}"
	)"; then
		search_space_path="${search_space_resolved}"
		if ! target_space_json="$(
			normalize_json_object_file \
				"${PYTHON_BIN}" \
				"${search_space_path}" 2>&1
		)"; then
			echo "[tune_cnn.sh] failed to parse search-space file: ${search_space_path}" >&2
			echo "[tune_cnn.sh] parse detail: ${target_space_json}" >&2
			exit 1
		fi
		TARGET_SEARCH_SPACE_JSON="${target_space_json}"
		echo "[tune_cnn.sh] target=${TARGET} search_space_file=${search_space_path}"
	else
		search_space_status=$?
		if [[ "${search_space_status}" -eq 2 ]]; then
			exit 1
		fi
		echo "[tune_cnn.sh] target=${TARGET} search_space_file=<embedded_${TARGET}>"
	fi

	for base_seed in "${SEED_VALUES[@]}"; do
		OUTPUT_DIR="${DATA_ROOT}/${SPECIES}/tuning/${TUNING_MODEL_NAME}/${TARGET}/${RUN_TIMESTAMP}_seed${base_seed}"
		mkdir -p "${OUTPUT_DIR}"
		TARGET_START_EPOCH="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
		TARGET_START_SECONDS="${SECONDS}"
		CONFIG_PATH="${OUTPUT_DIR}/hparam_search_config.json"
		TAG_JSON="$(intronmodel_json_string_or_null "${PYTHON_BIN}" "${TAG}")"
		TRAIN_POS_PATH_JSON="$(
			intronmodel_json_string_or_null \
				"${PYTHON_BIN}" \
				"$(intronmodel_resolve_species_template "${TRAIN_POS_PATH}" "${SPECIES}")"
		)"
		TRAIN_NEG_PATH_JSON="$(
			intronmodel_json_string_or_null \
				"${PYTHON_BIN}" \
				"$(intronmodel_resolve_species_template "${TRAIN_NEG_PATH}" "${SPECIES}")"
		)"
		cat > "${CONFIG_PATH}" <<JSON
{
  "project_root": "${PROJECT_ROOT}",
  "species": "${SPECIES}",
  "output_dir": "${OUTPUT_DIR}",
  "quick_trials": ${QUICK_TRIALS_TARGET},
  "quick_epochs": ${QUICK_EPOCHS},
  "top_k": ${TOP_K},
  "full_epochs": ${FULL_EPOCHS},
  "base_seed": ${base_seed},
  "gpu_ids": "${GPU_IDS}",
  "max_parallel_trials": "${MAX_PARALLEL_TRIALS}",
  "trial_process_mode": "${TRIAL_PROCESS_MODE}",
  "objective_metric": "${OBJECTIVE_METRIC}",
  "global_best_config_path": "${GLOBAL_BEST_CONFIG_PATH}",
  "seed_best_config_path": ${SEED_BEST_CONFIG_JSON},
  "search_algo": "${SEARCH_ALGO}",
  "history_top_n": ${HISTORY_TOP_N},
  "guided_random_fraction": ${GUIDED_RANDOM_FRACTION},
  "guided_mutation_rate": ${GUIDED_MUTATION_RATE},
  "min_batch_size": ${MIN_BATCH_SIZE},
  "max_oom_retries": ${MAX_OOM_RETRIES},
  "max_model_params": ${RESOLVED_MAX_MODEL_PARAMS},
  "base_args": {
    "model": "cnn",
    "species": "${SPECIES}",
    "train_target": "${TARGET}",
    "seed": ${base_seed},
    "donor_len": ${DONOR_LEN},
    "acceptor_len": ${ACCEPTOR_LEN},
    "val_frac": ${VAL_FRAC},
    "conv_depth": 3,
    "channel_candidates": "64,96,128,192,256,384,512",
    "kernel_candidates": "3,5,7,9,11,13,15",
    "max_pool_size": ${MAX_POOL_SIZE},
    "conv_stride": ${CONV_STRIDE},
    "head_type": "${HEAD_TYPE}",
    "device": "${DEVICE}",
    "visualize": "${VISUALIZE}",
    "name_fields": "${NAME_FIELDS}",
    "tag": ${TAG_JSON},
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
    "compile_mode": "off"
  },
  "full_overrides": {
    "epochs": ${FULL_EPOCHS},
    "compile_mode": "off"
  },
  "search_space": ${TARGET_SEARCH_SPACE_JSON}
}
JSON

		echo "[tune_cnn.sh] target=${TARGET} seed=${base_seed}"
		echo "[tune_cnn.sh] output_dir=${OUTPUT_DIR}"
			if ! intronmodel_run_with_process_title \
				"${PROCESS_TITLE}" \
				"${PYTHON_BIN}" \
				"${PROJECT_ROOT}/src/tools/hparam_search.py" \
				--config "${CONFIG_PATH}"; then
			target_elapsed_seconds=$((SECONDS - TARGET_START_SECONDS))
			target_elapsed_hms="$(format_elapsed "${target_elapsed_seconds}")"
			echo "[tune_cnn.sh] target=${TARGET} seed=${base_seed} "\
				"failed start=${TARGET_START_EPOCH} "\
				"elapsed=${target_elapsed_hms} (${target_elapsed_seconds}s)" >&2
			exit 1
			fi
		target_end_epoch="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
		target_elapsed_seconds=$((SECONDS - TARGET_START_SECONDS))
		target_elapsed_hms="$(format_elapsed "${target_elapsed_seconds}")"
		echo "[tune_cnn.sh] target=${TARGET} seed=${base_seed} "\
			"done start=${TARGET_START_EPOCH} end=${target_end_epoch} "\
			"elapsed=${target_elapsed_hms} (${target_elapsed_seconds}s)"
	done
done
