#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
	echo "[tune_cnn.sh] This script is config-only. Edit CONFIG and run without args." >&2
	exit 1
fi

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
DATA_ROOT="${INTRONMODEL_DATA_ROOT:-${PROJECT_ROOT}/data}"
MODEL_ROOT="${INTRONMODEL_MODEL_ROOT:-${PROJECT_ROOT}/model}"
export INTRONMODEL_MODEL_ROOT="${MODEL_ROOT}"
export INTRONMODEL_DATA_ROOT="${DATA_ROOT}"

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
	echo "[tune_cnn.sh] timing: start=${SCRIPT_START_EPOCH} "\
		"end=${script_end_epoch} elapsed=${elapsed_hms} "\
		"(${elapsed_seconds}s) exit=${exit_code}"
	return "${exit_code}"
}

trap 'print_script_timing' EXIT

resolve_species_case() {
	local raw_species="$1"
	local data_root="$2"

	if [[ -d "${data_root}/${raw_species}" ]]; then
		printf '%s\n' "${raw_species}"
		return 0
	fi

	local matches=()
	mapfile -t matches < <(
		find "${data_root}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
			| awk -v target="${raw_species}" 'tolower($0) == tolower(target)'
	)
	if [[ ${#matches[@]} -eq 1 ]]; then
		echo "[tune_cnn.sh] species case normalized: '${raw_species}' -> '${matches[0]}'" >&2
		printf '%s\n' "${matches[0]}"
		return 0
	fi
	if [[ ${#matches[@]} -gt 1 ]]; then
		echo "[tune_cnn.sh] ambiguous species '${raw_species}'." >&2
		printf '[tune_cnn.sh] case-insensitive matches: %s\n' "${matches[*]}" >&2
		return 1
	fi
	printf '%s\n' "${raw_species}"
	return 0
}

resolve_tune_targets() {
	local raw_targets="$1"
	local parts=()
	local resolved=()

	IFS=',' read -r -a parts <<< "${raw_targets}"
	for part in "${parts[@]}"; do
		local target
		target="$(printf '%s' "${part}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
		if [[ -z "${target}" ]]; then
			continue
		fi
		if [[ "${target}" != "donor" && "${target}" != "acceptor" ]]; then
			echo "[tune_cnn.sh] invalid target: ${target}" >&2
			echo "[tune_cnn.sh] TUNE_TARGETS must contain donor and/or acceptor." >&2
			return 1
		fi
		resolved+=("${target}")
	done

	if [[ ${#resolved[@]} -eq 0 ]]; then
		echo "[tune_cnn.sh] no valid tuning targets configured." >&2
		return 1
	fi

	printf '%s\n' "${resolved[@]}"
}

resolve_python_bin() {
	if command -v python3 >/dev/null 2>&1; then
		printf '%s\n' "python3"
		return 0
	fi
	if command -v python >/dev/null 2>&1; then
		printf '%s\n' "python"
		return 0
	fi
	echo "[tune_cnn.sh] python interpreter not found (python3/python)." >&2
	return 1
}

resolve_search_space_file() {
	local explicit_file="$1"
	local project_root="$2"
	local species="$3"
	local target="$4"

	if [[ -n "${explicit_file}" && "${explicit_file}" != "auto" ]]; then
		if [[ -f "${explicit_file}" ]]; then
			printf '%s\n' "${explicit_file}"
			return 0
		fi
		echo "[tune_cnn.sh] SEARCH_SPACE_FILE not found: ${explicit_file}" >&2
		return 2
	fi

	local target_file="${DATA_ROOT}/${species}/tuning/cnn/${target}/search_space.json"
	if [[ -f "${target_file}" ]]; then
		printf '%s\n' "${target_file}"
		return 0
	fi

	local species_file="${DATA_ROOT}/${species}/tuning/cnn/search_space.json"
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

# --------------------------
# CONFIG (edit here)
# --------------------------
SPECIES="Athal"
DONOR_LEN="100"
ACCEPTOR_LEN="100"
BASE_SEED="1337"

QUICK_TRIALS="24"
QUICK_EPOCHS="3"
TOP_K="5"
FULL_EPOCHS="20"
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
SEARCH_SPACE_FILE="auto"

VISUALIZE="none"
NAME_FIELDS="none"

DEFAULT_SEARCH_SPACE_JSON_DONOR="$(cat <<'JSON'
{
  "lr": {"type": "float", "min": 1e-4, "max": 3e-3, "scale": "log"},
  "batch_size": {
    "type": "categorical",
    "values": [128, 256, 512, 1024, 2048, 4096]
  },
  "dropout": {"type": "float", "min": 0.0, "max": 0.5, "scale": "linear"},
  "weight_decay": {"type": "float", "min": 1e-8, "max": 1e-2, "scale": "log"},
  "loss": {
    "type": "categorical",
    "values": ["weighted_bce", "focal", "asymmetric_focal"]
  },
  "conv_channels": {
    "type": "categorical",
    "values": [
      "96,192,384",
      "64,128",
      "64,128,256",
      "64,128,256,512",
      "128,256,512",
      "192,384,768,1536",
      "128,256,512,1024",
      "256,512,1024",
      "256,512,1024,2048",
      "384,768,1536",
      "512,1024,2048",
      "512,1024,2048,3072",
      "384,768,1536,3072",
      "256,512,1024,2048,3072"
    ]
  },
  "kernel_size": {
    "type": "categorical",
    "values": [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
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
  "lr": {"type": "float", "min": 8e-5, "max": 3e-3, "scale": "log"},
  "batch_size": {
    "type": "categorical",
    "values": [128, 256, 512, 1024, 2048, 4096, 8192]
  },
  "dropout": {"type": "float", "min": 0.0, "max": 0.55, "scale": "linear"},
  "weight_decay": {"type": "float", "min": 1e-8, "max": 2e-2, "scale": "log"},
  "loss": {
    "type": "categorical",
    "values": ["weighted_bce", "focal", "asymmetric_focal"]
  },
  "conv_channels": {
    "type": "categorical",
    "values": [
      "128,256,512",
      "128,256,512,1024",
      "192,384,768",
      "192,384,768,1536",
      "256,512,1024",
      "256,512,1024,2048",
      "256,512,1024,2048,3072",
      "384,768,1536",
      "384,768,1536,3072",
      "512,1024,2048",
      "512,1024,2048,3072",
      "768,1536,3072"
    ]
  },
  "kernel_size": {
    "type": "categorical",
    "values": [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]
  },
  "fc_hidden": {
    "type": "categorical",
    "values": [128, 256, 512, 1024, 1536, 2048, 3072, 4096, 6144]
  }
}
JSON
)"

SPECIES="$(resolve_species_case "${SPECIES}" "${DATA_ROOT}")"
mapfile -t TARGET_LIST < <(resolve_tune_targets "${TUNE_TARGETS}")
PYTHON_BIN="$(resolve_python_bin)"

if [[ "${QUICK_TRIALS_MODE}" != "fixed" && "${QUICK_TRIALS_MODE}" != "budget" ]]; then
	echo "[tune_cnn.sh] QUICK_TRIALS_MODE must be fixed|budget." >&2
	exit 1
fi
if [[ "${SEARCH_ALGO}" != "random" && "${SEARCH_ALGO}" != "history_guided" ]]; then
	echo "[tune_cnn.sh] SEARCH_ALGO must be random|history_guided." >&2
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

RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
echo "[tune_cnn.sh] species=${SPECIES}"
echo "[tune_cnn.sh] targets=${TARGET_LIST[*]}"

for TARGET in "${TARGET_LIST[@]}"; do
	OBJECTIVE_METRIC="${TARGET}_pr_auc"
	OUTPUT_DIR="${DATA_ROOT}/${SPECIES}/tuning/cnn/${TARGET}/${RUN_TIMESTAMP}"
	GLOBAL_BEST_CONFIG_PATH="${DATA_ROOT}/${SPECIES}/tuning/cnn/${TARGET}/best_config.json"
	mkdir -p "${OUTPUT_DIR}"
	TARGET_START_EPOCH="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
	TARGET_START_SECONDS="${SECONDS}"
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
			"${TARGET}"
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

	CONFIG_PATH="${OUTPUT_DIR}/hparam_search_config.json"
	cat > "${CONFIG_PATH}" <<JSON
{
  "project_root": "${PROJECT_ROOT}",
  "species": "${SPECIES}",
  "output_dir": "${OUTPUT_DIR}",
  "quick_trials": ${QUICK_TRIALS_TARGET},
  "quick_epochs": ${QUICK_EPOCHS},
  "top_k": ${TOP_K},
  "full_epochs": ${FULL_EPOCHS},
  "base_seed": ${BASE_SEED},
  "gpu_ids": "${GPU_IDS}",
  "max_parallel_trials": "${MAX_PARALLEL_TRIALS}",
  "objective_metric": "${OBJECTIVE_METRIC}",
  "global_best_config_path": "${GLOBAL_BEST_CONFIG_PATH}",
  "search_algo": "${SEARCH_ALGO}",
  "history_top_n": ${HISTORY_TOP_N},
  "guided_random_fraction": ${GUIDED_RANDOM_FRACTION},
  "guided_mutation_rate": ${GUIDED_MUTATION_RATE},
  "min_batch_size": ${MIN_BATCH_SIZE},
  "max_oom_retries": ${MAX_OOM_RETRIES},
  "base_args": {
    "model": "cnn",
    "species": "${SPECIES}",
    "train_target": "${TARGET}",
    "donor_len": ${DONOR_LEN},
    "acceptor_len": ${ACCEPTOR_LEN},
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
    "train_pos_path": null,
    "train_neg_path": null
  },
  "quick_overrides": {
    "epochs": ${QUICK_EPOCHS},
    "compile_mode": "off"
  },
  "full_overrides": {
    "epochs": ${FULL_EPOCHS},
    "compile_mode": "auto"
  },
  "search_space": ${TARGET_SEARCH_SPACE_JSON}
}
JSON

	echo "[tune_cnn.sh] target=${TARGET}"
	echo "[tune_cnn.sh] output_dir=${OUTPUT_DIR}"
	if ! "${PYTHON_BIN}" "${PROJECT_ROOT}/src/tools/hparam_search.py" \
		--config "${CONFIG_PATH}"; then
		target_elapsed_seconds=$((SECONDS - TARGET_START_SECONDS))
		target_elapsed_hms="$(format_elapsed "${target_elapsed_seconds}")"
		echo "[tune_cnn.sh] target=${TARGET} failed start=${TARGET_START_EPOCH} "\
			"elapsed=${target_elapsed_hms} (${target_elapsed_seconds}s)" >&2
		exit 1
	fi
	target_end_epoch="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
	target_elapsed_seconds=$((SECONDS - TARGET_START_SECONDS))
	target_elapsed_hms="$(format_elapsed "${target_elapsed_seconds}")"
	echo "[tune_cnn.sh] target=${TARGET} done start=${TARGET_START_EPOCH} "\
		"end=${target_end_epoch} elapsed=${target_elapsed_hms} "\
		"(${target_elapsed_seconds}s)"
done
