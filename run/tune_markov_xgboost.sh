#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
	echo "[tune_markov_xgboost.sh] This script is config-only. " \
		"Edit top CONFIG and run without args." >&2
	exit 1
fi

# --------------------------
# CONFIG (edit here)
# --------------------------
SPECIES="Dmel"
DONOR_LEN="100"
ACCEPTOR_LEN="100"
VAL_FRAC="0.1"
BASE_SEED="1337"

# markov_xgboost has no epoch-wise training schedule, so quick/full split
# gives little benefit. Keep TOP_K=1 to make the full phase a single
# confirmation rerun of the best quick trial.
QUICK_TRIALS="24"
QUICK_EPOCHS="1"
TOP_K="1"
FULL_EPOCHS="1"
SKIP_FULL_PHASE="1"
ENABLE_VISUALIZATION="0"

SEARCH_ALGO="history_guided"
HISTORY_TOP_N="128"
GUIDED_RANDOM_FRACTION="0.35"
GUIDED_MUTATION_RATE="0.25"

GPU_IDS="auto"
MAX_PARALLEL_TRIALS="auto"
TRIAL_PROCESS_MODE="subprocess"
TRIAL_STREAM_MODE="auto"

SEQUENCE_TRANSFORM="none"
MARKOV_ORDER="2"
MARKOV_ALPHA="0.5"
MARKOV_FEATURE_MODE="per_base"
MARKOV_CACHE_MODE="auto"
MARKOV_CACHE_DIR=""

OBJECTIVE_METRIC="pair_pr_auc"
SEARCH_SPACE_FILE="auto"
GLOBAL_BEST_CONFIG_PATH=""
SEED_BEST_CONFIG_PATH=""

XGB_N_ESTIMATORS="300"
XGB_MAX_DEPTH="4"
XGB_LEARNING_RATE="0.05"
XGB_SUBSAMPLE="0.9"
XGB_COLSAMPLE_BYTREE="1.0"
XGB_MIN_CHILD_WEIGHT="1.0"
XGB_REG_LAMBDA="1.0"
XGB_REG_ALPHA="0.0"
XGB_TREE_METHOD="hist"
XGB_N_JOBS="-1"

TRAIN_POS_PATH=""
TRAIN_NEG_PATH=""
TAG=""
DEVICE="cpu"
VISUALIZE="none"
NAME_FIELDS="none"
PROCESS_TITLE="${PROCESS_TITLE:-tune_markov_xgboost}"

MIN_BATCH_SIZE="1"
MAX_OOM_RETRIES="0"

DEFAULT_SEARCH_SPACE_JSON="$(cat <<'JSON'
{
  "donor_len": {"type": "int", "min": 40, "max": 100, "step": 10},
  "acceptor_len": {"type": "int", "min": 40, "max": 100, "step": 10},
  "xgb_n_estimators": {
    "type": "categorical",
    "values": [200, 300, 400, 600, 800]
  },
  "xgb_max_depth": {
    "type": "categorical",
    "values": [3, 4, 5, 6, 8]
  },
  "xgb_learning_rate": {
    "type": "float",
    "min": 0.01,
    "max": 0.2,
    "scale": "log"
  },
  "xgb_subsample": {
    "type": "float",
    "min": 0.6,
    "max": 1.0,
    "scale": "linear"
  },
  "xgb_colsample_bytree": {
    "type": "float",
    "min": 0.6,
    "max": 1.0,
    "scale": "linear"
  },
  "xgb_min_child_weight": {
    "type": "float",
    "min": 0.5,
    "max": 8.0,
    "scale": "log"
  },
  "xgb_reg_lambda": {
    "type": "float",
    "min": 0.001,
    "max": 20.0,
    "scale": "log"
  },
  "xgb_reg_alpha": {
    "type": "float",
    "min": 0.0001,
    "max": 5.0,
    "scale": "log"
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

intronmodel_start_timer "tune_markov_xgboost.sh"
trap 'intronmodel_print_timing' EXIT

format_elapsed() {
	intronmodel_format_elapsed "$1"
}

resolve_species_case() {
	intronmodel_resolve_species_case "$1" "$2" "tune_markov_xgboost.sh"
}

resolve_python_bin() {
	intronmodel_resolve_python_bin "tune_markov_xgboost.sh"
}

resolve_search_space_file() {
	local explicit_file="$1"
	local species="$2"

	if [[ -n "${explicit_file}" && "${explicit_file}" != "auto" ]]; then
		if [[ -f "${explicit_file}" ]]; then
			printf '%s\n' "${explicit_file}"
			return 0
		fi
		echo "[tune_markov_xgboost.sh] SEARCH_SPACE_FILE not found: ${explicit_file}" >&2
		return 2
	fi

	local target_file="${DATA_ROOT}/${species}/tuning/markov_xgboost/pair/search_space.json"
	if [[ -f "${target_file}" ]]; then
		printf '%s\n' "${target_file}"
		return 0
	fi

	local species_file="${DATA_ROOT}/${species}/tuning/markov_xgboost/search_space.json"
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

precompute_markov_features() {
	local python_bin="$1"
	local species="$2"
	local train_pos="$3"
	local train_neg="$4"
	local donor_len="$5"
	local acceptor_len="$6"
	local markov_order="$7"
	local markov_alpha="$8"
	local markov_feature_mode="$9"
	local val_frac="${10}"
	local seed="${11}"
	local sequence_transform="${12}"
	local markov_cache_mode="${13}"
	local markov_cache_dir="${14}"

	if [[ "${markov_cache_mode}" == "off" ]]; then
		echo "[tune_markov_xgboost.sh] markov cache mode is off; skip precompute."
		return 0
	fi

	PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
		"${python_bin}" - \
			"${species}" \
			"${train_pos}" \
			"${train_neg}" \
			"${donor_len}" \
			"${acceptor_len}" \
			"${markov_order}" \
			"${markov_alpha}" \
			"${markov_feature_mode}" \
			"${val_frac}" \
			"${seed}" \
			"${sequence_transform}" \
			"${markov_cache_mode}" \
			"${markov_cache_dir}" <<'PY'
from __future__ import annotations

import sys

from models.markov_xgboost import _load_or_build_markov_feature_bundle
from util.data_proc import resolve_train_paths

species = sys.argv[1]
train_pos_raw = sys.argv[2].strip() or None
train_neg_raw = sys.argv[3].strip() or None
donor_len = int(sys.argv[4])
acceptor_len = int(sys.argv[5])
markov_order = int(sys.argv[6])
markov_alpha = float(sys.argv[7])
markov_feature_mode = sys.argv[8]
val_frac = float(sys.argv[9])
seed = int(sys.argv[10])
sequence_transform = sys.argv[11]
markov_cache_mode = sys.argv[12]
markov_cache_dir = sys.argv[13]

train_pos_path, train_neg_path, _ = resolve_train_paths(
    species=species,
    train_pos_path=train_pos_raw,
    train_neg_path=train_neg_raw,
    donor_len=donor_len,
    acceptor_len=acceptor_len,
)
bundle, cache_hit, cache_path = _load_or_build_markov_feature_bundle(
    pos_path=train_pos_path,
    neg_path=train_neg_path,
    donor_len=donor_len,
    acceptor_len=acceptor_len,
    markov_order=markov_order,
    markov_alpha=markov_alpha,
    markov_feature_mode=markov_feature_mode,
    val_frac=val_frac,
    seed=seed,
    sequence_transform=sequence_transform,
    markov_cache_mode=markov_cache_mode,
    markov_cache_dir=markov_cache_dir,
)
print(
    "[tune_markov_xgboost.sh] markov precompute "
    f"cache_hit={cache_hit} cache_path={cache_path} "
    f"n_total={bundle.n_total} n_train={bundle.n_train} n_val={bundle.n_val}"
)
PY
}

SPECIES="$(resolve_species_case "${SPECIES}" "${DATA_ROOT}")"
PYTHON_BIN="$(resolve_python_bin)"
if [[ "${SEARCH_ALGO}" != "random" && "${SEARCH_ALGO}" != "history_guided" ]]; then
	echo "[tune_markov_xgboost.sh] SEARCH_ALGO must be random|history_guided." >&2
	exit 1
fi
if [[ "${OBJECTIVE_METRIC}" != "pair_pr_auc" \
	&& "${OBJECTIVE_METRIC}" != "pair_roc_auc" \
	&& "${OBJECTIVE_METRIC}" != "pair_max_f1" ]]; then
	echo "[tune_markov_xgboost.sh] OBJECTIVE_METRIC must be " \
		"pair_pr_auc|pair_roc_auc|pair_max_f1." >&2
	exit 1
fi

if [[ -z "${MARKOV_CACHE_DIR}" ]]; then
	RESOLVED_MARKOV_CACHE_DIR="${DATA_ROOT}/${SPECIES}/train/markov_xgboost_cache"
else
	RESOLVED_MARKOV_CACHE_DIR="${MARKOV_CACHE_DIR}"
fi

TRAIN_POS_PATH_RESOLVED="$(intronmodel_resolve_species_template "${TRAIN_POS_PATH}" "${SPECIES}")"
TRAIN_NEG_PATH_RESOLVED="$(intronmodel_resolve_species_template "${TRAIN_NEG_PATH}" "${SPECIES}")"

search_space_json="${DEFAULT_SEARCH_SPACE_JSON}"
search_space_path=""
if search_space_resolved="$(
	resolve_search_space_file "${SEARCH_SPACE_FILE}" "${SPECIES}"
)"; then
	search_space_path="${search_space_resolved}"
	if ! parsed_json="$(
		normalize_json_object_file "${PYTHON_BIN}" "${search_space_path}" 2>&1
	)"; then
		echo "[tune_markov_xgboost.sh] failed to parse search-space file: "\
			"${search_space_path}" >&2
		echo "[tune_markov_xgboost.sh] parse detail: ${parsed_json}" >&2
		exit 1
	fi
	search_space_json="${parsed_json}"
	echo "[tune_markov_xgboost.sh] search_space_file=${search_space_path}"
else
	search_space_status=$?
	if [[ "${search_space_status}" -eq 2 ]]; then
		exit 1
	fi
	echo "[tune_markov_xgboost.sh] search_space_file=<embedded_default>"
fi

if [[ -z "${GLOBAL_BEST_CONFIG_PATH}" ]]; then
	GLOBAL_BEST_CONFIG_PATH="${DATA_ROOT}/${SPECIES}/tuning/markov_xgboost/pair/best_config.json"
fi
SEED_BEST_CONFIG_JSON="null"
if [[ -n "${SEED_BEST_CONFIG_PATH}" ]]; then
	SEED_BEST_CONFIG_JSON="\"${SEED_BEST_CONFIG_PATH}\""
fi

echo "[tune_markov_xgboost.sh] species=${SPECIES}"
echo "[tune_markov_xgboost.sh] objective=${OBJECTIVE_METRIC}"
echo "[tune_markov_xgboost.sh] markov_cache_mode=${MARKOV_CACHE_MODE}"
echo "[tune_markov_xgboost.sh] markov_cache_dir=${RESOLVED_MARKOV_CACHE_DIR}"

precompute_markov_features \
	"${PYTHON_BIN}" \
	"${SPECIES}" \
	"${TRAIN_POS_PATH_RESOLVED}" \
	"${TRAIN_NEG_PATH_RESOLVED}" \
	"${DONOR_LEN}" \
	"${ACCEPTOR_LEN}" \
	"${MARKOV_ORDER}" \
	"${MARKOV_ALPHA}" \
	"${MARKOV_FEATURE_MODE}" \
	"${VAL_FRAC}" \
	"${BASE_SEED}" \
	"${SEQUENCE_TRANSFORM}" \
	"${MARKOV_CACHE_MODE}" \
	"${RESOLVED_MARKOV_CACHE_DIR}"

RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${DATA_ROOT}/${SPECIES}/tuning/markov_xgboost/pair/${RUN_TIMESTAMP}_seed${BASE_SEED}"
mkdir -p "${OUTPUT_DIR}"

TAG_JSON="$(intronmodel_json_string_or_null "${PYTHON_BIN}" "${TAG}")"
TRAIN_POS_PATH_JSON="$(
	intronmodel_json_string_or_null "${PYTHON_BIN}" "${TRAIN_POS_PATH_RESOLVED}"
)"
TRAIN_NEG_PATH_JSON="$(
	intronmodel_json_string_or_null "${PYTHON_BIN}" "${TRAIN_NEG_PATH_RESOLVED}"
)"
MARKOV_CACHE_DIR_JSON="$(
	intronmodel_json_string_or_null "${PYTHON_BIN}" "${RESOLVED_MARKOV_CACHE_DIR}"
)"

CONFIG_PATH="${OUTPUT_DIR}/hparam_search_config.json"
cat > "${CONFIG_PATH}" <<JSON
{
  "project_root": "${PROJECT_ROOT}",
  "species": "${SPECIES}",
  "output_dir": "${OUTPUT_DIR}",
  "quick_trials": ${QUICK_TRIALS},
  "quick_epochs": ${QUICK_EPOCHS},
  "top_k": ${TOP_K},
  "full_epochs": ${FULL_EPOCHS},
  "skip_full_phase": ${SKIP_FULL_PHASE},
  "enable_visualization": ${ENABLE_VISUALIZATION},
  "base_seed": ${BASE_SEED},
  "gpu_ids": "${GPU_IDS}",
  "max_parallel_trials": "${MAX_PARALLEL_TRIALS}",
  "trial_process_mode": "${TRIAL_PROCESS_MODE}",
  "trial_stream_mode": "${TRIAL_STREAM_MODE}",
  "objective_metric": "${OBJECTIVE_METRIC}",
  "global_best_config_path": "${GLOBAL_BEST_CONFIG_PATH}",
  "seed_best_config_path": ${SEED_BEST_CONFIG_JSON},
  "search_algo": "${SEARCH_ALGO}",
  "history_top_n": ${HISTORY_TOP_N},
  "guided_random_fraction": ${GUIDED_RANDOM_FRACTION},
  "guided_mutation_rate": ${GUIDED_MUTATION_RATE},
  "min_batch_size": ${MIN_BATCH_SIZE},
  "max_oom_retries": ${MAX_OOM_RETRIES},
  "base_args": {
    "model": "markov_xgboost",
    "species": "${SPECIES}",
    "train_target": "pair",
    "seed": ${BASE_SEED},
    "donor_len": ${DONOR_LEN},
    "acceptor_len": ${ACCEPTOR_LEN},
    "val_frac": ${VAL_FRAC},
    "sequence_transform": "${SEQUENCE_TRANSFORM}",
    "markov_order": ${MARKOV_ORDER},
    "markov_alpha": ${MARKOV_ALPHA},
    "markov_feature_mode": "${MARKOV_FEATURE_MODE}",
    "markov_cache_mode": "${MARKOV_CACHE_MODE}",
    "markov_cache_dir": ${MARKOV_CACHE_DIR_JSON},
    "xgb_n_estimators": ${XGB_N_ESTIMATORS},
    "xgb_max_depth": ${XGB_MAX_DEPTH},
    "xgb_learning_rate": ${XGB_LEARNING_RATE},
    "xgb_subsample": ${XGB_SUBSAMPLE},
    "xgb_colsample_bytree": ${XGB_COLSAMPLE_BYTREE},
    "xgb_min_child_weight": ${XGB_MIN_CHILD_WEIGHT},
    "xgb_reg_lambda": ${XGB_REG_LAMBDA},
    "xgb_reg_alpha": ${XGB_REG_ALPHA},
    "xgb_tree_method": "${XGB_TREE_METHOD}",
    "xgb_n_jobs": ${XGB_N_JOBS},
    "batch_size": 1,
    "device": "${DEVICE}",
    "visualize": "${VISUALIZE}",
    "name_fields": "${NAME_FIELDS}",
    "tag": ${TAG_JSON},
    "train_pos_path": ${TRAIN_POS_PATH_JSON},
    "train_neg_path": ${TRAIN_NEG_PATH_JSON}
  },
  "quick_overrides": {},
  "full_overrides": {},
  "search_space": ${search_space_json}
}
JSON

echo "[tune_markov_xgboost.sh] output_dir=${OUTPUT_DIR}"
if ! intronmodel_run_with_process_title \
	"${PROCESS_TITLE}" \
	"${PYTHON_BIN}" \
	"${PROJECT_ROOT}/src/tools/hparam_search.py" \
	--config "${CONFIG_PATH}"; then
	echo "[tune_markov_xgboost.sh] tuning failed." >&2
	exit 1
fi

elapsed_seconds=$((SECONDS - INTRONMODEL_SCRIPT_START_SECONDS))
elapsed_hms="$(format_elapsed "${elapsed_seconds}")"
echo "[tune_markov_xgboost.sh] completed in ${elapsed_hms} (${elapsed_seconds}s)."
