#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
	echo "[dnabert_pair.sh] This script is config-only. Edit top CONFIG and run without args." >&2
	exit 1
fi

# --------------------------
# CONFIG (edit here)
# --------------------------
# Frequently edited knobs are intentionally placed first in this block.
# Advanced runtime controls are kept below.
set -a
DNABERT_VARIANT="2"
SPECIES="Dmel"
TRUNC_MODE="on"
INTRONMODEL_AUTO_TMUX="on"
GPU_IDS="auto"
EPOCHS="auto"
MAX_EPOCHS="10"
EARLY_STOP_PATIENCE="2"
EARLY_STOP_MIN_DELTA="0.005"
SKIP_TRAINING="0"
CONTINUE_TRAINING="0"

PRETRAINED_MODEL_NAME=""
PRETRAINED_MODEL_RELATIVE_PATH_2="pretrained/dnabert2-117m-7bce263b15377fc15361f52cfab88f8b586abda0"
PRETRAINED_MODEL_RELATIVE_PATH_6="pretrained/dnabert6"
PRETRAINED_MODEL_RELATIVE_PATH_S="pretrained/dnabert-s"
PRETRAINED_REVISION=""
TRUST_REMOTE_CODE="1"

TRANSCRIPT_SCORE_AGG="min"
SOFTMIN_TAU="1.0"
SEED="1337"
NAME_FIELDS="tag"
PROCESS_TITLE="ETA"
# Optional output/data overrides for trunc-data runs.
TAG=""
SYNTHESIZE_MODE="off"
TRAIN_POS_PATH=""
TRAIN_NEG_PATH=""
MASK_TEST_TSV_PATH=""
VISUALIZE="true"
TRAIN_ONLY="0"
PRECOMPUTED_SITE_SCORE_TSV=""
CHECKPOINT_TOP_K="3"
CHECKPOINT_PRUNE_DRY_RUN="0"

DONOR_LEN="100"
ACCEPTOR_LEN="100"
BATCH_SIZE="64"
INFER_BATCH_SIZE="256"
LR="2e-5"
LOSS="weighted_bce"
MAX_TOKENS="auto"
DROPOUT="0.1"
HEAD_LAYER_NORM="1"
WEIGHT_DECAY="0.01"
ETA_MIN_RATIO="0.01"
LR_SCHEDULE="cosine"
WARMUP_RATIO="0.01"
ADAM_BETA1="0.9"
ADAM_BETA2="0.98"
ADAM_EPS="1e-8"
VAL_FRAC="0.1"
GRAD_CLIP="1.0"
POS_WEIGHT_CAP="20.0"
FOCAL_GAMMA="2.0"
FOCAL_ALPHA_POS=""
ASYM_GAMMA_POS="0.0"
ASYM_GAMMA_NEG="4.0"
ASYM_ALPHA_POS=""
USE_TUNED_HPARAMS="auto"
TUNED_HPARAMS_MODE="normal"
PAIR_TUNED_CONFIG_PATH=""
SHARED_TUNED_CONFIG_PATH=""

DEVICE="auto"
MAX_PARALLEL_TRIALS="auto"
USE_AMP="1"
AMP_DTYPE="auto"
COMPILE_MODE="on"
INFER_USE_AMP="1"
INFER_AMP_DTYPE="auto"
INFER_COMPILE="0"
INFER_COMPILE_MODE="off"
ALLOW_TF32="1"
CUDNN_BENCHMARK="1"
DETERMINISTIC="0"
NUM_WORKERS="auto"
PREFETCH_FACTOR="4"
PERSISTENT_WORKERS="1"
PIN_MEMORY="1"
MIN_BATCH_SIZE="64"
MAX_OOM_RETRIES="8"
MPS_MAX_BATCH_SIZE="1024"

set +a

# --------------------------
# Runtime implementation
# --------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/tuned_config.sh"
intronmodel_activate_conda "intronmodel"
intronmodel_init_paths "${BASH_SOURCE[0]}"

# Auto-run inside tmux on SSH so jobs survive disconnects.
# Set INTRONMODEL_AUTO_TMUX=off|on|auto (default: auto).
intronmodel_enable_auto_tmux "${PROJECT_ROOT}" "$0" "${BASH_SOURCE[0]##*/}"

resolve_dnabert_relative_path() {
	intronmodel_resolve_dnabert_relative_path "dnabert_pair.sh" "$@"
}

append_arg_if_set() {
	local flag="$1"
	local value="$2"
	if [[ -n "${value}" ]]; then
		args+=("--${flag}" "${value}")
	fi
}

append_flag_if_truthy() {
	local flag="$1"
	local value="$2"
	local normalized
	normalized="$(echo "${value}" | tr '[:upper:]' '[:lower:]' | xargs)"
	case "${normalized}" in
		1 | true | on | yes)
			args+=("--${flag}")
			;;
	esac
}

intronmodel_start_timer "dnabert_pair.sh"
trap 'intronmodel_print_timing' EXIT

if [[ -z "${PRETRAINED_MODEL_NAME}" ]]; then
	PRETRAINED_MODEL_RELATIVE_PATH="$(
		resolve_dnabert_relative_path \
			"${DNABERT_VARIANT}" \
			"${PRETRAINED_MODEL_RELATIVE_PATH_2}" \
			"${PRETRAINED_MODEL_RELATIVE_PATH_6}" \
			"${PRETRAINED_MODEL_RELATIVE_PATH_S}"
	)"
	export PRETRAINED_MODEL_RELATIVE_PATH
fi
if [[ "${TRUNC_MODE}" != "off" && "${TRUNC_MODE}" != "on" ]]; then
	echo "[dnabert_pair.sh] TRUNC_MODE must be off|on." >&2
	exit 1
fi
if [[ "${HEAD_LAYER_NORM}" != "0" && "${HEAD_LAYER_NORM}" != "1" ]]; then
	echo "[dnabert_pair.sh] HEAD_LAYER_NORM must be 0 or 1." >&2
	exit 1
fi
if [[ "${LR_SCHEDULE}" != "cosine" && "${LR_SCHEDULE}" != "linear" ]]; then
	echo "[dnabert_pair.sh] LR_SCHEDULE must be cosine|linear." >&2
	exit 1
fi
if ! [[ "${WARMUP_RATIO}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
	echo "[dnabert_pair.sh] WARMUP_RATIO must be numeric in [0,1)." >&2
	exit 1
fi
if ! awk -v x="${WARMUP_RATIO}" 'BEGIN{exit !(x>=0 && x<1)}'; then
	echo "[dnabert_pair.sh] WARMUP_RATIO must be in [0,1)." >&2
	exit 1
fi
if ! [[ "${ADAM_BETA1}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
	echo "[dnabert_pair.sh] ADAM_BETA1 must be numeric in (0,1)." >&2
	exit 1
fi
if ! awk -v x="${ADAM_BETA1}" 'BEGIN{exit !(x>0 && x<1)}'; then
	echo "[dnabert_pair.sh] ADAM_BETA1 must be in (0,1)." >&2
	exit 1
fi
if ! [[ "${ADAM_BETA2}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
	echo "[dnabert_pair.sh] ADAM_BETA2 must be numeric in (0,1)." >&2
	exit 1
fi
if ! awk -v x="${ADAM_BETA2}" 'BEGIN{exit !(x>0 && x<1)}'; then
	echo "[dnabert_pair.sh] ADAM_BETA2 must be in (0,1)." >&2
	exit 1
fi
if ! awk -v b1="${ADAM_BETA1}" -v b2="${ADAM_BETA2}" \
	'BEGIN{exit !(b1<b2)}'; then
	echo "[dnabert_pair.sh] ADAM_BETA1 must be smaller than ADAM_BETA2." >&2
	exit 1
fi
if ! [[ "${ADAM_EPS}" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
	echo "[dnabert_pair.sh] ADAM_EPS must be a positive number." >&2
	exit 1
fi
if ! awk -v x="${ADAM_EPS}" 'BEGIN{exit !(x>0)}'; then
	echo "[dnabert_pair.sh] ADAM_EPS must be > 0." >&2
	exit 1
fi
MASK_MODE="${TRUNC_MODE}"
export MASK_MODE

# Auto-generate versioning tag from DNABERT variant if TAG is empty
if [[ -z "${TAG}" ]]; then
	DNABERT_VARIANT_LOWER="${DNABERT_VARIANT,,}"
	TAG="dnabert_pair${DNABERT_VARIANT_LOWER}"
	export TAG
fi

USE_TUNED_HPARAMS_MODE="$(
	intronmodel_normalize_use_tuned_mode "${USE_TUNED_HPARAMS}" "dnabert_pair.sh"
)"
if [[ "${USE_TUNED_HPARAMS_MODE}" != "off" ]]; then
	RESOLVED_TUNED_TARGET="$(
		intronmodel_resolve_tuned_target "pair" "pair"
	)"
fi

tuned_model_name="$(
	intronmodel_resolve_pair_tuning_model_name "${MODEL:-dnabert_pair}"
)"
best_config_filename="$(
	intronmodel_resolve_pair_best_config_filename "${SYNTHESIZE_MODE}"
)"
tuned_path=""
tuned_output=""
tuned_args=()
use_wrapper_hparams="1"
args=()

append_flag_if_truthy "skip_train" "${SKIP_TRAINING}"
append_flag_if_truthy "continue_train" "${CONTINUE_TRAINING}"
append_flag_if_truthy "train_only" "${TRAIN_ONLY}"
if [[ -n "${TAG}" ]]; then
	append_arg_if_set "tag" "${TAG}"
fi
append_arg_if_set "train_pos_path" "${TRAIN_POS_PATH}"
append_arg_if_set "train_neg_path" "${TRAIN_NEG_PATH}"
append_arg_if_set "mask_test_tsv" "${MASK_TEST_TSV_PATH}"
if [[ "${SKIP_TRAINING}" == "1" && "${TRAIN_ONLY}" != "1" ]]; then
	intronmodel_append_versioned_output_args \
		"dnabert_pair.sh" "${SPECIES}" "${MODEL:-dnabert_pair}" args
fi

if [[ "${USE_TUNED_HPARAMS_MODE}" != "off" ]]; then
	tuned_path="$(
		intronmodel_resolve_tuned_config_path \
			"${DATA_ROOT}" \
			"${SPECIES}" \
			"${tuned_model_name}" \
			"${RESOLVED_TUNED_TARGET}" \
			"${PAIR_TUNED_CONFIG_PATH}" \
			"${SHARED_TUNED_CONFIG_PATH}" \
			"${best_config_filename}"
	)"
	if [[ -z "${tuned_path}" ]]; then
		if [[ "${USE_TUNED_HPARAMS_MODE}" == "required" ]]; then
			echo "[dnabert_pair.sh] tuned config is required but not found: "\
				"species=${SPECIES} target=${RESOLVED_TUNED_TARGET}" >&2
			exit 1
		fi
		echo "[dnabert_pair.sh] tuned config not found; using CONFIG defaults." >&2
	elif [[ ! -f "${tuned_path}" ]]; then
		if [[ "${USE_TUNED_HPARAMS_MODE}" == "required" ]]; then
			echo "[dnabert_pair.sh] tuned config path not found: ${tuned_path}" >&2
			exit 1
		fi
		echo "[dnabert_pair.sh] tuned config path not found: ${tuned_path}; "\
			"using CONFIG defaults." >&2
		tuned_path=""
	fi
fi

if [[ -n "${tuned_path}" ]]; then
	if ! tuned_output="$(intronmodel_load_tuned_overrides "${tuned_path}" 2>&1)"; then
		if [[ "${USE_TUNED_HPARAMS_MODE}" == "required" ]]; then
			echo "[dnabert_pair.sh] failed to load tuned config: ${tuned_path}" >&2
			echo "[dnabert_pair.sh] detail: ${tuned_output}" >&2
			exit 1
		fi
		echo "[dnabert_pair.sh] failed to load tuned config: ${tuned_path}; "\
			"using CONFIG defaults." >&2
	else
		use_wrapper_hparams="0"
		loaded_count=0
		while IFS= read -r line; do
			if [[ -z "${line}" ]]; then
				continue
			fi
			IFS=$'\t' read -r tuned_key tuned_value <<<"${line}"
			if [[ -z "${tuned_key}" || -z "${tuned_value}" ]]; then
				continue
			fi
			if [[ "${tuned_key}" == "tag" ]]; then
				continue
			fi
			tuned_args+=(--"${tuned_key}" "${tuned_value}")
			loaded_count=$((loaded_count + 1))
		done <<<"${tuned_output}"
		echo "[dnabert_pair.sh] tuned params loaded from ${tuned_path} "\
			"(count=${loaded_count})"
	fi
fi
append_arg_if_set "pair_tuned_config_path" "${tuned_path}"
if [[ -n "${tuned_path}" ]]; then
	PAIR_TUNED_CONFIG_PATH="${tuned_path}"
	export PAIR_TUNED_CONFIG_PATH
fi

if [[ "${use_wrapper_hparams}" == "1" ]]; then
	args+=(
		--donor_len "${DONOR_LEN}"
		--acceptor_len "${ACCEPTOR_LEN}"
		--seed "${SEED}"
		--batch_size "${BATCH_SIZE}"
		--lr "${LR}"
		--loss "${LOSS}"
		--max_tokens "${MAX_TOKENS}"
		--dropout "${DROPOUT}"
		--head_layer_norm "${HEAD_LAYER_NORM}"
		--weight_decay "${WEIGHT_DECAY}"
		--eta_min_ratio "${ETA_MIN_RATIO}"
		--lr_schedule "${LR_SCHEDULE}"
		--warmup_ratio "${WARMUP_RATIO}"
		--adam_beta1 "${ADAM_BETA1}"
		--adam_beta2 "${ADAM_BETA2}"
		--adam_eps "${ADAM_EPS}"
		--val_frac "${VAL_FRAC}"
		--grad_clip "${GRAD_CLIP}"
		--pos_weight_cap "${POS_WEIGHT_CAP}"
		--focal_gamma "${FOCAL_GAMMA}"
	)
	append_arg_if_set "focal_alpha_pos" "${FOCAL_ALPHA_POS}"
	append_arg_if_set "asym_gamma_pos" "${ASYM_GAMMA_POS}"
	append_arg_if_set "asym_gamma_neg" "${ASYM_GAMMA_NEG}"
	append_arg_if_set "asym_alpha_pos" "${ASYM_ALPHA_POS}"
fi

if [[ ${#tuned_args[@]} -gt 0 ]]; then
	args+=("${tuned_args[@]}")
fi

(
	export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
	if [[ ${#args[@]} -gt 0 ]]; then
		printf '[dnabert_pair.sh] prepared run args:'
		printf ' %q' "${args[@]}"
		printf '\n'
	fi
	intronmodel_run_with_process_title \
		"${PROCESS_TITLE}" \
		python3 "${PROJECT_ROOT}/src/tools/run_wrapper_pipeline.py" \
			--script-name "dnabert_pair.sh"
)
