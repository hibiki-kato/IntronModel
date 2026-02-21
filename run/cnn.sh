#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
	echo "[cnn.sh] This script is config-only. Edit CONFIG and run without args." >&2
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
	echo "[cnn.sh] timing: start=${SCRIPT_START_EPOCH} end=${script_end_epoch} "\
		"elapsed=${elapsed_hms} (${elapsed_seconds}s) exit=${exit_code}"
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
		echo "[cnn.sh] species case normalized: '${raw_species}' -> '${matches[0]}'" >&2
		printf '%s\n' "${matches[0]}"
		return 0
	fi
	if [[ ${#matches[@]} -gt 1 ]]; then
		echo "[cnn.sh] ambiguous species '${raw_species}'." >&2
		printf '[cnn.sh] case-insensitive matches: %s\n' "${matches[*]}" >&2
		return 1
	fi
	printf '%s\n' "${raw_species}"
	return 0
}

resolve_tuned_config_path() {
	local task="$1"
	local explicit_path="$2"
	local species="$3"
	local data_root="$4"
	local shared_path="$5"

	if [[ -n "${explicit_path}" ]]; then
		printf '%s\n' "${explicit_path}"
		return 0
	fi

	local task_path="${data_root}/${species}/tuning/cnn/${task}/best_config.json"
	if [[ -f "${task_path}" ]]; then
		printf '%s\n' "${task_path}"
		return 0
	fi

	if [[ -n "${shared_path}" && -f "${shared_path}" ]]; then
		printf '%s\n' "${shared_path}"
		return 0
	fi

	local legacy_path="${data_root}/${species}/tuning/cnn/best_config.json"
	if [[ -f "${legacy_path}" ]]; then
		printf '%s\n' "${legacy_path}"
		return 0
	fi

	return 1
}

extract_tuned_assignments() {
	local config_path="$1"
	local task_prefix="$2"
	python3 - "${config_path}" "${task_prefix}" <<'PY'
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
task_prefix = sys.argv[2]
payload = json.loads(config_path.read_text(encoding="utf-8"))
status = str(payload.get("status", "")).strip().lower()
if status != "ok":
    raise ValueError(f"Expected status='ok', got: {status or '<missing>'}")

sampled_params = payload.get("sampled_params")
if not isinstance(sampled_params, dict):
    raise ValueError("sampled_params is missing or invalid.")

key_map = {
    "batch_size": "BATCH_SIZE",
    "lr": "LR",
    "loss": "LOSS",
    "conv_channels": "CONV_CHANNELS",
    "kernel_size": "KERNEL_SIZE",
    "dropout": "DROPOUT",
    "fc_hidden": "FC_HIDDEN",
    "weight_decay": "WEIGHT_DECAY",
}

for key, suffix in key_map.items():
    if key not in sampled_params:
        continue
    value = sampled_params[key]
    if value is None:
        continue

    if isinstance(value, bool):
        value_text = "1" if value else "0"
    elif isinstance(value, int):
        value_text = str(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite value for '{key}'.")
        value_text = format(value, ".15g")
    else:
        value_text = str(value)

    print(f"{task_prefix}_{suffix}\t{value_text}")
PY
}

apply_tuned_overrides_for_task() {
	local task="$1"
	local config_path="$2"
	local task_prefix
	task_prefix="$(printf '%s' "${task}" | tr '[:lower:]' '[:upper:]')"

	if [[ ! -f "${config_path}" ]]; then
		echo "[cnn.sh] tuned ${task} config not found: ${config_path}" >&2
		return 1
	fi

	local extracted
	if ! extracted="$(
		extract_tuned_assignments "${config_path}" "${task_prefix}" 2>&1
	)"; then
		echo "[cnn.sh] failed to parse tuned ${task} config: ${config_path}" >&2
		echo "[cnn.sh] parse detail: ${extracted}" >&2
		return 1
	fi

	local applied_count=0
	local kept_manual_count=0
	local var_name
	local var_value
	while IFS=$'\t' read -r var_name var_value; do
		if [[ -z "${var_name}" ]]; then
			continue
		fi
		if [[ -n "${!var_name:-}" ]]; then
			kept_manual_count=$((kept_manual_count + 1))
			continue
		fi
		printf -v "${var_name}" '%s' "${var_value}"
		applied_count=$((applied_count + 1))
	done <<< "${extracted}"

	echo "[cnn.sh] tuned ${task} loaded from ${config_path} "\
		"(applied=${applied_count}, kept_manual=${kept_manual_count})"
	return 0
}

# --------------------------
# CONFIG (edit here)
# --------------------------
MODEL="cnn"
SPECIES="Mmus"
DONOR_LEN="100"
ACCEPTOR_LEN="100"

EPOCHS="25"
BATCH_SIZE="512"
LR="5e-4"
LOSS="focal"
CONV_CHANNELS="64,128,256"
KERNEL_SIZE="7"
DROPOUT="0.3"
FC_HIDDEN="128"
WEIGHT_DECAY="0.01"
ETA_MIN_RATIO="0.01"
VAL_FRAC="0.1"
GRAD_CLIP="5.0"
POS_WEIGHT_CAP="20.0"
FOCAL_GAMMA="2.0"
FOCAL_ALPHA_POS=""
ASYM_GAMMA_POS="0.0"
ASYM_GAMMA_NEG="4.0"
ASYM_ALPHA_POS=""
TRAIN_TARGET="both"
USE_TUNED_HPARAMS="auto"
DONOR_TUNED_CONFIG_PATH=""
ACCEPTOR_TUNED_CONFIG_PATH=""
SHARED_TUNED_CONFIG_PATH=""

DONOR_BATCH_SIZE=""
ACCEPTOR_BATCH_SIZE=""
DONOR_LR=""
ACCEPTOR_LR=""
DONOR_LOSS=""
ACCEPTOR_LOSS=""
DONOR_CONV_CHANNELS=""
ACCEPTOR_CONV_CHANNELS=""
DONOR_KERNEL_SIZE=""
ACCEPTOR_KERNEL_SIZE=""
DONOR_DROPOUT=""
ACCEPTOR_DROPOUT=""
DONOR_FC_HIDDEN=""
ACCEPTOR_FC_HIDDEN=""
DONOR_WEIGHT_DECAY=""
ACCEPTOR_WEIGHT_DECAY=""
DONOR_ETA_MIN_RATIO=""
ACCEPTOR_ETA_MIN_RATIO=""
DONOR_VAL_FRAC=""
ACCEPTOR_VAL_FRAC=""
DONOR_GRAD_CLIP=""
ACCEPTOR_GRAD_CLIP=""
DONOR_POS_WEIGHT_CAP=""
ACCEPTOR_POS_WEIGHT_CAP=""
DONOR_FOCAL_GAMMA=""
ACCEPTOR_FOCAL_GAMMA=""
DONOR_FOCAL_ALPHA_POS=""
ACCEPTOR_FOCAL_ALPHA_POS=""
DONOR_ASYM_GAMMA_POS=""
ACCEPTOR_ASYM_GAMMA_POS=""
DONOR_ASYM_GAMMA_NEG=""
ACCEPTOR_ASYM_GAMMA_NEG=""
DONOR_ASYM_ALPHA_POS=""
ACCEPTOR_ASYM_ALPHA_POS=""

INTRON_SCORE_OP="*"
TRANSCRIPT_AGG="min"
SOFTMIN_TAU="1.0"
SEED="1337"
NAME_FIELDS=""
VISUALIZE="true"
SKIP_TRAINING="0"
CONTINUE_TRAINING="1"
TRAIN_ONLY="0"
PRECOMPUTED_SITE_SCORE_TSV=""

PERF_MODE="max_throughput"
DEVICE="auto"
USE_AMP="1"
AMP_DTYPE="auto"
USE_COMPILE="auto"
ALLOW_TF32="1"
CUDNN_BENCHMARK="1"
DETERMINISTIC="0"
NUM_WORKERS="auto"
PREFETCH_FACTOR="4"
PERSISTENT_WORKERS="1"
PIN_MEMORY="1"
MIN_BATCH_SIZE="64"
MAX_OOM_RETRIES="8"
MPS_MAX_BATCH_SIZE="2048"

if [[ "${USE_COMPILE}" != "off" && "${USE_COMPILE}" != "on" \
	&& "${USE_COMPILE}" != "auto" ]]; then
	echo "[cnn.sh] USE_COMPILE must be off|on|auto." >&2
	exit 1
fi
if [[ "${TRAIN_ONLY}" != "0" && "${TRAIN_ONLY}" != "1" ]]; then
	echo "[cnn.sh] TRAIN_ONLY must be 0 or 1." >&2
	exit 1
fi
if [[ "${SKIP_TRAINING}" != "0" && "${SKIP_TRAINING}" != "1" ]]; then
	echo "[cnn.sh] SKIP_TRAINING must be 0 or 1." >&2
	exit 1
fi
if [[ "${CONTINUE_TRAINING}" != "0" && "${CONTINUE_TRAINING}" != "1" ]]; then
	echo "[cnn.sh] CONTINUE_TRAINING must be 0 or 1." >&2
	exit 1
fi
if [[ "${SKIP_TRAINING}" == "1" && "${CONTINUE_TRAINING}" == "1" ]]; then
	echo "[cnn.sh] CONTINUE_TRAINING=1 cannot be used with SKIP_TRAINING=1." >&2
	exit 1
fi
if [[ "${TRAIN_TARGET}" != "both" && "${TRAIN_TARGET}" != "donor" \
	&& "${TRAIN_TARGET}" != "acceptor" ]]; then
	echo "[cnn.sh] TRAIN_TARGET must be both|donor|acceptor." >&2
	exit 1
fi
if [[ "${USE_TUNED_HPARAMS}" != "off" && "${USE_TUNED_HPARAMS}" != "auto" \
	&& "${USE_TUNED_HPARAMS}" != "required" ]]; then
	echo "[cnn.sh] USE_TUNED_HPARAMS must be off|auto|required." >&2
	exit 1
fi
if [[ "${TRAIN_TARGET}" != "both" && "${TRAIN_ONLY}" != "1" ]]; then
	echo "[cnn.sh] TRAIN_TARGET donor/acceptor requires TRAIN_ONLY=1." >&2
	exit 1
fi
if ! [[ "${MPS_MAX_BATCH_SIZE}" =~ ^[0-9]+$ ]] || [[ "${MPS_MAX_BATCH_SIZE}" -le 0 ]]; then
	echo "[cnn.sh] MPS_MAX_BATCH_SIZE must be a positive integer." >&2
	exit 1
fi

export INTRONMODEL_MPS_MAX_BATCH_SIZE="${MPS_MAX_BATCH_SIZE}"

SPECIES="$(resolve_species_case "${SPECIES}" "${DATA_ROOT}")"

if [[ "${USE_TUNED_HPARAMS}" != "off" ]]; then
	if [[ -z "${SHARED_TUNED_CONFIG_PATH}" ]]; then
		SHARED_TUNED_CONFIG_PATH="${DATA_ROOT}/${SPECIES}/tuning/cnn/best_config.json"
	fi

	TUNED_TASKS=("donor" "acceptor")
	if [[ "${TRAIN_TARGET}" == "donor" || "${TRAIN_TARGET}" == "acceptor" ]]; then
		TUNED_TASKS=("${TRAIN_TARGET}")
	fi

	for tuned_task in "${TUNED_TASKS[@]}"; do
		task_config_path=""
		if [[ "${tuned_task}" == "donor" ]]; then
			explicit_task_path="${DONOR_TUNED_CONFIG_PATH}"
		else
			explicit_task_path="${ACCEPTOR_TUNED_CONFIG_PATH}"
		fi
		if resolve_path="$(
			resolve_tuned_config_path \
				"${tuned_task}" \
				"${explicit_task_path}" \
				"${SPECIES}" \
				"${DATA_ROOT}" \
				"${SHARED_TUNED_CONFIG_PATH}"
		)"; then
			task_config_path="${resolve_path}"
		else
			if [[ "${USE_TUNED_HPARAMS}" == "required" ]]; then
				echo "[cnn.sh] tuned ${tuned_task} config is required but not found." >&2
				exit 1
			fi
			echo "[cnn.sh] tuned ${tuned_task} config not found; "\
				"using CONFIG defaults." >&2
			continue
		fi

		if ! apply_tuned_overrides_for_task "${tuned_task}" "${task_config_path}"; then
			if [[ "${USE_TUNED_HPARAMS}" == "required" ]]; then
				exit 1
			fi
			echo "[cnn.sh] tuned ${tuned_task} load failed; "\
				"using CONFIG defaults." >&2
		fi
	done
fi

TEST_TSV="${DATA_ROOT}/${SPECIES}/raw/transcripts.tsv"
CLASS_FILE="${DATA_ROOT}/${SPECIES}/raw/transcript_class.txt"

OUTPUT_STEM="$({
	PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" python3 - <<PY
from util.data_proc import build_output_stem, parse_name_fields

model_name = "${MODEL}"
donor_len = int("${DONOR_LEN}")
acceptor_len = int("${ACCEPTOR_LEN}")
name_fields = parse_name_fields("${NAME_FIELDS}")
params = {
    "donor_len": donor_len,
    "acceptor_len": acceptor_len,
    "epochs": int("${EPOCHS}"),
    "batch_size": int("${BATCH_SIZE}"),
    "lr": float("${LR}"),
    "loss": "${LOSS}",
    "conv_channels": "${CONV_CHANNELS}" or None,
    "kernel_size": int("${KERNEL_SIZE}"),
    "dropout": float("${DROPOUT}"),
    "fc_hidden": int("${FC_HIDDEN}"),
    "weight_decay": float("${WEIGHT_DECAY}"),
    "eta_min_ratio": float("${ETA_MIN_RATIO}"),
    "grad_clip": float("${GRAD_CLIP}"),
    "val_frac": float("${VAL_FRAC}"),
    "intron_score_op": "${INTRON_SCORE_OP}",
    "transcript_score_agg": "${TRANSCRIPT_AGG}",
    "softmin_tau": float("${SOFTMIN_TAU}"),
    "seed": int("${SEED}"),
    "train_target": "${TRAIN_TARGET}",
}
print(
    build_output_stem(
        model_name=model_name,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        fallback_train_len=None,
        name_fields=name_fields,
        name_params=params,
    )
)
PY
} | tr -d '\n')"

OUTPUT_SITE_SCORE_TSV="${DATA_ROOT}/${SPECIES}/site_score/${OUTPUT_STEM}.tsv"
OUTPUT_TRANS_SCORE_TSV="${DATA_ROOT}/${SPECIES}/trans_score/${OUTPUT_STEM}.tsv"
OUTPUT_EVAL_SCORE_TXT="${DATA_ROOT}/${SPECIES}/eval_score/${OUTPUT_STEM}.txt"

RUN_ARGS=(
	--model "${MODEL}"
	--species "${SPECIES}"
	--donor_len "${DONOR_LEN}"
	--acceptor_len "${ACCEPTOR_LEN}"
	--epochs "${EPOCHS}"
	--train_target "${TRAIN_TARGET}"
	--batch_size "${BATCH_SIZE}"
	--lr "${LR}"
	--loss "${LOSS}"
	--conv_channels "${CONV_CHANNELS}"
	--kernel_size "${KERNEL_SIZE}"
	--dropout "${DROPOUT}"
	--fc_hidden "${FC_HIDDEN}"
	--weight_decay "${WEIGHT_DECAY}"
	--eta_min_ratio "${ETA_MIN_RATIO}"
	--grad_clip "${GRAD_CLIP}"
	--val_frac "${VAL_FRAC}"
	--pos_weight_cap "${POS_WEIGHT_CAP}"
	--focal_gamma "${FOCAL_GAMMA}"
	--name_fields "${NAME_FIELDS}"
	--intron_score_op "${INTRON_SCORE_OP}"
	--transcript_score_agg "${TRANSCRIPT_AGG}"
	--softmin_tau "${SOFTMIN_TAU}"
	--seed "${SEED}"
	--device "${DEVICE}"
	--visualize "${VISUALIZE}"
	--use_amp "${USE_AMP}"
	--amp_dtype "${AMP_DTYPE}"
	--compile_mode "${USE_COMPILE}"
	--allow_tf32 "${ALLOW_TF32}"
	--cudnn_benchmark "${CUDNN_BENCHMARK}"
	--deterministic "${DETERMINISTIC}"
	--num_workers "${NUM_WORKERS}"
	--prefetch_factor "${PREFETCH_FACTOR}"
	--persistent_workers "${PERSISTENT_WORKERS}"
	--pin_memory "${PIN_MEMORY}"
	--min_batch_size "${MIN_BATCH_SIZE}"
	--max_oom_retries "${MAX_OOM_RETRIES}"
	--test_tsv "${TEST_TSV}"
	--class_file "${CLASS_FILE}"
	--site_output_tsv "${OUTPUT_SITE_SCORE_TSV}"
	--transcript_output_tsv "${OUTPUT_TRANS_SCORE_TSV}"
	--eval_output_txt "${OUTPUT_EVAL_SCORE_TXT}"
)

if [[ -n "${FOCAL_ALPHA_POS}" ]]; then
	RUN_ARGS+=(--focal_alpha_pos "${FOCAL_ALPHA_POS}")
fi
if [[ -n "${DONOR_FOCAL_ALPHA_POS}" ]]; then
	RUN_ARGS+=(--donor_focal_alpha_pos "${DONOR_FOCAL_ALPHA_POS}")
fi
if [[ -n "${ACCEPTOR_FOCAL_ALPHA_POS}" ]]; then
	RUN_ARGS+=(--acceptor_focal_alpha_pos "${ACCEPTOR_FOCAL_ALPHA_POS}")
fi
if [[ -n "${ASYM_ALPHA_POS}" ]]; then
	RUN_ARGS+=(--asym_alpha_pos "${ASYM_ALPHA_POS}")
fi
if [[ -n "${DONOR_ASYM_ALPHA_POS}" ]]; then
	RUN_ARGS+=(--donor_asym_alpha_pos "${DONOR_ASYM_ALPHA_POS}")
fi
if [[ -n "${ACCEPTOR_ASYM_ALPHA_POS}" ]]; then
	RUN_ARGS+=(--acceptor_asym_alpha_pos "${ACCEPTOR_ASYM_ALPHA_POS}")
fi
if [[ -n "${ASYM_GAMMA_POS}" ]]; then
	RUN_ARGS+=(--asym_gamma_pos "${ASYM_GAMMA_POS}")
fi
if [[ -n "${ASYM_GAMMA_NEG}" ]]; then
	RUN_ARGS+=(--asym_gamma_neg "${ASYM_GAMMA_NEG}")
fi
if [[ -n "${DONOR_BATCH_SIZE}" ]]; then
	RUN_ARGS+=(--donor_batch_size "${DONOR_BATCH_SIZE}")
fi
if [[ -n "${ACCEPTOR_BATCH_SIZE}" ]]; then
	RUN_ARGS+=(--acceptor_batch_size "${ACCEPTOR_BATCH_SIZE}")
fi
if [[ -n "${DONOR_LR}" ]]; then
	RUN_ARGS+=(--donor_lr "${DONOR_LR}")
fi
if [[ -n "${ACCEPTOR_LR}" ]]; then
	RUN_ARGS+=(--acceptor_lr "${ACCEPTOR_LR}")
fi
if [[ -n "${DONOR_LOSS}" ]]; then
	RUN_ARGS+=(--donor_loss "${DONOR_LOSS}")
fi
if [[ -n "${ACCEPTOR_LOSS}" ]]; then
	RUN_ARGS+=(--acceptor_loss "${ACCEPTOR_LOSS}")
fi
if [[ -n "${DONOR_CONV_CHANNELS}" ]]; then
	RUN_ARGS+=(--donor_conv_channels "${DONOR_CONV_CHANNELS}")
fi
if [[ -n "${ACCEPTOR_CONV_CHANNELS}" ]]; then
	RUN_ARGS+=(--acceptor_conv_channels "${ACCEPTOR_CONV_CHANNELS}")
fi
if [[ -n "${DONOR_KERNEL_SIZE}" ]]; then
	RUN_ARGS+=(--donor_kernel_size "${DONOR_KERNEL_SIZE}")
fi
if [[ -n "${ACCEPTOR_KERNEL_SIZE}" ]]; then
	RUN_ARGS+=(--acceptor_kernel_size "${ACCEPTOR_KERNEL_SIZE}")
fi
if [[ -n "${DONOR_DROPOUT}" ]]; then
	RUN_ARGS+=(--donor_dropout "${DONOR_DROPOUT}")
fi
if [[ -n "${ACCEPTOR_DROPOUT}" ]]; then
	RUN_ARGS+=(--acceptor_dropout "${ACCEPTOR_DROPOUT}")
fi
if [[ -n "${DONOR_FC_HIDDEN}" ]]; then
	RUN_ARGS+=(--donor_fc_hidden "${DONOR_FC_HIDDEN}")
fi
if [[ -n "${ACCEPTOR_FC_HIDDEN}" ]]; then
	RUN_ARGS+=(--acceptor_fc_hidden "${ACCEPTOR_FC_HIDDEN}")
fi
if [[ -n "${DONOR_WEIGHT_DECAY}" ]]; then
	RUN_ARGS+=(--donor_weight_decay "${DONOR_WEIGHT_DECAY}")
fi
if [[ -n "${ACCEPTOR_WEIGHT_DECAY}" ]]; then
	RUN_ARGS+=(--acceptor_weight_decay "${ACCEPTOR_WEIGHT_DECAY}")
fi
if [[ -n "${DONOR_ETA_MIN_RATIO}" ]]; then
	RUN_ARGS+=(--donor_eta_min_ratio "${DONOR_ETA_MIN_RATIO}")
fi
if [[ -n "${ACCEPTOR_ETA_MIN_RATIO}" ]]; then
	RUN_ARGS+=(--acceptor_eta_min_ratio "${ACCEPTOR_ETA_MIN_RATIO}")
fi
if [[ -n "${DONOR_VAL_FRAC}" ]]; then
	RUN_ARGS+=(--donor_val_frac "${DONOR_VAL_FRAC}")
fi
if [[ -n "${ACCEPTOR_VAL_FRAC}" ]]; then
	RUN_ARGS+=(--acceptor_val_frac "${ACCEPTOR_VAL_FRAC}")
fi
if [[ -n "${DONOR_GRAD_CLIP}" ]]; then
	RUN_ARGS+=(--donor_grad_clip "${DONOR_GRAD_CLIP}")
fi
if [[ -n "${ACCEPTOR_GRAD_CLIP}" ]]; then
	RUN_ARGS+=(--acceptor_grad_clip "${ACCEPTOR_GRAD_CLIP}")
fi
if [[ -n "${DONOR_POS_WEIGHT_CAP}" ]]; then
	RUN_ARGS+=(--donor_pos_weight_cap "${DONOR_POS_WEIGHT_CAP}")
fi
if [[ -n "${ACCEPTOR_POS_WEIGHT_CAP}" ]]; then
	RUN_ARGS+=(--acceptor_pos_weight_cap "${ACCEPTOR_POS_WEIGHT_CAP}")
fi
if [[ -n "${DONOR_FOCAL_GAMMA}" ]]; then
	RUN_ARGS+=(--donor_focal_gamma "${DONOR_FOCAL_GAMMA}")
fi
if [[ -n "${ACCEPTOR_FOCAL_GAMMA}" ]]; then
	RUN_ARGS+=(--acceptor_focal_gamma "${ACCEPTOR_FOCAL_GAMMA}")
fi
if [[ -n "${DONOR_ASYM_GAMMA_POS}" ]]; then
	RUN_ARGS+=(--donor_asym_gamma_pos "${DONOR_ASYM_GAMMA_POS}")
fi
if [[ -n "${ACCEPTOR_ASYM_GAMMA_POS}" ]]; then
	RUN_ARGS+=(--acceptor_asym_gamma_pos "${ACCEPTOR_ASYM_GAMMA_POS}")
fi
if [[ -n "${DONOR_ASYM_GAMMA_NEG}" ]]; then
	RUN_ARGS+=(--donor_asym_gamma_neg "${DONOR_ASYM_GAMMA_NEG}")
fi
if [[ -n "${ACCEPTOR_ASYM_GAMMA_NEG}" ]]; then
	RUN_ARGS+=(--acceptor_asym_gamma_neg "${ACCEPTOR_ASYM_GAMMA_NEG}")
fi
if [[ "${SKIP_TRAINING}" == "1" ]]; then
	RUN_ARGS+=(--skip_train)
fi
if [[ "${CONTINUE_TRAINING}" == "1" ]]; then
	RUN_ARGS+=(--continue_train)
fi
if [[ "${TRAIN_ONLY}" == "1" ]]; then
	RUN_ARGS+=(--train_only)
fi
if [[ -n "${PRECOMPUTED_SITE_SCORE_TSV}" ]]; then
	RUN_ARGS+=(--site_score_tsv "${PRECOMPUTED_SITE_SCORE_TSV}")
fi

echo "[cnn.sh] Start unified pipeline"
echo "[cnn.sh] species=${SPECIES} perf_mode=${PERF_MODE} train_only=${TRAIN_ONLY}"
python3 "${PROJECT_ROOT}/src/run_model.py" "${RUN_ARGS[@]}"
echo "[cnn.sh] Done"
echo "[cnn.sh] site_score=${OUTPUT_SITE_SCORE_TSV}"
echo "[cnn.sh] transcript_score=${OUTPUT_TRANS_SCORE_TSV}"
echo "[cnn.sh] eval_score=${OUTPUT_EVAL_SCORE_TXT}"
