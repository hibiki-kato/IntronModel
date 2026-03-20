#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
	echo "[cnn_v2_pair.sh] This script is config-only." \
		"Edit top CONFIG and run without args." >&2
	exit 1
fi

# --------------------------
# CONFIG (edit here)
# --------------------------
set -a
SPECIES="Mmus,Athal,Dmel,Hsap"
DONOR_LEN="100"
ACCEPTOR_LEN="100"
SEQUENCE_TRANSFORM="none"
TRAIN_TARGET="pair"

EPOCHS="10"
MAX_EPOCHS="200"
EARLY_STOP_PATIENCE="12"
EARLY_STOP_MIN_DELTA="0.0"
BATCH_SIZE="512"
LR="5e-4"
LOSS="focal"
INPUT_MODE="onehot"   # onehot | kmer3 | bpe
PAIR_MODE="pair"      # pair | independent
EMBEDDING_DIM="32"
BPE_PRETRAINED_MODEL_NAME="zhihan1996/DNABERT-2-117M"
BPE_PRETRAINED_REVISION=""
BPE_TRUST_REMOTE_CODE="0"
DROPOUT="0.3"
WEIGHT_DECAY="0.01"
ETA_MIN_RATIO="0.01"
VAL_FRAC="0.2"
GRAD_CLIP="5.0"
POS_WEIGHT_CAP="20.0"
FOCAL_GAMMA="2.0"
FOCAL_ALPHA_POS=""
F1_LAMBDA="0.1"
ASYM_GAMMA_POS="0.0"
ASYM_GAMMA_NEG="4.0"
ASYM_ALPHA_POS=""

SEED="1337"
DEVICE="auto"
USE_AMP="1"
AMP_DTYPE="auto"
COMPILE_MODE="off"
INTRONMODEL_TORCH_COMPILE_STRATEGY="default-then-off"  # reduce-overhead only
INTRONMODEL_TORCH_COMPILE_STICKY_MODE="reduce-overhead"
INTRONMODEL_TORCH_COMPILE_DISABLED_MODES="max-autotune"
TORCHINDUCTOR_MAX_AUTOTUNE_GEMM="0"
ALLOW_TF32="1"
CUDNN_BENCHMARK="1"
DETERMINISTIC="0"
NUM_WORKERS="auto"
PREFETCH_FACTOR="4"
PERSISTENT_WORKERS="1"
PIN_MEMORY="1"
MIN_BATCH_SIZE="64"
MAX_OOM_RETRIES="8"

TEST_TSV_PATH=""
CLASS_FILE_PATH=""
REF_GFF_PATH=""
NAME_FIELDS="none"
TRANSCRIPT_SCORE_AGG="min"
SOFTMIN_TAU="1.0"
TAG=""
USE_TUNED_HPARAMS="auto"   # off | auto | required
TUNED_CONFIG_PATH=""
SHARED_TUNED_CONFIG_PATH=""
TUNED_TARGET="auto"        # auto | pair
set +a

# --------------------------
# Runtime implementation
# --------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"
intronmodel_activate_conda "intronmodel"
intronmodel_init_paths "${BASH_SOURCE[0]}"

normalize_use_tuned_mode() {
	local raw_mode="$1"
	local normalized
	normalized="$(echo "${raw_mode}" | tr '[:upper:]' '[:lower:]' | xargs)"
	case "${normalized}" in
		off | auto | required)
			printf '%s\n' "${normalized}"
			;;
		*)
			echo "[cnn_v2_pair.sh] USE_TUNED_HPARAMS must be off|auto|required." >&2
			exit 1
			;;
	esac
}

resolve_tuned_target() {
	local configured_target="$1"
	local normalized
	normalized="$(echo "${configured_target}" | tr '[:upper:]' '[:lower:]' | xargs)"
	if [[ "${normalized}" != "auto" && "${normalized}" != "" ]]; then
		printf '%s\n' "${normalized}"
		return 0
	fi
	printf 'pair\n'
}

resolve_tuned_config_path() {
	local species="$1"
	local tuned_target="$2"
	if [[ -n "${TUNED_CONFIG_PATH}" ]]; then
		printf '%s\n' "${TUNED_CONFIG_PATH}"
		return 0
	fi
	local task_path="${DATA_ROOT}/${species}/tuning/cnn_v2_pair/${tuned_target}/best_config.json"
	if [[ -f "${task_path}" ]]; then
		printf '%s\n' "${task_path}"
		return 0
	fi
	if [[ -n "${SHARED_TUNED_CONFIG_PATH}" ]]; then
		printf '%s\n' "${SHARED_TUNED_CONFIG_PATH}"
		return 0
	fi
	local shared_path="${DATA_ROOT}/${species}/tuning/cnn_v2_pair/best_config.json"
	if [[ -f "${shared_path}" ]]; then
		printf '%s\n' "${shared_path}"
		return 0
	fi
	printf ''
}

load_tuned_overrides() {
	local config_path="$1"
	python3 - "${config_path}" <<'PY'
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def _scalar_to_text(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite float value in sampled_params.")
        return format(value, ".15g")
    return str(value)


config_path = Path(sys.argv[1]).resolve()
payload = json.loads(config_path.read_text(encoding="utf-8"))
if not isinstance(payload, dict):
    raise ValueError("best_config payload must be an object.")
status = str(payload.get("status", "")).strip().lower()
if status != "ok":
    raise ValueError(f"Expected status='ok', got: {status or '<missing>'}")
sampled_params = payload.get("sampled_params")
if not isinstance(sampled_params, dict):
    raise ValueError("sampled_params is missing or invalid.")
for key in sorted(sampled_params):
    value = sampled_params[key]
    if value is None:
        continue
    print(f"{key}\t{_scalar_to_text(value)}")
PY
}

USE_TUNED_HPARAMS_MODE="$(normalize_use_tuned_mode "${USE_TUNED_HPARAMS}")"
if [[ "${USE_TUNED_HPARAMS_MODE}" != "off" ]]; then
	RESOLVED_TUNED_TARGET="$(resolve_tuned_target "${TUNED_TARGET}")"
fi

IFS=',' read -r -a SPECIES_LIST <<<"${SPECIES}"
for species_raw in "${SPECIES_LIST[@]}"; do
	species="$(echo "${species_raw}" | xargs)"
	if [[ -z "${species}" ]]; then
		continue
	fi

	args=(
		--model cnn_v2_pair
		--species "${species}"
		--donor_len "${DONOR_LEN}"
		--acceptor_len "${ACCEPTOR_LEN}"
		--device "${DEVICE}"
		--seed "${SEED}"
		--name_fields "${NAME_FIELDS}"
		--epochs "${EPOCHS}"
		--max_epochs "${MAX_EPOCHS}"
		--early_stop_patience "${EARLY_STOP_PATIENCE}"
		--early_stop_min_delta "${EARLY_STOP_MIN_DELTA}"
		--train_target "${TRAIN_TARGET}"
		--sequence_transform "${SEQUENCE_TRANSFORM}"
		--batch_size "${BATCH_SIZE}"
		--lr "${LR}"
		--loss "${LOSS}"
		--input_mode "${INPUT_MODE}"
		--pair_mode "${PAIR_MODE}"
		--embedding_dim "${EMBEDDING_DIM}"
		--bpe_pretrained_model_name "${BPE_PRETRAINED_MODEL_NAME}"
		--bpe_trust_remote_code "${BPE_TRUST_REMOTE_CODE}"
		--dropout "${DROPOUT}"
		--weight_decay "${WEIGHT_DECAY}"
		--eta_min_ratio "${ETA_MIN_RATIO}"
		--val_frac "${VAL_FRAC}"
		--grad_clip "${GRAD_CLIP}"
		--pos_weight_cap "${POS_WEIGHT_CAP}"
		--focal_gamma "${FOCAL_GAMMA}"
		--f1_lambda "${F1_LAMBDA}"
		--asym_gamma_pos "${ASYM_GAMMA_POS}"
		--asym_gamma_neg "${ASYM_GAMMA_NEG}"
		--use_amp "${USE_AMP}"
		--amp_dtype "${AMP_DTYPE}"
		--compile_mode "${COMPILE_MODE}"
		--allow_tf32 "${ALLOW_TF32}"
		--cudnn_benchmark "${CUDNN_BENCHMARK}"
		--deterministic "${DETERMINISTIC}"
		--num_workers "${NUM_WORKERS}"
		--prefetch_factor "${PREFETCH_FACTOR}"
		--persistent_workers "${PERSISTENT_WORKERS}"
		--pin_memory "${PIN_MEMORY}"
		--min_batch_size "${MIN_BATCH_SIZE}"
		--max_oom_retries "${MAX_OOM_RETRIES}"
		--transcript_score_agg "${TRANSCRIPT_SCORE_AGG}"
		--softmin_tau "${SOFTMIN_TAU}"
	)

	tuned_path=""
	tuned_output=""
	tuned_args=()
	if [[ "${USE_TUNED_HPARAMS_MODE}" != "off" ]]; then
		tuned_path="$(
			resolve_tuned_config_path \
				"${species}" \
				"${RESOLVED_TUNED_TARGET}"
		)"
		if [[ -z "${tuned_path}" ]]; then
			if [[ "${USE_TUNED_HPARAMS_MODE}" == "required" ]]; then
				echo "[cnn_v2_pair.sh] tuned config is required but not found: "\
					"species=${species} target=${RESOLVED_TUNED_TARGET}" >&2
				exit 1
			fi
			echo "[cnn_v2_pair.sh] tuned config not found; "\
				"using CONFIG defaults for species=${species}." >&2
		elif [[ ! -f "${tuned_path}" ]]; then
			if [[ "${USE_TUNED_HPARAMS_MODE}" == "required" ]]; then
				echo "[cnn_v2_pair.sh] tuned config path not found: ${tuned_path}" >&2
				exit 1
			fi
			echo "[cnn_v2_pair.sh] tuned config path not found: ${tuned_path}; "\
				"using CONFIG defaults for species=${species}." >&2
			tuned_path=""
		fi
	fi

	if [[ -n "${tuned_path}" ]]; then
		if ! tuned_output="$(load_tuned_overrides "${tuned_path}" 2>&1)"; then
			if [[ "${USE_TUNED_HPARAMS_MODE}" == "required" ]]; then
				echo "[cnn_v2_pair.sh] failed to load tuned config: ${tuned_path}" >&2
				echo "[cnn_v2_pair.sh] detail: ${tuned_output}" >&2
				exit 1
			fi
			echo "[cnn_v2_pair.sh] failed to load tuned config: ${tuned_path}; "\
				"using CONFIG defaults for species=${species}." >&2
		else
			loaded_count=0
			while IFS= read -r line; do
				if [[ -z "${line}" ]]; then
					continue
				fi
				IFS=$'\t' read -r tuned_key tuned_value <<<"${line}"
				if [[ -z "${tuned_key}" || -z "${tuned_value}" ]]; then
					continue
				fi
				tuned_args+=(--"${tuned_key}" "${tuned_value}")
				loaded_count=$((loaded_count + 1))
			done <<<"${tuned_output}"
			echo "[cnn_v2_pair.sh] tuned params loaded from ${tuned_path} "\
				"(species=${species}, count=${loaded_count})"
		fi
	fi

	if [[ -n "${BPE_PRETRAINED_REVISION}" ]]; then
		args+=(--bpe_pretrained_revision "${BPE_PRETRAINED_REVISION}")
	fi
	if [[ -n "${FOCAL_ALPHA_POS}" ]]; then
		args+=(--focal_alpha_pos "${FOCAL_ALPHA_POS}")
	fi
	if [[ -n "${ASYM_ALPHA_POS}" ]]; then
		args+=(--asym_alpha_pos "${ASYM_ALPHA_POS}")
	fi
	if [[ -n "${TEST_TSV_PATH}" ]]; then
		args+=(--test_tsv "${TEST_TSV_PATH}")
	fi
	if [[ -n "${CLASS_FILE_PATH}" ]]; then
		args+=(--class_file "${CLASS_FILE_PATH}")
	fi
	if [[ -n "${REF_GFF_PATH}" ]]; then
		args+=(--ref_gff "${REF_GFF_PATH}")
	fi
	if [[ -n "${TAG}" ]]; then
		args+=(--tag "${TAG}")
	fi
	if [[ ${#tuned_args[@]} -gt 0 ]]; then
		args+=("${tuned_args[@]}")
	fi

	echo "[cnn_v2_pair.sh] species=${species} input_mode=${INPUT_MODE} pair_mode=${PAIR_MODE}"
	PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
		python3 "${PROJECT_ROOT}/src/run_model.py" "${args[@]}"
done
