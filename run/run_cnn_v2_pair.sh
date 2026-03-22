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
FUSION_MODE="late"    # late | mid | early
EMBEDDING_DIM="32"
BPE_PRETRAINED_MODEL_NAME="zhihan1996/DNABERT-2-117M"
BPE_PRETRAINED_REVISION=""
BPE_TRUST_REMOTE_CODE="0"
CONV_CHANNELS=""
KERNEL_SIZES=""
DONOR_CONV_CHANNELS=""
ACCEPTOR_CONV_CHANNELS=""
DONOR_KERNEL_SIZES=""
ACCEPTOR_KERNEL_SIZES=""
MAX_POOL_SIZE="2"
CONV_STRIDE="1"
HEAD_TYPE="gap"
FC_HIDDEN="128"
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
GPU_IDS="auto"            # auto: detect visible GPUs for species parallel.
MAX_PARALLEL_TRIALS="auto"  # auto: use one concurrent species per GPU id.
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

run_species_once() {
	local species="$1"
	local assigned_gpu_id="${2-}"

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
		--batch_size "${BATCH_SIZE}"
		--lr "${LR}"
		--loss "${LOSS}"
		--input_mode "${INPUT_MODE}"
		--pair_mode "${PAIR_MODE}"
		--fusion_mode "${FUSION_MODE}"
		--embedding_dim "${EMBEDDING_DIM}"
		--bpe_pretrained_model_name "${BPE_PRETRAINED_MODEL_NAME}"
		--bpe_trust_remote_code "${BPE_TRUST_REMOTE_CODE}"
		--conv_channels "${CONV_CHANNELS}"
		--kernel_sizes "${KERNEL_SIZES}"
		--donor_conv_channels "${DONOR_CONV_CHANNELS}"
		--acceptor_conv_channels "${ACCEPTOR_CONV_CHANNELS}"
		--donor_kernel_sizes "${DONOR_KERNEL_SIZES}"
		--acceptor_kernel_sizes "${ACCEPTOR_KERNEL_SIZES}"
		--max_pool_size "${MAX_POOL_SIZE}"
		--conv_stride "${CONV_STRIDE}"
		--head_type "${HEAD_TYPE}"
		--fc_hidden "${FC_HIDDEN}"
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
	if [[ ${#tuned_args[@]} -gt 0 ]]; then
		args+=("${tuned_args[@]}")
	fi

	echo "[cnn_v2_pair.sh] species=${species} input_mode=${INPUT_MODE} pair_mode=${PAIR_MODE} fusion_mode=${FUSION_MODE}"
	local pythonpath="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
	if [[ -n "${assigned_gpu_id}" ]]; then
		CUDA_VISIBLE_DEVICES="${assigned_gpu_id}" \
			PYTHONPATH="${pythonpath}" \
			python3 "${PROJECT_ROOT}/src/run_model.py" "${args[@]}"
		return $?
	fi
	PYTHONPATH="${pythonpath}" \
	python3 "${PROJECT_ROOT}/src/run_model.py" "${args[@]}"
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


def _mask_to_sequence_transform(value: object) -> str:
    if isinstance(value, bool):
        normalized = "on" if value else "off"
    else:
        normalized = str(value).strip().lower()
    if normalized in {"on", "1", "true", "yes"}:
        return "mask_outside_intron_n"
    if normalized in {"off", "0", "false", "no"}:
        return "none"
    raise ValueError("mask must be on or off.")


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
context = payload.get("hparam_context")
fixed_run_args = None
if isinstance(context, dict):
    fixed_run_args = context.get("fixed_run_args")
if isinstance(fixed_run_args, dict):
    for key in sorted(fixed_run_args):
        value = fixed_run_args[key]
        if value is None:
            continue
        print(f"{key}\t{_scalar_to_text(value)}")
sequence_transform_value = sampled_params.pop("sequence_transform", None)
mask_value = sampled_params.pop("mask", None)
if mask_value is not None:
    print(f"sequence_transform\t{_mask_to_sequence_transform(mask_value)}")
elif sequence_transform_value is not None:
    print(
        "sequence_transform\t"
        f"{_scalar_to_text(sequence_transform_value)}"
    )
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

IFS=',' read -r -a SPECIES_LIST_RESOLVED <<<"${SPECIES}"
mapfile -t GPU_ID_LIST < <(
	intronmodel_resolve_gpu_ids "cnn_v2_pair.sh" "${GPU_IDS}" "${DEVICE}"
)
PARALLEL_SLOT_COUNT="$(
	intronmodel_resolve_parallel_slots \
		"cnn_v2_pair.sh" \
		"${MAX_PARALLEL_TRIALS}" \
		"${#GPU_ID_LIST[@]}"
)"
if [[ ${#SPECIES_LIST_RESOLVED[@]} -eq 0 ]]; then
	echo "[cnn_v2_pair.sh] SPECIES resolved to an empty list." >&2
	exit 1
fi
if [[ ${#SPECIES_LIST_RESOLVED[@]} -le 1 || ${#GPU_ID_LIST[@]} -le 1 || ${PARALLEL_SLOT_COUNT} -le 1 ]]; then
	serial_gpu_id=""
	if [[ ${#GPU_ID_LIST[@]} -gt 0 ]]; then
		serial_gpu_id="${GPU_ID_LIST[0]}"
	fi
	for species_raw in "${SPECIES_LIST_RESOLVED[@]}"; do
		species="$(echo "${species_raw}" | xargs)"
		if [[ -z "${species}" ]]; then
			continue
		fi
		run_species_once "${species}" "${serial_gpu_id}"
	done
else
	selected_gpu_ids=("${GPU_ID_LIST[@]:0:${PARALLEL_SLOT_COUNT}}")
	gpu_csv="$(IFS=,; echo "${selected_gpu_ids[*]}")"
	echo "[cnn_v2_pair.sh] species-parallel run across GPUs: ${gpu_csv}"
	declare -A pid_to_species=()
	declare -A pid_to_gpu=()
	available_gpu_ids=("${selected_gpu_ids[@]}")
	pending_species=("${SPECIES_LIST_RESOLVED[@]}")
	running_count=0
	stop_submitting=0
	first_error_code=0
	while [[ ${#pending_species[@]} -gt 0 || ${running_count} -gt 0 ]]; do
		while [[ ${#pending_species[@]} -gt 0 && ${#available_gpu_ids[@]} -gt 0 && ${stop_submitting} -eq 0 ]]; do
			species_raw="${pending_species[0]}"
			pending_species=("${pending_species[@]:1}")
			species="$(echo "${species_raw}" | xargs)"
			if [[ -z "${species}" ]]; then
				continue
			fi
			gpu_id="${available_gpu_ids[0]}"
			available_gpu_ids=("${available_gpu_ids[@]:1}")
			echo "[cnn_v2_pair.sh] species dispatch: ${species} -> gpu=${gpu_id}"
			run_species_once "${species}" "${gpu_id}" &
			pid=$!
			pid_to_species["${pid}"]="${species}"
			pid_to_gpu["${pid}"]="${gpu_id}"
			running_count=$((running_count + 1))
		done

		if [[ ${running_count} -eq 0 ]]; then
			break
		fi

		if wait -n -p completed_pid; then
			completed_code=0
		else
			completed_code=$?
		fi
		completed_species="${pid_to_species[$completed_pid]:-}"
		completed_gpu="${pid_to_gpu[$completed_pid]:-}"
		unset "pid_to_species[${completed_pid}]"
		unset "pid_to_gpu[${completed_pid}]"
		if [[ -n "${completed_gpu}" ]]; then
			available_gpu_ids+=("${completed_gpu}")
		fi
		running_count=$((running_count - 1))
		if [[ -n "${completed_species}" ]]; then
			echo "[cnn_v2_pair.sh] species complete: ${completed_species} gpu=${completed_gpu} exit=${completed_code}"
		fi
		if [[ ${completed_code} -ne 0 && ${first_error_code} -eq 0 ]]; then
			first_error_code="${completed_code}"
			stop_submitting=1
		fi
	done
	exit "${first_error_code}"
fi
