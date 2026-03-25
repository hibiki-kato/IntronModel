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
INTRONMODEL_AUTO_TMUX="on"  # off | on | auto
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
USE_TUNED_HPARAMS="required"   # off | auto | required
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
source "${SCRIPT_DIR}/lib/tuned_config.sh"
intronmodel_activate_conda "intronmodel"
intronmodel_init_paths "${BASH_SOURCE[0]}"
intronmodel_enable_auto_tmux "${PROJECT_ROOT}" "$0" "${BASH_SOURCE[0]##*/}"

intronmodel_abort_parallel_run() {
	trap - INT TERM HUP
	kill -TERM 0 2>/dev/null || true
	exit 130
}

trap 'intronmodel_abort_parallel_run' INT TERM HUP

run_species_once() {
	local species="$1"
	local assigned_gpu_id="${2-}"

	args=(
		--model cnn_v2_pair
		--species "${species}"
		--device "${DEVICE}"
		--name_fields "${NAME_FIELDS}"
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
			intronmodel_resolve_tuned_config_path \
				"${DATA_ROOT}" \
				"${species}" \
				"cnn_v2_pair" \
				"${RESOLVED_TUNED_TARGET}" \
				"${TUNED_CONFIG_PATH}" \
				"${SHARED_TUNED_CONFIG_PATH}"
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
		if ! tuned_output="$(intronmodel_load_tuned_overrides "${tuned_path}" 2>&1)"; then
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

	echo "[cnn_v2_pair.sh] species=${species}"
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

USE_TUNED_HPARAMS_MODE="$(
	intronmodel_normalize_use_tuned_mode "${USE_TUNED_HPARAMS}" "cnn_v2_pair.sh"
)"
if [[ "${USE_TUNED_HPARAMS_MODE}" != "off" ]]; then
	RESOLVED_TUNED_TARGET="$(
		intronmodel_resolve_tuned_target "${TUNED_TARGET}" "pair"
	)"
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
