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
MODEL="cnn_v2_pair"
SPECIES="${SPECIES:-Mmus,Athal,Dmel,Hsap}"
INTRONMODEL_AUTO_TMUX="${INTRONMODEL_AUTO_TMUX:-on}"  # off | on | auto
DONOR_LEN="100"
ACCEPTOR_LEN="100"
TRAIN_POS_PATH=""
TRAIN_NEG_PATH=""
TEST_TSV_PATH=""
CLASS_FILE_PATH=""
REF_GFF_PATH=""
EPOCHS="20"
MAX_EPOCHS="200"
EARLY_STOP_PATIENCE="12"
EARLY_STOP_MIN_DELTA="0.0"
BATCH_SIZE="512"
LR="5e-4"
LOSS="focal"
CONV_CHANNELS="64,128,256"
KERNEL_SIZES="7,7,7"
MAX_POOL_SIZE="2"
HEAD_TYPE="gap"
DROPOUT="0.3"
FC_HIDDEN="128"
WEIGHT_DECAY="0.01"
ETA_MIN_RATIO="0.01"
VAL_FRAC="0.25"
GRAD_CLIP="5.0"
POS_WEIGHT_CAP="20.0"
FOCAL_GAMMA="2.0"
FOCAL_ALPHA_POS=""
ASYM_GAMMA_POS="0.0"
ASYM_GAMMA_NEG="4.0"
ASYM_ALPHA_POS=""
TRAIN_TARGET="pair"
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
INFER_BATCH_SIZE="2048"
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
MIN_BATCH_SIZE="64"
MAX_OOM_RETRIES="8"
MPS_MAX_BATCH_SIZE="2048"
USE_TUNED_HPARAMS="required"   # off | auto | required
TUNED_CONFIG_PATH=""
SHARED_TUNED_CONFIG_PATH=""
TUNED_TARGET="auto"        # auto | pair
INTRON_SCORE_OP="*"
NAME_FIELDS="none"
TRANSCRIPT_SCORE_AGG="min"
SOFTMIN_TAU="1.0"
SEED="1337"
TAG=""
SYNTHESIZE_MODE="off"
VISUALIZE="true"
SKIP_TRAINING="0"
CONTINUE_TRAINING="0"
TRAIN_ONLY="0"
CHECKPOINT_TOP_K="3"
CHECKPOINT_PRUNE_DRY_RUN="0"
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

run_species_once() {
	local species="$1"
	local assigned_gpu_id="${2-}"
	local tuned_model_name
	local best_config_filename
	local synthesize_resolved
	local resolved_tag
	local resolved_train_pos_path
	local resolved_train_neg_path

	tuned_model_name="$(
		intronmodel_resolve_pair_tuning_model_name "${SYNTHESIZE_MODE}"
	)"
	best_config_filename="$(
		intronmodel_resolve_pair_best_config_filename "${SYNTHESIZE_MODE}"
	)"

	synthesize_resolved="$(
		intronmodel_resolve_pair_synthesize_defaults \
			"${species}" \
			"${SYNTHESIZE_MODE}" \
			"${TAG}" \
			"${TRAIN_POS_PATH}" \
			"${TRAIN_NEG_PATH}"
	)"
	IFS=$'\t' read -r resolved_tag resolved_train_pos_path \
		resolved_train_neg_path <<<"${synthesize_resolved}"
	resolved_train_paths="$(
		intronmodel_resolve_and_validate_train_paths \
			"run_cnn_v2_pair.sh" \
			"${species}" \
			"${resolved_train_pos_path}" \
			"${resolved_train_neg_path}"
	)"
	IFS=$'\t' read -r resolved_train_pos_path resolved_train_neg_path \
		<<<"${resolved_train_paths}"

	args=(
		--model "${MODEL}"
		--species "${species}"
		--donor_len "${DONOR_LEN}"
		--acceptor_len "${ACCEPTOR_LEN}"
		--device "${DEVICE}"
		--seed "${SEED}"
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
		--val_frac "${VAL_FRAC}"
		--epochs "${EPOCHS}"
		--max_epochs "${MAX_EPOCHS}"
		--early_stop_patience "${EARLY_STOP_PATIENCE}"
		--early_stop_min_delta "${EARLY_STOP_MIN_DELTA}"
		--batch_size "${BATCH_SIZE}"
		--lr "${LR}"
		--loss "${LOSS}"
		--conv_channels "${CONV_CHANNELS}"
		--kernel_sizes "${KERNEL_SIZES}"
		--max_pool_size "${MAX_POOL_SIZE}"
		--head_type "${HEAD_TYPE}"
		--dropout "${DROPOUT}"
		--fc_hidden "${FC_HIDDEN}"
		--weight_decay "${WEIGHT_DECAY}"
		--eta_min_ratio "${ETA_MIN_RATIO}"
		--grad_clip "${GRAD_CLIP}"
		--pos_weight_cap "${POS_WEIGHT_CAP}"
		--focal_gamma "${FOCAL_GAMMA}"
		--focal_alpha_pos "${FOCAL_ALPHA_POS}"
		--asym_gamma_pos "${ASYM_GAMMA_POS}"
		--asym_gamma_neg "${ASYM_GAMMA_NEG}"
		--asym_alpha_pos "${ASYM_ALPHA_POS}"
		--transcript_score_agg "${TRANSCRIPT_SCORE_AGG}"
		--softmin_tau "${SOFTMIN_TAU}"
		--train_target "${TRAIN_TARGET}"
		--intron_score_op "${INTRON_SCORE_OP}"
		--visualize "${VISUALIZE}"
		--checkpoint_top_k "${CHECKPOINT_TOP_K}"
		--checkpoint_prune_dry_run "${CHECKPOINT_PRUNE_DRY_RUN}"
		--infer_batch_size "${INFER_BATCH_SIZE}"
		--infer_use_amp "${INFER_USE_AMP}"
		--infer_amp_dtype "${INFER_AMP_DTYPE}"
		--infer_compile "${INFER_COMPILE}"
		--infer_compile_mode "${INFER_COMPILE_MODE}"
	)
	append_flag_if_truthy "skip_train" "${SKIP_TRAINING}"
	append_flag_if_truthy "continue_train" "${CONTINUE_TRAINING}"
	append_flag_if_truthy "train_only" "${TRAIN_ONLY}"
	if [[ -n "${resolved_tag}" ]]; then
		append_arg_if_set "tag" "${resolved_tag}"
	fi
	append_arg_if_set "train_pos_path" "${resolved_train_pos_path}"
	append_arg_if_set "train_neg_path" "${resolved_train_neg_path}"
	append_arg_if_set "test_tsv" "${TEST_TSV_PATH}"
	append_arg_if_set "class_file" "${CLASS_FILE_PATH}"
	append_arg_if_set "ref_gff" "${REF_GFF_PATH}"

	tuned_path=""
	tuned_output=""
	tuned_args=()
	if [[ "${USE_TUNED_HPARAMS_MODE}" != "off" ]]; then
		tuned_path="$(
			intronmodel_resolve_tuned_config_path \
				"${DATA_ROOT}" \
				"${species}" \
				"${tuned_model_name}" \
				"${RESOLVED_TUNED_TARGET}" \
				"${TUNED_CONFIG_PATH}" \
				"${SHARED_TUNED_CONFIG_PATH}" \
				"${best_config_filename}"
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
				# Keep wrapper-controlled tag handling so synth does not leak in.
				if [[ "${tuned_key}" == "tag" ]]; then
					continue
				fi
				tuned_args+=(--"${tuned_key}" "${tuned_value}")
				loaded_count=$((loaded_count + 1))
			done <<<"${tuned_output}"
			echo "[cnn_v2_pair.sh] tuned params loaded from ${tuned_path} "\
				"(species=${species}, count=${loaded_count})"
		fi
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
