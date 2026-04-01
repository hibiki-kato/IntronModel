#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
	echo "[cnn_v2.sh] This script is config-only." \
		"Edit top CONFIG and run without args." >&2
	exit 1
fi

# --------------------------
# CONFIG (edit here)
# --------------------------
set -a
MODEL="cnn_v2"
SPECIES="Dmel, Hsap, Mmus"
INTRONMODEL_AUTO_TMUX="off"
DEVICE="auto"
GPU_IDS="auto"            # auto: detect visible GPUs for species parallel.
MAX_PARALLEL_TRIALS="auto"  # auto: use one concurrent species per GPU id.
DONOR_LEN="100"
ACCEPTOR_LEN="100"
TRAIN_POS_PATH=""
TRAIN_NEG_PATH=""
TEST_TSV_PATH=""
CLASS_FILE_PATH=""
REF_GFF_PATH=""
EPOCHS="auto"
MAX_EPOCHS="20"
EARLY_STOP_PATIENCE="3"
EARLY_STOP_MIN_DELTA="0.001"
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
VAL_FRAC="0.2"
VALIDATION_METRIC="max_f1"  # pr_auc | roc_auc | max_f1 | acc@0.5
GRAD_CLIP="5.0"
POS_WEIGHT_CAP="20.0"
FOCAL_GAMMA="2.0"
FOCAL_ALPHA_POS=""
ASYM_GAMMA_POS="0.0"
ASYM_GAMMA_NEG="4.0"
ASYM_ALPHA_POS=""
USE_TUNED_HPARAMS="required"   # off | auto | required
DONOR_TUNED_CONFIG_PATH=""
ACCEPTOR_TUNED_CONFIG_PATH=""
INTRON_SCORE_OP="+"
NAME_FIELDS="none"
TRANSCRIPT_SCORE_AGG="min"
SOFTMIN_TAU="1.0"
SEED="1337"
TAG=""
VISUALIZE="true"
SKIP_TRAINING="0"
CONTINUE_TRAINING="0"
TRAIN_ONLY="0"
CHECKPOINT_TOP_K="3"
CHECKPOINT_PRUNE_DRY_RUN="0"
USE_AMP="1"
AMP_DTYPE="auto"
COMPILE_MODE="on"
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


append_versioned_output_args() {
	local script_tag="$1"
	local species="$2"
	local model_name="$3"
	local published_name=""

	published_name="$(
		intronmodel_resolve_latest_published_name \
			"${script_tag}" \
			"${species}" \
			"${model_name}"
	)"
	if [[ -z "${published_name}" ]]; then
		return 0
	fi

	args+=(
		--site_output_tsv "${DATA_ROOT}/${species}/site_score/${published_name}.tsv"
		--intron_output_tsv "${DATA_ROOT}/${species}/intron_score/${published_name}.tsv"
		--transcript_output_tsv "${DATA_ROOT}/${species}/trans_score/${published_name}.tsv"
		--eval_output_txt "${DATA_ROOT}/${species}/eval_score/${published_name}.txt"
		--metrics_json "${DATA_ROOT}/${species}/learning_metric/${published_name}.train.json"
	)
}

resolve_task_tuned_config_path() {
	local species="$1"
	local task_name="$2"
	local explicit_path="$3"
	local task_tuned_path="${explicit_path}"
	if [[ -z "${task_tuned_path}" ]]; then
		task_tuned_path="${DATA_ROOT}/${species}/tuning/cnn_v2/${task_name}/best_config.json"
	fi
	printf '%s\n' "${task_tuned_path}"
}

tuned_key_scope() {
	local tuned_key="$1"
	case "${tuned_key}" in
		batch_size | lr | loss | conv_channels | kernel_size | kernel_sizes \
		|max_pool_size | conv_stride | head_type | dropout | fc_hidden \
		|weight_decay | eta_min_ratio | val_frac | grad_clip \
		|pos_weight_cap | focal_gamma | focal_alpha_pos | f1_lambda \
		|asym_gamma_pos | asym_gamma_neg | asym_alpha_pos)
			printf '%s\n' "task"
			;;
		# run_model.py forces cnn_v2 into pair_mode=independent and delegates
		# donor/acceptor training to models/cnn, so cnn_v2-only token keys do not
		# participate in the effective runtime for this script.
		model | species | seed | train_target | donor_len | acceptor_len \
		|input_mode | pair_mode | sequence_transform | embedding_dim \
		|bpe_pretrained_model_name | bpe_pretrained_revision \
		|bpe_trust_remote_code)
			printf '%s\n' "ignore"
			;;
		*)
			printf '%s\n' "ignore"
			;;
	esac
}

append_tuned_args_for_task() {
	local species="$1"
	local task_name="$2"
	local explicit_path="$3"
	local -n task_arg_ref="$4"
	local -n shared_value_ref="$5"
	local -n shared_order_ref="$6"
	local config_path
	config_path="$(
		resolve_task_tuned_config_path \
			"${species}" \
			"${task_name}" \
			"${explicit_path}"
	)"

	if [[ ! -f "${config_path}" ]]; then
		if [[ "${USE_TUNED_HPARAMS_MODE}" == "required" ]]; then
			echo "[cnn_v2.sh] tuned config path not found: ${config_path}" >&2
			exit 1
		fi
		echo "[cnn_v2.sh] tuned config path not found: ${config_path}; "\
			"using CONFIG defaults for species=${species}, target=${task_name}." >&2
		return
	fi

	local tuned_output
	if ! tuned_output="$(intronmodel_load_tuned_overrides "${config_path}" 2>&1)"; then
		if [[ "${USE_TUNED_HPARAMS_MODE}" == "required" ]]; then
			echo "[cnn_v2.sh] failed to load tuned config: ${config_path}" >&2
			echo "[cnn_v2.sh] detail: ${tuned_output}" >&2
			exit 1
		fi
		echo "[cnn_v2.sh] failed to load tuned config: ${config_path}; "\
			"using CONFIG defaults for species=${species}, target=${task_name}." >&2
		return
	fi

	local loaded_count=0
	local line tuned_key tuned_value key_scope prefixed_key existing_value
	while IFS= read -r line; do
		if [[ -z "${line}" ]]; then
			continue
		fi
		IFS=$'\t' read -r tuned_key tuned_value <<<"${line}"
		if [[ -z "${tuned_key}" || -z "${tuned_value}" ]]; then
			continue
		fi
		key_scope="$(tuned_key_scope "${tuned_key}")"
		case "${key_scope}" in
			task)
				prefixed_key="${task_name}_${tuned_key}"
				task_arg_ref+=(--"${prefixed_key}" "${tuned_value}")
				loaded_count=$((loaded_count + 1))
				;;
			shared)
				existing_value="${shared_value_ref[${tuned_key}]:-}"
				if [[ -n "${existing_value}" && "${existing_value}" != "${tuned_value}" ]]; then
					echo "[cnn_v2.sh] conflicting shared tuned parameter ${tuned_key}: "\
						"${existing_value} vs ${tuned_value} "\
						"(species=${species}, target=${task_name})." >&2
					exit 1
				fi
				if [[ -z "${existing_value}" ]]; then
					shared_value_ref["${tuned_key}"]="${tuned_value}"
					shared_order_ref+=("${tuned_key}")
					loaded_count=$((loaded_count + 1))
				fi
				;;
		esac
	done <<<"${tuned_output}"

	echo "[cnn_v2.sh] tuned params loaded from ${config_path} "\
		"(species=${species}, target=${task_name}, count=${loaded_count})"
}

run_species_once() {
	local species="$1"
	local assigned_gpu_id="${2-}"

	local pythonpath="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
	local donor_tuned_args=()
	local acceptor_tuned_args=()
	local shared_tuned_args=()
	local shared_key
	declare -A shared_tuned_values=()
	local shared_tuned_order=()
	append_tuned_args_for_task \
		"${species}" \
		"donor" \
		"${DONOR_TUNED_CONFIG_PATH}" \
		donor_tuned_args \
		shared_tuned_values \
		shared_tuned_order
	append_tuned_args_for_task \
		"${species}" \
		"acceptor" \
		"${ACCEPTOR_TUNED_CONFIG_PATH}" \
		acceptor_tuned_args \
		shared_tuned_values \
		shared_tuned_order
	for shared_key in "${shared_tuned_order[@]}"; do
		shared_tuned_args+=(--"${shared_key}" "${shared_tuned_values[${shared_key}]}")
	done

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
		--transcript_score_agg "${TRANSCRIPT_SCORE_AGG}"
		--softmin_tau "${SOFTMIN_TAU}"
		--train_target "both"
		--validation_metric "${VALIDATION_METRIC}"
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
	append_arg_if_set "train_pos_path" "${TRAIN_POS_PATH}"
	append_arg_if_set "train_neg_path" "${TRAIN_NEG_PATH}"
	append_arg_if_set "test_tsv" "${TEST_TSV_PATH}"
	append_arg_if_set "class_file" "${CLASS_FILE_PATH}"
	append_arg_if_set "ref_gff" "${REF_GFF_PATH}"
	if [[ "${SKIP_TRAINING}" == "1" && "${TRAIN_ONLY}" != "1" ]]; then
		append_versioned_output_args "cnn_v2.sh" "${species}" "${MODEL}"
	fi

	if [[ ${#shared_tuned_args[@]} -gt 0 ]]; then
		args+=("${shared_tuned_args[@]}")
	fi
	if [[ ${#donor_tuned_args[@]} -gt 0 ]]; then
		args+=("${donor_tuned_args[@]}")
	fi
	if [[ ${#acceptor_tuned_args[@]} -gt 0 ]]; then
		args+=("${acceptor_tuned_args[@]}")
	fi

	echo "[cnn_v2.sh] species=${species} mode=independent tasks=donor,acceptor"
	if [[ -n "${assigned_gpu_id}" ]]; then
		CUDA_VISIBLE_DEVICES="${assigned_gpu_id}" \
			PYTHONPATH="${pythonpath}" \
			python3 "${PROJECT_ROOT}/src/run_model.py" "${args[@]}"
	else
		PYTHONPATH="${pythonpath}" \
			python3 "${PROJECT_ROOT}/src/run_model.py" "${args[@]}"
	fi
}

USE_TUNED_HPARAMS_MODE="$(
	intronmodel_normalize_use_tuned_mode "${USE_TUNED_HPARAMS}" "cnn_v2.sh"
)"

IFS=',' read -r -a SPECIES_LIST_RESOLVED <<<"${SPECIES}"
mapfile -t GPU_ID_LIST < <(
	intronmodel_resolve_gpu_ids "cnn_v2.sh" "${GPU_IDS}" "${DEVICE}"
)
PARALLEL_SLOT_COUNT="$(
	intronmodel_resolve_parallel_slots \
		"cnn_v2.sh" \
		"${MAX_PARALLEL_TRIALS}" \
		"${#GPU_ID_LIST[@]}"
)"
if [[ ${#SPECIES_LIST_RESOLVED[@]} -eq 0 ]]; then
	echo "[cnn_v2.sh] SPECIES resolved to an empty list." >&2
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
	echo "[cnn_v2.sh] species-parallel run across GPUs: ${gpu_csv}"
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
			echo "[cnn_v2.sh] species dispatch: ${species} -> gpu=${gpu_id}"
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
			echo "[cnn_v2.sh] species complete: ${completed_species} gpu=${completed_gpu} exit=${completed_code}"
		fi
		if [[ ${completed_code} -ne 0 && ${first_error_code} -eq 0 ]]; then
			first_error_code="${completed_code}"
			stop_submitting=1
		fi
	done
	exit "${first_error_code}"
fi
