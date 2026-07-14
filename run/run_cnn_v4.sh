#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
	echo "[cnn_v4.sh] This script is config-only." \
		"Edit top CONFIG and run without args." >&2
	exit 1
fi

# --------------------------
# CONFIG (edit here)
# --------------------------
set -a
MODEL="cnn_v4"
SPECIES="Athal, Dmel, Hsap, Mmus"
INTRONMODEL_AUTO_TMUX="on"
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
MAX_EPOCHS="100"
EARLY_STOP_PATIENCE="3"
EARLY_STOP_MIN_DELTA="0.001"
BATCH_SIZE="512"
LR="5e-4"
LOSS="focal"
CONV_CHANNELS="64,128,256"
KERNEL_SIZES="7,7,7"
HEAD_TYPE="gap"
BLOCK_DILATIONS="1,2,4,8"
RESIDUAL_CHANNELS="32,64,96,128"
DEFORMABLE_GROUPS="2"
DEFORMABLE_KERNEL_SIZE="3"
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
TAG="max"
VISUALIZE="true"
SKIP_TRAINING="0"
CONTINUE_TRAINING="0"
TRAIN_ONLY="0"
CHECKPOINT_TOP_K="3"
CHECKPOINT_PRUNE_DRY_RUN="0"
USE_AMP="1"
AMP_DTYPE="auto"
COMPILE_MODE="on"
INFER_BATCH_SIZE="2048"
INFER_USE_AMP="1"
INFER_AMP_DTYPE="auto"
INFER_COMPILE="0"
INFER_COMPILE_MODE="on"
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

trap 'intronmodel_abort_parallel_run' INT TERM HUP

append_arg_if_set() {
	intronmodel_append_arg_if_set args "$@"
}

append_flag_if_truthy() {
	intronmodel_append_flag_if_truthy args "$@"
}


resolve_task_tuned_config_path() {
	local species="$1"
	local task_name="$2"
	local explicit_path="$3"
	local task_tuned_path="${explicit_path}"
	if [[ -z "${task_tuned_path}" ]]; then
		task_tuned_path="${DATA_ROOT}/tuning/cnn_v4_shared/${task_name}/best_config.json"
	fi
	printf '%s\n' "${task_tuned_path}"
}

tuned_key_scope() {
	local tuned_key="$1"
	case "${tuned_key}" in
		batch_size | lr | loss | conv_channels | kernel_size | kernel_sizes \
		|conv_stride | head_type | dropout | fc_hidden \
		|block_dilations | residual_channels | deformable_groups | deformable_kernel_size \
		|weight_decay | eta_min_ratio | val_frac | grad_clip \
		|pos_weight_cap | focal_gamma | focal_alpha_pos | f1_lambda \
		|asym_gamma_pos | asym_gamma_neg | asym_alpha_pos)
			printf '%s\n' "task"
			;;
		# run_model.py keeps cnn_v4 in independent donor/acceptor mode. Pair-only
		# keys should not participate in the effective runtime for this script.
		model | species | seed | train_target \
		|input_mode | pair_mode | sequence_transform | embedding_dim \
		|bpe_pretrained_model_name | bpe_pretrained_revision \
		|bpe_trust_remote_code)
			printf '%s\n' "ignore"
			;;
		donor_len | acceptor_len)
			printf '%s\n' "shared"
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
	local -n resolved_config_path_ref="$7"
	local config_path
	config_path="$(
		resolve_task_tuned_config_path \
			"${species}" \
			"${task_name}" \
			"${explicit_path}"
	)"

	if [[ ! -f "${config_path}" ]]; then
		if [[ "${USE_TUNED_HPARAMS_MODE}" == "required" ]]; then
			echo "[cnn_v4.sh] tuned config path not found: ${config_path}" >&2
			exit 1
		fi
		echo "[cnn_v4.sh] tuned config path not found: ${config_path}; "\
			"using CONFIG defaults for species=${species}, target=${task_name}." >&2
		return
	fi

	resolved_config_path_ref="${config_path}"
	local tuned_output
	if ! tuned_output="$(intronmodel_load_tuned_overrides "${config_path}" 2>&1)"; then
		if [[ "${USE_TUNED_HPARAMS_MODE}" == "required" ]]; then
			echo "[cnn_v4.sh] failed to load tuned config: ${config_path}" >&2
			echo "[cnn_v4.sh] detail: ${tuned_output}" >&2
			exit 1
		fi
		echo "[cnn_v4.sh] failed to load tuned config: ${config_path}; "\
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
		if [[ "${tuned_key}" == donor_* || "${tuned_key}" == acceptor_* ]]; then
			if [[ "${tuned_key}" == "${task_name}_"* ]]; then
				task_arg_ref+=(--"${tuned_key}" "${tuned_value}")
				loaded_count=$((loaded_count + 1))
			fi
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
					echo "[cnn_v4.sh] conflicting shared tuned parameter ${tuned_key}: "\
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

	echo "[cnn_v4.sh] tuned params loaded from ${config_path} "\
		"(species=${species}, target=${task_name}, count=${loaded_count})"
}

run_species_once() {
	local species="$1"
	local assigned_gpu_id="${2-}"

	local donor_tuned_args=()
	local acceptor_tuned_args=()
	local donor_tuned_config_path=""
	local acceptor_tuned_config_path=""
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
		shared_tuned_order \
		donor_tuned_config_path
	append_tuned_args_for_task \
		"${species}" \
		"acceptor" \
		"${ACCEPTOR_TUNED_CONFIG_PATH}" \
		acceptor_tuned_args \
		shared_tuned_values \
		shared_tuned_order \
		acceptor_tuned_config_path
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
		--conv_channels "${CONV_CHANNELS}"
		--kernel_sizes "${KERNEL_SIZES}"
		--head_type "${HEAD_TYPE}"
		--block_dilations "${BLOCK_DILATIONS}"
		--residual_channels "${RESIDUAL_CHANNELS}"
		--deformable_groups "${DEFORMABLE_GROUPS}"
		--deformable_kernel_size "${DEFORMABLE_KERNEL_SIZE}"
		--dropout "${DROPOUT}"
		--fc_hidden "${FC_HIDDEN}"
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
		--epochs "${EPOCHS}"
		--max_epochs "${MAX_EPOCHS}"
		--early_stop_patience "${EARLY_STOP_PATIENCE}"
		--early_stop_min_delta "${EARLY_STOP_MIN_DELTA}"
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
	append_arg_if_set "donor_tuned_config_path" "${donor_tuned_config_path}"
	append_arg_if_set "acceptor_tuned_config_path" "${acceptor_tuned_config_path}"
	if [[ "${SKIP_TRAINING}" == "1" && "${TRAIN_ONLY}" != "1" ]]; then
		intronmodel_append_versioned_output_args \
			"cnn_v4.sh" "${species}" "${MODEL}" args
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

	echo "[cnn_v4.sh] species=${species} mode=independent tasks=donor,acceptor"
	intronmodel_run_model_with_optional_gpu "${PROJECT_ROOT}" "${assigned_gpu_id}" args
}

USE_TUNED_HPARAMS_MODE="$(
	intronmodel_normalize_use_tuned_mode "${USE_TUNED_HPARAMS}" "cnn_v4.sh"
)"

IFS=',' read -r -a SPECIES_LIST_RESOLVED <<<"${SPECIES}"
mapfile -t GPU_ID_LIST < <(
	intronmodel_resolve_gpu_ids "cnn_v4.sh" "${GPU_IDS}" "${DEVICE}"
)
PARALLEL_SLOT_COUNT="$(
	intronmodel_resolve_parallel_slots \
		"cnn_v4.sh" \
		"${MAX_PARALLEL_TRIALS}" \
		"${#GPU_ID_LIST[@]}"
)"
intronmodel_run_species_jobs \
	"cnn_v4.sh" \
	SPECIES_LIST_RESOLVED \
	GPU_ID_LIST \
	"${PARALLEL_SLOT_COUNT}" \
	run_species_once
