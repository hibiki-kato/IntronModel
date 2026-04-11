#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
	echo "[spliceformer_sc.sh] This script is config-only." \
		"Edit top CONFIG and run without args." >&2
	exit 1
fi

# --------------------------
# CONFIG (edit here)
# --------------------------
set -a
MODEL="spliceformer_sc"

# Species used for inference (one at a time by the pipeline).
# For training, set SPECIES_LIST below.
SPECIES="Hsap"

# Comma-separated species to pool for multi-species training.
# Leave empty to train on SPECIES only.
SPECIES_LIST="Athal,Dmel,Hsap,Mmus"

INTRONMODEL_AUTO_TMUX="off"
DEVICE="auto"
GPU_IDS="auto"
MAX_PARALLEL_TRIALS="auto"

DONOR_LEN="100"
ACCEPTOR_LEN="100"
TRAIN_POS_PATH=""
TRAIN_NEG_PATH=""
TEST_TSV_PATH=""
CLASS_FILE_PATH=""
REF_GFF_PATH=""

EPOCHS="auto"
MAX_EPOCHS="100"
EARLY_STOP_PATIENCE="5"
EARLY_STOP_MIN_DELTA="0.001"
BATCH_SIZE="512"
LR="1e-3"
LOSS="bce"
WEIGHT_DECAY="0.01"
ETA_MIN_RATIO="0.01"
VAL_FRAC="0.1"
VALIDATION_METRIC="pr_auc"   # pr_auc | roc_auc
GRAD_CLIP="5.0"
DROPOUT="0.1"
SEED="1337"

# Spliceformer-specific architecture
D_MODEL="32"
CNN_DILATIONS="1,2,4,8"
CNN_KERNEL_SIZE="11"
NHEAD="4"
NUM_TRANSFORMER_LAYERS="8"
DIM_FEEDFORWARD="512"
SPECIES_EMBED_DIM="256"
USE_FILM="0"          # 0=off  1=on  (FiLM species conditioning)
K_DONOR="256"
K_ACCEPTOR="256"
SPLICEFORMER_MODE="binary"   # binary | multiclass

# Runtime flags
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
TRANSCRIPT_SCORE_AGG="min"
SOFTMIN_TAU="1.0"
INTRON_SCORE_OP="+"
NAME_FIELDS="none"
set +a

# --------------------------
# Runtime implementation
# --------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"
intronmodel_activate_conda "intronmodel"
intronmodel_init_paths "${BASH_SOURCE[0]}"
intronmodel_enable_auto_tmux "${PROJECT_ROOT}" "$0" "${BASH_SOURCE[0]##*/}"

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

# ---------------------------------------------------------------------------
# Training — done once for all species combined.
# The SPECIES arg is set to the first non-empty entry in SPECIES_LIST so
# run_model.py can resolve the checkpoint directory.  SPECIES_LIST pools data.
# ---------------------------------------------------------------------------

TRAINING_SPECIES="${SPECIES}"
if [[ -n "${SPECIES_LIST}" ]]; then
	TRAINING_SPECIES="$(echo "${SPECIES_LIST}" | cut -d',' -f1 | xargs)"
fi

run_train_once() {
	local assigned_gpu_id="${1-}"
	local pythonpath="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

	args=(
		--model "${MODEL}"
		--species "${TRAINING_SPECIES}"
		--donor_len "${DONOR_LEN}"
		--acceptor_len "${ACCEPTOR_LEN}"
		--device "${DEVICE}"
		--seed "${SEED}"
		--batch_size "${BATCH_SIZE}"
		--lr "${LR}"
		--loss "${LOSS}"
		--weight_decay "${WEIGHT_DECAY}"
		--eta_min_ratio "${ETA_MIN_RATIO}"
		--val_frac "${VAL_FRAC}"
		--grad_clip "${GRAD_CLIP}"
		--dropout "${DROPOUT}"
		--epochs "${EPOCHS}"
		--max_epochs "${MAX_EPOCHS}"
		--early_stop_patience "${EARLY_STOP_PATIENCE}"
		--early_stop_min_delta "${EARLY_STOP_MIN_DELTA}"
		--validation_metric "${VALIDATION_METRIC}"
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
		--name_fields "${NAME_FIELDS}"
		# Spliceformer-specific
		--d_model "${D_MODEL}"
		--cnn_dilations "${CNN_DILATIONS}"
		--cnn_kernel_size "${CNN_KERNEL_SIZE}"
		--nhead "${NHEAD}"
		--num_transformer_layers "${NUM_TRANSFORMER_LAYERS}"
		--dim_feedforward "${DIM_FEEDFORWARD}"
		--species_embed_dim "${SPECIES_EMBED_DIM}"
		--use_film "${USE_FILM}"
		--k_donor "${K_DONOR}"
		--k_acceptor "${K_ACCEPTOR}"
		--spliceformer_mode "${SPLICEFORMER_MODE}"
		--train_only
	)
	append_arg_if_set "species_list" "${SPECIES_LIST}"
	append_arg_if_set "train_pos_path" "${TRAIN_POS_PATH}"
	append_arg_if_set "train_neg_path" "${TRAIN_NEG_PATH}"
	append_flag_if_truthy "skip_train" "${SKIP_TRAINING}"
	append_flag_if_truthy "continue_train" "${CONTINUE_TRAINING}"

	echo "[spliceformer_sc.sh] training on species_list=${SPECIES_LIST:-${TRAINING_SPECIES}}"
	if [[ -n "${assigned_gpu_id}" ]]; then
		CUDA_VISIBLE_DEVICES="${assigned_gpu_id}" \
			PYTHONPATH="${pythonpath}" \
			python3 "${PROJECT_ROOT}/src/run_model.py" "${args[@]}"
	else
		PYTHONPATH="${pythonpath}" \
			python3 "${PROJECT_ROOT}/src/run_model.py" "${args[@]}"
	fi
}

# ---------------------------------------------------------------------------
# Inference — run per-species after training.
# ---------------------------------------------------------------------------

run_infer_species() {
	local species="$1"
	local assigned_gpu_id="${2-}"
	local pythonpath="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

	args=(
		--model "${MODEL}"
		--species "${species}"
		--donor_len "${DONOR_LEN}"
		--acceptor_len "${ACCEPTOR_LEN}"
		--device "${DEVICE}"
		--seed "${SEED}"
		--batch_size "${BATCH_SIZE}"
		--lr "${LR}"
		--loss "${LOSS}"
		--dropout "${DROPOUT}"
		--name_fields "${NAME_FIELDS}"
		--transcript_score_agg "${TRANSCRIPT_SCORE_AGG}"
		--softmin_tau "${SOFTMIN_TAU}"
		--intron_score_op "${INTRON_SCORE_OP}"
		--visualize "${VISUALIZE}"
		--checkpoint_top_k "${CHECKPOINT_TOP_K}"
		--checkpoint_prune_dry_run "${CHECKPOINT_PRUNE_DRY_RUN}"
		--infer_batch_size "${INFER_BATCH_SIZE}"
		--infer_use_amp "${INFER_USE_AMP}"
		--infer_amp_dtype "${INFER_AMP_DTYPE}"
		--infer_compile "${INFER_COMPILE}"
		--infer_compile_mode "${INFER_COMPILE_MODE}"
		--d_model "${D_MODEL}"
		--cnn_dilations "${CNN_DILATIONS}"
		--cnn_kernel_size "${CNN_KERNEL_SIZE}"
		--nhead "${NHEAD}"
		--num_transformer_layers "${NUM_TRANSFORMER_LAYERS}"
		--dim_feedforward "${DIM_FEEDFORWARD}"
		--species_embed_dim "${SPECIES_EMBED_DIM}"
		--use_film "${USE_FILM}"
		--k_donor "${K_DONOR}"
		--k_acceptor "${K_ACCEPTOR}"
		--spliceformer_mode "${SPLICEFORMER_MODE}"
		--skip_train
	)
	append_arg_if_set "test_tsv" "${TEST_TSV_PATH}"
	append_arg_if_set "class_file" "${CLASS_FILE_PATH}"
	append_arg_if_set "ref_gff" "${REF_GFF_PATH}"

	echo "[spliceformer_sc.sh] infer species=${species}"
	if [[ -n "${assigned_gpu_id}" ]]; then
		CUDA_VISIBLE_DEVICES="${assigned_gpu_id}" \
			PYTHONPATH="${pythonpath}" \
			python3 "${PROJECT_ROOT}/src/run_model.py" "${args[@]}"
	else
		PYTHONPATH="${pythonpath}" \
			python3 "${PROJECT_ROOT}/src/run_model.py" "${args[@]}"
	fi
}

# ---------------------------------------------------------------------------
# Resolve GPU ID for training
# ---------------------------------------------------------------------------
GPU_ID_LIST=(
	$(
		intronmodel_resolve_gpu_ids "spliceformer_sc.sh" "${GPU_IDS}" "${DEVICE}"
	)
)
training_gpu_id=""
if [[ ${#GPU_ID_LIST[@]} -gt 0 ]]; then
	training_gpu_id="${GPU_ID_LIST[0]}"
fi

# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------
if [[ "${SKIP_TRAINING}" != "1" ]]; then
	run_train_once "${training_gpu_id}"
fi

if [[ "${TRAIN_ONLY}" != "1" ]]; then
	# Resolve inference species list: use SPECIES_LIST if set, else SPECIES
	if [[ -n "${SPECIES_LIST}" ]]; then
		IFS=',' read -ra INFER_SPECIES_LIST <<< "${SPECIES_LIST}"
	else
		INFER_SPECIES_LIST=("${SPECIES}")
	fi

	for species_raw in "${INFER_SPECIES_LIST[@]}"; do
		species="$(echo "${species_raw}" | xargs)"
		if [[ -z "${species}" ]]; then
			continue
		fi
		run_infer_species "${species}" "${training_gpu_id}"
	done
fi
