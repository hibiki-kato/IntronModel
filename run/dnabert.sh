#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
	echo "[dnabert.sh] This script is config-only. Edit CONFIG and run without args." >&2
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
	echo "[dnabert.sh] timing: start=${SCRIPT_START_EPOCH} end=${script_end_epoch} "\
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
		echo "[dnabert.sh] species case normalized: '${raw_species}' -> '${matches[0]}'" >&2
		printf '%s\n' "${matches[0]}"
		return 0
	fi
	if [[ ${#matches[@]} -gt 1 ]]; then
		echo "[dnabert.sh] ambiguous species '${raw_species}'." >&2
		printf '[dnabert.sh] case-insensitive matches: %s\n' "${matches[*]}" >&2
		return 1
	fi
	printf '%s\n' "${raw_species}"
	return 0
}

# --------------------------
# CONFIG (edit here)
# --------------------------
DNABERT_VARIANT="6"
SPECIES="Mmus"
DONOR_LEN="100"
ACCEPTOR_LEN="100"

PRETRAINED_MODEL_NAME="${MODEL_ROOT}/pretrained/dnabert2-117m-7bce263b15377fc15361f52cfab88f8b586abda0"
PRETRAINED_REVISION=""
TRUST_REMOTE_CODE="1"

EPOCHS="20"
BATCH_SIZE="64"
LR="2e-5"
LOSS="weighted_bce"
MAX_TOKENS="auto"
DROPOUT="0.1"
WEIGHT_DECAY="0.01"
ETA_MIN_RATIO="0.01"
VAL_FRAC="0.1"
GRAD_CLIP="1.0"
POS_WEIGHT_CAP="20.0"
FOCAL_GAMMA="2.0"
FOCAL_ALPHA_POS=""
ASYM_GAMMA_POS="0.0"
ASYM_GAMMA_NEG="4.0"
ASYM_ALPHA_POS=""
TRAIN_TARGET="both"

DONOR_BATCH_SIZE=""
ACCEPTOR_BATCH_SIZE=""
DONOR_LR=""
ACCEPTOR_LR=""
DONOR_LOSS=""
ACCEPTOR_LOSS=""
DONOR_MAX_TOKENS=""
ACCEPTOR_MAX_TOKENS=""
DONOR_DROPOUT=""
ACCEPTOR_DROPOUT=""
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
CONTINUE_TRAINING="0"
TRAIN_ONLY="0"
PRECOMPUTED_SITE_SCORE_TSV=""

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
MPS_MAX_BATCH_SIZE="1024"

if [[ "${DNABERT_VARIANT}" != "2" && "${DNABERT_VARIANT}" != "6" ]]; then
	echo "[dnabert.sh] DNABERT_VARIANT must be 2 or 6." >&2
	exit 1
fi
MODEL="dnabert${DNABERT_VARIANT}"

if [[ "${USE_COMPILE}" != "off" && "${USE_COMPILE}" != "on" \
	&& "${USE_COMPILE}" != "auto" ]]; then
	echo "[dnabert.sh] USE_COMPILE must be off|on|auto." >&2
	exit 1
fi
if [[ "${TRAIN_ONLY}" != "0" && "${TRAIN_ONLY}" != "1" ]]; then
	echo "[dnabert.sh] TRAIN_ONLY must be 0 or 1." >&2
	exit 1
fi
if [[ "${SKIP_TRAINING}" != "0" && "${SKIP_TRAINING}" != "1" ]]; then
	echo "[dnabert.sh] SKIP_TRAINING must be 0 or 1." >&2
	exit 1
fi
if [[ "${CONTINUE_TRAINING}" != "0" && "${CONTINUE_TRAINING}" != "1" ]]; then
	echo "[dnabert.sh] CONTINUE_TRAINING must be 0 or 1." >&2
	exit 1
fi
if [[ "${SKIP_TRAINING}" == "1" && "${CONTINUE_TRAINING}" == "1" ]]; then
	echo "[dnabert.sh] CONTINUE_TRAINING=1 cannot be used with SKIP_TRAINING=1." >&2
	exit 1
fi
if [[ "${TRAIN_TARGET}" != "both" && "${TRAIN_TARGET}" != "donor" \
	&& "${TRAIN_TARGET}" != "acceptor" ]]; then
	echo "[dnabert.sh] TRAIN_TARGET must be both|donor|acceptor." >&2
	exit 1
fi
if [[ "${TRAIN_TARGET}" != "both" && "${TRAIN_ONLY}" != "1" ]]; then
	echo "[dnabert.sh] TRAIN_TARGET donor/acceptor requires TRAIN_ONLY=1." >&2
	exit 1
fi
if [[ "${TRUST_REMOTE_CODE}" != "0" && "${TRUST_REMOTE_CODE}" != "1" ]]; then
	echo "[dnabert.sh] TRUST_REMOTE_CODE must be 0 or 1." >&2
	exit 1
fi
if ! [[ "${MPS_MAX_BATCH_SIZE}" =~ ^[0-9]+$ ]] \
	|| [[ "${MPS_MAX_BATCH_SIZE}" -le 0 ]]; then
	echo "[dnabert.sh] MPS_MAX_BATCH_SIZE must be a positive integer." >&2
	exit 1
fi

export INTRONMODEL_MPS_MAX_BATCH_SIZE="${MPS_MAX_BATCH_SIZE}"

SPECIES="$(resolve_species_case "${SPECIES}" "${DATA_ROOT}")"

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
    "dropout": float("${DROPOUT}"),
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
	--pretrained_model_name "${PRETRAINED_MODEL_NAME}"
	--pretrained_revision "${PRETRAINED_REVISION}"
	--trust_remote_code "${TRUST_REMOTE_CODE}"
	--epochs "${EPOCHS}"
	--train_target "${TRAIN_TARGET}"
	--batch_size "${BATCH_SIZE}"
	--lr "${LR}"
	--loss "${LOSS}"
	--max_tokens "${MAX_TOKENS}"
	--dropout "${DROPOUT}"
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
if [[ -n "${DONOR_MAX_TOKENS}" ]]; then
	RUN_ARGS+=(--donor_max_tokens "${DONOR_MAX_TOKENS}")
fi
if [[ -n "${ACCEPTOR_MAX_TOKENS}" ]]; then
	RUN_ARGS+=(--acceptor_max_tokens "${ACCEPTOR_MAX_TOKENS}")
fi
if [[ -n "${DONOR_DROPOUT}" ]]; then
	RUN_ARGS+=(--donor_dropout "${DONOR_DROPOUT}")
fi
if [[ -n "${ACCEPTOR_DROPOUT}" ]]; then
	RUN_ARGS+=(--acceptor_dropout "${ACCEPTOR_DROPOUT}")
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
if [[ -n "${DONOR_ASYM_ALPHA_POS}" ]]; then
	RUN_ARGS+=(--donor_asym_alpha_pos "${DONOR_ASYM_ALPHA_POS}")
fi
if [[ -n "${ACCEPTOR_ASYM_ALPHA_POS}" ]]; then
	RUN_ARGS+=(--acceptor_asym_alpha_pos "${ACCEPTOR_ASYM_ALPHA_POS}")
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

if [[ "${SKIP_TRAINING}" == "1" && -z "${PRECOMPUTED_SITE_SCORE_TSV}" ]]; then
	if ! PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
		python3 - "${RUN_ARGS[@]}" <<'PY'
import os
import sys

from run_model import (
    _build_checkpoint_paths,
    _build_checkpoint_stem_from_params,
    _infer_window_defaults,
    parse_args,
)

args = parse_args(sys.argv[1:])
donor_len, acceptor_len, inferred_train_len = _infer_window_defaults(
    species=args.species,
    donor_len=args.donor_len,
    acceptor_len=args.acceptor_len,
)
checkpoint_stem = _build_checkpoint_stem_from_params(
    model_name=args.model,
    donor_len=donor_len,
    acceptor_len=acceptor_len,
    inferred_train_len=inferred_train_len,
    raw_params=dict(vars(args)),
)
checkpoint_paths = _build_checkpoint_paths(args.species, checkpoint_stem)
train_target = str(getattr(args, "train_target", "both")).strip().lower()
required_tasks = ("donor", "acceptor") if train_target == "both" else (train_target,)
missing_paths = [
    checkpoint_paths[task]
    for task in required_tasks
    if not os.path.exists(checkpoint_paths[task])
]
if missing_paths:
    print(
        "[dnabert.sh] SKIP_TRAINING=1 requires existing checkpoints "
        "for the current config.",
        file=sys.stderr,
    )
    for missing_path in missing_paths:
        print(f"[dnabert.sh] missing checkpoint: {missing_path}", file=sys.stderr)
    print(
        "[dnabert.sh] Set SKIP_TRAINING=0 for a fresh run, "
        "or set PRECOMPUTED_SITE_SCORE_TSV to skip inference checkpoint loading.",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
	then
		exit 1
	fi
fi

echo "[dnabert.sh] Start unified pipeline"
echo "[dnabert.sh] model=${MODEL} species=${SPECIES} train_only=${TRAIN_ONLY}"
python3 "${PROJECT_ROOT}/src/run_model.py" "${RUN_ARGS[@]}"
echo "[dnabert.sh] Done"
echo "[dnabert.sh] site_score=${OUTPUT_SITE_SCORE_TSV}"
echo "[dnabert.sh] transcript_score=${OUTPUT_TRANS_SCORE_TSV}"
echo "[dnabert.sh] eval_score=${OUTPUT_EVAL_SCORE_TXT}"
