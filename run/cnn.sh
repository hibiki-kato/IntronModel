#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOT'
Usage: bash run/cnn.sh [options]

Options:
  --species <name>            Species folder under data/ (default: Dmel)
  --donor-len <int>           Donor window length (default: 100)
  --acceptor-len <int>        Acceptor window length (default: 100)
  --epochs <int>              Training epochs (default: 20)
  --batch-size <int>          Batch size (default: 512)
  --lr <float>                Learning rate (default: 5e-4)
  --loss <name>               Loss type (default: focal)
  --name-fields <csv>         Naming fields for outputs/checkpoints
  --intron-score-op <op>      Intron score operation: +|*|harmonic|min (default: *)
  --transcript-agg <name>     Transcript agg:
                              min|softmin|softmin_wavg|+|*|mean|avg|median|max
                              (default: min)
  --softmin-tau <float>       Temperature for transcript softmin/softmin_wavg
                              (default: 1.0)
  --seed <int>                Random seed (default: 1337)
  --device <name>             auto|cuda|mps|cpu (default: auto)
  --visualize <mode>          none|true|interactive (default: true)
  --site-score-tsv <path>     Use precomputed site scores (passes --site_score_tsv)
  --skip-training             Skip train stage
  --train-only                Run train stage only (skip infer/transcript/eval)
  -h, --help                  Show this help

Notes:
  - Paths are fixed to data/<species>/... for this CNN wrapper.
  - Model-specific hyperparameters are defined in this script body.
EOT
}

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

# Fixed pipeline target.
MODEL="cnn"
SPECIES="Dmel"
DONOR_LEN="100"
ACCEPTOR_LEN="100"
INTRON_SCORE_OP="*"
TRANSCRIPT_AGG="min"
SOFTMIN_TAU="1.0"
SEED="1337"
DEVICE="auto"
VISUALIZE="true"
SKIP_TRAINING="0"
TRAIN_ONLY="0"
PRECOMPUTED_SITE_SCORE_TSV=""
NAME_FIELDS=""

# CNN hyperparameters (edit here; not exposed as CLI options).
EPOCHS="20"
BATCH_SIZE="512"
LR="5e-4"
LOSS="focal"
CONV_CHANNELS="64,128,256"
KERNEL_SIZE="7"
DROPOUT="0.3"
FC_HIDDEN="128"
WEIGHT_DECAY="0.01"
GRAD_CLIP="5.0"
VAL_FRAC="0.1"
COMPILE_MODEL="0"
POS_WEIGHT_CAP="20.0"

while [[ $# -gt 0 ]]; do
	case "$1" in
	--species)
		SPECIES="$2"
		shift 2
		;;
	--donor-len)
		DONOR_LEN="$2"
		shift 2
		;;
	--acceptor-len)
		ACCEPTOR_LEN="$2"
		shift 2
		;;
	--epochs)
		EPOCHS="$2"
		shift 2
		;;
	--batch-size)
		BATCH_SIZE="$2"
		shift 2
		;;
	--lr)
		LR="$2"
		shift 2
		;;
	--loss)
		LOSS="$2"
		shift 2
		;;
	--name-fields)
		NAME_FIELDS="$2"
		shift 2
		;;
	--intron-score-op)
		INTRON_SCORE_OP="$2"
		shift 2
		;;
	--transcript-agg)
		TRANSCRIPT_AGG="$2"
		shift 2
		;;
	--softmin-tau)
		SOFTMIN_TAU="$2"
		shift 2
		;;
	--seed)
		SEED="$2"
		shift 2
		;;
	--device)
		DEVICE="$2"
		shift 2
		;;
	--visualize)
		VISUALIZE="$2"
		shift 2
		;;
	--site-score-tsv)
		PRECOMPUTED_SITE_SCORE_TSV="$2"
		shift 2
		;;
	--skip-training)
		SKIP_TRAINING="1"
		shift
		;;
	--train-only)
		TRAIN_ONLY="1"
		shift
		;;
	-h | --help)
		usage
		exit 0
		;;
	*)
		echo "Unknown argument: $1" >&2
		usage
		exit 1
		;;
	esac
done

TEST_TSV="${PROJECT_ROOT}/data/${SPECIES}/raw/transcripts.tsv"
CLASS_FILE="${PROJECT_ROOT}/data/${SPECIES}/raw/transcript_class.txt"

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
    "grad_clip": float("${GRAD_CLIP}"),
    "val_frac": float("${VAL_FRAC}"),
    "intron_score_op": "${INTRON_SCORE_OP}",
    "transcript_score_agg": "${TRANSCRIPT_AGG}",
    "softmin_tau": float("${SOFTMIN_TAU}"),
    "seed": int("${SEED}"),
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

OUTPUT_SITE_SCORE_TSV="${PROJECT_ROOT}/data/${SPECIES}/site_score/${OUTPUT_STEM}.tsv"
OUTPUT_TRANS_SCORE_TSV="${PROJECT_ROOT}/data/${SPECIES}/trans_score/${OUTPUT_STEM}.tsv"
OUTPUT_EVAL_SCORE_TXT="${PROJECT_ROOT}/data/${SPECIES}/eval_score/${OUTPUT_STEM}.txt"

RUN_ARGS=(
	--model "${MODEL}"
	--species "${SPECIES}"
	--donor_len "${DONOR_LEN}"
	--acceptor_len "${ACCEPTOR_LEN}"
	--epochs "${EPOCHS}"
	--batch_size "${BATCH_SIZE}"
	--lr "${LR}"
	--loss "${LOSS}"
	--conv_channels "${CONV_CHANNELS}"
	--kernel_size "${KERNEL_SIZE}"
	--dropout "${DROPOUT}"
	--fc_hidden "${FC_HIDDEN}"
	--weight_decay "${WEIGHT_DECAY}"
	--grad_clip "${GRAD_CLIP}"
	--val_frac "${VAL_FRAC}"
	--pos_weight_cap "${POS_WEIGHT_CAP}"
	--name_fields "${NAME_FIELDS}"
	--intron_score_op "${INTRON_SCORE_OP}"
	--transcript_score_agg "${TRANSCRIPT_AGG}"
	--softmin_tau "${SOFTMIN_TAU}"
	--seed "${SEED}"
	--device "${DEVICE}"
	--visualize "${VISUALIZE}"
	--test_tsv "${TEST_TSV}"
	--class_file "${CLASS_FILE}"
	--site_output_tsv "${OUTPUT_SITE_SCORE_TSV}"
	--transcript_output_tsv "${OUTPUT_TRANS_SCORE_TSV}"
	--eval_output_txt "${OUTPUT_EVAL_SCORE_TXT}"
)

if [[ "${COMPILE_MODEL}" == "1" ]]; then
	RUN_ARGS+=(--compile)
fi
if [[ "${SKIP_TRAINING}" == "1" ]]; then
	RUN_ARGS+=(--skip_train)
fi
if [[ "${TRAIN_ONLY}" == "1" ]]; then
	RUN_ARGS+=(--train_only)
fi
if [[ -n "${PRECOMPUTED_SITE_SCORE_TSV}" ]]; then
	RUN_ARGS+=(--site_score_tsv "${PRECOMPUTED_SITE_SCORE_TSV}")
fi

echo "[cnn.sh] Start unified pipeline"
echo "[cnn.sh] model=${MODEL} species=${SPECIES} loss=${LOSS} seed=${SEED}"
python3 "${PROJECT_ROOT}/src/run_model.py" "${RUN_ARGS[@]}"
echo "[cnn.sh] Done"
echo "[cnn.sh] site_score=${OUTPUT_SITE_SCORE_TSV}"
echo "[cnn.sh] transcript_score=${OUTPUT_TRANS_SCORE_TSV}"
echo "[cnn.sh] eval_score=${OUTPUT_EVAL_SCORE_TXT}"
