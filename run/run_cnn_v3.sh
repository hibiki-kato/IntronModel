#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
	echo "[cnn_v3.sh] This script is config-only." \
		"Edit top CONFIG and run without args." >&2
	exit 1
fi

# --------------------------
# CONFIG (edit here)
# --------------------------
set -a
SPECIES="Hsap,Mmus"
BASE_PAIR_CHECKPOINTS=""
DONOR_LEN="100"
ACCEPTOR_LEN="100"
SEQUENCE_TRANSFORM="none"
TRAIN_TARGET="pair"
META_HIDDEN_DIM="32"
META_DROPOUT="0.2"

EPOCHS="10"
MAX_EPOCHS="200"
EARLY_STOP_PATIENCE="12"
EARLY_STOP_MIN_DELTA="0.0"
BATCH_SIZE="512"
LR="5e-4"
LOSS="focal"
INPUT_MODE="bpe"    # onehot | kmer3 | bpe
PAIR_MODE="pair"    # pair only for cnn_v3
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
COMPILE_MODE="auto"
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
set +a

# --------------------------
# Runtime implementation
# --------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"
intronmodel_activate_conda "intronmodel"
intronmodel_init_paths "${BASH_SOURCE[0]}"

resolve_base_pair_checkpoints() {
	local species="$1"
	local explicit_value="$2"
	local python_bin
	python_bin="$(intronmodel_resolve_python_bin "run_cnn_v3.sh")"
	"${python_bin}" - "$PROJECT_ROOT" "$species" "$explicit_value" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path


def _dedupe_keep_order(values: list[str]) -> list[str]:
	seen: set[str] = set()
	out: list[str] = []
	for value in values:
		if value in seen:
			continue
		seen.add(value)
		out.append(value)
	return out


project_root = Path(sys.argv[1])
species = sys.argv[2]
explicit_value = sys.argv[3].strip()
if explicit_value:
	explicit_paths = _dedupe_keep_order(
		[token.strip() for token in explicit_value.split(",") if token.strip()]
	)
	if not explicit_paths:
		raise SystemExit(1)
	for path_text in explicit_paths:
		if not Path(path_text).exists():
			raise SystemExit(1)
	print(",".join(explicit_paths))
	raise SystemExit(0)

# Preferred source: model-separated tuning artifact for cnn_v2 pair.
best_config_path = (
	project_root
	/ "data"
	/ species
	/ "tuning"
	/ "cnn_v2"
	/ "pair"
	/ "best_config.json"
)
if best_config_path.is_file():
	try:
		payload = json.loads(best_config_path.read_text(encoding="utf-8"))
	except Exception:
		payload = None
	if isinstance(payload, dict):
		checkpoint_path = str(payload.get("pair_checkpoint_path", "")).strip()
		if checkpoint_path and Path(checkpoint_path).exists():
			print(checkpoint_path)
			raise SystemExit(0)

learning_metric_dir = project_root / "data" / species / "learning_metric"
if not learning_metric_dir.is_dir():
	raise SystemExit(1)

latest_by_model: dict[str, tuple[float, str]] = {}
for path in learning_metric_dir.glob("*.train.json"):
	try:
		payload = json.loads(path.read_text(encoding="utf-8"))
	except Exception:
		continue
	model_name = str(payload.get("model", "")).strip()
	checkpoint_path = str(payload.get("pair_checkpoint_path", "")).strip()
	if model_name != "cnn_v2" or checkpoint_path == "":
		continue
	if not Path(checkpoint_path).exists():
		continue
	mtime = path.stat().st_mtime
	previous = latest_by_model.get(model_name)
	if previous is None or mtime > previous[0]:
		latest_by_model[model_name] = (mtime, checkpoint_path)

resolved = _dedupe_keep_order(
	[checkpoint for _mtime, checkpoint in sorted(latest_by_model.values(), reverse=True)]
)
if not resolved:
	raise SystemExit(1)
print(",".join(resolved))
PY
}

IFS=',' read -r -a INFER_SPECIES_LIST <<<"${SPECIES}"

if [[ ${#INFER_SPECIES_LIST[@]} -eq 0 ]]; then
	echo "[cnn_v3.sh] INFER_SPECIES is empty." >&2
	exit 1
fi

run_index=0
for species_raw in "${INFER_SPECIES_LIST[@]}"; do
	species="$(echo "${species_raw}" | xargs)"
	if [[ -z "${species}" ]]; then
		continue
	fi
	resolved_base_pair_checkpoints="${BASE_PAIR_CHECKPOINTS}"
	if ! resolved_base_pair_checkpoints="$(resolve_base_pair_checkpoints "${species}" "${BASE_PAIR_CHECKPOINTS}")"; then
		echo "[cnn_v3.sh] Failed to resolve BASE_PAIR_CHECKPOINTS for species=${species}. Set BASE_PAIR_CHECKPOINTS explicitly." >&2
		exit 1
	fi

	args=(
		--model cnn_v3
		--species "${species}"
		--base_pair_checkpoints "${resolved_base_pair_checkpoints}"
		--meta_hidden_dim "${META_HIDDEN_DIM}"
		--meta_dropout "${META_DROPOUT}"
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

	if [[ ${run_index} -gt 0 ]]; then
		args+=(--skip_train)
	fi

	echo "[cnn_v3.sh] species=${species} base_pair_checkpoints=${resolved_base_pair_checkpoints} skip_train=$(( run_index > 0 ))"
	PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
		python3 "${PROJECT_ROOT}/src/run_model.py" "${args[@]}"

	run_index=$((run_index + 1))
done
