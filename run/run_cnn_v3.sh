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
TRAIN_SPECIES="Mmus,Hsap"
ARTIFACT_SPECIES="auto"
TRAIN_POS_PATH=""
TRAIN_NEG_PATH=""
BASE_PAIR_CHECKPOINTS=""
TRAIN_ONLY="1"

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
GPU_IDS="auto"  # auto: pin the single cross-species run to the first GPU.
USE_AMP="1"
AMP_DTYPE="auto"
COMPILE_MODE="off"
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

resolve_species_case() {
	intronmodel_resolve_species_case "$1" "$2" "cnn_v3.sh"
}

resolve_train_species_csv() {
	local data_root="$1"
	local species_csv="$2"
	local -a raw_species_list=()
	IFS=',' read -r -a raw_species_list <<<"${species_csv}"
	local -a resolved_species=()
	local raw_species=""
	local trimmed_species=""
	local canonical_species=""
	declare -A seen=()
	for raw_species in "${raw_species_list[@]}"; do
		trimmed_species="$(echo "${raw_species}" | xargs)"
		if [[ -z "${trimmed_species}" ]]; then
			continue
		fi
		canonical_species="$(
			resolve_species_case "${trimmed_species}" "${data_root}"
		)" || return 1
		if [[ -n "${seen[${canonical_species}]:-}" ]]; then
			continue
		fi
		seen["${canonical_species}"]="1"
		resolved_species+=("${canonical_species}")
	done
	if [[ ${#resolved_species[@]} -eq 0 ]]; then
		return 1
	fi
	printf '%s\n' "${resolved_species[@]}"
}

resolve_artifact_species_name() {
	local configured_value="$1"
	shift
	if [[ -n "${configured_value}" && "${configured_value}" != "auto" ]]; then
		printf '%s\n' "${configured_value}"
		return 0
	fi
	local joined=""
	joined="$(
		printf '%s\n' "$@" | awk 'NF' | LC_ALL=C sort -u | paste -sd '_' -
	)"
	if [[ -z "${joined}" ]]; then
		return 1
	fi
	printf 'cross/%s\n' "${joined}"
}

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

# Preferred source: model-separated tuning artifact for cnn_v2_pair.
preferred_best_paths = (
    project_root
    / "data"
    / species
    / "tuning"
    / "cnn_v2_pair"
    / "pair"
    / "best_config.json",
    # Backward-compatible fallback.
    project_root
    / "data"
    / species
    / "tuning"
    / "cnn_v2"
    / "pair"
    / "best_config.json",
)
for best_config_path in preferred_best_paths:
    if not best_config_path.is_file():
        continue
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
    if model_name not in {"cnn_v2", "cnn_v2_pair"} or checkpoint_path == "":
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

resolve_cross_base_pair_checkpoints() {
	local explicit_value="$1"
	shift
	local -a train_species_list=("$@")
	if [[ ${#train_species_list[@]} -eq 0 ]]; then
		return 1
	fi
	if [[ -n "${explicit_value}" ]]; then
		resolve_base_pair_checkpoints "${train_species_list[0]}" "${explicit_value}"
		return $?
	fi
	local -a merged=()
	local species_name=""
	local resolved_csv=""
	local token=""
	declare -A seen=()
	for species_name in "${train_species_list[@]}"; do
		resolved_csv="$(
			resolve_base_pair_checkpoints "${species_name}" ""
		)" || return 1
		IFS=',' read -r -a resolved_tokens <<<"${resolved_csv}"
		for token in "${resolved_tokens[@]}"; do
			token="$(echo "${token}" | xargs)"
			if [[ -z "${token}" ]]; then
				continue
			fi
			if [[ -n "${seen[${token}]:-}" ]]; then
				continue
			fi
			seen["${token}"]="1"
			merged+=("${token}")
		done
	done
	if [[ ${#merged[@]} -eq 0 ]]; then
		return 1
	fi
	local merged_csv=""
	merged_csv="$(IFS=','; echo "${merged[*]}")"
	printf '%s\n' "${merged_csv}"
}

resolve_cross_train_paths() {
	local python_bin
	python_bin="$(intronmodel_resolve_python_bin "run_cnn_v3.sh")"
	local project_root="$1"
	local data_root="$2"
	local artifact_species="$3"
	local donor_len="$4"
	local acceptor_len="$5"
	local train_pos_template="$6"
	local train_neg_template="$7"
	local train_species_csv="$8"
	"${python_bin}" - \
		"${project_root}" \
		"${data_root}" \
		"${artifact_species}" \
		"${donor_len}" \
		"${acceptor_len}" \
		"${train_pos_template}" \
		"${train_neg_template}" \
		"${train_species_csv}" <<'PY'
from __future__ import annotations

import os
import sys
from pathlib import Path

project_root = Path(sys.argv[1]).resolve()
data_root = Path(sys.argv[2]).resolve()
artifact_species = sys.argv[3].strip()
donor_len_raw = sys.argv[4].strip()
acceptor_len_raw = sys.argv[5].strip()
train_pos_template = sys.argv[6].strip()
train_neg_template = sys.argv[7].strip()
species_csv = sys.argv[8].strip()

if artifact_species == "":
    print("[cnn_v3.sh] ARTIFACT_SPECIES must not be empty.", file=sys.stderr)
    raise SystemExit(2)

species_list = [token.strip() for token in species_csv.split(",") if token.strip()]
if not species_list:
    print("[cnn_v3.sh] TRAIN_SPECIES must contain at least one value.", file=sys.stderr)
    raise SystemExit(2)

if (train_pos_template == "") != (train_neg_template == ""):
    print(
        "[cnn_v3.sh] TRAIN_POS_PATH and TRAIN_NEG_PATH must be set together.",
        file=sys.stderr,
    )
    raise SystemExit(2)


def _parse_optional_len(raw_value: str) -> int | None:
    text = raw_value.strip().lower()
    if text in {"", "none", "null"}:
        return None
    value = int(text)
    if value <= 0:
        raise ValueError("window length must be positive")
    return value


def _resolve_species_template(template: str, species: str) -> str:
    return (
        template.replace("${SPECIES}", species)
        .replace("{SPECIES}", species)
        .replace("{species}", species)
    )


def _copy_concat(inputs: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_file:
        for source_path in inputs:
            with Path(source_path).open("r", encoding="utf-8") as in_file:
                saw_content = False
                for line in in_file:
                    saw_content = True
                    out_file.write(line)
            if saw_content:
                with Path(source_path).open("rb") as raw_file:
                    raw_file.seek(0, os.SEEK_END)
                    if raw_file.tell() > 0:
                        raw_file.seek(-1, os.SEEK_END)
                        if raw_file.read(1) != b"\n":
                            out_file.write("\n")


donor_len = _parse_optional_len(donor_len_raw)
acceptor_len = _parse_optional_len(acceptor_len_raw)

sys.path.insert(0, str(project_root / "src"))
from util.data_proc import resolve_train_paths  # noqa: E402

resolved_pos_paths: list[str] = []
resolved_neg_paths: list[str] = []
for species in species_list:
    if train_pos_template != "":
        pos_path = _resolve_species_template(train_pos_template, species)
        neg_path = _resolve_species_template(train_neg_template, species)
    else:
        pos_path, neg_path, _ = resolve_train_paths(
            species=species,
            train_pos_path=None,
            train_neg_path=None,
            donor_len=donor_len,
            acceptor_len=acceptor_len,
        )
    if not Path(pos_path).is_file():
        print(
            f"[cnn_v3.sh] TRAIN_POS_PATH not found for species={species}: {pos_path}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if not Path(neg_path).is_file():
        print(
            f"[cnn_v3.sh] TRAIN_NEG_PATH not found for species={species}: {neg_path}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    resolved_pos_paths.append(pos_path)
    resolved_neg_paths.append(neg_path)

species_token = "_".join(sorted(species_list, key=str.casefold))
donor_token = "auto" if donor_len is None else str(donor_len)
acceptor_token = "auto" if acceptor_len is None else str(acceptor_len)
prefix = f"cross_{species_token}_d{donor_token}_a{acceptor_token}"
output_dir = data_root / artifact_species / "train"
merged_pos_path = output_dir / f"{prefix}.err"
merged_neg_path = output_dir / f"{prefix}.neg.err"
_copy_concat(resolved_pos_paths, merged_pos_path)
_copy_concat(resolved_neg_paths, merged_neg_path)
print(f"{merged_pos_path}\t{merged_neg_path}")
PY
}

if [[ "${TRAIN_ONLY}" != "0" && "${TRAIN_ONLY}" != "1" ]]; then
	echo "[cnn_v3.sh] TRAIN_ONLY must be 0 or 1." >&2
	exit 1
fi

mapfile -t TRAIN_SPECIES_LIST < <(
	resolve_train_species_csv "${DATA_ROOT}" "${TRAIN_SPECIES}"
) || {
	echo "[cnn_v3.sh] Failed to resolve TRAIN_SPECIES=${TRAIN_SPECIES}." >&2
	exit 1
}
if [[ ${#TRAIN_SPECIES_LIST[@]} -eq 0 ]]; then
	echo "[cnn_v3.sh] TRAIN_SPECIES resolved to an empty list." >&2
	exit 1
fi
TRAIN_SPECIES_RESOLVED_CSV="$(IFS=','; echo "${TRAIN_SPECIES_LIST[*]}")"
ARTIFACT_SPECIES_RESOLVED="$(
	resolve_artifact_species_name "${ARTIFACT_SPECIES}" "${TRAIN_SPECIES_LIST[@]}"
)"
if [[ -z "${ARTIFACT_SPECIES_RESOLVED}" ]]; then
	echo "[cnn_v3.sh] ARTIFACT_SPECIES resolved to empty." >&2
	exit 1
fi

if ! RESOLVED_BASE_PAIR_CHECKPOINTS="$(
	resolve_cross_base_pair_checkpoints \
		"${BASE_PAIR_CHECKPOINTS}" \
		"${TRAIN_SPECIES_LIST[@]}"
)"; then
	echo "[cnn_v3.sh] Failed to resolve BASE_PAIR_CHECKPOINTS for TRAIN_SPECIES=${TRAIN_SPECIES_RESOLVED_CSV}. Set BASE_PAIR_CHECKPOINTS explicitly." >&2
	exit 1
fi

if ! RESOLVED_CROSS_TRAIN_PATHS="$(
	resolve_cross_train_paths \
		"${PROJECT_ROOT}" \
		"${DATA_ROOT}" \
		"${ARTIFACT_SPECIES_RESOLVED}" \
		"${DONOR_LEN}" \
		"${ACCEPTOR_LEN}" \
		"${TRAIN_POS_PATH}" \
		"${TRAIN_NEG_PATH}" \
		"${TRAIN_SPECIES_RESOLVED_CSV}"
)"; then
	exit 1
fi
IFS=$'\t' read -r CROSS_TRAIN_POS_PATH CROSS_TRAIN_NEG_PATH <<< \
	"${RESOLVED_CROSS_TRAIN_PATHS}"
if [[ -z "${CROSS_TRAIN_POS_PATH}" || -z "${CROSS_TRAIN_NEG_PATH}" ]]; then
	echo "[cnn_v3.sh] Failed to build cross-species train files." >&2
	exit 1
fi

mapfile -t GPU_ID_LIST < <(
	intronmodel_resolve_gpu_ids "cnn_v3.sh" "${GPU_IDS}" "${DEVICE}"
)
ASSIGNED_GPU_ID=""
if [[ ${#GPU_ID_LIST[@]} -gt 0 ]]; then
	ASSIGNED_GPU_ID="${GPU_ID_LIST[0]}"
fi

args=(
	--model cnn_v3
	--species "${ARTIFACT_SPECIES_RESOLVED}"
	--base_pair_checkpoints "${RESOLVED_BASE_PAIR_CHECKPOINTS}"
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
	--train_pos_path "${CROSS_TRAIN_POS_PATH}"
	--train_neg_path "${CROSS_TRAIN_NEG_PATH}"
	--train_only "${TRAIN_ONLY}"
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

echo "[cnn_v3.sh] train_species=${TRAIN_SPECIES_RESOLVED_CSV} artifact_species=${ARTIFACT_SPECIES_RESOLVED} train_only=${TRAIN_ONLY}"
echo "[cnn_v3.sh] base_pair_checkpoints=${RESOLVED_BASE_PAIR_CHECKPOINTS}"
if [[ -n "${ASSIGNED_GPU_ID}" ]]; then
	CUDA_VISIBLE_DEVICES="${ASSIGNED_GPU_ID}" \
		PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
		python3 "${PROJECT_ROOT}/src/run_model.py" "${args[@]}"
else
	PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
		python3 "${PROJECT_ROOT}/src/run_model.py" "${args[@]}"
fi
