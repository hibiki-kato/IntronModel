#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOT'
Usage: bash run/run_isolated_mmus_rna60_pipeline.sh [options]

Options:
  --query-gtf <path>         Query GTF-like input path
                             (default: raw/GCF_000001635.27_GRCm39_genomic.fna.gtf.rna60)
  --species <name>           Species name (default: Mmus)
  --source-data-root <path>  Source data root used for raw assets and tuning configs
                             (default: <repo>/data)
  --source-model-root <path> Source model root used for checkpoints
                             (default: <repo>/model)
  --work-root <path>         Isolated workspace root
                             (default: <repo>/temp/isolated_<species>_rna60)
  --device <name>            DEVICE for inference wrappers (default: auto)
  --gpu-ids <csv|auto>       GPU_IDS for inference wrappers (default: auto)
  --max-parallel <n|auto>    MAX_PARALLEL_TRIALS for inference wrappers
                             (default: auto)
  --train-on-missing         If tuned best config is missing, train the model
                             and continue (default: on)
  --no-train-on-missing      Disable train fallback on missing tuned config
  --skip-inference           Stop after data-prep steps
  --allow-missing-models     Skip models whose tuned config is missing
  -h, --help                 Show this help

What this script does (isolated under --work-root):
  1) transcript_label generation
  2) prepare 100bp + masked + truncated data
  3) intron unique/map assets
  4) inference for Mmus models excluding Markov
EOT
}

# --------------------------
# USER DEFAULTS (optional edit)
# --------------------------
CONDA_ENV="intronmodel"
QUERY_GTF_INPUT="raw/GCF_000001635.27_GRCm39_genomic.fna.gtf.rna60"
SPECIES="Mmus"
SOURCE_DATA_ROOT=""
SOURCE_MODEL_ROOT=""
WORK_ROOT=""
DONOR_LEN="100"
ACCEPTOR_LEN="100"
FLANK_BP="10"
EXON_CONTEXT_BP="3"
DEVICE_SETTING="auto"
GPU_IDS_SETTING="auto"
MAX_PARALLEL_SETTING="auto"
RUN_INFERENCE="1"
ALLOW_MISSING_MODELS="0"
TRAIN_ON_MISSING="1"
# Set to "1" to force-skip Step 1-7, or "0" to skip only when outputs exist.
SKIP_PREP_STEPS="1"
# Set to "1" to skip inference when output score files already exist.
SKIP_EXISTING_INFER="1"
# Comma-separated inference model allowlist.
# Use "all" to run every model block below.
# Available keys:
#   cnn,cnn_pair,bilstm_pair,cnn_resdil,tcn,bert,reservoir,dnabert6,dnabert2_pair,cnn_v2
INFER_MODELS="cnn_v2,dnabert2_pair"

# --------------------------
# Runtime implementation
# --------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"
intronmodel_activate_conda "${CONDA_ENV}"
intronmodel_init_paths "${BASH_SOURCE[0]}"

# Keep this orchestration fully non-interactive and detached from tmux bootstrap.
INTRONMODEL_AUTO_TMUX="off"

while [[ $# -gt 0 ]]; do
	case "$1" in
	--query-gtf)
		QUERY_GTF_INPUT="$2"
		shift 2
		;;
	--species)
		SPECIES="$2"
		shift 2
		;;
	--source-data-root)
		SOURCE_DATA_ROOT="$2"
		shift 2
		;;
	--source-model-root)
		SOURCE_MODEL_ROOT="$2"
		shift 2
		;;
	--work-root)
		WORK_ROOT="$2"
		shift 2
		;;
	--device)
		DEVICE_SETTING="$2"
		shift 2
		;;
	--gpu-ids)
		GPU_IDS_SETTING="$2"
		shift 2
		;;
	--max-parallel)
		MAX_PARALLEL_SETTING="$2"
		shift 2
		;;
	--skip-inference)
		RUN_INFERENCE="0"
		shift
		;;
	--train-on-missing)
		TRAIN_ON_MISSING="1"
		shift
		;;
	--no-train-on-missing)
		TRAIN_ON_MISSING="0"
		shift
		;;
	--allow-missing-models)
		ALLOW_MISSING_MODELS="1"
		shift
		;;
	-h | --help)
		usage
		exit 0
		;;
	*)
		echo "[isolated_pipeline] Unknown argument: $1" >&2
		usage >&2
		exit 1
		;;
	esac
done

log() {
	printf '[isolated_pipeline] %s\n' "$*"
}

die() {
	printf '[isolated_pipeline] ERROR: %s\n' "$*" >&2
	exit 1
}

if [[ -z "${SOURCE_DATA_ROOT}" ]]; then
	SOURCE_DATA_ROOT="${PROJECT_ROOT}/data"
fi
if [[ -z "${SOURCE_MODEL_ROOT}" ]]; then
	SOURCE_MODEL_ROOT="${PROJECT_ROOT}/model"
fi
if [[ -z "${WORK_ROOT}" ]]; then
	WORK_ROOT="${PROJECT_ROOT}/temp/isolated_${SPECIES}_rna60"
fi

resolve_existing_file() {
	local raw_path="$1"
	local source_data_root="$2"
	local species="$3"
	local -a candidates=()
	local base_name=""

	base_name="$(basename "${raw_path}")"
	candidates=(
		"${raw_path}"
		"${PROJECT_ROOT}/${raw_path}"
		"${source_data_root}/${raw_path}"
		"${source_data_root}/${species}/${raw_path}"
		"${source_data_root}/${species}/raw/${base_name}"
	)

	local candidate=""
	for candidate in "${candidates[@]}"; do
		if [[ -f "${candidate}" ]]; then
			realpath "${candidate}"
			return 0
		fi
	done
	return 1
}

resolve_existing_dir() {
	local raw_path="$1"
	if [[ -d "${raw_path}" ]]; then
		realpath "${raw_path}"
		return 0
	fi
	if [[ -d "${PROJECT_ROOT}/${raw_path}" ]]; then
		realpath "${PROJECT_ROOT}/${raw_path}"
		return 0
	fi
	return 1
}

SOURCE_DATA_ROOT="$(resolve_existing_dir "${SOURCE_DATA_ROOT}" \
	|| die "source data root not found: ${SOURCE_DATA_ROOT}")"
SOURCE_MODEL_ROOT="$(resolve_existing_dir "${SOURCE_MODEL_ROOT}" \
	|| die "source model root not found: ${SOURCE_MODEL_ROOT}")"
QUERY_GTF_ABS="$(resolve_existing_file "${QUERY_GTF_INPUT}" "${SOURCE_DATA_ROOT}" "${SPECIES}" \
	|| die "query gtf input not found: ${QUERY_GTF_INPUT}")"

SOURCE_RAW_DIR="${SOURCE_DATA_ROOT}/${SPECIES}/raw"
[[ -d "${SOURCE_RAW_DIR}" ]] || die "source raw dir not found: ${SOURCE_RAW_DIR}"

pick_first_file() {
	local dir_path="$1"
	shift
	local pattern=""
	local -a matches=()
	for pattern in "$@"; do
		shopt -s nullglob
		matches=("${dir_path}"/${pattern})
		shopt -u nullglob
		if [[ ${#matches[@]} -gt 0 ]]; then
			printf '%s\n' "${matches[0]}"
			return 0
		fi
	done
	return 1
}

SOURCE_FASTA="$(pick_first_file "${SOURCE_RAW_DIR}" "*.clean.fna" "*.fna" \
	|| die "fasta not found under ${SOURCE_RAW_DIR}")"
SOURCE_REF_ANNOT="$(pick_first_file "${SOURCE_RAW_DIR}" \
	"*.fix.gff" "*.gff.fix" "*.gff" "*.gff3" \
	|| die "reference annotation not found under ${SOURCE_RAW_DIR}")"
SOURCE_100BP_POS="${SOURCE_RAW_DIR}/100bp.err"
SOURCE_100BP_NEG="${SOURCE_RAW_DIR}/100bp.neg.err"
[[ -f "${SOURCE_100BP_POS}" ]] || die "missing source file: ${SOURCE_100BP_POS}"
[[ -f "${SOURCE_100BP_NEG}" ]] || die "missing source file: ${SOURCE_100BP_NEG}"

ISOLATED_DATA_ROOT="${WORK_ROOT}/data"
ISOLATED_SPECIES_DIR="${ISOLATED_DATA_ROOT}/${SPECIES}"
ISOLATED_RAW_DIR="${ISOLATED_SPECIES_DIR}/raw"
ISOLATED_PROCESSED_DIR="${ISOLATED_SPECIES_DIR}/processed"
mkdir -p "${ISOLATED_RAW_DIR}" "${ISOLATED_PROCESSED_DIR}"

copy_asset() {
	local src_path="$1"
	local dst_path="$2"
	if [[ -L "${dst_path}" ]]; then
		# Replace legacy symlink output with a real copied file.
		rm -f "${dst_path}"
	fi
	if [[ -e "${dst_path}" && ! -f "${dst_path}" ]]; then
		die "destination exists but is not a regular file: ${dst_path}"
	fi
	if [[ ! -f "${dst_path}" || "${src_path}" -nt "${dst_path}" ]]; then
		cp -f "${src_path}" "${dst_path}"
	fi
}

ISOLATED_FASTA="${ISOLATED_RAW_DIR}/$(basename "${SOURCE_FASTA}")"
ISOLATED_REF_ANNOT="${ISOLATED_RAW_DIR}/$(basename "${SOURCE_REF_ANNOT}")"
ISOLATED_QUERY_GTF="${ISOLATED_RAW_DIR}/query.rna60.gtf"
ISOLATED_FASTA_GTF="${ISOLATED_FASTA}.gtf"
ISOLATED_100BP_POS="${ISOLATED_RAW_DIR}/100bp.err"
ISOLATED_100BP_NEG="${ISOLATED_RAW_DIR}/100bp.neg.err"
ISOLATED_CLASS_FILE="${ISOLATED_RAW_DIR}/transcript_class.txt"

copy_asset "${SOURCE_FASTA}" "${ISOLATED_FASTA}"
copy_asset "${SOURCE_REF_ANNOT}" "${ISOLATED_REF_ANNOT}"
copy_asset "${QUERY_GTF_ABS}" "${ISOLATED_QUERY_GTF}"
# make_intron_training_data_from_err expects <fasta>.gtf exactly.
copy_asset "${QUERY_GTF_ABS}" "${ISOLATED_FASTA_GTF}"
copy_asset "${SOURCE_100BP_POS}" "${ISOLATED_100BP_POS}"
copy_asset "${SOURCE_100BP_NEG}" "${ISOLATED_100BP_NEG}"

run_with_isolated_root() {
	INTRONMODEL_AUTO_TMUX=off \
		INTRONMODEL_DATA_ROOT="${ISOLATED_DATA_ROOT}" \
		INTRONMODEL_MODEL_ROOT="${SOURCE_MODEL_ROOT}" \
		"$@"
}

all_files_exist() {
	local path=""
	for path in "$@"; do
		if [[ ! -f "${path}" ]]; then
			return 1
		fi
	done
	return 0
}

TRANSCRIPTS_TSV="${ISOLATED_PROCESSED_DIR}/transcripts.tsv"
INTRON_EVAL_TSV="${ISOLATED_PROCESSED_DIR}/intron_eval_flank10.tsv"
INTRON_POS_TSV="${ISOLATED_PROCESSED_DIR}/intron_full_flank10.pos.tsv"
INTRON_QC_TSV="${ISOLATED_PROCESSED_DIR}/intron_full_flank10.pos.qc.tsv"
INTRON_NEG_REQ_TSV="${ISOLATED_PROCESSED_DIR}/intron_full_flank10.neg_coordinate_request.tsv"
TRIMMED_POS="${ISOLATED_PROCESSED_DIR}/100bp_trimmed.err"
TRIMMED_NEG="${ISOLATED_PROCESSED_DIR}/100bp_trimmed.neg.err"
TRIMMED_NPAD_POS="${ISOLATED_PROCESSED_DIR}/100bp_trimmed_npad.err"
TRIMMED_NPAD_NEG="${ISOLATED_PROCESSED_DIR}/100bp_trimmed_npad.neg.err"
UNIQUE_TSV="${ISOLATED_PROCESSED_DIR}/transcripts.unique.tsv"
UNIQUE_MAP_TSV="${ISOLATED_PROCESSED_DIR}/transcripts.unique.map.tsv"
UNIQUE_LABELED_TSV="${ISOLATED_PROCESSED_DIR}/intron_eval_flank10.unique.tsv"
UNIQUE_CATALOG_TSV="${ISOLATED_PROCESSED_DIR}/intron_unique_catalog.tsv"
UNIQUE_MASK_TSV="${ISOLATED_PROCESSED_DIR}/transcripts.unique.mask.tsv"
UNIQUE_TRUNC_TSV="${ISOLATED_PROCESSED_DIR}/transcripts.unique.trunc.tsv"

log "work_root=${WORK_ROOT}"
log "isolated_data_root=${ISOLATED_DATA_ROOT}"
log "source_model_root=${SOURCE_MODEL_ROOT}"
log "species=${SPECIES}"
log "query_gtf=${QUERY_GTF_ABS}"
log "skip_prep_steps=${SKIP_PREP_STEPS}"
log "skip_existing_infer=${SKIP_EXISTING_INFER}"

if [[ "${SKIP_PREP_STEPS}" == "1" ]]; then
	log "Step 1/7: skip (SKIP_PREP_STEPS=1)"
elif all_files_exist "${TRANSCRIPTS_TSV}"; then
	log "Step 1/7: skip (already exists: ${TRANSCRIPTS_TSV})"
else
	log "Step 1/7: build transcripts.tsv (100bp windows)"
	run_with_isolated_root \
		bash "${PROJECT_ROOT}/run/make_test_data.sh" \
			--species "${SPECIES}" \
			--fasta "${ISOLATED_FASTA}" \
			--gtf "${ISOLATED_QUERY_GTF}" \
			--out-tsv "${TRANSCRIPTS_TSV}" \
			--donor-len "${DONOR_LEN}" \
			--acceptor-len "${ACCEPTOR_LEN}"
fi

if [[ "${SKIP_PREP_STEPS}" == "1" ]]; then
	log "Step 2/7: skip (SKIP_PREP_STEPS=1)"
elif all_files_exist "${ISOLATED_CLASS_FILE}"; then
	log "Step 2/7: skip (already exists: ${ISOLATED_CLASS_FILE})"
else
	log "Step 2/7: build transcript_label (transcript_class.txt)"
	run_with_isolated_root \
		bash "${PROJECT_ROOT}/run/make_transcript_class.sh" \
			--species "${SPECIES}" \
			--data-root "${ISOLATED_DATA_ROOT}" \
			--gtf "${ISOLATED_QUERY_GTF}" \
			--reference-annotation "${ISOLATED_REF_ANNOT}" \
			--out-name "transcript_class.txt"
fi

if [[ "${SKIP_PREP_STEPS}" == "1" ]]; then
	log "Step 3/7: skip (SKIP_PREP_STEPS=1)"
elif all_files_exist "${INTRON_EVAL_TSV}"; then
	log "Step 3/7: skip (already exists: ${INTRON_EVAL_TSV})"
else
	log "Step 3/7: build labeled intron eval TSV"
	run_with_isolated_root \
		bash "${PROJECT_ROOT}/run/make_labeled_intron_eval_data.sh" \
			--species "${SPECIES}" \
			--data-root "${ISOLATED_DATA_ROOT}" \
			--fasta "${ISOLATED_FASTA}" \
			--query-gtf "${ISOLATED_QUERY_GTF}" \
			--reference-annotation "${ISOLATED_REF_ANNOT}" \
			--out-name "intron_eval_flank10.tsv" \
			--donor-len "${DONOR_LEN}" \
			--acceptor-len "${ACCEPTOR_LEN}" \
			--flank-bp "${FLANK_BP}"
fi

if [[ "${SKIP_PREP_STEPS}" == "1" ]]; then
	log "Step 4/7: skip (SKIP_PREP_STEPS=1)"
elif all_files_exist "${INTRON_POS_TSV}" "${INTRON_QC_TSV}" "${INTRON_NEG_REQ_TSV}"; then
	log "Step 4/7: skip (intron training utility TSVs already exist)"
else
	log "Step 4/7: build 100bp-derived train utility data"
	run_with_isolated_root \
		bash "${PROJECT_ROOT}/run/make_intron_training_data.sh" \
			--species "${SPECIES}" \
			--data-root "${ISOLATED_DATA_ROOT}" \
			--flank-bp "${FLANK_BP}" \
			--no-strict
fi

if [[ "${SKIP_PREP_STEPS}" == "1" ]]; then
	log "Step 5/7: skip (SKIP_PREP_STEPS=1)"
elif all_files_exist \
	"${TRIMMED_POS}" \
	"${TRIMMED_NEG}" \
	"${TRIMMED_NPAD_POS}" \
	"${TRIMMED_NPAD_NEG}"; then
	log "Step 5/7: skip (trimmed/masked pair files already exist)"
else
	log "Step 5/7: build truncated/masked 100bp pair data"
	run_with_isolated_root \
		bash "${PROJECT_ROOT}/run/make_trimmed_pair_data.sh" \
			--species "${SPECIES}" \
			--data-root "${ISOLATED_DATA_ROOT}" \
			--out-pos-name "100bp_trimmed.err" \
			--out-neg-name "100bp_trimmed.neg.err" \
			--exon-context-bp "${EXON_CONTEXT_BP}"
	run_with_isolated_root \
		bash "${PROJECT_ROOT}/run/make_trimmed_pair_data.sh" \
			--species "${SPECIES}" \
			--data-root "${ISOLATED_DATA_ROOT}" \
			--out-pos-name "100bp_trimmed_npad.err" \
			--out-neg-name "100bp_trimmed_npad.neg.err" \
			--exon-context-bp "${EXON_CONTEXT_BP}" \
			--pad-with-n
fi

if [[ "${SKIP_PREP_STEPS}" == "1" ]]; then
	log "Step 6/7: skip (SKIP_PREP_STEPS=1)"
elif all_files_exist \
	"${UNIQUE_TSV}" \
	"${UNIQUE_MAP_TSV}" \
	"${UNIQUE_LABELED_TSV}" \
	"${UNIQUE_CATALOG_TSV}"; then
	log "Step 6/7: skip (unique/map assets already exist)"
else
	log "Step 6/7: build unique intron assets (unique/map)"
	run_with_isolated_root \
		bash "${PROJECT_ROOT}/run/make_unique_intron_assets.sh" \
			--species "${SPECIES}" \
			--data-root "${ISOLATED_DATA_ROOT}" \
			--overwrite
fi

if [[ "${SKIP_PREP_STEPS}" == "1" ]]; then
	log "Step 7/7: skip (SKIP_PREP_STEPS=1)"
elif all_files_exist "${UNIQUE_MASK_TSV}" "${UNIQUE_TRUNC_TSV}"; then
	log "Step 7/7: skip (unique mask/trunc variants already exist)"
else
	log "Step 7/7: build unique mask/trunc sequence variants"
	run_with_isolated_root \
		python3 "${PROJECT_ROOT}/src/tools/build_unique_sequence_variants.py" \
			--species "${SPECIES}" \
			--data-root "${ISOLATED_DATA_ROOT}" \
			--exon-context-bp "${EXON_CONTEXT_BP}" \
			--overwrite
fi

if [[ "${SKIP_PREP_STEPS}" == "1" ]]; then
	if ! all_files_exist \
		"${ISOLATED_CLASS_FILE}" \
		"${TRANSCRIPTS_TSV}" \
		"${INTRON_EVAL_TSV}" \
		"${INTRON_POS_TSV}" \
		"${INTRON_QC_TSV}" \
		"${INTRON_NEG_REQ_TSV}" \
		"${TRIMMED_POS}" \
		"${TRIMMED_NEG}" \
		"${TRIMMED_NPAD_POS}" \
		"${TRIMMED_NPAD_NEG}" \
		"${UNIQUE_TSV}" \
		"${UNIQUE_MAP_TSV}" \
		"${UNIQUE_LABELED_TSV}" \
		"${UNIQUE_CATALOG_TSV}" \
		"${UNIQUE_MASK_TSV}" \
		"${UNIQUE_TRUNC_TSV}"; then
		die "SKIP_PREP_STEPS=1 but required prep outputs are missing under ${ISOLATED_SPECIES_DIR}. Set SKIP_PREP_STEPS=0 once, or reuse a prepared --work-root."
	fi
fi

extract_config_block() {
	local wrapper_path="$1"
	awk '
		/^set -a$/ {in_block=1; next}
		/^set \+a$/ {in_block=0}
		in_block {print}
	' "${wrapper_path}"
}

all_tuned_configs_exist() {
	local path=""
	for path in "$@"; do
		if [[ ! -f "${path}" ]]; then
			return 1
		fi
	done
	return 0
}

missing_tuned_configs_text() {
	local path=""
	local first="1"
	for path in "$@"; do
		if [[ -f "${path}" ]]; then
			continue
		fi
		if [[ "${first}" == "1" ]]; then
			printf '%s' "${path}"
			first="0"
		else
			printf ', %s' "${path}"
		fi
	done
}

model_is_selected() {
	local model_key="${1,,}"
	local raw_list="${INFER_MODELS,,}"
	local normalized="${raw_list//[[:space:]]/}"
	if [[ -z "${normalized}" || "${normalized}" == "all" ]]; then
		return 0
	fi
	local token=""
	IFS=',' read -r -a model_tokens <<< "${normalized}"
	for token in "${model_tokens[@]}"; do
		if [[ "${token}" == "${model_key}" ]]; then
			return 0
		fi
	done
	return 1
}

missing_action_for_model() {
	local label="$1"
	local missing_text="$2"
	local supports_train_fallback="$3"

	if [[ "${TRAIN_ON_MISSING}" == "1" && "${supports_train_fallback}" == "1" ]]; then
		log "Missing tuned config for ${label}: ${missing_text}; run training fallback."
		printf 'train\n'
		return 0
	fi
	if [[ "${ALLOW_MISSING_MODELS}" == "1" ]]; then
		log "Skip ${label}: missing tuned config (${missing_text})"
		printf 'skip\n'
		return 0
	fi
	if [[ "${TRAIN_ON_MISSING}" == "1" && "${supports_train_fallback}" != "1" ]]; then
		die "Missing tuned config for ${label}: ${missing_text} (auto-train not supported)"
	fi
	die "Missing tuned config for ${label}: ${missing_text}"
}

run_wrapper_pipeline_mode() {
	local mode="$1"
	local wrapper_script="$2"
	local script_name="$3"
	shift 3
	local pipeline_species="${SPECIES}"
	local pipeline_donor_len="${DONOR_LEN}"
	local pipeline_acceptor_len="${ACCEPTOR_LEN}"
	local pipeline_class_file="${ISOLATED_CLASS_FILE}"
	local config_block
	config_block="$(extract_config_block "${wrapper_script}")"
	(
		set -euo pipefail
		set -a
		# shellcheck disable=SC2016
		eval "${config_block}"
		set +a

		export INTRONMODEL_DATA_ROOT="${ISOLATED_DATA_ROOT}"
		export INTRONMODEL_MODEL_ROOT="${SOURCE_MODEL_ROOT}"
		export INTRONMODEL_AUTO_TMUX="off"
		export SPECIES="${pipeline_species}"
		export DONOR_LEN="${pipeline_donor_len}"
		export ACCEPTOR_LEN="${pipeline_acceptor_len}"
		if [[ "${mode}" == "infer" ]]; then
			export SKIP_TRAINING="1"
			export USE_TUNED_HPARAMS="required"
		elif [[ "${mode}" == "train" ]]; then
			export SKIP_TRAINING="0"
			export USE_TUNED_HPARAMS="auto"
		else
			echo "[isolated_pipeline] invalid wrapper mode: ${mode}" >&2
			exit 2
		fi
		export CONTINUE_TRAINING="0"
		export TRAIN_ONLY="0"
		export TUNED_HPARAMS_MODE="normal"
		export VISUALIZE="none"
		export DEVICE="${DEVICE_SETTING}"
		export GPU_IDS="${GPU_IDS_SETTING}"
		export MAX_PARALLEL_TRIALS="${MAX_PARALLEL_SETTING}"
		export COMPILE_MODE="off"
		export INFER_COMPILE="0"
		export INFER_COMPILE_MODE="off"
		export CLASS_FILE_PATH="${pipeline_class_file}"
		export TEST_TSV_PATH=""
		export TRAIN_POS_PATH=""
		export TRAIN_NEG_PATH=""
		export PRECOMPUTED_SITE_SCORE_TSV=""
		export CHECKPOINT_TOP_K="3"
		export CHECKPOINT_PRUNE_DRY_RUN="0"
		export NAME_FIELDS=""
		export TAG=""
		export SKIP_EXISTING_INFER="${SKIP_EXISTING_INFER}"

		local assignment=""
		for assignment in "$@"; do
			export "${assignment}"
		done

		PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
			python3 "${PROJECT_ROOT}/src/tools/run_wrapper_pipeline.py" \
				--script-name "${script_name}"
	)
}

run_wrapper_infer() {
	run_wrapper_pipeline_mode "infer" "$@"
}

run_wrapper_train_fallback() {
	run_wrapper_pipeline_mode "train" "$@"
}

run_legacy_cnn_v2_infer() {
	local best_config_path="$1"
	log "Run legacy cnn_v2 inference from tuned config: ${best_config_path}"
	INTRONMODEL_DATA_ROOT="${ISOLATED_DATA_ROOT}" \
		INTRONMODEL_MODEL_ROOT="${SOURCE_MODEL_ROOT}" \
		PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
		python3 - \
			"${PROJECT_ROOT}" \
			"${SPECIES}" \
			"${best_config_path}" \
			"${DONOR_LEN}" \
			"${ACCEPTOR_LEN}" \
			"${ISOLATED_CLASS_FILE}" \
			"${ISOLATED_PROCESSED_DIR}/transcripts.unique.tsv" <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

project_root = Path(sys.argv[1]).resolve()
species = sys.argv[2]
best_config_path = Path(sys.argv[3]).resolve()
donor_len = int(sys.argv[4])
acceptor_len = int(sys.argv[5])
class_file = Path(sys.argv[6]).resolve()
test_tsv = Path(sys.argv[7]).resolve()

if not best_config_path.is_file():
    raise FileNotFoundError(f"best_config not found: {best_config_path}")
if not class_file.is_file():
    raise FileNotFoundError(f"class_file not found: {class_file}")
if not test_tsv.is_file():
    raise FileNotFoundError(f"test_tsv not found: {test_tsv}")

sys.path.insert(0, str(project_root / "src"))

from run_model import (  # noqa: E402
    _build_checkpoint_paths,
    _build_checkpoint_stem_from_params,
    _infer_window_defaults,
    parse_args,
)
from util.checkpoint_io import extract_task_checkpoint_path, read_json_object  # noqa: E402


def scalar_to_text(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(value, ".15g")
    return str(value)


payload = read_json_object(best_config_path)
if payload is None or payload.get("status") != "ok":
    raise ValueError(f"Invalid best_config payload: {best_config_path}")

pair_ckpt = extract_task_checkpoint_path(
    payload,
    task="pair",
    base_dir=best_config_path.parent,
)
if pair_ckpt is None or not pair_ckpt.exists():
    raise FileNotFoundError(
        f"pair checkpoint not found in best_config: {best_config_path}"
    )

sampled_params_obj = payload.get("sampled_params")
sampled_params = sampled_params_obj if isinstance(sampled_params_obj, dict) else {}

run_args: list[str] = [
    "--model",
    "cnn_v2",
    "--species",
    species,
    "--donor_len",
    str(donor_len),
    "--acceptor_len",
    str(acceptor_len),
    "--train_target",
    "pair",
    "--sequence_transform",
    "none",
    "--name_fields",
    "tag",
    "--tag",
    "legacy_cnn_v2_iso",
    "--seed",
    "1337",
    "--device",
    "auto",
    "--compile_mode",
    "off",
    "--infer_compile",
    "0",
    "--infer_compile_mode",
    "off",
    "--skip_train",
    "--test_tsv",
    str(test_tsv),
    "--class_file",
    str(class_file),
]

for key in sorted(sampled_params):
    value = sampled_params[key]
    if value is None:
        continue
    run_args.extend([f"--{key}", scalar_to_text(value)])

parsed = parse_args(run_args)
resolved_donor_len, resolved_acceptor_len, inferred_train_len = _infer_window_defaults(
    species=parsed.species,
    donor_len=parsed.donor_len,
    acceptor_len=parsed.acceptor_len,
)
stem = _build_checkpoint_stem_from_params(
    model_name=parsed.model,
    donor_len=resolved_donor_len,
    acceptor_len=resolved_acceptor_len,
    inferred_train_len=inferred_train_len,
    raw_params=dict(vars(parsed)),
)
strict_pair_path = Path(
    _build_checkpoint_paths(parsed.species, stem, tasks=("pair",))["pair"]
).resolve()

if not strict_pair_path.exists():
    strict_pair_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(pair_ckpt, strict_pair_path)
    except OSError:
        shutil.copy2(pair_ckpt, strict_pair_path)

command = [sys.executable, str(project_root / "src" / "run_model.py"), *run_args]
subprocess.run(command, check=True)
PY
}

if [[ "${RUN_INFERENCE}" != "1" ]]; then
	log "skip-inference requested; preparation completed."
	log "isolated_data_root=${ISOLATED_DATA_ROOT}"
	exit 0
fi

TUNING_ROOT="${SOURCE_DATA_ROOT}/${SPECIES}/tuning"

CNN_DONOR_CFG="${TUNING_ROOT}/cnn_mask/donor/best_config.json"
CNN_ACCEPTOR_CFG="${TUNING_ROOT}/cnn_mask/acceptor/best_config.json"
CNN_PAIR_CFG="${TUNING_ROOT}/cnn_pair_mask/pair/best_config.json"
BILSTM_PAIR_CFG="${TUNING_ROOT}/bilstm_pair_mask/pair/best_config.json"
CNN_RESDIL_DONOR_CFG="${TUNING_ROOT}/cnn_resdil_mask/donor/best_config.json"
CNN_RESDIL_ACCEPTOR_CFG="${TUNING_ROOT}/cnn_resdil_mask/acceptor/best_config.json"
TCN_DONOR_CFG="${TUNING_ROOT}/tcn/donor/best_config.json"
TCN_ACCEPTOR_CFG="${TUNING_ROOT}/tcn/acceptor/best_config.json"
BERT_DONOR_CFG="${TUNING_ROOT}/bert/donor/best_config.json"
BERT_ACCEPTOR_CFG="${TUNING_ROOT}/bert/acceptor/best_config.json"
RESERVOIR_DONOR_CFG="${TUNING_ROOT}/reservoir/donor/best_config.json"
RESERVOIR_ACCEPTOR_CFG="${TUNING_ROOT}/reservoir/acceptor/best_config.json"
DNABERT6_DONOR_CFG="${TUNING_ROOT}/dnabert6/donor/best_config.json"
DNABERT6_ACCEPTOR_CFG="${TUNING_ROOT}/dnabert6/acceptor/best_config.json"
DNABERT2_PAIR_TRUNC_CFG="${TUNING_ROOT}/dnabert2_pair_trunc/pair/best_config.json"
CNN_V2_PAIR_CFG="${TUNING_ROOT}/cnn_v2_pair/pair/best_config.json"

log "Inference: wrappers (Markov excluded)"
log "infer_models=${INFER_MODELS}"

if model_is_selected "cnn"; then
	if all_tuned_configs_exist "${CNN_DONOR_CFG}" "${CNN_ACCEPTOR_CFG}"; then
		run_wrapper_infer "${PROJECT_ROOT}/run/run_cnn_v2.sh" "cnn_v2.sh" \
			"MODEL=cnn_v2" \
			"MASK_MODE=on" \
			"DONOR_TUNED_CONFIG_PATH=${CNN_DONOR_CFG}" \
			"ACCEPTOR_TUNED_CONFIG_PATH=${CNN_ACCEPTOR_CFG}"
	else
		missing_text="$(
			missing_tuned_configs_text "${CNN_DONOR_CFG}" "${CNN_ACCEPTOR_CFG}"
		)"
		action="$(missing_action_for_model "cnn" "${missing_text}" "1")"
		if [[ "${action}" == "train" ]]; then
			run_wrapper_train_fallback "${PROJECT_ROOT}/run/run_cnn_v2.sh" "cnn_v2.sh" \
				"MODEL=cnn_v2" \
				"MASK_MODE=on" \
				"DONOR_TUNED_CONFIG_PATH=${CNN_DONOR_CFG}" \
				"ACCEPTOR_TUNED_CONFIG_PATH=${CNN_ACCEPTOR_CFG}"
		fi
	fi
else
	log "Skip cnn: not selected"
fi

if model_is_selected "cnn_pair"; then
	if all_tuned_configs_exist "${CNN_PAIR_CFG}"; then
		run_wrapper_infer "${PROJECT_ROOT}/run/run_cnn_v2_pair.sh" "cnn_v2_pair.sh" \
			"MODEL=cnn_v2_pair" \
			"MASK_MODE=on" \
			"PAIR_TUNED_CONFIG_PATH=${CNN_PAIR_CFG}"
	else
		missing_text="$(missing_tuned_configs_text "${CNN_PAIR_CFG}")"
		action="$(missing_action_for_model "cnn_pair" "${missing_text}" "1")"
		if [[ "${action}" == "train" ]]; then
			run_wrapper_train_fallback "${PROJECT_ROOT}/run/run_cnn_v2_pair.sh" "cnn_v2_pair.sh" \
				"MODEL=cnn_v2_pair" \
				"MASK_MODE=on" \
				"PAIR_TUNED_CONFIG_PATH=${CNN_PAIR_CFG}"
		fi
	fi
else
	log "Skip cnn_pair: not selected"
fi

if model_is_selected "bilstm_pair"; then
	if all_tuned_configs_exist "${BILSTM_PAIR_CFG}"; then
		run_wrapper_infer "${PROJECT_ROOT}/archive/run/bilstm_pair/run_bilstm_pair.sh" "bilstm_pair.sh" \
			"MASK_MODE=on" \
			"PAIR_TUNED_CONFIG_PATH=${BILSTM_PAIR_CFG}"
	else
		missing_text="$(missing_tuned_configs_text "${BILSTM_PAIR_CFG}")"
		action="$(missing_action_for_model "bilstm_pair" "${missing_text}" "1")"
		if [[ "${action}" == "train" ]]; then
			run_wrapper_train_fallback "${PROJECT_ROOT}/archive/run/bilstm_pair/run_bilstm_pair.sh" "bilstm_pair.sh" \
				"MASK_MODE=on" \
				"PAIR_TUNED_CONFIG_PATH=${BILSTM_PAIR_CFG}"
		fi
	fi
else
	log "Skip bilstm_pair: not selected"
fi

if model_is_selected "cnn_resdil"; then
	if all_tuned_configs_exist "${CNN_RESDIL_DONOR_CFG}" "${CNN_RESDIL_ACCEPTOR_CFG}"; then
		if ! run_wrapper_infer "${PROJECT_ROOT}/archive/run/cnn/run_cnn_resdil.sh" "cnn_resdil.sh" \
			"MODEL=cnn_resdil" \
			"MASK_MODE=on" \
			"DONOR_TUNED_CONFIG_PATH=${CNN_RESDIL_DONOR_CFG}" \
			"ACCEPTOR_TUNED_CONFIG_PATH=${CNN_RESDIL_ACCEPTOR_CFG}"; then
			if [[ "${TRAIN_ON_MISSING}" == "1" ]]; then
				log "cnn_resdil infer failed; retry with train fallback."
				run_wrapper_train_fallback "${PROJECT_ROOT}/archive/run/cnn/run_cnn_resdil.sh" "cnn_resdil.sh" \
					"MODEL=cnn_resdil" \
					"MASK_MODE=on" \
					"DONOR_TUNED_CONFIG_PATH=${CNN_RESDIL_DONOR_CFG}" \
					"ACCEPTOR_TUNED_CONFIG_PATH=${CNN_RESDIL_ACCEPTOR_CFG}"
			else
				die "cnn_resdil inference failed. Set TRAIN_ON_MISSING=1 to allow retraining fallback."
			fi
		fi
	else
		missing_text="$(
			missing_tuned_configs_text \
				"${CNN_RESDIL_DONOR_CFG}" \
				"${CNN_RESDIL_ACCEPTOR_CFG}"
		)"
		action="$(missing_action_for_model "cnn_resdil" "${missing_text}" "1")"
		if [[ "${action}" == "train" ]]; then
			run_wrapper_train_fallback "${PROJECT_ROOT}/archive/run/cnn/run_cnn_resdil.sh" "cnn_resdil.sh" \
				"MODEL=cnn_resdil" \
				"MASK_MODE=on" \
				"DONOR_TUNED_CONFIG_PATH=${CNN_RESDIL_DONOR_CFG}" \
				"ACCEPTOR_TUNED_CONFIG_PATH=${CNN_RESDIL_ACCEPTOR_CFG}"
		fi
	fi
else
	log "Skip cnn_resdil: not selected"
fi

if model_is_selected "tcn"; then
	if all_tuned_configs_exist "${TCN_DONOR_CFG}" "${TCN_ACCEPTOR_CFG}"; then
		run_wrapper_infer "${PROJECT_ROOT}/archive/run/tcn/run_tcn.sh" "tcn.sh" \
			"MODEL=tcn" \
			"MASK_MODE=off" \
			"DONOR_TUNED_CONFIG_PATH=${TCN_DONOR_CFG}" \
			"ACCEPTOR_TUNED_CONFIG_PATH=${TCN_ACCEPTOR_CFG}"
	else
		missing_text="$(missing_tuned_configs_text "${TCN_DONOR_CFG}" "${TCN_ACCEPTOR_CFG}")"
		action="$(missing_action_for_model "tcn" "${missing_text}" "1")"
		if [[ "${action}" == "train" ]]; then
			run_wrapper_train_fallback "${PROJECT_ROOT}/archive/run/tcn/run_tcn.sh" "tcn.sh" \
				"MODEL=tcn" \
				"MASK_MODE=off" \
				"DONOR_TUNED_CONFIG_PATH=${TCN_DONOR_CFG}" \
				"ACCEPTOR_TUNED_CONFIG_PATH=${TCN_ACCEPTOR_CFG}"
		fi
	fi
else
	log "Skip tcn: not selected"
fi

if model_is_selected "bert"; then
	if all_tuned_configs_exist "${BERT_DONOR_CFG}" "${BERT_ACCEPTOR_CFG}"; then
		run_wrapper_infer "${PROJECT_ROOT}/archive/run/bert/run_bert.sh" "bert.sh" \
			"MODEL=bert" \
			"MASK_MODE=off" \
			"DONOR_TUNED_CONFIG_PATH=${BERT_DONOR_CFG}" \
			"ACCEPTOR_TUNED_CONFIG_PATH=${BERT_ACCEPTOR_CFG}"
	else
		missing_text="$(
			missing_tuned_configs_text "${BERT_DONOR_CFG}" "${BERT_ACCEPTOR_CFG}"
		)"
		action="$(missing_action_for_model "bert" "${missing_text}" "1")"
		if [[ "${action}" == "train" ]]; then
			run_wrapper_train_fallback "${PROJECT_ROOT}/archive/run/bert/run_bert.sh" "bert.sh" \
				"MODEL=bert" \
				"MASK_MODE=off" \
				"DONOR_TUNED_CONFIG_PATH=${BERT_DONOR_CFG}" \
				"ACCEPTOR_TUNED_CONFIG_PATH=${BERT_ACCEPTOR_CFG}"
		fi
	fi
else
	log "Skip bert: not selected"
fi

if model_is_selected "reservoir"; then
	if all_tuned_configs_exist "${RESERVOIR_DONOR_CFG}" "${RESERVOIR_ACCEPTOR_CFG}"; then
		run_wrapper_infer "${PROJECT_ROOT}/archive/run/reservoir/run_reservoir.sh" "reservoir.sh" \
			"MODEL=reservoir" \
			"MASK_MODE=off" \
			"DONOR_TUNED_CONFIG_PATH=${RESERVOIR_DONOR_CFG}" \
			"ACCEPTOR_TUNED_CONFIG_PATH=${RESERVOIR_ACCEPTOR_CFG}"
	else
		missing_text="$(
			missing_tuned_configs_text \
				"${RESERVOIR_DONOR_CFG}" \
				"${RESERVOIR_ACCEPTOR_CFG}"
		)"
		action="$(missing_action_for_model "reservoir" "${missing_text}" "1")"
		if [[ "${action}" == "train" ]]; then
			run_wrapper_train_fallback "${PROJECT_ROOT}/archive/run/reservoir/run_reservoir.sh" "reservoir.sh" \
				"MODEL=reservoir" \
				"MASK_MODE=off" \
				"DONOR_TUNED_CONFIG_PATH=${RESERVOIR_DONOR_CFG}" \
				"ACCEPTOR_TUNED_CONFIG_PATH=${RESERVOIR_ACCEPTOR_CFG}"
		fi
	fi
else
	log "Skip reservoir: not selected"
fi

if model_is_selected "dnabert6"; then
	if all_tuned_configs_exist "${DNABERT6_DONOR_CFG}" "${DNABERT6_ACCEPTOR_CFG}"; then
		run_wrapper_infer "${PROJECT_ROOT}/archive/run/dnabert/run_dnabert.sh" "dnabert.sh" \
			"DNABERT_VARIANT=6" \
			"TRUST_REMOTE_CODE=1" \
			"MASK_MODE=off" \
			"PRETRAINED_MODEL_NAME=${SOURCE_MODEL_ROOT}/pretrained/dnabert6" \
			"DONOR_TUNED_CONFIG_PATH=${DNABERT6_DONOR_CFG}" \
			"ACCEPTOR_TUNED_CONFIG_PATH=${DNABERT6_ACCEPTOR_CFG}"
	else
		missing_text="$(
			missing_tuned_configs_text \
				"${DNABERT6_DONOR_CFG}" \
				"${DNABERT6_ACCEPTOR_CFG}"
		)"
		action="$(missing_action_for_model "dnabert6" "${missing_text}" "1")"
		if [[ "${action}" == "train" ]]; then
			run_wrapper_train_fallback "${PROJECT_ROOT}/archive/run/dnabert/run_dnabert.sh" "dnabert.sh" \
				"DNABERT_VARIANT=6" \
				"TRUST_REMOTE_CODE=1" \
				"MASK_MODE=off" \
				"PRETRAINED_MODEL_NAME=${SOURCE_MODEL_ROOT}/pretrained/dnabert6" \
				"DONOR_TUNED_CONFIG_PATH=${DNABERT6_DONOR_CFG}" \
				"ACCEPTOR_TUNED_CONFIG_PATH=${DNABERT6_ACCEPTOR_CFG}"
		fi
	fi
else
	log "Skip dnabert6: not selected"
fi

if model_is_selected "dnabert2_pair"; then
	if all_tuned_configs_exist "${DNABERT2_PAIR_TRUNC_CFG}"; then
		if ! run_wrapper_infer "${PROJECT_ROOT}/archive/run/dnabert/run_dnabert_pair.sh" "dnabert_pair.sh" \
			"DNABERT_VARIANT=2" \
			"TRUST_REMOTE_CODE=1" \
			"TRUNC_MODE=on" \
			"MASK_MODE=on" \
			"PRETRAINED_MODEL_NAME=${SOURCE_MODEL_ROOT}/pretrained/dnabert2-117m-7bce263b15377fc15361f52cfab88f8b586abda0" \
			"PAIR_TUNED_CONFIG_PATH=${DNABERT2_PAIR_TRUNC_CFG}"; then
			if [[ "${TRAIN_ON_MISSING}" == "1" ]]; then
				log "dnabert2_pair infer failed; retry with train fallback."
				run_wrapper_train_fallback "${PROJECT_ROOT}/archive/run/dnabert/run_dnabert_pair.sh" "dnabert_pair.sh" \
					"DNABERT_VARIANT=2" \
					"TRUST_REMOTE_CODE=1" \
					"TRUNC_MODE=on" \
					"MASK_MODE=on" \
					"PRETRAINED_MODEL_NAME=${SOURCE_MODEL_ROOT}/pretrained/dnabert2-117m-7bce263b15377fc15361f52cfab88f8b586abda0" \
					"PAIR_TUNED_CONFIG_PATH=${DNABERT2_PAIR_TRUNC_CFG}"
			else
				die "dnabert2_pair inference failed. Set TRAIN_ON_MISSING=1 to allow retraining fallback."
			fi
		fi
	else
		missing_text="$(missing_tuned_configs_text "${DNABERT2_PAIR_TRUNC_CFG}")"
		action="$(missing_action_for_model "dnabert2_pair_trunc" "${missing_text}" "1")"
		if [[ "${action}" == "train" ]]; then
			run_wrapper_train_fallback "${PROJECT_ROOT}/archive/run/dnabert/run_dnabert_pair.sh" "dnabert_pair.sh" \
				"DNABERT_VARIANT=2" \
				"TRUST_REMOTE_CODE=1" \
				"TRUNC_MODE=on" \
				"MASK_MODE=on" \
				"PRETRAINED_MODEL_NAME=${SOURCE_MODEL_ROOT}/pretrained/dnabert2-117m-7bce263b15377fc15361f52cfab88f8b586abda0" \
				"PAIR_TUNED_CONFIG_PATH=${DNABERT2_PAIR_TRUNC_CFG}"
		fi
	fi
else
	log "Skip dnabert2_pair: not selected"
fi

if model_is_selected "cnn_v2"; then
	if all_tuned_configs_exist "${CNN_V2_PAIR_CFG}"; then
		run_legacy_cnn_v2_infer "${CNN_V2_PAIR_CFG}"
	else
		missing_text="$(missing_tuned_configs_text "${CNN_V2_PAIR_CFG}")"
		action="$(missing_action_for_model "legacy cnn_v2" "${missing_text}" "0")"
		if [[ "${action}" == "train" ]]; then
			die "Internal error: legacy cnn_v2 does not support train fallback."
		fi
	fi
else
	log "Skip cnn_v2: not selected"
fi

log "Done."
log "All outputs are isolated under: ${ISOLATED_SPECIES_DIR}"
log "Key files:"
log "  - ${ISOLATED_RAW_DIR}/transcript_class.txt"
log "  - ${ISOLATED_PROCESSED_DIR}/transcripts.tsv"
log "  - ${ISOLATED_PROCESSED_DIR}/transcripts.unique.tsv"
log "  - ${ISOLATED_PROCESSED_DIR}/transcripts.unique.mask.tsv"
log "  - ${ISOLATED_PROCESSED_DIR}/transcripts.unique.trunc.tsv"
