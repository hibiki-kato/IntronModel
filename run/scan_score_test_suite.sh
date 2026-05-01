#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
	echo "[scan_score_test_suite.sh] This script is config-only." \
		"Edit the CONFIG block and run without args." >&2
	exit 1
fi

# --------------------------
# CONFIG (edit here)
# --------------------------
CONDA_ENV="${CONDA_ENV:-intronmodel}"
# cnn_v3 is the current runnable default in this workspace.
# cnn_v2 checkpoints here are placeholder text files, so only pin cnn_v2 manually
# if you have restored the real artifacts.
MODEL="${MODEL:-dnabert2}"
SPECIES="${SPECIES:-Dmel}"
TAG="${TAG:-h}"
SUITE_ROOT="${SUITE_ROOT:-}"
STUDENTS_DIR="${STUDENTS_DIR:-}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-512}"
# Leave empty for normal cnn runs; versioned artifacts will resolve the latest
# live checkpoint automatically. Set this only to pin a specific published version.
BEST_CONFIG_PATH="${BEST_CONFIG_PATH:-}"
# Optional DNABERT pair model for score adjustment.
# Set to e.g. "cnn_pair_v2" or "cnn_pair_v3" to enable pair filtering.
# Leave empty to keep pure site-score outputs.
PAIR_MODEL="${PAIR_MODEL:-}"
PAIR_BATCH_SIZE="${PAIR_BATCH_SIZE:-}"
PAIR_CHECKPOINT_PATH="${PAIR_CHECKPOINT_PATH:-}"
SITE_KEEP_THRESHOLD="${SITE_KEEP_THRESHOLD:-0.5}"
# Must stay strictly positive for downstream log transforms.
PAIR_INACTIVE_SCORE="${PAIR_INACTIVE_SCORE:-1e-12}"

# --------------------------
# Runtime implementation
# --------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"
intronmodel_activate_conda "${CONDA_ENV}"
intronmodel_init_paths "${BASH_SOURCE[0]}"

if [[ -z "${SUITE_ROOT}" ]]; then
	SUITE_ROOT="${PROJECT_ROOT}/score_test_suite"
fi
if [[ -z "${STUDENTS_DIR}" ]]; then
	STUDENTS_DIR="${SUITE_ROOT}/Students"
fi
if [[ -z "${TAG}" ]]; then
	echo "[scan_score_test_suite.sh] TAG must be non-empty." >&2
	exit 1
fi

# Allow explicit disable values from inherited shell env.
if [[ "${PAIR_MODEL}" == "none" ]] || [[ "${PAIR_MODEL}" == "off" ]] || [[ "${PAIR_MODEL}" == "0" ]]; then
	PAIR_MODEL=""
fi

args=(
	--data-root "${DATA_ROOT}"
	--species "${SPECIES}"
	--model "${MODEL}"
	--suite-root "${SUITE_ROOT}"
	--students-dir "${STUDENTS_DIR}"
	--tag "${TAG}"
	--device "${DEVICE}"
	--batch-size "${BATCH_SIZE}"
)

if [[ -n "${BEST_CONFIG_PATH}" ]]; then
	args+=(--best-config-path "${BEST_CONFIG_PATH}")
fi

PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
	python3 "${PROJECT_ROOT}/src/tools/scan_splice_candidate_sites.py" \
		"${args[@]}"

if [[ -z "${PAIR_MODEL}" ]]; then
	exit 0
fi

pair_batch_size="${PAIR_BATCH_SIZE}"
if [[ -z "${pair_batch_size}" ]]; then
	pair_batch_size="${BATCH_SIZE}"
fi

for case_dir in "${SUITE_ROOT}"/cds-* "${SUITE_ROOT}"/rna-*; do
	[[ -d "${case_dir}" ]] || continue
	case_name="$(basename "${case_dir}")"
	fasta_path="${case_dir}/${case_name}.fa"
	donor_path="${STUDENTS_DIR}/out.gt.${case_name}.${TAG}.txt"
	acceptor_path="${STUDENTS_DIR}/out.ag.${case_name}.${TAG}.txt"

	pair_args=(
		--fasta "${fasta_path}"
		--donor-input "${donor_path}"
		--acceptor-input "${acceptor_path}"
		--donor-output "${donor_path}"
		--acceptor-output "${acceptor_path}"
		--species "${SPECIES}"
		--model-name "${PAIR_MODEL}"
		--site-score-mode max_pair
		--site-keep-threshold "${SITE_KEEP_THRESHOLD}"
		--inactive-score "${PAIR_INACTIVE_SCORE}"
		--device "${DEVICE}"
		--batch-size "${pair_batch_size}"
		--missing-pair-model-mode error
	)
	if [[ -n "${PAIR_CHECKPOINT_PATH}" ]]; then
		pair_args+=(--pair-checkpoint-path "${PAIR_CHECKPOINT_PATH}")
	fi

	echo "[scan_score_test_suite.sh] pair max_pair case=${case_name} model=${PAIR_MODEL}" >&2
	PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
		python3 "${PROJECT_ROOT}/src/tools/filter_score_test_suite_pairs.py" \
			"${pair_args[@]}"
done
