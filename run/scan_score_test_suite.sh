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
MODEL="${MODEL:-cnn_v2}"
SPECIES="${SPECIES:-Dmel}"
TAG="${TAG:-h}"
SUITE_ROOT="${SUITE_ROOT:-}"
STUDENTS_DIR="${STUDENTS_DIR:-}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-512}"
BEST_CONFIG_PATH="${BEST_CONFIG_PATH:-}"

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
