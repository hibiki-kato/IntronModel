#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
	echo "[scan_score_test_suite_integrated.sh] This script is config-only." \
		"Edit the CONFIG block and run without args." >&2
	exit 1
fi

# --------------------------
# CONFIG (edit here)
# --------------------------
MODEL="${MODEL:-dnabert2}"
PAIR_MODEL="${PAIR_MODEL:-dnabert2_pair}"

CONDA_ENV="${CONDA_ENV:-intronmodel}"
SUITE_ROOT="${SUITE_ROOT:-}"
SPECIES="${SPECIES:-Dmel}"
TAG="${TAG:-h}"
STUDENTS_DIR="${STUDENTS_DIR:-}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-512}"
BEST_CONFIG_PATH="${BEST_CONFIG_PATH:-}"
PAIR_BATCH_SIZE="${PAIR_BATCH_SIZE:-}"
PAIR_CHECKPOINT_PATH="${PAIR_CHECKPOINT_PATH:-}"
SITE_KEEP_THRESHOLD="${SITE_KEEP_THRESHOLD:-0.01}"
PAIR_INACTIVE_SCORE="${PAIR_INACTIVE_SCORE:-1e-12}"

# Final evaluation
RUN_EVAL="${RUN_EVAL:-1}"

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

echo "[scan_score_test_suite_integrated.sh] Stage1 scoring via scan_score_test_suite.sh" >&2
SUITE_ROOT="${SUITE_ROOT}" \
SPECIES="${SPECIES}" \
TAG="${TAG}" \
STUDENTS_DIR="${STUDENTS_DIR}" \
DEVICE="${DEVICE}" \
BATCH_SIZE="${BATCH_SIZE}" \
MODEL="${MODEL}" \
BEST_CONFIG_PATH="${BEST_CONFIG_PATH}" \
PAIR_MODEL="${PAIR_MODEL}" \
PAIR_BATCH_SIZE="${PAIR_BATCH_SIZE}" \
PAIR_CHECKPOINT_PATH="${PAIR_CHECKPOINT_PATH}" \
SITE_KEEP_THRESHOLD="${SITE_KEEP_THRESHOLD}" \
PAIR_INACTIVE_SCORE="${PAIR_INACTIVE_SCORE}" \
	/usr/bin/env bash "${PROJECT_ROOT}/run/scan_score_test_suite.sh"

if [[ "${RUN_EVAL}" == "1" ]]; then
	echo "[scan_score_test_suite_integrated.sh] Stage4 evaluation" >&2
	/usr/bin/env bash "${SUITE_ROOT}/run_test_llms.h.sh"
fi
