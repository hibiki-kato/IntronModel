#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
	echo "[split_cnn_v2_best.sh] This script is config-only." \
		"Edit top CONFIG and run without args." >&2
	exit 1
fi

# --------------------------
# CONFIG (edit here)
# --------------------------
set -a
SPECIES="${SPECIES:-Athal,Dmel,Hsap,Mmus}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN="${DRY_RUN:-0}"
set +a

# --------------------------
# Runtime implementation
# --------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"
intronmodel_init_paths "${BASH_SOURCE[0]}"

args=(
	--species "${SPECIES}"
	--project-root "${PROJECT_ROOT}"
)
if [[ "${DRY_RUN}" == "1" ]]; then
	args+=(--dry-run)
fi

PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
	"${PYTHON_BIN}" \
	"${PROJECT_ROOT}/src/tools/split_cnn_v2_best.py" \
	"${args[@]}"
