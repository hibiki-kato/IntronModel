#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
	echo "[test_legacy_cnn_vs_v2.sh] This script is config-only." \
		"Edit top CONFIG and run without args." >&2
	exit 1
fi

# --------------------------
# CONFIG (edit here)
# --------------------------
set -a
SPECIES="${SPECIES:-Athal,Dmel,Hsap,Mmus}"
VARIANTS="${VARIANTS:-cnn_pair,cnn_pair_mask}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-1337}"
PROMOTE_EPSILON="${PROMOTE_EPSILON:-0.0}"
PROMOTE_IF_BETTER="${PROMOTE_IF_BETTER:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN="${DRY_RUN:-0}"
MAX_RUNS="${MAX_RUNS:-0}"
set +a

# --------------------------
# Runtime implementation
# --------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"
intronmodel_activate_conda "intronmodel"
intronmodel_init_paths "${BASH_SOURCE[0]}"

args=(
	--species "${SPECIES}"
	--variants "${VARIANTS}"
	--device "${DEVICE}"
	--seed "${SEED}"
	--promote-epsilon "${PROMOTE_EPSILON}"
	--python-bin "${PYTHON_BIN}"
)

if [[ "${MAX_RUNS}" != "0" ]]; then
	args+=(--max-runs "${MAX_RUNS}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
	args+=(--dry-run)
fi
if [[ "${PROMOTE_IF_BETTER}" == "1" ]]; then
	args+=(--promote-if-better)
fi

PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
	"${PYTHON_BIN}" \
	"${PROJECT_ROOT}/src/tools/merge_cnn_legacy_into_v2.py" \
	"${args[@]}"
