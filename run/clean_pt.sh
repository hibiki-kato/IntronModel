#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
	echo "[run_clean_pt.sh] This script is config-only. Edit top CONFIG and run without args." >&2
	exit 1
fi

# --------------------------
# CONFIG (edit here)
# --------------------------
CONDA_ENV="intronmodel"
SPECIES=""
MODEL=""
DRY_RUN="0"
INTRONMODEL_AUTO_TMUX="off"

# Optional explicit roots. Leave empty to use project defaults.
DATA_ROOT_OVERRIDE=""
MODEL_ROOT_OVERRIDE=""

# --------------------------
# Runtime implementation
# --------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"
intronmodel_activate_conda "${CONDA_ENV}"
intronmodel_init_paths "${BASH_SOURCE[0]}"

# Auto-run inside tmux on SSH so jobs survive disconnects.
# Set INTRONMODEL_AUTO_TMUX=off|on|auto (default: auto).
intronmodel_enable_auto_tmux "${PROJECT_ROOT}" "$0" "${BASH_SOURCE[0]##*/}"

intronmodel_start_timer "run_clean_pt.sh"
trap 'intronmodel_print_timing' EXIT

if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
	echo "[run_clean_pt.sh] DRY_RUN must be 0 or 1." >&2
	exit 2
fi

python_bin="$(intronmodel_resolve_python_bin "run_clean_pt.sh")"
data_root="${DATA_ROOT}"
model_root="${MODEL_ROOT}"
if [[ -n "${DATA_ROOT_OVERRIDE}" ]]; then
	data_root="${DATA_ROOT_OVERRIDE}"
fi
if [[ -n "${MODEL_ROOT_OVERRIDE}" ]]; then
	model_root="${MODEL_ROOT_OVERRIDE}"
fi

cmd=(
	"${python_bin}" "${PROJECT_ROOT}/src/tools/prune_missing_rank_checkpoints.py"
	--data_root "${data_root}"
	--model_root "${model_root}"
	--dry_run "${DRY_RUN}"
)
if [[ -n "${SPECIES}" ]]; then
	cmd+=(--species "${SPECIES}")
fi
if [[ -n "${MODEL}" ]]; then
	cmd+=(--model "${MODEL}")
fi

echo "[run_clean_pt.sh] data_root=${data_root} model_root=${model_root}"
if [[ -n "${SPECIES}" ]]; then
	echo "[run_clean_pt.sh] species=${SPECIES}"
fi
if [[ -n "${MODEL}" ]]; then
	echo "[run_clean_pt.sh] model=${MODEL}"
fi
echo "[run_clean_pt.sh] dry_run=${DRY_RUN}"

PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" "${cmd[@]}"
