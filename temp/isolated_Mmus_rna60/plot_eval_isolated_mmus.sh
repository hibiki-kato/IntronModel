#!/usr/bin/env bash
set -euo pipefail

# --------------------------
# CONFIG (edit directly)
# --------------------------
CONDA_ENV="intronmodel"
USE_CONDA_ACTIVATE="1"
SPECIES="Mmus"
INTERACTIVE="0"
OUTPUT_PNG=""
X_MIN=""
X_MAX=""
Y_MIN=""
Y_MAX=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/run/lib/common.sh"
if [[ "${USE_CONDA_ACTIVATE}" == "1" ]]; then
	intronmodel_activate_conda "${CONDA_ENV}"
fi

if [[ -z "${MPLCONFIGDIR:-}" ]]; then
	MPLCONFIGDIR="${TMPDIR:-/tmp}/intronmodel-mpl-cache"
	mkdir -p "${MPLCONFIGDIR}"
	export MPLCONFIGDIR
fi
if [[ -z "${XDG_CACHE_HOME:-}" ]]; then
	XDG_CACHE_HOME="${TMPDIR:-/tmp}/intronmodel-cache"
	mkdir -p "${XDG_CACHE_HOME}"
	export XDG_CACHE_HOME
fi

EVAL_DIR="${SCRIPT_DIR}/data/${SPECIES}/eval_score"
if [[ ! -d "${EVAL_DIR}" ]]; then
	echo "[plot_eval_isolated_mmus] eval_score directory not found: ${EVAL_DIR}" >&2
	exit 1
fi

if [[ -z "${OUTPUT_PNG}" ]]; then
	OUTPUT_PNG="${SCRIPT_DIR}/data/${SPECIES}/${SPECIES}_snpr.png"
fi

set_defaults_for_species() {
	local sp="$1"
	case "${sp}" in
	Athal)
		DEFAULT_X_MIN="10.0"
		DEFAULT_X_MAX="52.0"
		DEFAULT_Y_MIN="48.0"
		DEFAULT_Y_MAX="75.0"
		;;
	Dmel)
		DEFAULT_X_MIN="40.0"
		DEFAULT_X_MAX="52.0"
		DEFAULT_Y_MIN="39.0"
		DEFAULT_Y_MAX="55.0"
		;;
	Mmus)
		DEFAULT_X_MIN="40.0"
		DEFAULT_X_MAX="50.0"
		DEFAULT_Y_MIN="12.0"
		DEFAULT_Y_MAX="15.0"
		;;
	Hsap)
		DEFAULT_X_MIN="10.0"
		DEFAULT_X_MAX="19.0"
		DEFAULT_Y_MIN="26.0"
		DEFAULT_Y_MAX="38.0"
		;;
	*)
		DEFAULT_X_MIN="40.0"
		DEFAULT_X_MAX="50.0"
		DEFAULT_Y_MIN="40.0"
		DEFAULT_Y_MAX="50.0"
		;;
	esac
}

set_defaults_for_species "${SPECIES}"
X_MIN_FINAL="${X_MIN:-${DEFAULT_X_MIN}}"
X_MAX_FINAL="${X_MAX:-${DEFAULT_X_MAX}}"
Y_MIN_FINAL="${Y_MIN:-${DEFAULT_Y_MIN}}"
Y_MAX_FINAL="${Y_MAX:-${DEFAULT_Y_MAX}}"

if [[ "${INTERACTIVE}" != "0" && "${INTERACTIVE}" != "1" ]]; then
	echo "[plot_eval_isolated_mmus] INTERACTIVE must be 0 or 1" >&2
	exit 1
fi

PYTHON_BIN="$(intronmodel_resolve_python_bin "plot_eval_isolated_mmus.sh")"

echo "[plot_eval_isolated_mmus] species=${SPECIES}"
echo "[plot_eval_isolated_mmus] eval_dir=${EVAL_DIR}"
echo "[plot_eval_isolated_mmus] output_png=${OUTPUT_PNG}"
echo "[plot_eval_isolated_mmus] bounds x=(${X_MIN_FINAL}, ${X_MAX_FINAL}) y=(${Y_MIN_FINAL}, ${Y_MAX_FINAL})"

"${PYTHON_BIN}" - "${PROJECT_ROOT}" "${SPECIES}" "${EVAL_DIR}" "${OUTPUT_PNG}" \
	"${INTERACTIVE}" "${X_MIN_FINAL}" "${X_MAX_FINAL}" "${Y_MIN_FINAL}" "${Y_MAX_FINAL}" <<'PY'
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

project_root = Path(sys.argv[1])
species = sys.argv[2]
eval_dir = Path(sys.argv[3])
output_png = sys.argv[4]
interactive = sys.argv[5] == "1"
x_min = float(sys.argv[6])
x_max = float(sys.argv[7])
y_min = float(sys.argv[8])
y_max = float(sys.argv[9])

module_path = project_root / "src" / "evaluate_scores.py"
spec = importlib.util.spec_from_file_location("evaluate_scores", module_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"failed to load module: {module_path}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Reuse the existing plotting implementation, but lock eval dir to isolated root.
def _resolve_eval_dir_override(_: str) -> str:
    return str(eval_dir)

module.resolve_eval_dir = _resolve_eval_dir_override
module.plot_eval_scores(
    species=species,
    output_png=output_png,
    interactive=interactive,
    x_min=x_min,
    x_max=x_max,
    y_min=y_min,
    y_max=y_max,
)
PY
