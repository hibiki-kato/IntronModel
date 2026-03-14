#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOT'
Usage: bash run/make_unique_intron_assets.sh [options]

Options:
  --species <csv>            Species list (default: Dmel,Mmus,Athal,Hsap)
  --data-root <path>         Data root (default: <repo>/data)
  --labeled-name <name>      Source labeled TSV under processed/
                             (default: intron_eval_flank10.tsv)
  --train-pos-path <path>    Optional path template for positive train TSV
                             (supports {species})
  --train-neg-path <path>    Optional path template for negative ERR
                             (supports {species})
  --overwrite                Overwrite existing outputs
  -h, --help                 Show this help
EOT
}

# --------------------------
# USER DEFAULTS (optional edit)
# --------------------------
CONDA_ENV="intronmodel"
SPECIES="Dmel,Mmus,Athal,Hsap"
DATA_ROOT=""
LABELED_NAME="intron_eval_flank10.tsv"
TRAIN_POS_PATH=""
TRAIN_NEG_PATH=""
OVERWRITE="0"

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

while [[ $# -gt 0 ]]; do
	case "$1" in
	--species)
		SPECIES="$2"
		shift 2
		;;
	--data-root)
		DATA_ROOT="$2"
		shift 2
		;;
	--labeled-name)
		LABELED_NAME="$2"
		shift 2
		;;
	--train-pos-path)
		TRAIN_POS_PATH="$2"
		shift 2
		;;
	--train-neg-path)
		TRAIN_NEG_PATH="$2"
		shift 2
		;;
	--overwrite)
		OVERWRITE="1"
		shift
		;;
	-h | --help)
		usage
		exit 0
		;;
	*)
		echo "Unknown argument: $1" >&2
		usage
		exit 1
		;;
	esac
done

if [[ -z "${DATA_ROOT}" ]]; then
	DATA_ROOT="${PROJECT_ROOT}/data"
fi

python_bin="$(intronmodel_resolve_python_bin "make_unique_intron_assets.sh")"

args=(
	"${python_bin}" "${PROJECT_ROOT}/src/tools/build_unique_intron_assets.py"
	--species "${SPECIES}"
	--data-root "${DATA_ROOT}"
	--labeled-name "${LABELED_NAME}"
)

if [[ -n "${TRAIN_POS_PATH}" ]]; then
	args+=(--train-pos-path "${TRAIN_POS_PATH}")
fi
if [[ -n "${TRAIN_NEG_PATH}" ]]; then
	args+=(--train-neg-path "${TRAIN_NEG_PATH}")
fi
if [[ "${OVERWRITE}" == "1" ]]; then
	args+=(--overwrite)
fi

echo "[make_unique_intron_assets] species=${SPECIES} data_root=${DATA_ROOT}"
PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
	"${args[@]}"
