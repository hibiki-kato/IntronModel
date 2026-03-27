#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOT'
Usage: bash run/make_random_intron_and_trans_scores.sh [options]

Options:
  --species <csv>         Species list (default: Dmel,Mmus,Athal,Hsap)
  --data-root <path>      Data root (default: <repo>/data)
  --output-stem <name>    Output stem under intron_score/ and trans_score/
                          (default: random)
  --seed <int>            Base random seed (default: 20260327)
  -h, --help              Show this help
EOT
}

# --------------------------
# USER DEFAULTS (optional edit)
# --------------------------
CONDA_ENV="intronmodel"
SPECIES="Dmel,Mmus,Athal,Hsap"
DATA_ROOT=""
OUTPUT_STEM="random"
SEED="20260327"

# --------------------------
# Runtime implementation
# --------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"
intronmodel_activate_conda "${CONDA_ENV}"
intronmodel_init_paths "${BASH_SOURCE[0]}"

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
	--output-stem)
		OUTPUT_STEM="$2"
		shift 2
		;;
	--seed)
		SEED="$2"
		shift 2
		;;
	-h | --help)
		usage
		exit 0
		;;
	*)
		echo "Unknown argument: $1" >&2
		usage >&2
		exit 1
		;;
	esac
done

if [[ -z "${DATA_ROOT}" ]]; then
	DATA_ROOT="${PROJECT_ROOT}/data"
fi

echo "[make_random_intron_and_trans_scores] species=${SPECIES}"
echo "[make_random_intron_and_trans_scores] data_root=${DATA_ROOT}"
echo "[make_random_intron_and_trans_scores] output_stem=${OUTPUT_STEM}"
echo "[make_random_intron_and_trans_scores] seed=${SEED}"

python_bin="$(intronmodel_resolve_python_bin "make_random_intron_and_trans_scores.sh")"
PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
	"${python_bin}" "${PROJECT_ROOT}/src/tools/build_random_intron_and_trans_scores.py" \
	--data-root "${DATA_ROOT}" \
	--species "${SPECIES}" \
	--output-stem "${OUTPUT_STEM}" \
	--seed "${SEED}"
