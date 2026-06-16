#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOT'
Usage: bash run/make_jim_site_scores.sh [options]

Options:
  --species <name[,name...]>  Target species under data/ (default: all)
  --pattern <glob>            Input glob under site_score/ (default: *.tsv)
  --dry-run                   Print planned outputs without writing files
  --conda-env <name>          Conda environment name (default: intronmodel)
  -h, --help                  Show this help
EOT
}

CONDA_ENV="intronmodel"
SPECIES=""
PATTERN="*.tsv"
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
	case "$1" in
	--species)
		SPECIES="$2"
		shift 2
		;;
	--pattern)
		PATTERN="$2"
		shift 2
		;;
	--dry-run)
		DRY_RUN="1"
		shift
		;;
	--conda-env)
		CONDA_ENV="$2"
		shift 2
		;;
	-h | --help)
		usage
		exit 0
		;;
	*)
		echo "Unknown argument: $1" >&2
		usage >&2
		exit 2
		;;
	esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"
intronmodel_activate_conda "${CONDA_ENV}"
intronmodel_init_paths "${BASH_SOURCE[0]}"

RUN_ARGS=(
	"${PROJECT_ROOT}/src/tools/export_site_scores_for_jim.py"
	--data-root "${DATA_ROOT}"
	--pattern "${PATTERN}"
	--dry-run "${DRY_RUN}"
)

if [[ -n "${SPECIES}" ]]; then
	RUN_ARGS+=(--species "${SPECIES}")
fi

echo "[make_jim_site_scores] data_root=${DATA_ROOT} species=${SPECIES:-all} pattern=${PATTERN} dry_run=${DRY_RUN}"
python3 "${RUN_ARGS[@]}"

