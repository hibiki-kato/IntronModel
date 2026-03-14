#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOT'
Usage: bash run/make_unique_scores_temporary.sh [options]

Options:
  --species <csv>            Species list (default: Dmel,Mmus,Athal,Hsap)
  --data-root <path>         Data root (default: <repo>/data)
  --site-pattern <glob>      Pattern under site_score/ (default: *.tsv)
  --intron-pattern <glob>    Pattern under intron_score/ (default: *.tsv)
  --tolerance <float>        Allowed max diff in one unique group (default: 1e-4)
  --dry-run                  Validate only; do not overwrite files
  -h, --help                 Show this help
EOT
}

CONDA_ENV="intronmodel"
SPECIES="Dmel,Mmus,Athal,Hsap"
DATA_ROOT=""
SITE_PATTERN="*.tsv"
INTRON_PATTERN="*.tsv"
TOLERANCE="1e-4"
DRY_RUN="0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"
intronmodel_activate_conda "${CONDA_ENV}"
intronmodel_init_paths "${BASH_SOURCE[0]}"

# Auto-run inside tmux on SSH so jobs survive disconnects.
# Set INTRONMODEL_AUTO_TMUX=off|on|auto (default: off).
: "${INTRONMODEL_AUTO_TMUX:=off}"
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
	--site-pattern)
		SITE_PATTERN="$2"
		shift 2
		;;
	--intron-pattern)
		INTRON_PATTERN="$2"
		shift 2
		;;
	--tolerance)
		TOLERANCE="$2"
		shift 2
		;;
	--dry-run)
		DRY_RUN="1"
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

python_bin="$(intronmodel_resolve_python_bin "make_unique_scores_temporary.sh")"
args=(
	"${python_bin}" "${PROJECT_ROOT}/src/tools/uniqueify_scores_temporary.py"
	--species "${SPECIES}"
	--data-root "${DATA_ROOT}"
	--site-pattern "${SITE_PATTERN}"
	--intron-pattern "${INTRON_PATTERN}"
	--tolerance "${TOLERANCE}"
	--dry-run "${DRY_RUN}"
)

echo "[make_unique_scores_temporary] species=${SPECIES} data_root=${DATA_ROOT} "\
"site_pattern=${SITE_PATTERN} intron_pattern=${INTRON_PATTERN} "\
"tolerance=${TOLERANCE} dry_run=${DRY_RUN}"
PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
	"${args[@]}"
