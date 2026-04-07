#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOT'
Usage: bash run/versioning_rollout.sh [options]

Options:
  --species <csv>              Optional species whitelist (example: Dmel,Hsap)
  --model <csv>                Optional model whitelist (example: cnn_v2,cnn_pair_v3)
  --apply-seed                 Seed missing .01 publications when possible
  --allow-seed-with-history    Call seed even when history already exists
  --repair-paths               Rewrite publication metadata paths repo-relatively
  --data-root <path>           Data root (default: <repo>/data)
  --project-root <path>        Project root (default: auto-detected)
  -h, --help                   Show this help
EOT
}

CONDA_ENV="intronmodel"
SPECIES=""
MODEL=""
APPLY_SEED="0"
ALLOW_SEED_WITH_HISTORY="0"
REPAIR_PATHS="0"
DATA_ROOT=""
PROJECT_ROOT_OVERRIDE=""

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
	--model)
		MODEL="$2"
		shift 2
		;;
	--apply-seed)
		APPLY_SEED="1"
		shift
		;;
	--allow-seed-with-history)
		ALLOW_SEED_WITH_HISTORY="1"
		shift
		;;
	--repair-paths)
		REPAIR_PATHS="1"
		shift
		;;
	--data-root)
		DATA_ROOT="$2"
		shift 2
		;;
	--project-root)
		PROJECT_ROOT_OVERRIDE="$2"
		shift 2
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

if [[ -n "${PROJECT_ROOT_OVERRIDE}" ]]; then
	PROJECT_ROOT="${PROJECT_ROOT_OVERRIDE}"
fi
if [[ -z "${DATA_ROOT}" ]]; then
	DATA_ROOT="${PROJECT_ROOT}/data"
fi

python_bin="$(intronmodel_resolve_python_bin "versioning_rollout.sh")"

echo "[versioning_rollout] data_root=${DATA_ROOT} species=${SPECIES:-all} model=${MODEL:-all} apply_seed=${APPLY_SEED} repair_paths=${REPAIR_PATHS}"
PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
	"${python_bin}" "${PROJECT_ROOT}/src/tools/versioning_rollout.py" \
	--project_root "${PROJECT_ROOT}" \
	--data_root "${DATA_ROOT}" \
	--species "${SPECIES}" \
	--model "${MODEL}" \
	--apply_seed "${APPLY_SEED}" \
	--allow_seed_with_history "${ALLOW_SEED_WITH_HISTORY}" \
	--repair_paths "${REPAIR_PATHS}"
