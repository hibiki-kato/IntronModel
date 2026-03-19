#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOT'
Usage: bash run/share_eval_to_chirag.sh [options]

Two-way sync between data/<species>/eval_score and plotting workspace.
For conflicting files, the newer mtime side is kept.
Files that exist only on one side are copied to the other side.
No files are deleted.

Options:
  --species <name>      Athal|Dmel|Hsap|Mmus|all (default: all)
  --source-root <path>  Local data root (default: ./data)
  --dest-root <path>    Plotting root base path
                        (default: Google Drive Genomics_Plotting path)
  -h, --help            Show this help

Examples:
  bash run/share_eval_to_chirag.sh
  bash run/share_eval_to_chirag.sh --species Dmel
EOT
}

SPECIES="all"
SOURCE_ROOT="data"
DEST_ROOT="/Users/hibiki/Library/CloudStorage/GoogleDrive-hibiki@umd.edu/.shortcut-targets-by-id/19OxNEr9a8VSRk55Q1BUL5UwPb873WbYe/Genomics_Plotting"

while [[ $# -gt 0 ]]; do
	case "$1" in
	--species)
		SPECIES="$2"
		shift 2
		;;
	--source-root)
		SOURCE_ROOT="$2"
		shift 2
		;;
	--dest-root)
		DEST_ROOT="$2"
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ "${SOURCE_ROOT}" != /* ]]; then
	SOURCE_ROOT="${PROJECT_ROOT}/${SOURCE_ROOT}"
fi

sync_one_species() {
	local species="$1"
	local local_dir="${SOURCE_ROOT}/${species}/eval_score"
	local drive_dir="${DEST_ROOT}/${species}"

	mkdir -p "${local_dir}"
	mkdir -p "${drive_dir}"

	# Pass 1: local -> drive (skip when drive file is newer).
	rsync -au --exclude '.DS_Store' "${local_dir}/" "${drive_dir}/"
	# Pass 2: drive -> local (skip when local file is newer).
	rsync -au --exclude '.DS_Store' "${drive_dir}/" "${local_dir}/"

	echo "[sync_eval_score_to_plotting] species=${species}"
	echo "[sync_eval_score_to_plotting] local=${local_dir}"
	echo "[sync_eval_score_to_plotting] drive=${drive_dir}"
}

if [[ "${SPECIES}" == "all" ]]; then
	sync_one_species "Athal"
	sync_one_species "Dmel"
	sync_one_species "Hsap"
	sync_one_species "Mmus"
else
	case "${SPECIES}" in
	Athal | Dmel | Hsap | Mmus)
		sync_one_species "${SPECIES}"
		;;
	*)
		echo "Invalid --species value: ${SPECIES}" >&2
		exit 2
		;;
	esac
fi

echo "[sync_eval_score_to_plotting] completed"
