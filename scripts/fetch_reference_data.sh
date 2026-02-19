#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOT'
Usage: bash scripts/fetch_reference_data.sh [options]

Copy or symlink external reference files into data/<species>/raw.

Options:
  --species <name>      Athal|Dmel|Mmus|all (default: all)
  --source-root <path>  External root containing <species>/raw files (required)
  --target-root <path>  Repository data root (default: ./data)
  --mode <name>         copy|symlink (default: copy)
  -h, --help            Show this help

Expected source layout:
  <source-root>/<species>/raw/<reference files>
EOT
}

SPECIES="all"
SOURCE_ROOT=""
TARGET_ROOT="data"
MODE="copy"

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
	--target-root)
		TARGET_ROOT="$2"
		shift 2
		;;
	--mode)
		MODE="$2"
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

if [[ -z "${SOURCE_ROOT}" ]]; then
	echo "--source-root is required." >&2
	exit 2
fi
if [[ "${MODE}" != "copy" && "${MODE}" != "symlink" ]]; then
	echo "--mode must be copy or symlink." >&2
	exit 3
fi
if [[ ! -d "${SOURCE_ROOT}" ]]; then
	echo "Source root not found: ${SOURCE_ROOT}" >&2
	exit 4
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ "${TARGET_ROOT}" != /* ]]; then
	TARGET_ROOT="${PROJECT_ROOT}/${TARGET_ROOT}"
fi

copy_or_link() {
	local src="$1"
	local dst="$2"
	if [[ "${MODE}" == "symlink" ]]; then
		ln -sfn "${src}" "${dst}"
	else
		cp -f "${src}" "${dst}"
	fi
}

copy_species() {
	local species="$1"
	local src_raw="${SOURCE_ROOT}/${species}/raw"
	local dst_raw="${TARGET_ROOT}/${species}/raw"
	if [[ ! -d "${src_raw}" ]]; then
		echo "Missing source raw directory: ${src_raw}" >&2
		return 1
	fi

	mkdir -p "${dst_raw}"
	shopt -s nullglob
	local files=("${src_raw}"/*.fna "${src_raw}"/*.gtf "${src_raw}"/*.gff \
		"${src_raw}"/*.gff3 "${src_raw}"/*.gff.*)
	shopt -u nullglob

	if [[ ${#files[@]} -eq 0 ]]; then
		echo "No reference files found under: ${src_raw}" >&2
		return 1
	fi

	for src_path in "${files[@]}"; do
		local filename
		filename="$(basename "${src_path}")"
		copy_or_link "${src_path}" "${dst_raw}/${filename}"
	done

	echo "[fetch_reference_data] species=${species} mode=${MODE} files=${#files[@]}"
}

if [[ "${SPECIES}" == "all" ]]; then
	copy_species "Athal"
	copy_species "Dmel"
	copy_species "Mmus"
else
	case "${SPECIES}" in
	Athal | Dmel | Mmus)
		copy_species "${SPECIES}"
		;;
	*)
		echo "Invalid --species value: ${SPECIES}" >&2
		exit 5
		;;
	esac
fi
