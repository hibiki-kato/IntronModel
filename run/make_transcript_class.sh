#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOT'
Usage: bash run/make_transcript_class.sh [options]

Options:
  --species <csv>              Species list (default: Dmel,Mmus,Athal,Hsap)
  --data-root <path>           Data root (default: <repo>/data)
  --gtf <path>                 Override query GTF path
  --reference-annotation <path>
                               Override reference annotation path
  --out-name <filename>        Output filename under data/<species>/raw
                               (default: transcript_class.txt)
  --keep-temp                  Keep gffcompare temporary files
  -h, --help                   Show this help

Notes:
  - Query GTF priority: <fasta>.gtf, then *.fna.gtf, then *.gtf.
  - Reference annotation priority: *.fix.gff, *.gff.fix, *.gff, *.gff3.
  - Requires gffcompare to be available on PATH.
EOT
}

# --------------------------
# USER DEFAULTS (optional edit)
# --------------------------
CONDA_ENV="intronmodel"
SPECIES="Dmel,Mmus,Athal,Hsap"
DATA_ROOT=""
QUERY_GTF_PATH=""
REFERENCE_ANNOTATION_PATH=""
OUT_NAME="transcript_class.txt"
KEEP_TEMP="0"

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
	--gtf)
		QUERY_GTF_PATH="$2"
		shift 2
		;;
	--reference-annotation)
		REFERENCE_ANNOTATION_PATH="$2"
		shift 2
		;;
	--out-name)
		OUT_NAME="$2"
		shift 2
		;;
	--keep-temp)
		KEEP_TEMP="1"
		shift
		;;
	-h | --help)
		usage
		exit 0
		;;
	*)
		echo "Unknown option: $1" >&2
		usage >&2
		exit 1
		;;
	esac
done

if [[ -z "${DATA_ROOT}" ]]; then
	DATA_ROOT="${PROJECT_ROOT}/data"
fi

if ! command -v gffcompare &>/dev/null; then
	echo "[make_transcript_class.sh] gffcompare not found on PATH." >&2
	exit 2
fi

resolve_query_gtf() {
	local raw_dir="$1"
	local fasta="$2"
	local direct_gtf="${fasta}.gtf"
	local candidates_fna_gtf=()
	local candidates_gtf=()

	if [[ -f "${direct_gtf}" ]]; then
		echo "${direct_gtf}"
		return 0
	fi

	shopt -s nullglob
	candidates_fna_gtf=("${raw_dir}"/*.fna.gtf)
	candidates_gtf=("${raw_dir}"/*.gtf)
	shopt -u nullglob

	if [[ ${#candidates_fna_gtf[@]} -gt 0 ]]; then
		echo "${candidates_fna_gtf[0]}"
		return 0
	fi
	if [[ ${#candidates_gtf[@]} -gt 0 ]]; then
		echo "${candidates_gtf[0]}"
		return 0
	fi
	return 1
}

resolve_fasta() {
	local raw_dir="$1"
	local clean_fna=()
	local fna=()

	shopt -s nullglob
	clean_fna=("${raw_dir}"/*.clean.fna)
	fna=("${raw_dir}"/*.fna)
	shopt -u nullglob

	if [[ ${#clean_fna[@]} -gt 0 ]]; then
		echo "${clean_fna[0]}"
		return 0
	fi
	if [[ ${#fna[@]} -gt 0 ]]; then
		echo "${fna[0]}"
		return 0
	fi
	return 1
}

resolve_reference_annotation() {
	local raw_dir="$1"
	local preferred=()
	local gff=()
	local gff3=()

	shopt -s nullglob
	preferred=("${raw_dir}"/*.fix.gff "${raw_dir}"/*.gff.fix)
	gff=("${raw_dir}"/*.gff)
	gff3=("${raw_dir}"/*.gff3)
	shopt -u nullglob

	if [[ ${#preferred[@]} -gt 0 ]]; then
		for candidate in "${preferred[@]}"; do
			if [[ -f "${candidate}" ]]; then
				echo "${candidate}"
				return 0
			fi
		done
	fi
	if [[ ${#gff[@]} -gt 0 ]]; then
		for candidate in "${gff[@]}"; do
			if [[ -f "${candidate}" ]]; then
				echo "${candidate}"
				return 0
			fi
		done
	fi
	if [[ ${#gff3[@]} -gt 0 ]]; then
		for candidate in "${gff3[@]}"; do
			if [[ -f "${candidate}" ]]; then
				echo "${candidate}"
				return 0
			fi
		done
	fi
	return 1
}

python_bin="$(intronmodel_resolve_python_bin "make_transcript_class")"

IFS=',' read -r -a species_tokens <<< "${SPECIES}"
for raw_species in "${species_tokens[@]}"; do
	token="$(printf '%s' "${raw_species}" | tr -d '[:space:]')"
	if [[ -z "${token}" ]]; then
		continue
	fi

	species="$(intronmodel_resolve_species_case \
		"${token}" "${DATA_ROOT}" "make_transcript_class")"
	raw_dir="${DATA_ROOT}/${species}/raw"
	if [[ ! -d "${raw_dir}" ]]; then
		echo "Raw directory not found: ${raw_dir}" >&2
		exit 3
	fi

	query_gtf="${QUERY_GTF_PATH}"
	ref_annotation="${REFERENCE_ANNOTATION_PATH}"

	if [[ -z "${query_gtf}" ]]; then
		fasta="$(resolve_fasta "${raw_dir}" || true)"
		query_gtf="$(resolve_query_gtf "${raw_dir}" "${fasta}" || true)"
	fi
	if [[ -z "${ref_annotation}" ]]; then
		ref_annotation="$(resolve_reference_annotation "${raw_dir}" || true)"
	fi

	if [[ -z "${query_gtf}" || ! -f "${query_gtf}" ]]; then
		echo "Query GTF not found for species=${species}" >&2
		exit 4
	fi
	if [[ -z "${ref_annotation}" || ! -f "${ref_annotation}" ]]; then
		echo "Reference annotation not found for species=${species}" >&2
		exit 5
	fi

	out_path="${raw_dir}/${OUT_NAME}"
	tmp_prefix="${raw_dir}/gffcompare_tmp_$$"
	tracking_path="${tmp_prefix}.tracking"

	echo "[make_transcript_class.sh] species=${species}"
	echo "[make_transcript_class.sh] query_gtf=${query_gtf}"
	echo "[make_transcript_class.sh] reference_annotation=${ref_annotation}"
	echo "[make_transcript_class.sh] out=${out_path}"
	echo "[make_transcript_class.sh] running gffcompare..."

	gffcompare \
		-r "${ref_annotation}" \
		"${query_gtf}" \
		-o "${tmp_prefix}"

	if [[ ! -f "${tracking_path}" ]]; then
		echo "gffcompare tracking not found: ${tracking_path}" >&2
		exit 6
	fi

	"${python_bin}" \
		"${PROJECT_ROOT}/src/util/make_transcript_class_from_tmap.py" \
		--tracking "${tracking_path}" \
		--out "${out_path}"

	if [[ "${KEEP_TEMP}" == "0" ]]; then
		rm -f "${tmp_prefix}".* "${tmp_prefix}"
		echo "[make_transcript_class.sh] cleaned up gffcompare temp files"
	fi
done
