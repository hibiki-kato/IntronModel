#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOT'
Usage: bash run/make_labeled_intron_eval_data.sh [options]

Options:
  --species <csv>              Species list (default: Dmel,Mmus,Athal,Hsap)
  --data-root <path>           Data root (default: <repo>/data)
  --fasta <path>               Override FASTA path
  --query-gtf <path>           Override query GTF path
  --reference-annotation <path>
                               Override reference annotation path
  --out-name <filename>        Output TSV name under data/<species>/processed
                               (default: intron_eval_flank10.tsv)
  --donor-len <int>            Donor window length (default: 100)
  --acceptor-len <int>         Acceptor window length (default: 100)
  --flank-bp <int>             Flank bp for intron sequence (default: 10)
  --limit <int>                Max rows per species (default: 0; no limit)
  -h, --help                   Show this help

Notes:
  - Query GTF priority: <fasta>.gtf, then *.fna.gtf, then *.gtf.
  - Reference annotation priority: *.fix.gff, *.gff.fix, *.gff, *.gff3.
EOT
}

# --------------------------
# USER DEFAULTS (optional edit)
# --------------------------
CONDA_ENV="intronmodel"
SPECIES="Dmel,Mmus,Athal,Hsap"
DATA_ROOT=""
FASTA_PATH=""
QUERY_GTF_PATH=""
REFERENCE_ANNOTATION_PATH=""
OUT_NAME="intron_eval_flank10.tsv"
DONOR_LEN="100"
ACCEPTOR_LEN="100"
FLANK_BP="10"
LIMIT="0"

# --------------------------
# Runtime implementation
# --------------------------
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
	--fasta)
		FASTA_PATH="$2"
		shift 2
		;;
	--query-gtf)
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
	--donor-len)
		DONOR_LEN="$2"
		shift 2
		;;
	--acceptor-len)
		ACCEPTOR_LEN="$2"
		shift 2
		;;
	--flank-bp)
		FLANK_BP="$2"
		shift 2
		;;
	--limit)
		LIMIT="$2"
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

if [[ -z "${DATA_ROOT}" ]]; then
	DATA_ROOT="${PROJECT_ROOT}/data"
fi

resolve_fasta() {
	local raw_dir="$1"
	local candidates_clean=()
	local candidates_fna=()

	shopt -s nullglob
	candidates_clean=("${raw_dir}"/*.clean.fna)
	candidates_fna=("${raw_dir}"/*.fna)
	shopt -u nullglob

	if [[ ${#candidates_clean[@]} -gt 0 ]]; then
		echo "${candidates_clean[0]}"
		return 0
	fi
	if [[ ${#candidates_fna[@]} -gt 0 ]]; then
		echo "${candidates_fna[0]}"
		return 0
	fi
	return 1
}

resolve_query_gtf() {
	local raw_dir="$1"
	local fasta_path="$2"
	local direct_gtf="${fasta_path}.gtf"
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

python_bin="$(intronmodel_resolve_python_bin "make_labeled_intron_eval_data")"

IFS=',' read -r -a species_tokens <<< "${SPECIES}"
for raw_species in "${species_tokens[@]}"; do
	token="$(printf '%s' "${raw_species}" | tr -d '[:space:]')"
	if [[ -z "${token}" ]]; then
		continue
	fi

	species="$(intronmodel_resolve_species_case \
		"${token}" "${DATA_ROOT}" "make_labeled_intron_eval_data")"
	raw_dir="${DATA_ROOT}/${species}/raw"
	processed_dir="${DATA_ROOT}/${species}/processed"
	if [[ ! -d "${raw_dir}" ]]; then
		echo "Raw directory not found: ${raw_dir}" >&2
		exit 2
	fi
	mkdir -p "${processed_dir}"

	fasta="${FASTA_PATH}"
	query_gtf="${QUERY_GTF_PATH}"
	ref_annotation="${REFERENCE_ANNOTATION_PATH}"
	out_tsv="${processed_dir}/${OUT_NAME}"

	if [[ -z "${fasta}" ]]; then
		fasta="$(resolve_fasta "${raw_dir}" || true)"
	fi
	if [[ -z "${query_gtf}" ]]; then
		query_gtf="$(resolve_query_gtf "${raw_dir}" "${fasta}" || true)"
	fi
	if [[ -z "${ref_annotation}" ]]; then
		ref_annotation="$(resolve_reference_annotation "${raw_dir}" || true)"
	fi

	if [[ -z "${fasta}" || ! -f "${fasta}" ]]; then
		echo "FASTA not found for species=${species}" >&2
		exit 3
	fi
	if [[ -z "${query_gtf}" || ! -f "${query_gtf}" ]]; then
		echo "Query GTF not found for species=${species}" >&2
		exit 4
	fi
	if [[ -z "${ref_annotation}" || ! -f "${ref_annotation}" ]]; then
		echo "Reference annotation not found for species=${species}" >&2
		exit 5
	fi

	echo "[make_labeled_intron_eval_data] species=${species}"
	echo "[make_labeled_intron_eval_data] fasta=${fasta}"
	echo "[make_labeled_intron_eval_data] query_gtf=${query_gtf}"
	echo "[make_labeled_intron_eval_data] reference_annotation=${ref_annotation}"
	echo "[make_labeled_intron_eval_data] out_tsv=${out_tsv}"

	"${python_bin}" "${PROJECT_ROOT}/src/util/make_labeled_intron_eval_data.py" \
		--species "${species}" \
		--fasta "${fasta}" \
		--query-gtf "${query_gtf}" \
		--reference-annotation "${ref_annotation}" \
		--out-tsv "${out_tsv}" \
		--donor-len "${DONOR_LEN}" \
		--acceptor-len "${ACCEPTOR_LEN}" \
		--flank-bp "${FLANK_BP}" \
		--limit "${LIMIT}"
done
