#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOT'
Usage: bash run/gffcompare_counts.sh [options]

Options:
  --species <name>     Species folder under data/ (default: Dmel)
  --conda-env <name>   Conda environment name (default: intronmodel)
  -h, --help           Show this help

Notes:
  - Input files are auto-detected from data/<species>/raw.
  - gffcompare runs as: gffcompare -r <reference.gff> -o <tmp_prefix> <query.gtf>
  - Output file is data/<species>/raw/gffcompare_counts.txt with:
      good<TAB>...
      total<TAB>...
      ref<TAB>...
  - Temporary gffcompare products are deleted automatically.
EOT
}

SPECIES="Dmel"
CONDA_ENV="intronmodel"

while [[ $# -gt 0 ]]; do
	case "$1" in
	--species)
		SPECIES="$2"
		shift 2
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
		usage
		exit 1
		;;
	esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RAW_DIR="${PROJECT_ROOT}/data/${SPECIES}/raw"
OUT_COUNTS="${RAW_DIR}/gffcompare_counts.txt"

if [[ ! -d "${RAW_DIR}" ]]; then
	echo "Raw directory not found: ${RAW_DIR}" >&2
	exit 3
fi

# Ensure conda is available in non-interactive shells.
if command -v conda >/dev/null 2>&1; then
	CONDA_BASE="$(conda info --base 2>/dev/null || true)"
	if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
		# shellcheck source=/dev/null
		source "${CONDA_BASE}/etc/profile.d/conda.sh"
	fi
fi

if ! command -v conda >/dev/null 2>&1; then
	echo "conda not found in PATH (needed for gffcompare env)" >&2
	exit 6
fi

shopt -s nullglob
gtf_candidates=("${RAW_DIR}"/*.gtf)
gff_candidates=("${RAW_DIR}"/*.gff "${RAW_DIR}"/*.gff3 "${RAW_DIR}"/*.gff.*)
shopt -u nullglob

if [[ ${#gtf_candidates[@]} -lt 1 ]]; then
	echo "No .gtf files found under: ${RAW_DIR}" >&2
	exit 4
fi
if [[ ${#gff_candidates[@]} -lt 1 ]]; then
	echo "No .gff/.gff3/.gff.* files found under: ${RAW_DIR}" >&2
	exit 5
fi

# Query transcript GTF: prefer *.fna.gtf, else the first .gtf.
QUERY_GTF=""
for f in "${gtf_candidates[@]}"; do
	if [[ "${f}" == *.fna.gtf ]]; then
		QUERY_GTF="${f}"
		break
	fi
done
if [[ -z "${QUERY_GTF}" ]]; then
	QUERY_GTF="${gtf_candidates[0]}"
fi

# Reference annotation: prefer *.fix.gff or *.gff.fix, else first gff-like.
REF_GFF=""
for f in "${gff_candidates[@]}"; do
	if [[ "${f}" == *.fix.gff || "${f}" == *.gff.fix ]]; then
		REF_GFF="${f}"
		break
	fi
done
if [[ -z "${REF_GFF}" ]]; then
	REF_GFF="${gff_candidates[0]}"
fi

TMP_DIR="$(mktemp -d "${RAW_DIR}/.gffcompare_tmp.XXXXXX")"
OUT_STEM="gffcompare_tmp_${$}_$(date +%s)"
OUT_PREFIX="${TMP_DIR}/${OUT_STEM}"
QUERY_BASENAME="$(basename "${QUERY_GTF}")"
QUERY_TMAP="${RAW_DIR}/${OUT_STEM}.${QUERY_BASENAME}.tmap"
QUERY_REFMAP="${RAW_DIR}/${OUT_STEM}.${QUERY_BASENAME}.refmap"

cleanup() {
	rm -rf "${TMP_DIR}" || true
	rm -f "${QUERY_TMAP}" "${QUERY_REFMAP}" || true
}
trap cleanup EXIT

echo "[gffcompare_counts] species=${SPECIES}" >&2
echo "[gffcompare_counts] reference=${REF_GFF}" >&2
echo "[gffcompare_counts] query=${QUERY_GTF}" >&2

# Run inside TMP_DIR because gffcompare may emit .tmap/.refmap to CWD.
(
	cd "${TMP_DIR}"
	conda run -n "${CONDA_ENV}" gffcompare -r "${REF_GFF}" -o "${OUT_STEM}" \
		"${QUERY_GTF}" >/dev/null
)

STATS_FILE="${OUT_PREFIX}.stats"
if [[ ! -f "${STATS_FILE}" ]]; then
	echo "Expected gffcompare stats not found: ${STATS_FILE}" >&2
	exit 7
fi

if [[ ! -f "${QUERY_TMAP}" ]]; then
	echo "Expected gffcompare tmap not found: ${QUERY_TMAP}" >&2
	exit 8
fi
TMAP_FILE="${QUERY_TMAP}"

GOOD="$({
	awk '
		/^[[:space:]]*Matching[[:space:]]+intron[[:space:]]+chains:/ {
			for (i = 1; i <= NF; i++) {
				if ($i ~ /^[0-9]+$/) {
					print $i
					exit
				}
			}
		}
	' "${STATS_FILE}"
})"

REF="$({
	awk '
		/^[[:space:]#]*Reference[[:space:]]+mRNAs[[:space:]]*:/ {
			for (i = 1; i <= NF; i++) {
				if ($i ~ /^[0-9]+$/) {
					print $i
					exit
				}
			}
		}
	' "${STATS_FILE}"
})"

TOTAL="$({
	awk -F '\t' '
		NR == 1 {
			for (i = 1; i <= NF; i++) {
				if ($i == "class_code" || $i == "#class_code") {
					class_idx = i
				}
				if ($i == "num_exons") {
					exon_idx = i
				}
			}
			if (class_idx == 0 || exon_idx == 0) {
				exit 2
			}
			next
		}
		{
			exons = $exon_idx + 0
			code = $class_idx
			if (exons > 1 && code != "c") {
				total += 1
			}
		}
		END {
			if (NR < 2) {
				exit 3
			}
			print total + 0
		}
	' "${TMAP_FILE}"
})"

if [[ -z "${GOOD}" || -z "${TOTAL}" || -z "${REF}" ]]; then
	echo "Failed to parse one or more counts from gffcompare outputs." >&2
	exit 9
fi

{
	printf 'good\t%s\n' "${GOOD}"
	printf 'total\t%s\n' "${TOTAL}"
	printf 'ref\t%s\n' "${REF}"
} >"${OUT_COUNTS}"

echo "[gffcompare_counts] wrote ${OUT_COUNTS}" >&2
echo "[gffcompare_counts] good=${GOOD} total=${TOTAL} ref=${REF}" >&2
