#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOT'
Usage: bash run/make_test_data.sh [options]

Options:
  --species <name>        Species folder under data/ (default: Dmel)
  --donor-len <int>       Donor window length (default: 100)
  --acceptor-len <int>    Acceptor window length (default: 100)
  --fasta <path>          Override FASTA path (.fna)
  --gtf <path>            Override GTF path
  --out-tsv <path>        Output TSV path
  --feature <name>        GTF feature to use (default: exon)
  --limit <int>           Max rows to write (default: 0; no limit)
  -h, --help              Show this help

Notes:
  - By default, FASTA/GTF are auto-detected from data/<species>/raw.
  - FASTA priority: *.clean.fna, then *.fna.
  - GTF priority: <fasta>.gtf, then *.fna.gtf, then *.gtf.
EOT
}

# --------------------------
# USER DEFAULTS (optional edit)
# --------------------------
CONDA_ENV="intronmodel"
SPECIES="Dmel"
DONOR_LEN="100"
ACCEPTOR_LEN="100"
FASTA_PATH=""
GTF_PATH=""
OUT_TSV=""
FEATURE="exon"
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
# Set INTRONMODEL_AUTO_TMUX=off|on|auto (default: auto).
intronmodel_enable_auto_tmux "${PROJECT_ROOT}" "$0" "${BASH_SOURCE[0]##*/}"

while [[ $# -gt 0 ]]; do
	case "$1" in
	--species)
		SPECIES="$2"
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
	--fasta)
		FASTA_PATH="$2"
		shift 2
		;;
	--gtf)
		GTF_PATH="$2"
		shift 2
		;;
	--out-tsv)
		OUT_TSV="$2"
		shift 2
		;;
	--feature)
		FEATURE="$2"
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

RAW_DIR="${DATA_ROOT}/${SPECIES}/raw"
if [[ ! -d "${RAW_DIR}" ]]; then
	echo "Raw directory not found: ${RAW_DIR}" >&2
	exit 2
fi

if [[ -z "${FASTA_PATH}" ]]; then
	shopt -s nullglob
	clean_fna_candidates=("${RAW_DIR}"/*.clean.fna)
	fna_candidates=("${RAW_DIR}"/*.fna)
	shopt -u nullglob

	if [[ ${#clean_fna_candidates[@]} -gt 0 ]]; then
		FASTA_PATH="${clean_fna_candidates[0]}"
	elif [[ ${#fna_candidates[@]} -gt 0 ]]; then
		FASTA_PATH="${fna_candidates[0]}"
	else
		echo "No FASTA file found in ${RAW_DIR} (*.clean.fna or *.fna)." >&2
		exit 3
	fi
fi

if [[ -z "${GTF_PATH}" ]]; then
	default_gtf="${FASTA_PATH}.gtf"
	if [[ -f "${default_gtf}" ]]; then
		GTF_PATH="${default_gtf}"
	else
		shopt -s nullglob
		fna_gtf_candidates=("${RAW_DIR}"/*.fna.gtf)
		gtf_candidates=("${RAW_DIR}"/*.gtf)
		shopt -u nullglob

		if [[ ${#fna_gtf_candidates[@]} -gt 0 ]]; then
			GTF_PATH="${fna_gtf_candidates[0]}"
		elif [[ ${#gtf_candidates[@]} -gt 0 ]]; then
			GTF_PATH="${gtf_candidates[0]}"
		else
			echo "No GTF file found in ${RAW_DIR} (*.fna.gtf or *.gtf)." >&2
			exit 4
		fi
	fi
fi

if [[ -z "${OUT_TSV}" ]]; then
	OUT_TSV="${RAW_DIR}/transcripts.tsv"
fi

echo "[make_test_data.sh] species=${SPECIES}"
echo "[make_test_data.sh] fasta=${FASTA_PATH}"
echo "[make_test_data.sh] gtf=${GTF_PATH}"
echo "[make_test_data.sh] out_tsv=${OUT_TSV}"

python3 "${PROJECT_ROOT}/src/util/make_test_data_from_gtf.py" \
	--fasta "${FASTA_PATH}" \
	--gtf "${GTF_PATH}" \
	--out_tsv "${OUT_TSV}" \
	--donor_len "${DONOR_LEN}" \
	--acceptor_len "${ACCEPTOR_LEN}" \
	--feature "${FEATURE}" \
	--limit "${LIMIT}"
