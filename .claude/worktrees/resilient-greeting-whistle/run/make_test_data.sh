#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOT'
Usage: bash run/make_test_data.sh [options]

Options:
  --species <name>        Species folder under data/ (default: Dmel)
  --donor-upstream-bp <int>     Donor upstream context (default: 100)
  --donor-downstream-bp <int>   Donor downstream context (default: 100)
  --acceptor-upstream-bp <int>  Acceptor upstream context (default: 100)
  --acceptor-downstream-bp <int> Acceptor downstream context (default: 100)
  --donor-len <int>       Legacy donor total length override
  --acceptor-len <int>    Legacy acceptor total length override
  --clip-short-intron     Keep intronic context within intron length
  --fasta <path>          Override FASTA path (.fna)
  --gtf <path>            Override GTF path
  --out-tsv <path>        Output TSV path (default: data/<species>/processed/transcripts.tsv)
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
DONOR_UPSTREAM_BP="100"
DONOR_DOWNSTREAM_BP="100"
ACCEPTOR_UPSTREAM_BP="100"
ACCEPTOR_DOWNSTREAM_BP="100"
DONOR_LEN=""
ACCEPTOR_LEN=""
FASTA_PATH=""
GTF_PATH=""
OUT_TSV=""
FEATURE="exon"
LIMIT="0"
CLIP_SHORT_INTRON="0"

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
	--donor-len)
		DONOR_LEN="$2"
		shift 2
		;;
	--donor-upstream-bp)
		DONOR_UPSTREAM_BP="$2"
		shift 2
		;;
	--donor-downstream-bp)
		DONOR_DOWNSTREAM_BP="$2"
		shift 2
		;;
	--acceptor-len)
		ACCEPTOR_LEN="$2"
		shift 2
		;;
	--acceptor-upstream-bp)
		ACCEPTOR_UPSTREAM_BP="$2"
		shift 2
		;;
	--acceptor-downstream-bp)
		ACCEPTOR_DOWNSTREAM_BP="$2"
		shift 2
		;;
	--fasta)
		FASTA_PATH="$2"
		shift 2
		;;
	--clip-short-intron)
		CLIP_SHORT_INTRON="1"
		shift
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
PROCESSED_DIR="${DATA_ROOT}/${SPECIES}/processed"
if [[ ! -d "${RAW_DIR}" ]]; then
	echo "Raw directory not found: ${RAW_DIR}" >&2
	exit 2
fi
mkdir -p "${PROCESSED_DIR}"

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
	OUT_TSV="${PROCESSED_DIR}/transcripts.tsv"
fi

echo "[make_test_data.sh] species=${SPECIES}"
echo "[make_test_data.sh] fasta=${FASTA_PATH}"
echo "[make_test_data.sh] gtf=${GTF_PATH}"
echo "[make_test_data.sh] out_tsv=${OUT_TSV}"
echo "[make_test_data.sh] clip_short_intron=${CLIP_SHORT_INTRON}"
if [[ -n "${DONOR_LEN}" || -n "${ACCEPTOR_LEN}" ]]; then
	echo "[make_test_data.sh] legacy donor_len=${DONOR_LEN} acceptor_len=${ACCEPTOR_LEN}"
else
	echo "[make_test_data.sh] donor_upstream_bp=${DONOR_UPSTREAM_BP} donor_downstream_bp=${DONOR_DOWNSTREAM_BP}"
	echo "[make_test_data.sh] acceptor_upstream_bp=${ACCEPTOR_UPSTREAM_BP} acceptor_downstream_bp=${ACCEPTOR_DOWNSTREAM_BP}"
fi

args=(
	--fasta "${FASTA_PATH}"
	--gtf "${GTF_PATH}"
	--out_tsv "${OUT_TSV}"
	--feature "${FEATURE}"
	--limit "${LIMIT}"
)

if [[ -n "${DONOR_LEN}" || -n "${ACCEPTOR_LEN}" ]]; then
	if [[ -z "${DONOR_LEN}" || -z "${ACCEPTOR_LEN}" ]]; then
		echo "Both --donor-len and --acceptor-len are required in legacy mode." >&2
		exit 5
	fi
	args+=(
		--donor_len "${DONOR_LEN}"
		--acceptor_len "${ACCEPTOR_LEN}"
	)
else
	args+=(
		--donor_upstream_bp "${DONOR_UPSTREAM_BP}"
		--donor_downstream_bp "${DONOR_DOWNSTREAM_BP}"
		--acceptor_upstream_bp "${ACCEPTOR_UPSTREAM_BP}"
		--acceptor_downstream_bp "${ACCEPTOR_DOWNSTREAM_BP}"
	)
fi

if [[ "${CLIP_SHORT_INTRON}" == "1" ]]; then
	args+=(--clip-short-intron)
fi

python3 "${PROJECT_ROOT}/src/util/make_test_data_from_gtf.py" "${args[@]}"
