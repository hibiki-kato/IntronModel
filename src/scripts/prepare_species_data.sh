#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOT'
Usage: bash src/scripts/prepare_species_data.sh [options]

Prepare species-level data directories and generated files.

Options:
	--species <name>        Athal|Dmel|Hsap|Mmus (default: Dmel)
  --donor-len <int>       Donor window length (default: 100)
  --acceptor-len <int>    Acceptor window length (default: 100)
  --clip-short-intron     Keep intronic context within intron length
  --source-root <path>    Optional source root for raw data import
  --target-root <path>    Data root path (default: INTRONMODEL_DATA_ROOT or ./data)
  -h, --help              Show this help

Notes:
  - This script does not create train/*.err files.
  - Place training files separately if train stage is required.
EOT
}

SPECIES="Dmel"
DONOR_LEN="100"
ACCEPTOR_LEN="100"
SOURCE_ROOT=""
TARGET_ROOT="${INTRONMODEL_DATA_ROOT:-data}"
CLIP_SHORT_INTRON="0"

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
	--source-root)
		SOURCE_ROOT="$2"
		shift 2
		;;
	--clip-short-intron)
		CLIP_SHORT_INTRON="1"
		shift
		;;
	--target-root)
		TARGET_ROOT="$2"
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

case "${SPECIES}" in
Athal | Dmel | Hsap | Mmus)
	;;
*)
	echo "Invalid --species value: ${SPECIES}" >&2
	exit 2
	;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ "${TARGET_ROOT}" != /* ]]; then
	TARGET_ROOT="${PROJECT_ROOT}/${TARGET_ROOT}"
fi

SPECIES_DIR="${TARGET_ROOT}/${SPECIES}"
mkdir -p "${SPECIES_DIR}/raw"
mkdir -p "${SPECIES_DIR}/train"
mkdir -p "${SPECIES_DIR}/site_score"
mkdir -p "${SPECIES_DIR}/intron_score"
mkdir -p "${SPECIES_DIR}/learning_metric"
mkdir -p "${SPECIES_DIR}/trans_score"
mkdir -p "${SPECIES_DIR}/eval_score"

if [[ -n "${SOURCE_ROOT}" ]]; then
	bash "${PROJECT_ROOT}/src/scripts/fetch_reference_data.sh" \
		--species "${SPECIES}" \
		--source-root "${SOURCE_ROOT}" \
		--target-root "${TARGET_ROOT}" \
		--mode copy
fi

make_test_args=(
	--species "${SPECIES}"
	--donor-len "${DONOR_LEN}"
	--acceptor-len "${ACCEPTOR_LEN}"
	--out-tsv "${SPECIES_DIR}/raw/transcripts.tsv"
)
if [[ "${CLIP_SHORT_INTRON}" == "1" ]]; then
	make_test_args+=(--clip-short-intron)
fi

bash "${PROJECT_ROOT}/run/make_test_data.sh" "${make_test_args[@]}"

echo "[prepare_species_data] prepared directories under ${SPECIES_DIR}"
echo "[prepare_species_data] generated raw/transcripts.tsv"

if [[ ! -f "${SPECIES_DIR}/train/${DONOR_LEN}bp.err" ]]; then
	echo "[prepare_species_data] warning: missing ${SPECIES_DIR}/train/${DONOR_LEN}bp.err"
fi
if [[ ! -f "${SPECIES_DIR}/train/${DONOR_LEN}bp.neg.err" ]]; then
	echo "[prepare_species_data] warning: missing" \
		"${SPECIES_DIR}/train/${DONOR_LEN}bp.neg.err"
fi
