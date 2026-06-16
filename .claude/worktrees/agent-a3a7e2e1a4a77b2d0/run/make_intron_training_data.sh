#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOT'
Usage: bash run/make_intron_training_data.sh [options]

Options:
  --species <csv>              Species list (default: Dmel,Mmus,Athal)
  --data-root <path>           Data root (default: <repo>/data)
  --flank-bp <int>             Flank bp on both intron ends (default: 10)
  --pos-input-name <name>      Positive input filename (default: 100bp.err)
  --out-pos-name <name>        Positive output TSV name
                               (default: intron_full_flank10.pos.tsv)
  --out-qc-name <name>         QC output TSV name
                               (default: intron_full_flank10.pos.qc.tsv)
  --out-neg-request-name <name>
                               Negative coordinate-request TSV name
                               (default: intron_full_flank10.neg_coordinate_request.tsv)
  --no-strict                  Allow unmatched/ambiguous/mismatch rows
  -h, --help                   Show this help

Notes:
  - Existing model training/inference pipeline is not modified.
  - This script only generates dataset TSV files under data/<species>/processed.
EOT
}

# --------------------------
# USER DEFAULTS (optional edit)
# --------------------------
CONDA_ENV="intronmodel"
SPECIES="Dmel,Mmus,Athal,Hsap"
DATA_ROOT=""
FLANK_BP="10"
POS_INPUT_NAME="100bp.err"
OUT_POS_NAME="intron_full_flank10.pos.tsv"
OUT_QC_NAME="intron_full_flank10.pos.qc.tsv"
OUT_NEG_REQUEST_NAME="intron_full_flank10.neg_coordinate_request.tsv"
STRICT="1"

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
	--flank-bp)
		FLANK_BP="$2"
		shift 2
		;;
	--pos-input-name)
		POS_INPUT_NAME="$2"
		shift 2
		;;
	--out-pos-name)
		OUT_POS_NAME="$2"
		shift 2
		;;
	--out-qc-name)
		OUT_QC_NAME="$2"
		shift 2
		;;
	--out-neg-request-name)
		OUT_NEG_REQUEST_NAME="$2"
		shift 2
		;;
	--no-strict)
		STRICT="0"
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

echo "[make_intron_training_data.sh] species=${SPECIES}"
echo "[make_intron_training_data.sh] data_root=${DATA_ROOT}"
echo "[make_intron_training_data.sh] flank_bp=${FLANK_BP}"
echo "[make_intron_training_data.sh] pos_input_name=${POS_INPUT_NAME}"
echo "[make_intron_training_data.sh] out_pos_name=${OUT_POS_NAME}"
echo "[make_intron_training_data.sh] out_qc_name=${OUT_QC_NAME}"
echo "[make_intron_training_data.sh] out_neg_request_name=${OUT_NEG_REQUEST_NAME}"
echo "[make_intron_training_data.sh] strict=${STRICT}"

args=(
	--species "${SPECIES}"
	--data-root "${DATA_ROOT}"
	--flank-bp "${FLANK_BP}"
	--pos-input-name "${POS_INPUT_NAME}"
	--out-pos-name "${OUT_POS_NAME}"
	--out-qc-name "${OUT_QC_NAME}"
	--out-neg-request-name "${OUT_NEG_REQUEST_NAME}"
)

if [[ "${STRICT}" == "0" ]]; then
	args+=(--no-strict)
fi

python3 "${PROJECT_ROOT}/src/util/make_intron_training_data_from_err.py" "${args[@]}"
