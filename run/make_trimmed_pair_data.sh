#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOT'
Usage: bash run/make_trimmed_pair_data.sh [options]

Options:
  --species <csv>           Species list (default: Dmel,Mmus,Athal)
  --data-root <path>        Data root (default: <repo>/data)
  --pos-input-name <name>   Positive source name (default: 100bp.err)
  --neg-input-name <name>   Negative source name (default: 100bp.neg.err)
  --out-pos-name <name>     Output positive name (default: 100bp_trimmed.err)
  --out-neg-name <name>     Output negative name
                            (default: 100bp_trimmed.neg.err)
  --exon-context-bp <int>   Boundary context bp (default: 3)
  --pad-with-n              Preserve fixed length by N-padding trimmed regions
  --no-strict               Skip malformed rows when possible
  -h, --help                Show help

Notes:
  - This script trims donor/acceptor pair sequences using the right-most value
    (intron half-length token) in pair records.
  - Existing model code is not modified.
EOT
}

# --------------------------
# USER DEFAULTS (optional edit)
# --------------------------
CONDA_ENV="intronmodel"
SPECIES="Dmel,Mmus,Athal,Hsap"
DATA_ROOT=""
POS_INPUT_NAME="100bp.err"
NEG_INPUT_NAME="100bp.neg.err"
OUT_POS_NAME="100bp_trimmed.err"
OUT_NEG_NAME="100bp_trimmed.neg.err"
EXON_CONTEXT_BP="3"
PAD_WITH_N="0"
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
	--pos-input-name)
		POS_INPUT_NAME="$2"
		shift 2
		;;
	--neg-input-name)
		NEG_INPUT_NAME="$2"
		shift 2
		;;
	--out-pos-name)
		OUT_POS_NAME="$2"
		shift 2
		;;
	--out-neg-name)
		OUT_NEG_NAME="$2"
		shift 2
		;;
	--exon-context-bp)
		EXON_CONTEXT_BP="$2"
		shift 2
		;;
	--pad-with-n)
		PAD_WITH_N="1"
		shift
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

echo "[make_trimmed_pair_data.sh] species=${SPECIES}"
echo "[make_trimmed_pair_data.sh] data_root=${DATA_ROOT}"
echo "[make_trimmed_pair_data.sh] pos_input_name=${POS_INPUT_NAME}"
echo "[make_trimmed_pair_data.sh] neg_input_name=${NEG_INPUT_NAME}"
echo "[make_trimmed_pair_data.sh] out_pos_name=${OUT_POS_NAME}"
echo "[make_trimmed_pair_data.sh] out_neg_name=${OUT_NEG_NAME}"
echo "[make_trimmed_pair_data.sh] exon_context_bp=${EXON_CONTEXT_BP}"
echo "[make_trimmed_pair_data.sh] pad_with_n=${PAD_WITH_N}"
echo "[make_trimmed_pair_data.sh] strict=${STRICT}"

args=(
	--species "${SPECIES}"
	--data-root "${DATA_ROOT}"
	--pos-input-name "${POS_INPUT_NAME}"
	--neg-input-name "${NEG_INPUT_NAME}"
	--out-pos-name "${OUT_POS_NAME}"
	--out-neg-name "${OUT_NEG_NAME}"
	--exon-context-bp "${EXON_CONTEXT_BP}"
)

if [[ "${STRICT}" == "0" ]]; then
	args+=(--no-strict)
fi

if [[ "${PAD_WITH_N}" == "1" ]]; then
	args+=(--pad-with-n)
fi

python3 "${PROJECT_ROOT}/src/util/make_trimmed_pair_data.py" "${args[@]}"
