#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOT'
Usage: bash run/make_mixed_pair_neg_data.sh [options]

Options:
  --species <csv>                 Species list (default: Dmel,Mmus,Athal,Hsap)
  --data-root <path>              Data root (default: <repo>/data)
  --pos-input-name <name>         Positive source name (default: 100bp.err)
  --neg-input-name <name>         Negative source name (default: 100bp.neg.err)
  --output-name <name>            Output filename in processed dir
                                  (default: 100bp_mixed_one_side.neg.err)
  --mix-mode <mode>               both | donor_pos | acceptor_pos
                                  (default: both)
  --samples-per-negative <int>    Generated rows per negative per enabled side
                                  (default: 1)
  --seed <int>                    Random seed (default: 1337)
  --no-shuffle                    Disable final shuffle
  --no-strict                     Skip malformed rows when possible
  -h, --help                      Show help

Notes:
  - The output consists of DEBUG pair rows where one side is from a true pair
    and the opposite side is sampled from negative donor/acceptor site rows.
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
OUTPUT_NAME="100bp_mixed_one_side.neg.err"
MIX_MODE="both"
SAMPLES_PER_NEGATIVE="1"
SEED="1337"
SHUFFLE="1"
STRICT="1"
INTRONMODEL_AUTO_TMUX=off

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
	--output-name)
		OUTPUT_NAME="$2"
		shift 2
		;;
	--mix-mode)
		MIX_MODE="$2"
		shift 2
		;;
	--samples-per-negative | --samples-per-positive)
		SAMPLES_PER_NEGATIVE="$2"
		shift 2
		;;
	--seed)
		SEED="$2"
		shift 2
		;;
	--no-shuffle)
		SHUFFLE="0"
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

echo "[make_mixed_pair_neg_data.sh] species=${SPECIES}"
echo "[make_mixed_pair_neg_data.sh] data_root=${DATA_ROOT}"
echo "[make_mixed_pair_neg_data.sh] pos_input_name=${POS_INPUT_NAME}"
echo "[make_mixed_pair_neg_data.sh] neg_input_name=${NEG_INPUT_NAME}"
echo "[make_mixed_pair_neg_data.sh] output_name=${OUTPUT_NAME}"
echo "[make_mixed_pair_neg_data.sh] mix_mode=${MIX_MODE}"
echo "[make_mixed_pair_neg_data.sh] samples_per_negative=${SAMPLES_PER_NEGATIVE}"
echo "[make_mixed_pair_neg_data.sh] seed=${SEED}"
echo "[make_mixed_pair_neg_data.sh] shuffle=${SHUFFLE}"
echo "[make_mixed_pair_neg_data.sh] strict=${STRICT}"

args=(
	--species "${SPECIES}"
	--data-root "${DATA_ROOT}"
	--pos-input-name "${POS_INPUT_NAME}"
	--neg-input-name "${NEG_INPUT_NAME}"
	--output-name "${OUTPUT_NAME}"
	--mix-mode "${MIX_MODE}"
	--samples-per-negative "${SAMPLES_PER_NEGATIVE}"
	--seed "${SEED}"
)

if [[ "${SHUFFLE}" == "0" ]]; then
	args+=(--no-shuffle)
fi
if [[ "${STRICT}" == "0" ]]; then
	args+=(--no-strict)
fi

PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
	python3 "${PROJECT_ROOT}/src/util/make_mixed_pair_neg_data.py" "${args[@]}"
