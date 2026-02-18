#!/usr/bin/env bash
set -euo pipefail

# Ensure conda is available in non-interactive shells
if command -v conda >/dev/null 2>&1; then
	CONDA_BASE="$(conda info --base 2>/dev/null || true)"
	if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
		# shellcheck source=/dev/null
		source "${CONDA_BASE}/etc/profile.d/conda.sh"
	fi
fi

conda activate intronmodel

# Resolve script directory and run python with robust paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "${SCRIPT_DIR}/../src/cnn.py" \
	--species Dmel \
	--train_pos_path "${SCRIPT_DIR}/../data/Dmel/train/100bp.err" \
	--train_neg_path "${SCRIPT_DIR}/../data/Dmel/train/100bp.neg.err" \
	--test_tsv "${SCRIPT_DIR}/../data/Dmel/raw/transcripts.tsv" \
	--output_tsv "${SCRIPT_DIR}/../data/Dmel/trans_score/cnn100bp.tsv" \
	--donor_len 100 \
	--donor_left 3 \
	--acceptor_len 100 \
	--acceptor_right 3 \
	"$@"
