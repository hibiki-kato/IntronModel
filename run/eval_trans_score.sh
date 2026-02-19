#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOT'
Usage: bash run/eval_trans_score.sh

This script uses internal configuration only.
Edit the CONFIG block in this file to change species, score files,
visualization mode, or environment settings.

Optional:
  -h, --help   Show this help
EOT
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
	usage
	exit 0
fi
if [[ $# -gt 0 ]]; then
	echo "This script does not accept CLI options. Edit the CONFIG block instead." >&2
	exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --------------------------
# CONFIG (edit here)
# --------------------------
CONDA_ENV="intronmodel"
USE_CONDA_ACTIVATE="1"
VISUALIZE="none"
OUTPUT_PNG=""
TARGET_SPECIES=("Athal" "Mmus")
SCORE_INPUTS=("Markov.txt")
CLASS_FILE_OVERRIDE=""
COUNTS_FILE_OVERRIDE=""

if [[ "${VISUALIZE}" != "none" && "${VISUALIZE}" != "true" \
	&& "${VISUALIZE}" != "interactive" ]]; then
	echo "Invalid VISUALIZE value: ${VISUALIZE}" >&2
	exit 1
fi
if [[ ${#TARGET_SPECIES[@]} -eq 0 ]]; then
	echo "TARGET_SPECIES must contain at least one species." >&2
	exit 1
fi
if [[ ${#SCORE_INPUTS[@]} -eq 0 ]]; then
	echo "SCORE_INPUTS must contain at least one score file name/path." >&2
	exit 1
fi

if [[ "${USE_CONDA_ACTIVATE}" == "1" ]] && command -v conda >/dev/null 2>&1; then
	CONDA_BASE="$(conda info --base 2>/dev/null || true)"
	if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
		# shellcheck source=/dev/null
		source "${CONDA_BASE}/etc/profile.d/conda.sh"
	fi
	conda activate "${CONDA_ENV}"
fi

if [[ -z "${MPLCONFIGDIR:-}" ]]; then
	MPLCONFIGDIR="${TMPDIR:-/tmp}/intronmodel-mpl-cache"
	mkdir -p "${MPLCONFIGDIR}"
	export MPLCONFIGDIR
fi
if [[ -z "${XDG_CACHE_HOME:-}" ]]; then
	XDG_CACHE_HOME="${TMPDIR:-/tmp}/intronmodel-cache"
	mkdir -p "${XDG_CACHE_HOME}"
	export XDG_CACHE_HOME
fi

resolve_score_file() {
	local input_value="$1"
	local trans_score_dir="$2"

	if [[ -f "${input_value}" ]]; then
		echo "${input_value}"
		return 0
	fi

	local candidate="${trans_score_dir}/${input_value}"
	if [[ -f "${candidate}" ]]; then
		echo "${candidate}"
		return 0
	fi
	if [[ "${input_value}" != *.tsv && -f "${candidate}.tsv" ]]; then
		echo "${candidate}.tsv"
		return 0
	fi
	if [[ "${input_value}" != *.txt && -f "${candidate}.txt" ]]; then
		echo "${candidate}.txt"
		return 0
	fi

	return 1
}

for species in "${TARGET_SPECIES[@]}"; do
	RAW_DIR="${PROJECT_ROOT}/data/${species}/raw"
	TRANS_SCORE_DIR="${PROJECT_ROOT}/data/${species}/trans_score"
	EVAL_SCORE_DIR="${PROJECT_ROOT}/data/${species}/eval_score"

	CLASS_FILE="${CLASS_FILE_OVERRIDE}"
	COUNTS_FILE="${COUNTS_FILE_OVERRIDE}"
	if [[ -z "${CLASS_FILE}" ]]; then
		CLASS_FILE="${RAW_DIR}/transcript_class.txt"
	fi
	if [[ -z "${COUNTS_FILE}" ]]; then
		COUNTS_FILE="${RAW_DIR}/gffcompare_counts.txt"
	fi

	if [[ ! -d "${TRANS_SCORE_DIR}" ]]; then
		echo "trans_score directory not found: ${TRANS_SCORE_DIR}" >&2
		exit 2
	fi
	if [[ ! -f "${CLASS_FILE}" ]]; then
		echo "class file not found: ${CLASS_FILE}" >&2
		exit 3
	fi
	if [[ ! -f "${COUNTS_FILE}" ]]; then
		echo "counts file not found: ${COUNTS_FILE}" >&2
		exit 4
	fi

	GOOD="$(awk -F '\t' '$1 == "good" {print $2}' "${COUNTS_FILE}")"
	TOTAL="$(awk -F '\t' '$1 == "total" {print $2}' "${COUNTS_FILE}")"
	REF="$(awk -F '\t' '$1 == "ref" {print $2}' "${COUNTS_FILE}")"

	if [[ ! "${GOOD}" =~ ^[0-9]+$ || ! "${TOTAL}" =~ ^[0-9]+$ \
		|| ! "${REF}" =~ ^[0-9]+$ ]]; then
		echo "Invalid counts in ${COUNTS_FILE}. Expected good/total/ref." >&2
		exit 5
	fi

	mkdir -p "${EVAL_SCORE_DIR}"

	for input_value in "${SCORE_INPUTS[@]}"; do
		score_file="$(resolve_score_file "${input_value}" "${TRANS_SCORE_DIR}" || true)"
		if [[ -z "${score_file}" ]]; then
			echo "score file not found for species=${species}: ${input_value}" >&2
			echo "Checked direct path and ${TRANS_SCORE_DIR}/ with .tsv/.txt." >&2
			exit 6
		fi

		stem="$(basename "${score_file}")"
		stem="${stem%.*}"
		output_file="${EVAL_SCORE_DIR}/${stem}.txt"

		RUN_ARGS=(
			eval
			"${CLASS_FILE}"
			"${score_file}"
			--output_file "${output_file}"
			--species "${species}"
			--good "${GOOD}"
			--total "${TOTAL}"
			--ref "${REF}"
			--visualize "${VISUALIZE}"
		)
		if [[ -n "${OUTPUT_PNG}" ]]; then
			RUN_ARGS+=(--output_png "${OUTPUT_PNG}")
		fi

		echo "[eval_trans_score] species=${species} file=${score_file}"
		echo "[eval_trans_score] counts good=${GOOD} total=${TOTAL} ref=${REF}"
		python3 "${PROJECT_ROOT}/src/evaluate_scores.py" "${RUN_ARGS[@]}"
		echo "[eval_trans_score] wrote ${output_file}"
	done
done
