#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOT'
Usage: bash run/eval_trans_score.sh

This script uses internal configuration only.
Edit the top CONFIG block in this file to change species, score files,
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
	echo "This script does not accept CLI options. Edit the top CONFIG block instead." >&2
	exit 1
fi

# --------------------------
# CONFIG (edit here)
# --------------------------
# Frequently edited knobs are intentionally placed first in this block.
CONDA_ENV="intronmodel"
USE_CONDA_ACTIVATE="1"
VISUALIZE="true"
OUTPUT_PNG=""
TARGET_SPECIES=("Dmel")
SCORE_INPUTS=("cnn100bp.tsv" "Markov.txt" "LLM100f.tsv")
CLASS_FILE_OVERRIDE=""
REF_GFF_OVERRIDE=""

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

# --------------------------
# Runtime implementation
# --------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${INTRONMODEL_DATA_ROOT:-${PROJECT_ROOT}/data}"
MODEL_ROOT="${INTRONMODEL_MODEL_ROOT:-${PROJECT_ROOT}/model}"
export INTRONMODEL_MODEL_ROOT="${MODEL_ROOT}"
export INTRONMODEL_DATA_ROOT="${DATA_ROOT}"

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

resolve_ref_gff() {
	local raw_dir="$1"
	local gff_candidates=()
	local regular_candidates=()
	local preferred_candidates=()

	shopt -s nullglob
	gff_candidates=("${raw_dir}"/*.gff "${raw_dir}"/*.gff3 "${raw_dir}"/*.gff.*)
	shopt -u nullglob

	if [[ ${#gff_candidates[@]} -eq 0 ]]; then
		return 1
	fi

	for candidate in "${gff_candidates[@]}"; do
		if [[ ! -f "${candidate}" ]]; then
			continue
		fi
		regular_candidates+=("${candidate}")
		if [[ "${candidate}" == *.fix.gff || "${candidate}" == *.gff.fix ]]; then
			preferred_candidates+=("${candidate}")
		fi
	done

	if [[ ${#regular_candidates[@]} -eq 0 ]]; then
		return 1
	fi
	if [[ ${#preferred_candidates[@]} -gt 0 ]]; then
		echo "${preferred_candidates[0]}"
		return 0
	fi
	echo "${regular_candidates[0]}"
	return 0
}

for species in "${TARGET_SPECIES[@]}"; do
	RAW_DIR="${DATA_ROOT}/${species}/raw"
	TRANS_SCORE_DIR="${DATA_ROOT}/${species}/trans_score"
	EVAL_SCORE_DIR="${DATA_ROOT}/${species}/eval_score"

	CLASS_FILE="${CLASS_FILE_OVERRIDE}"
	REF_GFF="${REF_GFF_OVERRIDE}"
	if [[ -z "${CLASS_FILE}" ]]; then
		CLASS_FILE="${RAW_DIR}/transcript_class.txt"
	fi
	if [[ -z "${REF_GFF}" ]]; then
		REF_GFF="$(resolve_ref_gff "${RAW_DIR}" || true)"
	fi

	if [[ ! -d "${TRANS_SCORE_DIR}" ]]; then
		echo "trans_score directory not found: ${TRANS_SCORE_DIR}" >&2
		exit 2
	fi
	if [[ ! -f "${CLASS_FILE}" ]]; then
		echo "class file not found: ${CLASS_FILE}" >&2
		exit 3
	fi
	if [[ -z "${REF_GFF}" || ! -f "${REF_GFF}" ]]; then
		echo "reference gff not found for species=${species}" >&2
		exit 4
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
			"${REF_GFF}"
			--output_file "${output_file}"
			--species "${species}"
			--visualize "${VISUALIZE}"
		)
		if [[ -n "${OUTPUT_PNG}" ]]; then
			RUN_ARGS+=(--output_png "${OUTPUT_PNG}")
		fi

		echo "[eval_trans_score] species=${species} file=${score_file}"
		echo "[eval_trans_score] ref_gff=${REF_GFF}"
		python3 "${PROJECT_ROOT}/src/evaluate_scores.py" "${RUN_ARGS[@]}"
		echo "[eval_trans_score] wrote ${output_file}"
	done
done
