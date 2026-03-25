#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOT'
Usage: bash run/eval_trans_score.sh [options]

This script uses internal CONFIG defaults, but you can override targets
from CLI to process multiple species/scores in one run.

Options:
  --species <csv>    Comma-separated species list (e.g. Hsap,Mmus)
  --scores <csv>     Comma-separated score files/paths
  --score <value>    One score file/path (repeatable)

Behavior:
	- If SCORE_INPUTS is empty, all *.tsv and *.txt files under
		data/<species>/trans_score are evaluated.

Optional:
  -h, --help   Show this help
EOT
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
	usage
	exit 0
fi

# --------------------------
# CONFIG (edit here)
# --------------------------
# Frequently edited knobs are intentionally placed first in this block.
CONDA_ENV="intronmodel"
USE_CONDA_ACTIVATE="1"
VISUALIZE="true"
OUTPUT_PNG=""
X_MIN=""
X_MAX=""
Y_MIN=""
Y_MAX=""
TARGET_SPECIES=("Athal" "Dmel" "Mmus" "Hsap")
SCORE_INPUTS=("cnn_v2")
CLASS_FILE_OVERRIDE=""
REF_GFF_OVERRIDE=""
INTRONMODEL_AUTO_TMUX=off

CLI_SPECIES_CSV=""
CLI_SCORES_CSV=""
CLI_SCORE_INPUTS=()

append_csv_items() {
	local csv_text="$1"
	local raw_item=""
	IFS=',' read -r -a _csv_items <<<"${csv_text}"
	for raw_item in "${_csv_items[@]}"; do
		item="$(printf '%s' "${raw_item}" | tr -d '[:space:]')"
		if [[ -n "${item}" ]]; then
			CLI_SCORE_INPUTS+=("${item}")
		fi
	done
}

while [[ $# -gt 0 ]]; do
	case "$1" in
	--species)
		CLI_SPECIES_CSV="$2"
		shift 2
		;;
	--scores)
		CLI_SCORES_CSV="$2"
		shift 2
		;;
	--score)
		CLI_SCORE_INPUTS+=("$2")
		shift 2
		;;
	*)
		echo "Unknown option: $1" >&2
		usage >&2
		exit 1
		;;
	esac
done

if [[ -n "${CLI_SPECIES_CSV}" ]]; then
	TARGET_SPECIES=()
	IFS=',' read -r -a _species_items <<<"${CLI_SPECIES_CSV}"
	for raw_item in "${_species_items[@]}"; do
		item="$(printf '%s' "${raw_item}" | tr -d '[:space:]')"
		if [[ -n "${item}" ]]; then
			TARGET_SPECIES+=("${item}")
		fi
	done
fi

if [[ -n "${CLI_SCORES_CSV}" ]]; then
	append_csv_items "${CLI_SCORES_CSV}"
fi
if [[ ${#CLI_SCORE_INPUTS[@]} -gt 0 ]]; then
	SCORE_INPUTS=("${CLI_SCORE_INPUTS[@]}")
fi


if [[ "${VISUALIZE}" != "none" && "${VISUALIZE}" != "true" \
	&& "${VISUALIZE}" != "interactive" ]]; then
	echo "Invalid VISUALIZE value: ${VISUALIZE}" >&2
	exit 1
fi
if [[ ${#TARGET_SPECIES[@]} -eq 0 ]]; then
	echo "TARGET_SPECIES must contain at least one species." >&2
	exit 1
fi
for score_input in "${SCORE_INPUTS[@]}"; do
	if [[ -z "${score_input}" ]]; then
		echo "SCORE_INPUTS contains an empty item." >&2
		exit 1
	fi
done

# --------------------------
# Runtime implementation
# --------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"
intronmodel_init_paths "${BASH_SOURCE[0]}"

# Auto-run inside tmux on SSH so jobs survive disconnects.
# Set INTRONMODEL_AUTO_TMUX=off|on|auto (default: auto).
intronmodel_enable_auto_tmux "${PROJECT_ROOT}" "$0" "${BASH_SOURCE[0]##*/}"

if [[ "${USE_CONDA_ACTIVATE}" == "1" ]]; then
	intronmodel_activate_conda "${CONDA_ENV}"
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

collect_all_score_inputs() {
	local trans_score_dir="$1"

	find "${trans_score_dir}" -maxdepth 1 -type f \
		\( -name '*.tsv' -o -name '*.txt' \) -printf '%f\n' | sort
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

set_defaults_for_species() {
	local sp="$1"
	case "${sp}" in
	Athal)
		DEFAULT_X_MIN="10.0"
		DEFAULT_X_MAX="52.0"
		DEFAULT_Y_MIN="48.0"
		DEFAULT_Y_MAX="75.0"
		;;
	Dmel)
		DEFAULT_X_MIN="40.0"
		DEFAULT_X_MAX="52.0"
		DEFAULT_Y_MIN="40.0"
		DEFAULT_Y_MAX="55.0"
		;;
	Mmus)
		DEFAULT_X_MIN="10.0"
		DEFAULT_X_MAX="18.0"
		DEFAULT_Y_MIN="40.0"
		DEFAULT_Y_MAX="46.0"
		;;
	Hsap)
		DEFAULT_X_MIN="10.0"
		DEFAULT_X_MAX="19.0"
		DEFAULT_Y_MIN="26.0"
		DEFAULT_Y_MAX="35.0"
		;;
	*)
		DEFAULT_X_MIN="40.0"
		DEFAULT_X_MAX="50.0"
		DEFAULT_Y_MIN="40.0"
		DEFAULT_Y_MAX="50.0"
		;;
	esac
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
	set_defaults_for_species "${species}"
	x_min_final="${X_MIN:-${DEFAULT_X_MIN}}"
	x_max_final="${X_MAX:-${DEFAULT_X_MAX}}"
	y_min_final="${Y_MIN:-${DEFAULT_Y_MIN}}"
	y_max_final="${Y_MAX:-${DEFAULT_Y_MAX}}"

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

	score_inputs_for_species=()
	if [[ ${#SCORE_INPUTS[@]} -eq 0 ]]; then
		mapfile -t score_inputs_for_species < <(
			collect_all_score_inputs "${TRANS_SCORE_DIR}"
		)
		if [[ ${#score_inputs_for_species[@]} -eq 0 ]]; then
			echo "No score files found under: ${TRANS_SCORE_DIR}" >&2
			echo "Expected at least one *.tsv or *.txt." >&2
			exit 5
		fi
		echo "[eval_trans_score] species=${species} evaluating all score files"
	else
		score_inputs_for_species=("${SCORE_INPUTS[@]}")
	fi

	for input_value in "${score_inputs_for_species[@]}"; do
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
			--x_min "${x_min_final}"
			--x_max "${x_max_final}"
			--y_min "${y_min_final}"
			--y_max "${y_max_final}"
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
