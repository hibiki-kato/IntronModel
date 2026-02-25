#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOT'
Usage: bash run/plot_eval.sh [options]

Options:
  --species <name>      Species folder under data/ (default: Dmel)
  --output-png <path>   Output PNG path (default: data/<species>/<species>_snpr.png)
  --interactive         Show plot interactively
  --x-min <float>       X-axis minimum (species default if omitted)
  --x-max <float>       X-axis maximum (species default if omitted)
  --y-min <float>       Y-axis minimum (species default if omitted)
  --y-max <float>       Y-axis maximum (species default if omitted)
  -h, --help            Show this help

Species default ranges:
  Athal: x=[10.0, 52.0], y=[48.0, 75.0]
  Dmel : x=[40.0, 52.0], y=[40.0, 55.0]
  Mmus : x=[10.0, 18.0], y=[40.0, 46.0]
EOT
}

# --------------------------
# USER DEFAULTS (optional edit)
# --------------------------
CONDA_ENV="intronmodel"
SPECIES="Mmus, Athal, Dmel"
OUTPUT_PNG=""
INTERACTIVE="1"
X_MIN=""
X_MAX=""
Y_MIN=""
Y_MAX=""

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
	--output-png)
		OUTPUT_PNG="$2"
		shift 2
		;;
	--interactive)
		INTERACTIVE="1"
		shift
		;;
	--x-min)
		X_MIN="$2"
		shift 2
		;;
	--x-max)
		X_MAX="$2"
		shift 2
		;;
	--y-min)
		Y_MIN="$2"
		shift 2
		;;
	--y-max)
		Y_MAX="$2"
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
	*)
		if [[ -z "${X_MIN}" || -z "${X_MAX}" || -z "${Y_MIN}" || -z "${Y_MAX}" ]]; then
			echo "[plot_eval.sh] Unknown species '${sp}'." >&2
			echo "[plot_eval.sh] Use one of: Athal, Dmel, Mmus; or provide all of" >&2
			echo "[plot_eval.sh] --x-min --x-max --y-min --y-max explicitly." >&2
			return 1
		fi
		DEFAULT_X_MIN="40.0"
		DEFAULT_X_MAX="50.0"
		DEFAULT_Y_MIN="40.0"
		DEFAULT_Y_MAX="50.0"
		;;
	esac
	return 0
}


run_for_one_species() {
	local sp="$1"
	local out_png_override="$2"
	# compute defaults for this species
	if ! set_defaults_for_species "${sp}"; then
		return 1
	fi

	local x_min_final="${X_MIN:-}"
	local x_max_final="${X_MAX:-}"
	local y_min_final="${Y_MIN:-}"
	local y_max_final="${Y_MAX:-}"
	if [[ -z "${x_min_final}" ]]; then
		x_min_final="${DEFAULT_X_MIN}"
	fi
	if [[ -z "${x_max_final}" ]]; then
		x_max_final="${DEFAULT_X_MAX}"
	fi
	if [[ -z "${y_min_final}" ]]; then
		y_min_final="${DEFAULT_Y_MIN}"
	fi
	if [[ -z "${y_max_final}" ]]; then
		y_max_final="${DEFAULT_Y_MAX}"
	fi

	local run_args=(
		plot
		"${sp}"
		--x_min "${x_min_final}"
		--x_max "${x_max_final}"
		--y_min "${y_min_final}"
		--y_max "${y_max_final}"
	)
	if [[ -n "${out_png_override}" ]]; then
		run_args+=(--output_png "${out_png_override}")
	fi
	if [[ "${INTERACTIVE}" == "1" ]]; then
		run_args+=(--interactive)
	fi

	echo "[plot_eval.sh] species=${sp} x=(${x_min_final},${x_max_final}) y=(${y_min_final},${y_max_final})"
	python3 "${PROJECT_ROOT}/src/evaluate_scores.py" "${run_args[@]}"
}

# If SPECIES is comma-separated, run per species
if [[ "${SPECIES}" == *","* ]]; then
	IFS=',' read -ra SPEC_ARR <<< "${SPECIES}"
	for sp in "${SPEC_ARR[@]}"; do
		# trim whitespace
		sp_trimmed="$(echo "${sp}" | sed -e 's/^\s*//' -e 's/\s*$//')"
		if ! run_for_one_species "${sp_trimmed}" "${OUTPUT_PNG}"; then
			echo "[plot_eval.sh] Failed for species ${sp_trimmed}" >&2
		fi
	done
else
	run_for_one_species "${SPECIES}" "${OUTPUT_PNG}"
fi
