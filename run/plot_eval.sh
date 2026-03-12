#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOT'
Usage: bash run/plot_eval.sh [options]

Options:
  --species <name>      Species folder under data/ (default: Dmel)
                        Use commas to launch multiple species
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
  Hsap : x=[10.0, 19.0], y=[26.0, 35.0]
  Mmus : x=[10.0, 18.0], y=[40.0, 46.0]
EOT
}

# --------------------------
# USER DEFAULTS (optional edit)
# --------------------------
CONDA_ENV="intronmodel"
SPECIES="Mmus, Athal, Dmel, Hsap"
OUTPUT_PNG=""
INTERACTIVE="0"
X_MIN=""
X_MAX=""
Y_MIN=""
Y_MAX=""
INTRONMODEL_AUTO_TMUX="off"

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
		if [[ -z "${X_MIN}" || -z "${X_MAX}" || -z "${Y_MIN}" || -z "${Y_MAX}" ]]; then
			echo "[plot_eval.sh] Unknown species '${sp}'." >&2
			echo "[plot_eval.sh] Use one of: Athal, Dmel, Hsap, Mmus; or provide all of" >&2
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


trim_species_name() {
	local raw_name="$1"

	echo "${raw_name}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
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


run_species_list_sequential() {
	local out_png_override="$1"
	shift

	local exit_code=0
	local sp=""
	local sp_trimmed=""

	for sp in "$@"; do
		sp_trimmed="$(trim_species_name "${sp}")"
		if ! run_for_one_species "${sp_trimmed}" "${out_png_override}"; then
			echo "[plot_eval.sh] Failed for species ${sp_trimmed}" >&2
			exit_code=1
		fi
	done
	return "${exit_code}"
}


run_species_list_parallel() {
	local out_png_override="$1"
	shift

	local pids=()
	local species_names=()
	local sp=""
	local sp_trimmed=""
	local idx=0
	local exit_code=0

	for sp in "$@"; do
		sp_trimmed="$(trim_species_name "${sp}")"
		run_for_one_species "${sp_trimmed}" "${out_png_override}" &
		pids+=("$!")
		species_names+=("${sp_trimmed}")
	done

	for idx in "${!pids[@]}"; do
		if ! wait "${pids[$idx]}"; then
			echo "[plot_eval.sh] Failed for species ${species_names[$idx]}" >&2
			exit_code=1
		fi
	done
	return "${exit_code}"
}


run_species_selection() {
	local species_arg="$1"
	local out_png_override="$2"

	if [[ "${species_arg}" != *","* ]]; then
		run_for_one_species "${species_arg}" "${out_png_override}"
		return $?
	fi

	local spec_arr=()
	IFS=',' read -r -a spec_arr <<< "${species_arg}"
	if [[ "${INTERACTIVE}" == "1" ]]; then
		run_species_list_parallel "${out_png_override}" "${spec_arr[@]}"
		return $?
	fi
	run_species_list_sequential "${out_png_override}" "${spec_arr[@]}"
}


main() {
	local script_dir=""

	script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
	# shellcheck source=/dev/null
	source "${script_dir}/lib/common.sh"
	intronmodel_activate_conda "${CONDA_ENV}"
	intronmodel_init_paths "${BASH_SOURCE[0]}"

	# Auto-run inside tmux on SSH so jobs survive disconnects.
	# Set INTRONMODEL_AUTO_TMUX=off|on|auto (default: auto).
	intronmodel_enable_auto_tmux "${PROJECT_ROOT}" "$0" "${BASH_SOURCE[0]##*/}" "$@"

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

	run_species_selection "${SPECIES}" "${OUTPUT_PNG}"
}


if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
	main "$@"
fi
