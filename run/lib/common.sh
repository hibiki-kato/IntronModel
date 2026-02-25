#!/usr/bin/env bash

intronmodel_activate_conda() {
	local env_name="${1:-intronmodel}"

	if command -v conda >/dev/null 2>&1; then
		local conda_base
		conda_base="$(conda info --base 2>/dev/null || true)"
		if [[ -n "${conda_base}" && -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
			# shellcheck source=/dev/null
			source "${conda_base}/etc/profile.d/conda.sh"
		fi
	fi
	conda activate "${env_name}"
}


intronmodel_init_paths() {
	local script_path="$1"
	SCRIPT_DIR="$(cd "$(dirname "${script_path}")" && pwd)"
	PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
	DATA_ROOT="${INTRONMODEL_DATA_ROOT:-${PROJECT_ROOT}/data}"
	MODEL_ROOT="${INTRONMODEL_MODEL_ROOT:-${PROJECT_ROOT}/model}"
	export INTRONMODEL_MODEL_ROOT="${MODEL_ROOT}"
	export INTRONMODEL_DATA_ROOT="${DATA_ROOT}"
}


intronmodel_format_elapsed() {
	local total_seconds="$1"
	local hours=$((total_seconds / 3600))
	local minutes=$(((total_seconds % 3600) / 60))
	local seconds=$((total_seconds % 60))
	printf '%02d:%02d:%02d' "${hours}" "${minutes}" "${seconds}"
}


intronmodel_start_timer() {
	INTRONMODEL_SCRIPT_TAG="$1"
	INTRONMODEL_SCRIPT_START_EPOCH="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
	INTRONMODEL_SCRIPT_START_SECONDS="${SECONDS}"
}


intronmodel_enable_auto_tmux() {
	local project_root="$1"
	local entrypoint="$2"
	local script_name="$3"
	# shellcheck source=/dev/null
	source "${project_root}/run/_auto_tmux.sh"
	intronmodel_auto_tmux "${entrypoint}" "${script_name}"
}


intronmodel_print_timing() {
	local exit_code="$?"
	local script_end_epoch
	script_end_epoch="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
	local elapsed_seconds=$((SECONDS - INTRONMODEL_SCRIPT_START_SECONDS))
	local elapsed_hms
	elapsed_hms="$(intronmodel_format_elapsed "${elapsed_seconds}")"
	echo "[${INTRONMODEL_SCRIPT_TAG}] timing: "\
		"start=${INTRONMODEL_SCRIPT_START_EPOCH} end=${script_end_epoch} "\
		"elapsed=${elapsed_hms} (${elapsed_seconds}s) exit=${exit_code}"
	return "${exit_code}"
}


intronmodel_resolve_species_case() {
	local raw_species="$1"
	local data_root="$2"
	local script_tag="$3"

	if [[ -d "${data_root}/${raw_species}" ]]; then
		printf '%s\n' "${raw_species}"
		return 0
	fi

	local matches=()
	mapfile -t matches < <(
		find "${data_root}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
			| awk -v target="${raw_species}" 'tolower($0) == tolower(target)'
	)
	if [[ ${#matches[@]} -eq 1 ]]; then
		if [[ -n "${script_tag}" ]]; then
			echo "[${script_tag}] species case normalized: "\
				"'${raw_species}' -> '${matches[0]}'" >&2
		fi
		printf '%s\n' "${matches[0]}"
		return 0
	fi
	if [[ ${#matches[@]} -gt 1 ]]; then
		if [[ -n "${script_tag}" ]]; then
			echo "[${script_tag}] ambiguous species '${raw_species}'." >&2
			printf '[%s] case-insensitive matches: %s\n' \
				"${script_tag}" "${matches[*]}" >&2
			return 1
		fi
		printf '%s\n' "${raw_species}"
		return 0
	fi
	printf '%s\n' "${raw_species}"
	return 0
}


intronmodel_resolve_tune_targets() {
	local raw_targets="$1"
	local script_tag="$2"
	local parts=()
	local resolved=()

	IFS=',' read -r -a parts <<< "${raw_targets}"
	for part in "${parts[@]}"; do
		local target
		target="$(printf '%s' "${part}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
		if [[ -z "${target}" ]]; then
			continue
		fi
		if [[ "${target}" != "donor" && "${target}" != "acceptor" ]]; then
			echo "[${script_tag}] invalid target: ${target}" >&2
			echo "[${script_tag}] TUNE_TARGETS must contain donor and/or acceptor." >&2
			return 1
		fi
		resolved+=("${target}")
	done

	if [[ ${#resolved[@]} -eq 0 ]]; then
		echo "[${script_tag}] no valid tuning targets configured." >&2
		return 1
	fi
	printf '%s\n' "${resolved[@]}"
}


intronmodel_resolve_python_bin() {
	local script_tag="$1"
	if command -v python3 >/dev/null 2>&1; then
		printf '%s\n' "python3"
		return 0
	fi
	if command -v python >/dev/null 2>&1; then
		printf '%s\n' "python"
		return 0
	fi
	echo "[${script_tag}] python interpreter not found (python3/python)." >&2
	return 1
}
