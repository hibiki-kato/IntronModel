#!/usr/bin/env bash

intronmodel_abort_parallel_run() {
	trap - INT TERM HUP
	kill -TERM 0 2>/dev/null || true
	exit 130
}

intronmodel_append_arg_if_set() {
	local args_name="$1"
	local flag="$2"
	local value="$3"
	if [[ -n "${value}" ]]; then
		local -n args_ref="${args_name}"
		args_ref+=("--${flag}" "${value}")
	fi
}

intronmodel_append_flag_if_truthy() {
	local args_name="$1"
	local flag="$2"
	local value="$3"
	local normalized
	normalized="$(printf '%s' "${value}" | tr '[:upper:]' '[:lower:]' | xargs)"
	case "${normalized}" in
		1 | true | on | yes)
			local -n args_ref="${args_name}"
			args_ref+=("--${flag}")
			;;
	esac
}

intronmodel_run_model_with_optional_gpu() {
	local project_root="$1"
	local assigned_gpu_id="$2"
	local args_name="$3"
	local -n args_ref="${args_name}"

	local pythonpath="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
	if [[ -n "${assigned_gpu_id}" ]]; then
		CUDA_VISIBLE_DEVICES="${assigned_gpu_id}" \
			PYTHONPATH="${pythonpath}" \
			python3 "${project_root}/src/run_model.py" "${args_ref[@]}"
		return $?
	fi
	PYTHONPATH="${pythonpath}" \
		python3 "${project_root}/src/run_model.py" "${args_ref[@]}"
}

intronmodel_normalize_json_object_file() {
	local python_bin="$1"
	local json_path="$2"

	"${python_bin}" - "${json_path}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if not isinstance(payload, dict):
    raise ValueError("Search-space file must contain a JSON object.")
print(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
PY
}

intronmodel_run_double_descent_plot() {
	local python_bin="$1"
	local project_root="$2"
	local species_name="$3"
	local target_name="$4"
	local model_name="$5"

	"${python_bin}" "${project_root}/src/tools/plot_tuning_double_descent.py" \
		--project_root "${project_root}" \
		--species "${species_name}" \
		--target "${target_name}" \
		--model "${model_name}" || true
}

intronmodel_append_unique_values() {
	local array_name="$1"
	shift || true
	local -n target_ref="${array_name}"
	local candidate
	local existing
	local found

	for candidate in "$@"; do
		if [[ -z "${candidate}" ]]; then
			continue
		fi
		found=0
		for existing in "${target_ref[@]}"; do
			if [[ "${existing}" == "${candidate}" ]]; then
				found=1
				break
			fi
		done
		if [[ "${found}" -eq 0 ]]; then
			target_ref+=("${candidate}")
		fi
	done
}

intronmodel_remove_value_from_csv() {
	local value_csv="$1"
	local remove_value="$2"
	local parts=()
	local kept=()
	local value

	IFS=',' read -r -a parts <<< "${value_csv}"
	for value in "${parts[@]}"; do
		if [[ -z "${value}" || "${value}" == "${remove_value}" ]]; then
			continue
		fi
		kept+=("${value}")
	done
	(
		IFS=,
		printf '%s\n' "${kept[*]}"
	)
}

intronmodel_run_species_jobs() {
	local script_tag="$1"
	local species_array_name="$2"
	local gpu_array_name="$3"
	local parallel_slot_count="$4"
	local runner_fn="$5"

	local -n species_list="${species_array_name}"
	local -n gpu_id_list="${gpu_array_name}"

	if [[ ${#species_list[@]} -eq 0 ]]; then
		echo "[${script_tag}] SPECIES resolved to an empty list." >&2
		return 1
	fi

	if [[ ${#species_list[@]} -le 1 || ${#gpu_id_list[@]} -le 1 || ${parallel_slot_count} -le 1 ]]; then
		local serial_gpu_id=""
		if [[ ${#gpu_id_list[@]} -gt 0 ]]; then
			serial_gpu_id="${gpu_id_list[0]}"
		fi
		local species_raw
		local species_value
		local serial_code
		for species_raw in "${species_list[@]}"; do
			species_value="$(printf '%s' "${species_raw}" | xargs)"
			if [[ -z "${species_value}" ]]; then
				continue
			fi
			if "${runner_fn}" "${species_value}" "${serial_gpu_id}"; then
				continue
			else
				serial_code=$?
			fi
			echo "[${script_tag}] species failed: ${species_value} gpu=${serial_gpu_id} exit=${serial_code}" >&2
			return "${serial_code}"
		done
		return 0
	fi

	local selected_gpu_ids=("${gpu_id_list[@]:0:${parallel_slot_count}}")
	local gpu_csv
	gpu_csv="$(IFS=,; echo "${selected_gpu_ids[*]}")"
	echo "[${script_tag}] species-parallel run across GPUs: ${gpu_csv}"

	declare -A pid_to_species=()
	declare -A pid_to_gpu=()
	local available_gpu_ids=("${selected_gpu_ids[@]}")
	local pending_species=("${species_list[@]}")
	local running_count=0
	local stop_submitting=0
	local first_error_code=0
	local species_raw
	local species_value
	local gpu_id
	local pid
	local completed_pid
	local completed_code
	local completed_species
	local completed_gpu

	while [[ ${#pending_species[@]} -gt 0 || ${running_count} -gt 0 ]]; do
		while [[ ${#pending_species[@]} -gt 0 && ${#available_gpu_ids[@]} -gt 0 && ${stop_submitting} -eq 0 ]]; do
			species_raw="${pending_species[0]}"
			pending_species=("${pending_species[@]:1}")
			species_value="$(printf '%s' "${species_raw}" | xargs)"
			if [[ -z "${species_value}" ]]; then
				continue
			fi
			gpu_id="${available_gpu_ids[0]}"
			available_gpu_ids=("${available_gpu_ids[@]:1}")
			echo "[${script_tag}] species dispatch: ${species_value} -> gpu=${gpu_id}"
			"${runner_fn}" "${species_value}" "${gpu_id}" &
			pid=$!
			pid_to_species["${pid}"]="${species_value}"
			pid_to_gpu["${pid}"]="${gpu_id}"
			running_count=$((running_count + 1))
		done

		if [[ ${running_count} -eq 0 ]]; then
			break
		fi

		if wait -n -p completed_pid; then
			completed_code=0
		else
			completed_code=$?
		fi
		completed_species="${pid_to_species[$completed_pid]:-}"
		completed_gpu="${pid_to_gpu[$completed_pid]:-}"
		unset "pid_to_species[${completed_pid}]"
		unset "pid_to_gpu[${completed_pid}]"
		if [[ -n "${completed_gpu}" ]]; then
			available_gpu_ids+=("${completed_gpu}")
		fi
		running_count=$((running_count - 1))
		if [[ -n "${completed_species}" ]]; then
			echo "[${script_tag}] species complete: ${completed_species} gpu=${completed_gpu} exit=${completed_code}"
		fi
		if [[ ${completed_code} -ne 0 && ${first_error_code} -eq 0 ]]; then
			first_error_code="${completed_code}"
			stop_submitting=1
		fi
	done
	return "${first_error_code}"
}
