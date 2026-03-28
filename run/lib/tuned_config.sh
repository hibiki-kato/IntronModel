#!/usr/bin/env bash

intronmodel_normalize_use_tuned_mode() {
	local raw_mode="$1"
	local script_name="$2"
	local normalized
	normalized="$(echo "${raw_mode}" | tr '[:upper:]' '[:lower:]' | xargs)"
	case "${normalized}" in
		off | auto | required)
			printf '%s\n' "${normalized}"
			;;
		*)
			echo "[${script_name}] USE_TUNED_HPARAMS must be off|auto|required." >&2
			return 1
			;;
	esac
}


intronmodel_resolve_tuned_target() {
	local configured_target="$1"
	local default_target="$2"
	local normalized
	normalized="$(echo "${configured_target}" | tr '[:upper:]' '[:lower:]' | xargs)"
	if [[ -n "${normalized}" && "${normalized}" != "auto" ]]; then
		printf '%s\n' "${normalized}"
		return 0
	fi
	printf '%s\n' "${default_target}"
}


intronmodel_resolve_tuned_config_path() {
	local data_root="$1"
	local species="$2"
	local tuned_model_name="$3"
	local tuned_target="$4"
	local explicit_path="$5"
	local shared_path="$6"
	local best_config_filename="${7:-best_config.json}"

	if [[ -n "${explicit_path}" ]]; then
		printf '%s\n' "${explicit_path}"
		return 0
	fi

	local task_path="${data_root}/${species}/tuning/${tuned_model_name}/${tuned_target}/${best_config_filename}"
	if [[ -f "${task_path}" ]]; then
		printf '%s\n' "${task_path}"
		return 0
	fi

	local legacy_tuned_model_name="${tuned_model_name}"
	case "${tuned_model_name}" in
		cnn_pair_v2)
			legacy_tuned_model_name="cnn_v2_pair"
			;;
	esac
	if [[ "${legacy_tuned_model_name}" != "${tuned_model_name}" ]]; then
		local legacy_task_path="${data_root}/${species}/tuning/${legacy_tuned_model_name}/${tuned_target}/${best_config_filename}"
		if [[ -f "${legacy_task_path}" ]]; then
			printf '%s\n' "${legacy_task_path}"
			return 0
		fi
	fi

	local use_task_only_configs="0"
	case "${tuned_model_name}" in
		cnn_v2 | cnn_v2_pair | cnn_pair_v2)
			use_task_only_configs="1"
			;;
	esac

	if [[ "${use_task_only_configs}" != "1" && -n "${shared_path}" ]]; then
		local shared_candidate="${shared_path}"
		if [[ -f "${shared_candidate}" ]]; then
			printf '%s\n' "${shared_candidate}"
			return 0
		fi
	fi

	if [[ "${use_task_only_configs}" != "1" && "${best_config_filename}" == "best_config.json" ]]; then
		local legacy_path="${data_root}/${species}/tuning/${tuned_model_name}/best_config.json"
		if [[ -f "${legacy_path}" ]]; then
			printf '%s\n' "${legacy_path}"
			return 0
		fi
	fi

	return 0
}


intronmodel_load_tuned_overrides() {
	local config_path="$1"
	python3 - "${config_path}" <<'PY'
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def _scalar_to_text(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite float value in tuned config.")
        return format(value, ".15g")
    return str(value)


def _mask_to_sequence_transform(value: object) -> str:
    if isinstance(value, bool):
        normalized = "on" if value else "off"
    else:
        normalized = str(value).strip().lower()
    if normalized in {"on", "1", "true", "yes"}:
        return "mask_outside_intron_n"
    if normalized in {"off", "0", "false", "no"}:
        return "none"
    raise ValueError("mask must be on or off.")


config_path = Path(sys.argv[1]).resolve()
payload = json.loads(config_path.read_text(encoding="utf-8"))
if not isinstance(payload, dict):
    raise ValueError("best_config payload must be an object.")
status = str(payload.get("status", "")).strip().lower()
if status != "ok":
    raise ValueError(f"Expected status='ok', got: {status or '<missing>'}")

context = payload.get("hparam_context")
fixed_run_args = None
if isinstance(context, dict):
    fixed_run_args = context.get("fixed_run_args")
sampled_params = payload.get("sampled_params")
if not isinstance(sampled_params, dict):
    raise ValueError("sampled_params is missing or invalid.")
pair_mode_value = None
if isinstance(fixed_run_args, dict):
    pair_mode_value = fixed_run_args.get("pair_mode")
if pair_mode_value is None and isinstance(sampled_params, dict):
    pair_mode_value = sampled_params.get("pair_mode")
independent_mode = (
    isinstance(pair_mode_value, str)
    and pair_mode_value.strip().lower() == "independent"
)
if isinstance(fixed_run_args, dict):
    for key in sorted(fixed_run_args):
        if independent_mode and key in {"mask", "sequence_transform"}:
            continue
        value = fixed_run_args[key]
        if value is None:
            continue
        print(f"{key}\t{_scalar_to_text(value)}")
sequence_transform_value = sampled_params.pop("sequence_transform", None)
mask_value = sampled_params.pop("mask", None)
if independent_mode:
    print("sequence_transform\tnone")
elif mask_value is not None:
    print(f"sequence_transform\t{_mask_to_sequence_transform(mask_value)}")
elif sequence_transform_value is not None:
    print(
        "sequence_transform\t"
        f"{_scalar_to_text(sequence_transform_value)}"
    )
for key in sorted(sampled_params):
    value = sampled_params[key]
    if value is None:
        continue
    print(f"{key}\t{_scalar_to_text(value)}")
PY
}
