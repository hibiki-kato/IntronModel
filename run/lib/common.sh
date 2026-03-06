#!/usr/bin/env bash

_intronmodel_source_conda_sh() {
	local conda_sh_path="$1"
	if [[ -z "${conda_sh_path}" || ! -f "${conda_sh_path}" ]]; then
		return 1
	fi
	# shellcheck source=/dev/null
	source "${conda_sh_path}"
	return 0
}


_intronmodel_conda_base_from_exec() {
	local conda_exec_path="$1"
	if [[ -z "${conda_exec_path}" ]]; then
		return 1
	fi
	local resolved_path
	resolved_path="$(readlink -f "${conda_exec_path}" 2>/dev/null || true)"
	if [[ -z "${resolved_path}" ]]; then
		resolved_path="${conda_exec_path}"
	fi
	local bin_dir
	bin_dir="$(dirname "${resolved_path}")"
	local bin_name
	bin_name="$(basename "${bin_dir}")"
	if [[ "${bin_name}" != "bin" && "${bin_name}" != "condabin" ]]; then
		return 1
	fi
	printf '%s\n' "$(dirname "${bin_dir}")"
	return 0
}


intronmodel_activate_conda() {
	local env_name="${1:-intronmodel}"
	local conda_base=""
	local conda_exec_path=""
	local candidate
	local fallback_paths=()

	if [[ -n "${INTRONMODEL_CONDA_SH:-}" ]]; then
		fallback_paths+=("${INTRONMODEL_CONDA_SH}")
	fi
	if [[ -n "${CONDA_EXE:-}" ]]; then
		fallback_paths+=(
			"$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"
		)
	fi

	if command -v conda >/dev/null 2>&1; then
		conda_exec_path="$(command -v conda)"
		conda_base="$(
			CONDA_NO_PLUGINS=true conda info --base 2>/dev/null || true
		)"
		if [[ -z "${conda_base}" ]]; then
			conda_base="$(
				_intronmodel_conda_base_from_exec "${conda_exec_path}" || true
			)"
		fi
		if [[ -n "${conda_base}" ]]; then
			fallback_paths+=("${conda_base}/etc/profile.d/conda.sh")
		fi
	fi

	fallback_paths+=(
		"${HOME}/miniforge3/etc/profile.d/conda.sh"
		"${HOME}/mambaforge/etc/profile.d/conda.sh"
		"${HOME}/miniconda3/etc/profile.d/conda.sh"
		"${HOME}/anaconda3/etc/profile.d/conda.sh"
		"/export/${USER}/miniforge3/etc/profile.d/conda.sh"
		"/export/${USER}/mambaforge/etc/profile.d/conda.sh"
		"/export/${USER}/miniconda3/etc/profile.d/conda.sh"
		"/export/${USER}/anaconda3/etc/profile.d/conda.sh"
	)

	for candidate in "${fallback_paths[@]}"; do
		_intronmodel_source_conda_sh "${candidate}" || true
		if command -v conda >/dev/null 2>&1 \
			&& conda activate "${env_name}" >/dev/null 2>&1; then
			return 0
		fi
	done
	if ! command -v conda >/dev/null 2>&1; then
		echo "[common.sh] conda command not found." \
			"Set INTRONMODEL_CONDA_SH or add conda to PATH." >&2
		return 127
	fi
	echo "[common.sh] failed to activate conda env '${env_name}'." \
		"Set INTRONMODEL_CONDA_SH explicitly if needed." >&2
	return 1
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


intronmodel_resolve_max_model_params() {
	local script_tag="$1"
	local setting="$2"
	local gpu_ids_setting="$3"
	local fallback_params="$4"
	local mem_fraction="$5"
	local reserve_mib="$6"
	local bytes_per_param="$7"
	local model_factor="$8"
	local py_bin="${9:-python3}"

	"${py_bin}" - \
		"${script_tag}" \
		"${setting}" \
		"${gpu_ids_setting}" \
		"${fallback_params}" \
		"${mem_fraction}" \
		"${reserve_mib}" \
		"${bytes_per_param}" \
		"${model_factor}" <<'PY'
from __future__ import annotations

import re
import subprocess
import sys


def _fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(2)


def _parse_positive_int(text: str, name: str) -> int:
    raw = text.strip()
    if not re.fullmatch(r"\d+", raw):
        _fail(f"[{script_tag}] {name} must be a positive integer or auto.")
    value = int(raw)
    if value <= 0:
        _fail(f"[{script_tag}] {name} must be > 0.")
    return value


def _parse_non_negative_int(text: str, name: str) -> int:
    raw = text.strip()
    if not re.fullmatch(r"\d+", raw):
        _fail(f"[{script_tag}] {name} must be a non-negative integer.")
    return int(raw)


def _parse_positive_float(text: str, name: str) -> float:
    raw = text.strip()
    try:
        value = float(raw)
    except ValueError:
        _fail(f"[{script_tag}] {name} must be a positive number.")
    if value <= 0.0:
        _fail(f"[{script_tag}] {name} must be > 0.")
    return value


def _query_gpu_total_mib() -> list[int]:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    totals: list[int] = []
    for line in result.stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            value = int(text)
        except ValueError:
            continue
        if value > 0:
            totals.append(value)
    return totals


def _resolve_selected_indices(raw: str, count: int) -> list[int]:
    text = raw.strip().lower()
    if text == "" or text == "auto":
        return list(range(count))

    selected: list[int] = []
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        if not re.fullmatch(r"\d+", item):
            _fail(f"[{script_tag}] GPU_IDS must be auto or comma-separated integers.")
        idx = int(item)
        if idx < 0 or idx >= count:
            _fail(
                f"[{script_tag}] GPU index out of range: {idx} "
                f"(detected {count} GPUs)."
            )
        selected.append(idx)

    if not selected:
        _fail(f"[{script_tag}] GPU_IDS resolved to an empty set.")
    return selected


script_tag = sys.argv[1]
setting = sys.argv[2].strip()
gpu_ids_setting = sys.argv[3]
fallback_params = _parse_positive_int(sys.argv[4], "MAX_MODEL_PARAMS fallback")
mem_fraction = _parse_positive_float(
    sys.argv[5],
    "MAX_MODEL_PARAMS_MEM_FRACTION",
)
if mem_fraction > 1.0:
    _fail(
        f"[{script_tag}] MAX_MODEL_PARAMS_MEM_FRACTION must be <= 1.0 "
        f"(got {mem_fraction})."
    )
reserve_mib = _parse_non_negative_int(
    sys.argv[6],
    "MAX_MODEL_PARAMS_RESERVE_MIB",
)
bytes_per_param = _parse_positive_float(
    sys.argv[7],
    "MAX_MODEL_PARAMS_BYTES_PER_PARAM",
)
model_factor = _parse_positive_float(
    sys.argv[8],
    "MAX_MODEL_PARAMS_MODEL_FACTOR",
)

if setting.lower() != "auto":
    resolved = _parse_positive_int(setting, "MAX_MODEL_PARAMS")
    print(resolved)
    print(
        f"[{script_tag}] MAX_MODEL_PARAMS fixed: {resolved}.",
        file=sys.stderr,
    )
    raise SystemExit(0)

totals_mib = _query_gpu_total_mib()
if not totals_mib:
    print(fallback_params)
    print(
        f"[{script_tag}] MAX_MODEL_PARAMS auto: nvidia-smi unavailable; "
        f"fallback={fallback_params}.",
        file=sys.stderr,
    )
    raise SystemExit(0)

selected_indices = _resolve_selected_indices(gpu_ids_setting, len(totals_mib))
selected_totals = [totals_mib[idx] for idx in selected_indices]
min_total_mib = min(selected_totals)

usable_mib = (float(min_total_mib) * mem_fraction) - float(reserve_mib)
if usable_mib <= 0.0:
    print(fallback_params)
    print(
        f"[{script_tag}] MAX_MODEL_PARAMS auto: usable VRAM <= 0; "
        f"fallback={fallback_params}.",
        file=sys.stderr,
    )
    raise SystemExit(0)

usable_bytes = usable_mib * 1024.0 * 1024.0
estimated_params = int((usable_bytes / bytes_per_param) * model_factor)
if estimated_params <= 0:
    print(fallback_params)
    print(
        f"[{script_tag}] MAX_MODEL_PARAMS auto: estimate <= 0; "
        f"fallback={fallback_params}.",
        file=sys.stderr,
    )
    raise SystemExit(0)

print(estimated_params)
print(
    f"[{script_tag}] MAX_MODEL_PARAMS auto: "
    f"selected_gpu_indices={selected_indices} "
    f"min_total_mib={min_total_mib} "
    f"mem_fraction={mem_fraction:.2f} reserve_mib={reserve_mib} "
    f"bytes_per_param={bytes_per_param:.1f} model_factor={model_factor:.3f} "
    f"resolved={estimated_params}.",
    file=sys.stderr,
)
PY
}
