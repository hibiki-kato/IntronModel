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


_intronmodel_prepare_conda_config() {
	local reason=""
	local config_home=""
	local conda_config_dir=""
	local user_condarc=""

	if [[ -n "${CONDARC:-}" ]]; then
		if [[ -e "${CONDARC}" && ! -r "${CONDARC}" ]]; then
			reason="unreadable CONDARC='${CONDARC}'"
		else
			return 0
		fi
	fi

	if [[ -n "${XDG_CONFIG_HOME:-}" ]]; then
		config_home="${XDG_CONFIG_HOME}"
	else
		config_home="${HOME}/.config"
	fi
	conda_config_dir="${config_home}/conda"
	user_condarc="${conda_config_dir}/.condarc"

	if [[ -z "${reason}" ]]; then
		if [[ -d "${config_home}" && ! -x "${config_home}" ]]; then
			reason="unsearchable config home '${config_home}'"
		elif [[ -d "${conda_config_dir}" && ! -x "${conda_config_dir}" ]]; then
			reason="unsearchable conda config dir '${conda_config_dir}'"
		elif [[ -e "${user_condarc}" && ! -r "${user_condarc}" ]]; then
			reason="unreadable conda config '${user_condarc}'"
		fi
	fi

	if [[ -z "${reason}" ]]; then
		return 0
	fi

	local fallback_root
	fallback_root="${TMPDIR:-/tmp}/intronmodel-conda-${USER:-$(id -un)}"
	local fallback_condarc="${fallback_root}/.condarc"
	mkdir -p "${fallback_root}"
	if [[ ! -f "${fallback_condarc}" ]]; then
		: >"${fallback_condarc}"
	fi
	export CONDARC="${fallback_condarc}"
	echo "[common.sh] ${reason}; using CONDARC='${CONDARC}'." >&2
}


intronmodel_activate_conda() {
	local env_name="${1:-intronmodel}"
	local conda_base=""
	local conda_exec_path=""
	local candidate
	local fallback_paths=()

	_intronmodel_prepare_conda_config

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
	intronmodel_configure_compile_defaults
	intronmodel_configure_hf_cache
}


intronmodel_configure_compile_defaults() {
	if [[ -z "${INTRONMODEL_TORCH_COMPILE_STRATEGY:-}" ]]; then
		export INTRONMODEL_TORCH_COMPILE_STRATEGY="default-then-off"
	fi
	if [[ -z "${INTRONMODEL_TORCH_COMPILE_STICKY_MODE:-}" ]]; then
		export INTRONMODEL_TORCH_COMPILE_STICKY_MODE="reduce-overhead"
	fi
	if [[ -z "${INTRONMODEL_TORCH_COMPILE_DISABLED_MODES:-}" ]]; then
		export INTRONMODEL_TORCH_COMPILE_DISABLED_MODES="max-autotune"
	fi
}


intronmodel_configure_hf_cache() {
	local default_cache_root
	local default_config_root
	default_cache_root="${TMPDIR:-/tmp}/intronmodel-cache-${USER}"
	default_config_root="${default_cache_root}/config"

	# Use one writable cache root for Hugging Face artifacts when callers do not
	# set explicit cache paths. This avoids permission issues on shared HOME/AFS
	# for Hugging Face, Matplotlib, and torch.compile artifact caches.
	if [[ -z "${XDG_CACHE_HOME:-}" ]]; then
		XDG_CACHE_HOME="${default_cache_root}"
	fi
	mkdir -p "${XDG_CACHE_HOME}"
	export XDG_CACHE_HOME

	if [[ -z "${XDG_CONFIG_HOME:-}" ]]; then
		XDG_CONFIG_HOME="${default_config_root}"
	fi
	mkdir -p "${XDG_CONFIG_HOME}"
	export XDG_CONFIG_HOME

	if [[ -z "${MPLCONFIGDIR:-}" ]]; then
		MPLCONFIGDIR="${XDG_CONFIG_HOME}/matplotlib"
	fi
	mkdir -p "${MPLCONFIGDIR}"
	export MPLCONFIGDIR

	if [[ -z "${TORCHINDUCTOR_CACHE_DIR:-}" ]]; then
		TORCHINDUCTOR_CACHE_DIR="${XDG_CACHE_HOME}/torchinductor"
	fi
	mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"
	export TORCHINDUCTOR_CACHE_DIR

	if [[ -z "${TRITON_CACHE_DIR:-}" ]]; then
		TRITON_CACHE_DIR="${XDG_CACHE_HOME}/triton"
	fi
	mkdir -p "${TRITON_CACHE_DIR}"
	export TRITON_CACHE_DIR

	if [[ -z "${HF_HOME:-}" ]]; then
		HF_HOME="${XDG_CACHE_HOME}/huggingface"
	fi
	mkdir -p "${HF_HOME}"
	export HF_HOME

	if [[ -z "${TRANSFORMERS_CACHE:-}" ]]; then
		TRANSFORMERS_CACHE="${HF_HOME}/hub"
	fi
	mkdir -p "${TRANSFORMERS_CACHE}"
	export TRANSFORMERS_CACHE

	if [[ -z "${HF_MODULES_CACHE:-}" ]]; then
		HF_MODULES_CACHE="${HF_HOME}/modules"
	fi
	mkdir -p "${HF_MODULES_CACHE}"
	export HF_MODULES_CACHE
}


intronmodel_format_elapsed() {
	local total_seconds="$1"
	local hours=$((total_seconds / 3600))
	local minutes=$(((total_seconds % 3600) / 60))
	local seconds=$((total_seconds % 60))
	printf '%02d:%02d:%02d' "${hours}" "${minutes}" "${seconds}"
}


intronmodel_format_eta_epoch() {
	local epoch_seconds="$1"

	if date -d "@${epoch_seconds}" '+%m/%d %-H:%M' >/dev/null 2>&1; then
		date -d "@${epoch_seconds}" '+%m/%d %-H:%M'
		return 0
	fi
	if date -r "${epoch_seconds}" '+%m/%d %-H:%M' >/dev/null 2>&1; then
		date -r "${epoch_seconds}" '+%m/%d %-H:%M'
		return 0
	fi

	python3 - "${epoch_seconds}" <<'PY'
from __future__ import annotations

from datetime import datetime
import sys

epoch_seconds = int(sys.argv[1])
eta_dt = datetime.fromtimestamp(epoch_seconds)
print(f"{eta_dt:%m/%d} {eta_dt.hour}:{eta_dt:%M}")
PY
}


intronmodel_build_eta_process_title() {
	local eta_label="$1"

	printf 'ETA:%s\n' "${eta_label}"
}


intronmodel_eta_prefix() {
	local eta_scope="$1"

	if [[ "${eta_scope}" == "gpu" ]]; then
		printf '%s\n' "GPU_free_in"
		return 0
	fi
	printf '%s\n' "ETA_remaining"
}


intronmodel_resolve_eta_scope() {
	local script_tag="$1"
	local gpu_ids_setting="$2"
	local parallel_setting="$3"
	local device_setting="$4"
	local job_count="$5"
	local py_bin="${6:-python3}"

	if [[ ! "${job_count}" =~ ^[0-9]+$ ]]; then
		echo "[${script_tag}] job count must be an integer." >&2
		return 1
	fi

	local -a gpu_ids=()
	mapfile -t gpu_ids < <(
		intronmodel_resolve_gpu_ids \
			"${script_tag}" \
			"${gpu_ids_setting}" \
			"${device_setting}" \
			"${py_bin}"
	)
	local gpu_slot_count="${#gpu_ids[@]}"
	local parallel_slot_count
	parallel_slot_count="$(
		intronmodel_resolve_parallel_slots \
			"${script_tag}" \
			"${parallel_setting}" \
			"${gpu_slot_count}"
	)" || return 1

	if (( gpu_slot_count <= 0 )); then
		printf '%s\n' "species"
		return 0
	fi
	if (( parallel_slot_count < job_count )); then
		printf '%s\n' "gpu"
		return 0
	fi
	printf '%s\n' "species"
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
	shift 3
	# shellcheck source=/dev/null
	source "${project_root}/run/lib/auto_tmux.sh"
	intronmodel_auto_tmux "${entrypoint}" "${script_name}" "$@"
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


intronmodel_run_with_process_title() {
	local process_title="${1-}"
	shift || true

	if [[ $# -eq 0 ]]; then
		echo "[common.sh] intronmodel_run_with_process_title requires a command." >&2
		return 2
	fi
	if [[ -z "${process_title}" ]]; then
		"$@"
		return $?
	fi

	(
		export INTRONMODEL_PROCESS_TITLE="${process_title}"
		"$@"
	)
}


intronmodel_json_string_or_null() {
	local py_bin="$1"
	local raw_value="${2-}"

	if [[ -z "${raw_value}" ]]; then
		printf '%s\n' "null"
		return 0
	fi

	"${py_bin}" - "${raw_value}" <<'PY'
from __future__ import annotations

import json
import sys

print(json.dumps(sys.argv[1], ensure_ascii=False))
PY
}


intronmodel_resolve_species_template() {
	local raw_value="${1-}"
	local species="$2"
	local resolved="${raw_value}"

	resolved="${resolved//\$\{SPECIES\}/${species}}"
	resolved="${resolved//\{SPECIES\}/${species}}"
	resolved="${resolved//\{species\}/${species}}"
	printf '%s\n' "${resolved}"
}


intronmodel_resolve_and_validate_train_paths() {
	local script_tag="$1"
	local species="$2"
	local train_pos_path="${3-}"
	local train_neg_path="${4-}"

	local resolved_pos=""
	local resolved_neg=""
	if [[ -n "${train_pos_path}" ]]; then
		resolved_pos="$(
			intronmodel_resolve_species_template "${train_pos_path}" "${species}"
		)"
	fi
	if [[ -n "${train_neg_path}" ]]; then
		resolved_neg="$(
			intronmodel_resolve_species_template "${train_neg_path}" "${species}"
		)"
	fi

	if [[ -z "${resolved_pos}" && -z "${resolved_neg}" ]]; then
		printf '\t\n'
		return 0
	fi
	if [[ -z "${resolved_pos}" || -z "${resolved_neg}" ]]; then
		echo "[${script_tag}] TRAIN_POS_PATH and TRAIN_NEG_PATH must be set together." >&2
		return 1
	fi
	if [[ ! -f "${resolved_pos}" ]]; then
		echo "[${script_tag}] TRAIN_POS_PATH not found for species=${species}:" \
			"${resolved_pos}" >&2
		return 1
	fi
	if [[ ! -f "${resolved_neg}" ]]; then
		echo "[${script_tag}] TRAIN_NEG_PATH not found for species=${species}:" \
			"${resolved_neg}" >&2
		return 1
	fi
	printf '%s\t%s\n' "${resolved_pos}" "${resolved_neg}"
}


intronmodel_resolve_pair_synthesize_defaults() {
	local species="$1"
	local synthesize_mode="$2"
	local tag_value="${3-}"
	local train_pos_path="${4-}"
	local train_neg_path="${5-}"
	local normalized_mode

	normalized_mode="$(printf '%s' "${synthesize_mode}" | tr '[:upper:]' '[:lower:]' \
		| xargs)"
	if [[ "${normalized_mode}" != "on" ]]; then
		printf '%s\t%s\t%s\n' "${tag_value}" "${train_pos_path}" "${train_neg_path}"
		return 0
	fi

	local resolved_tag="${tag_value}"
	if [[ -z "${resolved_tag}" ]]; then
		resolved_tag="synth"
	elif [[ "${resolved_tag}" != *"synth"* ]]; then
		resolved_tag="${resolved_tag}_synth"
	fi

	local resolved_pos="${train_pos_path}"
	local resolved_neg="${train_neg_path}"
	if [[ -z "${resolved_pos}" ]]; then
		resolved_pos="${DATA_ROOT}/${species}/raw/100bp.err"
	fi
	if [[ -z "${resolved_neg}" ]]; then
		resolved_neg="${DATA_ROOT}/${species}/processed/100bp_mixed_one_side.neg.err"
	fi

	printf '%s\t%s\t%s\n' "${resolved_tag}" "${resolved_pos}" "${resolved_neg}"
}


intronmodel_resolve_pair_best_config_filename() {
	local synthesize_mode="${1-}"
	local normalized_mode

	normalized_mode="$(printf '%s' "${synthesize_mode}" | tr '[:upper:]' '[:lower:]' \
		| xargs)"
	if [[ "${normalized_mode}" == "on" ]]; then
		printf '%s\n' "best_synth_config.json"
		return 0
	fi
	printf '%s\n' "best_config.json"
}


intronmodel_resolve_synth_tuning_model_name() {
	local base_model_name="$1"
	local synthesize_mode="${2-}"
	local normalized_mode

	normalized_mode="$(printf '%s' "${synthesize_mode}" | tr '[:upper:]' '[:lower:]' \
		| xargs)"
	if [[ "${normalized_mode}" == "on" ]]; then
		printf '%s_synth\n' "${base_model_name}"
		return 0
	fi
	printf '%s\n' "${base_model_name}"
}


intronmodel_resolve_pair_tuning_model_name() {
	intronmodel_resolve_synth_tuning_model_name "cnn_v2_pair" "${1-}"
}


intronmodel_resolve_pair_best_config_path() {
	local data_root="$1"
	local species="$2"
	local tuning_model_name="$3"
	local synthesize_mode="${4-}"
	local best_config_filename

	best_config_filename="$(
		intronmodel_resolve_pair_best_config_filename "${synthesize_mode}"
	)"
	printf '%s\n' \
		"${data_root}/${species}/tuning/${tuning_model_name}/pair/${best_config_filename}"
}


intronmodel_resolve_seed_list() {
	local script_tag="$1"
	local base_seed="$2"
	local raw_seed_list="${3-}"
	local py_bin="${4:-python3}"

	"${py_bin}" - \
		"${script_tag}" \
		"${base_seed}" \
		"${raw_seed_list}" <<'PY'
from __future__ import annotations

import re
import sys


def _fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(2)


def _parse_seed(text: str, *, name: str) -> int:
    stripped = text.strip()
    if not re.fullmatch(r"\d+", stripped):
        _fail(f"[{script_tag}] {name} must be a non-negative integer.")
    return int(stripped)


script_tag = sys.argv[1]
base_seed = _parse_seed(sys.argv[2], name="BASE_SEED")
raw_seed_list = sys.argv[3].strip()

if raw_seed_list:
    print(
        f"[{script_tag}] SEED_LIST is ignored. "
        "Using BASE_SEED only for a single tuning run.",
        file=sys.stderr,
    )

print(base_seed)
PY
}


intronmodel_resolve_gpu_ids() {
	local script_tag="$1"
	local gpu_ids_setting="$2"
	local device_setting="${3:-auto}"
	local py_bin="${4:-python3}"

	"${py_bin}" - \
		"${script_tag}" \
		"${gpu_ids_setting}" \
		"${device_setting}" <<'PY'
from __future__ import annotations

import os
import re
import subprocess
import sys


def _fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(2)


script_tag = sys.argv[1]
raw_setting = sys.argv[2].strip()
device_setting = sys.argv[3].strip().lower()

if device_setting not in {"", "auto", "cuda", "cpu", "mps"}:
    _fail(f"[{script_tag}] DEVICE must be auto|cuda|cpu|mps.")
if device_setting in {"cpu", "mps"}:
    raise SystemExit(0)

if raw_setting == "" or raw_setting.lower() == "auto":
    env_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env_visible != "":
        selected = [part.strip() for part in env_visible.split(",") if part.strip()]
        print("\n".join(selected))
        raise SystemExit(0)

    command = [
        "nvidia-smi",
        "--query-gpu=index",
        "--format=csv,noheader",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        raise SystemExit(0)
    if result.returncode != 0:
        raise SystemExit(0)
    selected = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    print("\n".join(selected))
    raise SystemExit(0)

selected: list[str] = []
for token in raw_setting.split(","):
    item = token.strip()
    if item == "":
        continue
    if not re.fullmatch(r"\d+", item):
        _fail(f"[{script_tag}] GPU_IDS must be auto or comma-separated integers.")
    selected.append(str(int(item)))

if not selected:
    _fail(f"[{script_tag}] GPU_IDS resolved to an empty set.")

print("\n".join(selected))
PY
}


intronmodel_resolve_parallel_slots() {
	local script_tag="$1"
	local parallel_setting="$2"
	local available_slots="$3"

	if [[ ! "${available_slots}" =~ ^[0-9]+$ ]]; then
		echo "[${script_tag}] available slot count must be an integer." >&2
		return 1
	fi
	if (( available_slots <= 0 )); then
		printf '0\n'
		return 0
	fi

	local normalized
	normalized="$(printf '%s' "${parallel_setting}" | tr '[:upper:]' '[:lower:]' | xargs)"
	if [[ -z "${normalized}" || "${normalized}" == "auto" ]]; then
		printf '%s\n' "${available_slots}"
		return 0
	fi
	if ! [[ "${normalized}" =~ ^[0-9]+$ ]]; then
		echo "[${script_tag}] MAX_PARALLEL_TRIALS must be auto or a positive integer." >&2
		return 1
	fi

	local resolved="${normalized}"
	if (( resolved <= 0 )); then
		echo "[${script_tag}] MAX_PARALLEL_TRIALS must be > 0." >&2
		return 1
	fi
	if (( resolved > available_slots )); then
		resolved="${available_slots}"
	fi
	printf '%s\n' "${resolved}"
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
