#!/usr/bin/env bash

# Shared helpers for cross-species best-config fallback in tuning scripts.

trim_whitespace() {
	local raw="$1"
	# shellcheck disable=SC2001
	echo "${raw}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
}

extract_target_override() {
	local raw_map="$1"
	local target="$2"
	local -a items=()
	local item=""
	local key=""
	local value=""
	local fallback=""

	IFS=',' read -r -a items <<< "${raw_map}"
	for item in "${items[@]}"; do
		item="$(trim_whitespace "${item}")"
		if [[ -z "${item}" || "${item}" != *=* ]]; then
			continue
		fi
		key="$(trim_whitespace "${item%%=*}")"
		value="$(trim_whitespace "${item#*=}")"
		if [[ -z "${value}" ]]; then
			continue
		fi
		if [[ "${key}" == "*" ]]; then
			fallback="${value}"
			continue
		fi
		if [[ "${key}" == "${target}" ]]; then
			echo "${value}"
			return 0
		fi
	done
	echo "${fallback}"
}

collect_best_candidates() {
	local python_bin="$1"
	local data_root="$2"
	local model_name="$3"
	local target="$4"
	local species="$5"

	"${python_bin}" - "${data_root}" "${model_name}" "${target}" "${species}" <<'PY'
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def read_score(path: Path) -> float:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return float("-inf")
    if not isinstance(payload, dict):
        return float("-inf")
    status = payload.get("status")
    if status != "ok":
        return float("-inf")
    value = payload.get("objective_score")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return float("-inf")


data_root = Path(sys.argv[1])
model_name = sys.argv[2]
target = sys.argv[3]
excluded_species = sys.argv[4]

rows: list[tuple[float, str, Path]] = []
candidate_species_dirs: list[tuple[str, Path]] = []
for first_level in sorted(data_root.iterdir()):
    if not first_level.is_dir():
        continue
    if (first_level / "tuning").is_dir():
        candidate_species_dirs.append((first_level.name, first_level))
    for second_level in sorted(first_level.iterdir()):
        if not second_level.is_dir():
            continue
        if not (second_level / "tuning").is_dir():
            continue
        nested_species = f"{first_level.name}/{second_level.name}"
        candidate_species_dirs.append((nested_species, second_level))

for species, species_dir in candidate_species_dirs:
    if species == excluded_species:
        continue
    candidate = species_dir / "tuning" / model_name / target / "best_config.json"
    if not candidate.exists():
        continue
    score = read_score(candidate)
    rows.append((score, species, candidate.resolve()))

rows.sort(
    key=lambda row: (
        math.isfinite(row[0]),
        row[0] if math.isfinite(row[0]) else float("-inf"),
    ),
    reverse=True,
)
for score, species, path in rows:
    if math.isfinite(score):
        score_text = f"{score:.6f}"
    else:
        score_text = "nan"
    print(f"{species}\t{path}\t{score_text}")
PY
}

resolve_cross_species_best_seed() {
	local script_tag="$1"
	local python_bin="$2"
	local data_root="$3"
	local model_name="$4"
	local species="$5"
	local target="$6"
	local local_best_path="$7"
	local fallback_mode="$8"
	local override_map="$9"
	local preferred_species_csv="${10}"

	local mode
	mode="$(printf '%s' "${fallback_mode}" | tr '[:upper:]' '[:lower:]')"
	if [[ -z "${mode}" ]]; then
		mode="auto"
	fi
	if [[ "${mode}" == "off" || "${mode}" == "none" ]]; then
		echo ""
		return 0
	fi
	if [[ "${mode}" != "auto" && "${mode}" != "interactive" ]]; then
		echo "[${script_tag}] CROSS_SPECIES_BEST_MODE must be auto|interactive|off." >&2
		return 2
	fi

	if [[ -f "${local_best_path}" ]]; then
		echo ""
		return 0
	fi

	local manual_selector
	manual_selector="$(extract_target_override "${override_map}" "${target}")"

	local -a candidates=()
	if mapfile -t candidates < <(
		collect_best_candidates \
			"${python_bin}" \
			"${data_root}" \
			"${model_name}" \
			"${target}" \
			"${species}"
	); then
		:
	else
		echo "[${script_tag}] failed to collect cross-species best candidates." >&2
		echo ""
		return 0
	fi

	if [[ ${#candidates[@]} -eq 0 ]]; then
		echo "[${script_tag}] no cross-species best_config candidate: model=${model_name} target=${target}" >&2
		echo ""
		return 0
	fi

	echo "[${script_tag}] cross-species best candidates: model=${model_name} target=${target}" >&2
	local row=""
	local idx=0
	for row in "${candidates[@]}"; do
		IFS=$'\t' read -r cand_species cand_path cand_score <<< "${row}"
		echo "[${script_tag}]   [$idx] species=${cand_species} score=${cand_score} path=${cand_path}" >&2
		idx=$((idx + 1))
	done

	local resolved_path=""
	if [[ -n "${manual_selector}" ]]; then
		if [[ -f "${manual_selector}" ]]; then
			resolved_path="${manual_selector}"
		elif [[ -f "${data_root}/${manual_selector}/tuning/${model_name}/${target}/best_config.json" ]]; then
			resolved_path="${data_root}/${manual_selector}/tuning/${model_name}/${target}/best_config.json"
		else
			echo "[${script_tag}] invalid CROSS_SPECIES_BEST_OVERRIDE for ${target}: ${manual_selector}" >&2
			return 2
		fi
		echo "[${script_tag}] selected by override: ${resolved_path}" >&2
		echo "${resolved_path}"
		return 0
	fi

	if [[ "${mode}" == "interactive" ]]; then
		if [[ ! -t 0 ]]; then
			echo "[${script_tag}] interactive mode requires a TTY; skipping seed fallback." >&2
			echo ""
			return 0
		fi
		echo "[${script_tag}] select candidate index for ${species}/${target} (empty=skip):" >&2
		local selection=""
		read -r selection
		selection="$(trim_whitespace "${selection}")"
		if [[ -z "${selection}" ]]; then
			echo ""
			return 0
		fi
		if ! [[ "${selection}" =~ ^[0-9]+$ ]]; then
			echo "[${script_tag}] invalid selection: ${selection}" >&2
			return 2
		fi
		if [[ "${selection}" -ge "${#candidates[@]}" ]]; then
			echo "[${script_tag}] selection out of range: ${selection}" >&2
			return 2
		fi
		IFS=$'\t' read -r _ resolved_path _ <<< "${candidates[selection]}"
		echo "[${script_tag}] selected interactively: ${resolved_path}" >&2
		echo "${resolved_path}"
		return 0
	fi

	if [[ -n "${preferred_species_csv}" ]]; then
		local wanted=""
		local -a wanted_species=()
		IFS=',' read -r -a wanted_species <<< "${preferred_species_csv}"
		for wanted in "${wanted_species[@]}"; do
			wanted="$(trim_whitespace "${wanted}")"
			if [[ -z "${wanted}" ]]; then
				continue
			fi
			for row in "${candidates[@]}"; do
				IFS=$'\t' read -r cand_species cand_path _ <<< "${row}"
				if [[ "${cand_species}" == "${wanted}" ]]; then
					echo "[${script_tag}] selected by CROSS_SPECIES_BEST_PREFERRED_SPECIES: ${cand_path}" >&2
					echo "${cand_path}"
					return 0
				fi
			done
		done
	fi

	IFS=$'\t' read -r _ resolved_path _ <<< "${candidates[0]}"
	echo "[${script_tag}] selected automatically: ${resolved_path}" >&2
	echo "${resolved_path}"
}
