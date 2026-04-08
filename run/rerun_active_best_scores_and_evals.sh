#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOT'
Usage: bash run/rerun_active_best_scores_and_evals.sh

This script reruns the saved best active models for all species using
``--skip_train``, then rebuilds the downstream eval outputs and plots:
  1) best cnn_v2 inference from saved checkpoints
  2) best cnn_v3 inference from saved checkpoints
  3) best cnn_pair_v2 inference from saved checkpoints
  4) best cnn_pair_v3 inference from saved checkpoints
  5) run/plot_eval.sh
  6) run/eval_trans_score.sh
  7) run/eval_intron_pr_auc.sh

Edit the CONFIG block below before running if you want to narrow the species
set or disable any step.
EOT
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
	usage
	exit 0
fi

# --------------------------
# CONFIG (edit here)
# --------------------------
CONDA_ENV="intronmodel"
SPECIES="Athal,Dmel,Hsap,Mmus"
RUN_CNN_V2="1"
RUN_CNN_V3="1"
RUN_CNN_V2_PAIR="1"
RUN_CNN_V3_PAIR="1"
RUN_PLOT_EVAL="1"
RUN_TRANS_EVAL="1"
RUN_INTRON_EVAL="1"
INTRONMODEL_AUTO_TMUX="off"

# --------------------------
# Runtime implementation
# --------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"
intronmodel_activate_conda "${CONDA_ENV}"
intronmodel_init_paths "${BASH_SOURCE[0]}"
intronmodel_enable_auto_tmux "${PROJECT_ROOT}" "$0" "${BASH_SOURCE[0]##*/}"

run_best_infer() {
	local model_name="$1"
	local best_config_path="$2"
	local species="$3"
	local label="$4"

	echo "[rerun_active_best_scores_and_evals] ${label}"
	PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
		python3 - \
			"${PROJECT_ROOT}" \
			"${model_name}" \
			"${best_config_path}" \
			"${species}" <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch

project_root = Path(sys.argv[1]).resolve()
model_name = sys.argv[2]
best_config_path = Path(sys.argv[3]).resolve()
species = sys.argv[4]

if not best_config_path.is_file():
    raise FileNotFoundError(f"best_config not found: {best_config_path}")

sys.path.insert(0, str(project_root / "src"))

from run_model import (  # noqa: E402
    _build_checkpoint_paths,
    _build_checkpoint_stem_from_params,
    _infer_window_defaults,
    parse_args,
)
from util.checkpoint_io import (  # noqa: E402
    extract_task_checkpoint_path,
    read_json_object,
)
from util.model_task_paths import checkpoint_tasks_for_model  # noqa: E402


def _scalar_to_text(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(value, ".15g")
    return str(value)


payload = read_json_object(best_config_path)
if payload is None or str(payload.get("status", "")).strip().lower() != "ok":
    raise ValueError(f"Invalid best_config payload: {best_config_path}")

hparam_context = payload.get("hparam_context")
if not isinstance(hparam_context, dict):
    raise ValueError(f"Invalid hparam_context payload: {best_config_path}")

fixed_run_args = hparam_context.get("fixed_run_args")
if not isinstance(fixed_run_args, dict):
    raise ValueError(f"Invalid fixed_run_args payload: {best_config_path}")

sampled_params = payload.get("sampled_params")
if not isinstance(sampled_params, dict):
    raise ValueError(f"Invalid sampled_params payload: {best_config_path}")

run_args: list[str] = [
    "--model",
    model_name,
    "--species",
    species,
    "--skip_train",
    "--name_fields",
    "none",
    "--device",
    "auto",
    "--use_amp",
    "1",
    "--amp_dtype",
    "auto",
    "--compile_mode",
    "off",
    "--allow_tf32",
    "1",
    "--cudnn_benchmark",
    "1",
    "--deterministic",
    "0",
    "--num_workers",
    "auto",
    "--prefetch_factor",
    "4",
    "--persistent_workers",
    "1",
    "--pin_memory",
    "1",
    "--min_batch_size",
    "64",
    "--max_oom_retries",
    "8",
    "--transcript_score_agg",
    "min",
    "--softmin_tau",
    "1.0",
]

model_tasks = checkpoint_tasks_for_model(model_name)

full_epochs = hparam_context.get("full_epochs")
if isinstance(full_epochs, int) and full_epochs > 0:
    run_args.extend(["--epochs", str(full_epochs)])

for key in ("seed", "pair_mode", "val_frac"):
    value = fixed_run_args.get(key)
    if value is None:
        continue
    run_args.extend([f"--{key}", _scalar_to_text(value)])

train_target_value = fixed_run_args.get("train_target")
if (
    train_target_value is not None
    and len(model_tasks) == 1
    and model_tasks[0] == "pair"
):
    run_args.extend(["--train_target", _scalar_to_text(train_target_value)])

for key in sorted(sampled_params):
    if key == "mask":
        continue
    value = sampled_params[key]
    if value is None:
        continue
    run_args.extend([f"--{key}", _scalar_to_text(value)])

parsed = parse_args(run_args)
model_tasks = checkpoint_tasks_for_model(str(parsed.model))
resolved_donor_len, resolved_acceptor_len, inferred_train_len = _infer_window_defaults(
    species=parsed.species,
    donor_len=parsed.donor_len,
    acceptor_len=parsed.acceptor_len,
)
checkpoint_stem = _build_checkpoint_stem_from_params(
    model_name=str(parsed.model),
    donor_len=resolved_donor_len,
    acceptor_len=resolved_acceptor_len,
    inferred_train_len=inferred_train_len,
    raw_params=dict(vars(parsed)),
)
checkpoint_paths = _build_checkpoint_paths(
    parsed.species,
    checkpoint_stem,
    tasks=model_tasks,
)

for task in model_tasks:
    strict_path = Path(checkpoint_paths[task]).resolve()
    if strict_path.exists():
        continue
    candidate_paths: list[Path] = []
    direct_checkpoint = extract_task_checkpoint_path(
        payload,
        task=task,
        base_dir=best_config_path.parent,
    )
    if direct_checkpoint is not None and direct_checkpoint.exists():
        candidate_paths.append(direct_checkpoint)

    task_dir = project_root / "model" / species / task
    if task_dir.is_dir():
        candidate_paths.extend(
            sorted(task_dir.glob("*.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
        )

    best_checkpoint: Path | None = None
    seen_paths: set[Path] = set()
    for candidate in candidate_paths:
        resolved = candidate.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        try:
            torch.load(resolved, map_location="cpu", weights_only=False)
        except Exception:
            continue
        best_checkpoint = resolved
        break

    if best_checkpoint is None:
        continue
    strict_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(best_checkpoint, strict_path)
    except OSError:
        shutil.copy2(best_checkpoint, strict_path)

subprocess.run(
    [sys.executable, str(project_root / "src" / "run_model.py"), *run_args],
    check=True,
)
PY
}

run_for_species() {
	local species="$1"
	local species_label="$2"
	local data_root="${PROJECT_ROOT}/data"

	resolve_best_config_path() {
		local model_dir="$1"
		shift
		local target=""
		for target in "$@"; do
			local candidate="${data_root}/${species}/tuning/${model_dir}/${target}/best_config.json"
			if [[ -f "${candidate}" ]]; then
				printf '%s\n' "${candidate}"
				return 0
			fi
		done
		return 1
	}

	local cnn_v2_best=""
	local cnn_v3_best=""
	local cnn_pair_v2_best=""
	local cnn_pair_v3_best=""
	cnn_v2_best="$(
		resolve_best_config_path "cnn_v2" "donor" "acceptor" || true
	)"
	cnn_v3_best="$(
		resolve_best_config_path "cnn_v3" "donor" "acceptor" || true
	)"
	cnn_pair_v2_best="$(
		resolve_best_config_path "cnn_pair_v2" "pair" || true
	)"
	cnn_pair_v3_best="$(
		resolve_best_config_path "cnn_pair_v3" "pair" || true
	)"

	if [[ "${RUN_CNN_V2}" == "1" ]]; then
		if [[ -n "${cnn_v2_best}" ]]; then
			run_best_infer "cnn_v2" "${cnn_v2_best}" "${species}" \
				"${species_label}: rerun cnn_v2 best"
		else
			echo "[rerun_active_best_scores_and_evals] skip cnn_v2 for ${species}:"
			echo "  missing best_config under data/${species}/tuning/cnn_v2/{donor,acceptor}/best_config.json"
		fi
	fi

	if [[ "${RUN_CNN_V3}" == "1" ]]; then
		if [[ -n "${cnn_v3_best}" ]]; then
			run_best_infer "cnn_v3" "${cnn_v3_best}" "${species}" \
				"${species_label}: rerun cnn_v3 best"
		else
			echo "[rerun_active_best_scores_and_evals] skip cnn_v3 for ${species}:"
			echo "  missing best_config under data/${species}/tuning/cnn_v3/{donor,acceptor}/best_config.json"
		fi
	fi

	if [[ "${RUN_CNN_V2_PAIR}" == "1" ]]; then
		if [[ -n "${cnn_pair_v2_best}" ]]; then
			run_best_infer "cnn_pair_v2" "${cnn_pair_v2_best}" "${species}" \
				"${species_label}: rerun cnn_pair_v2 best"
		else
			echo "[rerun_active_best_scores_and_evals] skip cnn_pair_v2 for ${species}:"
			echo "  missing best_config under data/${species}/tuning/cnn_pair_v2/pair/best_config.json"
		fi
	fi

	if [[ "${RUN_CNN_V3_PAIR}" == "1" ]]; then
		if [[ -n "${cnn_pair_v3_best}" ]]; then
			run_best_infer "cnn_pair_v3" "${cnn_pair_v3_best}" "${species}" \
				"${species_label}: rerun cnn_pair_v3 best"
		else
			echo "[rerun_active_best_scores_and_evals] skip cnn_pair_v3 for ${species}:"
			echo "  missing best_config under data/${species}/tuning/cnn_pair_v3/pair/best_config.json"
		fi
	fi
}

IFS=',' read -r -a SPECIES_TOKENS <<< "${SPECIES}"
for species_raw in "${SPECIES_TOKENS[@]}"; do
	species="$(printf '%s' "${species_raw}" | tr -d '[:space:]')"
	if [[ -z "${species}" ]]; then
		continue
	fi
	run_for_species "${species}" "${species}"
done

if [[ "${RUN_PLOT_EVAL}" == "1" ]]; then
	echo "[rerun_active_best_scores_and_evals] rebuild eval plots"
	bash "${PROJECT_ROOT}/run/plot_eval.sh" --species "${SPECIES}"
fi

if [[ "${RUN_TRANS_EVAL}" == "1" ]]; then
	echo "[rerun_active_best_scores_and_evals] rerun transcript eval"
	bash "${PROJECT_ROOT}/run/eval_trans_score.sh" --species "${SPECIES}"
fi

if [[ "${RUN_INTRON_EVAL}" == "1" ]]; then
	echo "[rerun_active_best_scores_and_evals] rerun intron pr auc eval"
	IFS=',' read -r -a INTRON_SPECIES_TOKENS <<< "${SPECIES}"
	for species_raw in "${INTRON_SPECIES_TOKENS[@]}"; do
		species="$(printf '%s' "${species_raw}" | tr -d '[:space:]')"
		if [[ -z "${species}" ]]; then
			continue
		fi
		bash "${PROJECT_ROOT}/run/eval_intron_pr_auc.sh" \
			--species "${species}" \
			--site-score-tsv "cnn_v2" \
			--score-source auto \
			--intron-score-op "*"
	done
fi

echo "[rerun_active_best_scores_and_evals] done"
