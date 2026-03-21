#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="/export/hibiki/miniforge3/envs/intronmodel/bin/python3"
SPECIES=("Athal" "Dmel" "Hsap" "Mmus")
GPUS=("4" "5" "6" "7")
LOG_DIR="$PROJECT_ROOT/run/tmp_cnn_v2_pair_late_logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG_DIR="$LOG_DIR/late_from_best_4gpu_$STAMP"

mkdir -p "$RUN_LOG_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  printf '[late_from_best_4gpu] missing python: %s\n' "$PYTHON_BIN" >&2
  exit 1
fi

build_args() {
  local project_root="$1"
  local species="$2"
  "$PYTHON_BIN" - "$project_root" "$species" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path


def load_sampled_params(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("status", "")).strip().lower() != "ok":
        raise ValueError(f"Expected status='ok' in {path}")
    sampled_params = payload.get("sampled_params")
    if not isinstance(sampled_params, dict):
        raise ValueError(f"Missing sampled_params in {path}")
    return sampled_params


def as_text(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(value, ".15g")
    return str(value)


def as_list_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return ",".join(as_text(item) for item in value)
    return as_text(value)


project_root = Path(sys.argv[1])
species = sys.argv[2]
tuning_root = project_root / "data" / species / "tuning"
donor_path = tuning_root / "cnn_v2" / "donor" / "best_config.json"
acceptor_path = tuning_root / "cnn_v2" / "acceptor" / "best_config.json"
pair_path = tuning_root / "cnn_v2_pair" / "pair" / "best_config.json"

donor_sampled = load_sampled_params(donor_path)
acceptor_sampled = load_sampled_params(acceptor_path)
pair_sampled = load_sampled_params(pair_path)

args = [
    "--model",
    "cnn_v2_pair",
    "--species",
    species,
    "--donor_len",
    as_text(pair_sampled.get("donor_len", 100)),
    "--acceptor_len",
    as_text(pair_sampled.get("acceptor_len", 100)),
    "--device",
    "auto",
    "--seed",
    "1337",
    "--name_fields",
    "none",
    "--epochs",
    "10",
    "--max_epochs",
    "200",
    "--early_stop_patience",
    "12",
    "--early_stop_min_delta",
    "0.0",
    "--train_target",
    "pair",
    "--sequence_transform",
    as_text(pair_sampled.get("sequence_transform", "none")),
    "--batch_size",
    as_text(pair_sampled.get("batch_size", 512)),
    "--lr",
    as_text(pair_sampled.get("lr", 5e-4)),
    "--loss",
    as_text(pair_sampled.get("loss", "focal")),
    "--input_mode",
    as_text(pair_sampled.get("input_mode", "onehot")),
    "--pair_mode",
    "pair",
    "--fusion_mode",
    "late",
    "--embedding_dim",
    as_text(pair_sampled.get("embedding_dim", 32)),
    "--bpe_pretrained_model_name",
    "zhihan1996/DNABERT-2-117M",
    "--bpe_trust_remote_code",
    "0",
    "--conv_channels",
    "",
    "--kernel_sizes",
    "",
    "--donor_conv_channels",
    as_list_text(donor_sampled.get("conv_channels")),
    "--acceptor_conv_channels",
    as_list_text(acceptor_sampled.get("conv_channels")),
    "--donor_kernel_sizes",
    as_list_text(donor_sampled.get("kernel_sizes")),
    "--acceptor_kernel_sizes",
    as_list_text(acceptor_sampled.get("kernel_sizes")),
    "--max_pool_size",
    as_text(pair_sampled.get("max_pool_size", 2)),
    "--conv_stride",
    as_text(pair_sampled.get("conv_stride", 1)),
    "--head_type",
    as_text(pair_sampled.get("head_type", "gap")),
    "--fc_hidden",
    as_text(pair_sampled.get("fc_hidden", 128)),
    "--dropout",
    as_text(pair_sampled.get("dropout", 0.3)),
    "--weight_decay",
    as_text(pair_sampled.get("weight_decay", 0.01)),
    "--eta_min_ratio",
    "0.01",
    "--val_frac",
    "0.2",
    "--grad_clip",
    "5.0",
    "--pos_weight_cap",
    "20.0",
    "--focal_gamma",
    "2.0",
    "--f1_lambda",
    as_text(pair_sampled.get("f1_lambda", 0.1)),
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
    "--tag",
    "late_from_cnn_v2_best_4gpu",
]

for arg in args:
    print(arg)
PY
}

cleanup() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}

trap cleanup EXIT INT TERM

declare -a PIDS=()
declare -a PID_SPECIES=()
declare -a PID_GPUS=()
declare -a PID_LOGS=()

printf '[late_from_best_4gpu] launching species jobs:\n'
for index in "${!SPECIES[@]}"; do
  species="${SPECIES[$index]}"
  gpu_id="${GPUS[$index]}"
  log_path="$RUN_LOG_DIR/${species}.log"
  mapfile -t run_args < <(build_args "$PROJECT_ROOT" "$species")
  printf '[late_from_best_4gpu] start species=%s gpu=%s log=%s\n' \
    "$species" "$gpu_id" "$log_path"
  CUDA_VISIBLE_DEVICES="$gpu_id" \
    PYTHONPATH="$PROJECT_ROOT/src" \
    "$PYTHON_BIN" "$PROJECT_ROOT/src/run_model.py" \
    "${run_args[@]}" \
    >"$log_path" 2>&1 &
  pid="$!"
  PIDS+=("$pid")
  PID_SPECIES+=("$species")
  PID_GPUS+=("$gpu_id")
  PID_LOGS+=("$log_path")
done

printf '[late_from_best_4gpu] logs=%s\n' "$RUN_LOG_DIR"

while (( ${#PIDS[@]} > 0 )); do
  finished_index=-1
  for index in "${!PIDS[@]}"; do
    pid="${PIDS[$index]}"
    if kill -0 "$pid" 2>/dev/null; then
      continue
    fi
    wait "$pid"
    rc="$?"
    if (( rc != 0 )); then
      printf '[late_from_best_4gpu] failed species=%s gpu=%s rc=%s log=%s\n' \
        "${PID_SPECIES[$index]}" "${PID_GPUS[$index]}" "$rc" \
        "${PID_LOGS[$index]}"
      exit "$rc"
    fi
    printf '[late_from_best_4gpu] done species=%s gpu=%s log=%s\n' \
      "${PID_SPECIES[$index]}" "${PID_GPUS[$index]}" \
      "${PID_LOGS[$index]}"
    unset 'PIDS[index]'
    unset 'PID_SPECIES[index]'
    unset 'PID_GPUS[index]'
    unset 'PID_LOGS[index]'
    PIDS=("${PIDS[@]}")
    PID_SPECIES=("${PID_SPECIES[@]}")
    PID_GPUS=("${PID_GPUS[@]}")
    PID_LOGS=("${PID_LOGS[@]}")
    finished_index=0
    break
  done
  if (( finished_index == -1 )); then
    sleep 30
  fi
done

printf '[late_from_best_4gpu] all species completed successfully\n'
