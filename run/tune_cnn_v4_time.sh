#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
    echo "[tune_cnn_v4_time.sh] This script is config-only. Edit top CONFIG and run without args." >&2
    exit 1
fi

# --------------------------
# CONFIG (edit here)
# --------------------------
INTRONMODEL_AUTO_TMUX="on"
# The cohort used to rank one species-independent architecture/hparam set.
# Quick smoke search: TUNING_SPECIES=("Dmel")
# Production shared search: TUNING_SPECIES=("Hsap" "Dmel" "Mmus" "Athal")
TUNING_SPECIES=("Hsap" "Dmel" "Mmus" "Athal")
# Set to 0 for an exploratory cohort: retain its run-local best_config.json,
# without replacing data/tuning/cnn_v4_shared/<task>/best_config.json.
PUBLISH_BEST="1"
TARGET_ORDER=("donor" "acceptor")
TRIALS="24"
EPOCHS="12"
BASE_SEED="1337"
OBJECTIVE_METRIC="max_f1"  # pr_auc | roc_auc | max_f1
DEVICE="auto"
BATCH_SIZE="512"
MIN_BATCH_SIZE="64"
# site_flank100 stores 100 nt on each side of the splice site (200 nt total).
DONOR_LEN="200"
ACCEPTOR_LEN="200"
TRAIN_POS_PATH=""
TRAIN_NEG_PATH=""
VAL_FRAC="0.2"
LOSS="focal"
USE_AMP="1"
AMP_DTYPE="auto"
COMPILE_MODE="off"

DEFAULT_SEARCH_SPACE_JSON='{
  "lr": {"type": "float", "min": 8e-5, "max": 3e-3, "scale": "log"},
  "batch_size": {"type": "categorical", "values": [64, 128, 256, 512, 1024]},
  "dropout": {"type": "float", "min": 0.0, "max": 0.5, "scale": "linear"},
  "weight_decay": {"type": "float", "min": 1e-8, "max": 2e-2, "scale": "log"},
  "conv_channels": {"type": "categorical", "values": ["32,64", "48,96", "64,128"]},
  "kernel_sizes": {"type": "categorical", "values": ["5,5", "7,5", "7,7"]},
  "block_dilations": {"type": "categorical", "values": ["1,2", "1,4", "1,2,4"]},
  "residual_channels": {"type": "categorical", "values": ["16,32", "24,48", "32,64"]},
  "deformable_groups": {"type": "categorical", "values": [1, 2, 4]},
  "deformable_kernel_size": {"type": "categorical", "values": [3, 5]},
  "fc_hidden": {"type": "categorical", "values": [64, 128, 192]},
  "head_type": {"type": "categorical", "values": ["gap", "center"]}
}'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"
intronmodel_activate_conda "intronmodel"
intronmodel_init_paths "${BASH_SOURCE[0]}"
intronmodel_enable_auto_tmux "${PROJECT_ROOT}" "$0" "${BASH_SOURCE[0]##*/}"
PYTHON_BIN="$(intronmodel_resolve_python_bin "tune_cnn_v4_time.sh")"
export PYTHONPATH="${PROJECT_ROOT}/../..:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! "${TRIALS}" =~ ^[1-9][0-9]*$ || ! "${EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[tune_cnn_v4_time.sh] TRIALS and EPOCHS must be positive integers." >&2
    exit 1
fi
if [[ ! "${PUBLISH_BEST}" =~ ^[01]$ ]]; then
    echo "[tune_cnn_v4_time.sh] PUBLISH_BEST must be 0 or 1." >&2
    exit 1
fi
if [[ ${#TUNING_SPECIES[@]} -eq 0 ]]; then
    echo "[tune_cnn_v4_time.sh] TUNING_SPECIES must contain at least one species." >&2
    exit 1
fi
declare -A seen_tuning_species=()
for species in "${TUNING_SPECIES[@]}"; do
    if [[ -z "${species}" || -n "${seen_tuning_species[${species}]+x}" ]]; then
        echo "[tune_cnn_v4_time.sh] TUNING_SPECIES entries must be non-empty and unique." >&2
        exit 1
    fi
    seen_tuning_species["${species}"]=1
done

for task in "${TARGET_ORDER[@]}"; do
    if [[ "${task}" != "donor" && "${task}" != "acceptor" ]]; then
        echo "[tune_cnn_v4_time.sh] TARGET_ORDER values must be donor|acceptor." >&2
        exit 1
    fi
    output_dir="${DATA_ROOT}/tuning/cnn_v4_shared/${task}/runs/$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "${output_dir}"
    config_path="${output_dir}/config.json"
    "${PYTHON_BIN}" - "${config_path}" "${PROJECT_ROOT}" "${DATA_ROOT}" "${output_dir}" \
        "${task}" "${TRIALS}" "${EPOCHS}" "${BASE_SEED}" "${OBJECTIVE_METRIC}" \
        "${DEFAULT_SEARCH_SPACE_JSON}" "${DEVICE}" "${BATCH_SIZE}" "${MIN_BATCH_SIZE}" \
        "${DONOR_LEN}" "${ACCEPTOR_LEN}" "${VAL_FRAC}" "${LOSS}" "${USE_AMP}" \
        "${AMP_DTYPE}" "${COMPILE_MODE}" "${TRAIN_POS_PATH}" "${TRAIN_NEG_PATH}" \
        "${PUBLISH_BEST}" "${TUNING_SPECIES[@]}" <<'PYCONFIG'
from __future__ import annotations
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
project_root, data_root, output_dir, task = sys.argv[2:6]
trials, epochs, seed = map(int, sys.argv[6:9])
objective_metric, search_space = sys.argv[9:11]
device, batch_size, min_batch_size = sys.argv[11:14]
donor_len, acceptor_len, val_frac, loss = sys.argv[14:18]
use_amp, amp_dtype, compile_mode = sys.argv[18:21]
train_pos_path, train_neg_path = sys.argv[21:23]
publish_best = bool(int(sys.argv[23]))
species = sys.argv[24:]
base_args = {
    "model": "cnn_v4",
    "device": device,
    "batch_size": int(batch_size),
    "min_batch_size": int(min_batch_size),
    "donor_len": int(donor_len),
    "acceptor_len": int(acceptor_len),
    "val_frac": float(val_frac),
    "loss": loss,
    "use_amp": int(use_amp),
    "amp_dtype": amp_dtype,
    "compile_mode": compile_mode,
    "pair_mode": "independent",
}
if train_pos_path:
    base_args["train_pos_path"] = train_pos_path
if train_neg_path:
    base_args["train_neg_path"] = train_neg_path
config_path.write_text(json.dumps({
    "project_root": project_root,
    "data_root": data_root,
    "output_dir": output_dir,
    "species": species,
    "publish_best": publish_best,
    "task": task,
    "trials": trials,
    "epochs": epochs,
    "seed": seed,
    "objective_metric": objective_metric,
    "base_args": base_args,
    "search_space": json.loads(search_space),
}, indent=2) + "\n", encoding="utf-8")
PYCONFIG
    "${PYTHON_BIN}" "${PROJECT_ROOT}/src/tools/shared_hparam_search.py" --config "${config_path}"
done
