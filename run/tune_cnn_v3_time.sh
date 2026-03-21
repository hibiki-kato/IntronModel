#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
	echo "[tune_cnn_v3_time.sh] This script is config-only." \
		"Edit top CONFIG and run without args." >&2
	exit 1
fi

# --------------------------
# CONFIG (edit here)
# --------------------------
# Frequently edited knobs are intentionally placed first in this block.
# Advanced fallback defaults are kept below.
TIME_BUDGET_MINUTES="100"

INTRONMODEL_AUTO_TMUX="off"
# Optional output/data overrides for tagged or mask-data tuning runs.
TAG=""
TRAIN_POS_PATH=""
TRAIN_NEG_PATH=""
BASE_PAIR_CHECKPOINTS=""
MASK_MODE="off"
CHEAT_MODE="off"
DONOR_LEN="100"
ACCEPTOR_LEN="100"
VAL_FRAC="0.2"
BASE_SEED="1337"
# Deprecated: SEED_LIST is ignored. Only BASE_SEED is used.
SEED_LIST=""
PROCESS_TITLE="ETA"

QUICK_TRIALS="12"
QUICK_EPOCHS="2"
TOP_K="3"
FULL_EPOCHS="8"
QUICK_COMPILE_MODE="off"
FULL_COMPILE_MODE="off"

GPU_IDS="auto"
# auto: use one concurrent trial per configured GPU_IDS entry.
MAX_PARALLEL_TRIALS="auto"

DEVICE="auto"
USE_AMP="1"
AMP_DTYPE="auto"
ALLOW_TF32="1"
CUDNN_BENCHMARK="1"
DETERMINISTIC="0"
NUM_WORKERS="auto"
PREFETCH_FACTOR="4"
PERSISTENT_WORKERS="1"
PIN_MEMORY="1"
MIN_BATCH_SIZE="64"
MAX_OOM_RETRIES="5"
MAX_MODEL_PARAMS="auto"
MAX_MODEL_PARAMS_FALLBACK="300000000"
MAX_MODEL_PARAMS_MEM_FRACTION="0.80"
MAX_MODEL_PARAMS_RESERVE_MIB="2048"
MAX_MODEL_PARAMS_BYTES_PER_PARAM="32"
MAX_MODEL_PARAMS_MODEL_FACTOR="0.90"

VISUALIZE="none"
NAME_FIELDS="none"
SEQUENCE_TRANSFORM="none"
UPDATE_DOUBLE_DESCENT_PLOT="0"

SEARCH_ALGO="history_guided"
HISTORY_TOP_N="512"
GUIDED_RANDOM_FRACTION="0.20"
GUIDED_MUTATION_RATE="0.35"
SEARCH_SPACE_FILE="auto"
MAX_POOL_SIZE="2"
CONV_STRIDE="1"
HEAD_TYPE="gap"

CROSS_SPECIES_BEST_MODE="off"
CROSS_SPECIES_BEST_OVERRIDE=""
CROSS_SPECIES_BEST_PREFERRED_SPECIES=""

# Source species used for one cross-species cnn_v3 model.
TRAIN_SPECIES=(
	"Mmus"
	"Hsap"
)
# Artifact namespace for outputs/checkpoints under data/<artifact_species>/...
# Set to "auto" to derive "cross/<species1>_<species2>...".
ARTIFACT_SPECIES="auto"

DEFAULT_SEARCH_SPACE_JSON_PAIR="$(cat <<'JSON'
{
  "donor_len": {"type": "int", "min": 40, "max": 100, "step": 10},
  "acceptor_len": {"type": "int", "min": 40, "max": 100, "step": 10},
  "lr": {"type": "float", "min": 8e-5, "max": 3e-3, "scale": "log"},
	"batch_size": {
		"type": "categorical",
		"values": [128, 256, 512, 1024, 2048]
	},
  "dropout": {"type": "float", "min": 0.0, "max": 0.55, "scale": "linear"},
  "weight_decay": {"type": "float", "min": 1e-8, "max": 2e-2, "scale": "log"},
	"meta_hidden_dim": {
		"type": "categorical",
		"values": [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
	},
	"meta_dropout": {
		"type": "float",
		"min": 0.0,
		"max": 0.6,
		"scale": "linear"
	},
	"input_mode": {
		"type": "categorical",
		"values": ["onehot", "kmer3", "bpe"]
	},
	"pair_mode": {
		"type": "categorical",
		"values": ["pair"]
	},
	"sequence_transform": {
		"type": "categorical",
		"values": ["none", "mask_outside_intron_n", "truncate_outside_intron"]
	},
	"embedding_dim": {
		"type": "categorical",
		"values": [32, 48, 64]
	},
  "loss": {
    "type": "categorical",
    "values": ["weighted_bce", "focal", "asymmetric_focal", "f1", "weighted_bce_f1", "focal_f1"]
  },
  "f1_lambda": {"type": "float", "min": 0.02, "max": 0.5, "scale": "log"}
}
JSON
)"


# --------------------------
# Runtime implementation
# --------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"
intronmodel_activate_conda "intronmodel"
intronmodel_init_paths "${BASH_SOURCE[0]}"

# Auto-run inside tmux on SSH so jobs survive disconnects.
# Set INTRONMODEL_AUTO_TMUX=off|on|auto (default: auto).
intronmodel_enable_auto_tmux "${PROJECT_ROOT}" "$0" "${BASH_SOURCE[0]##*/}"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/tuning_cross_species_best.sh"

# Keep process title fixed during tune_time runs.
export INTRONMODEL_DISABLE_ETA_PROCESS_TITLE="1"

format_elapsed() {
	intronmodel_format_elapsed "$1"
}

format_eta() {
	intronmodel_format_eta_epoch "$1"
}

build_eta_process_title() {
	intronmodel_build_eta_process_title "$1"
}

resolve_species_case() {
	intronmodel_resolve_species_case "$1" "$2" ""
}

resolve_python_bin() {
	intronmodel_resolve_python_bin "tune_cnn_v3_time.sh"
}

resolve_seed_list() {
	intronmodel_resolve_seed_list \
		"tune_cnn_v3_time.sh" \
		"${BASE_SEED}" \
		"${SEED_LIST}" \
		"${PYTHON_BIN}"
}

resolve_base_pair_checkpoints() {
	local python_bin="$1"
	local project_root="$2"
	local species="$3"
	local explicit_value="$4"
	"${python_bin}" - "$project_root" "$species" "$explicit_value" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path


def _dedupe_keep_order(values: list[str]) -> list[str]:
	seen: set[str] = set()
	out: list[str] = []
	for value in values:
		if value in seen:
			continue
		seen.add(value)
		out.append(value)
	return out


project_root = Path(sys.argv[1])
species = sys.argv[2]
explicit_value = sys.argv[3].strip()
if explicit_value:
	explicit_paths = _dedupe_keep_order(
		[token.strip() for token in explicit_value.split(",") if token.strip()]
	)
	if not explicit_paths:
		raise SystemExit(1)
	for path_text in explicit_paths:
		if not Path(path_text).exists():
			raise SystemExit(1)
	print(",".join(explicit_paths))
	raise SystemExit(0)

# Preferred source: model-separated tuning artifact for cnn_v2_pair.
preferred_best_paths = (
	project_root
	/ "data"
	/ species
	/ "tuning"
	/ "cnn_v2_pair"
	/ "pair"
	/ "best_config.json",
	# Backward-compatible fallback.
	project_root
	/ "data"
	/ species
	/ "tuning"
	/ "cnn_v2"
	/ "pair"
	/ "best_config.json",
)
for best_config_path in preferred_best_paths:
	if not best_config_path.is_file():
		continue
	try:
		payload = json.loads(best_config_path.read_text(encoding="utf-8"))
	except Exception:
		payload = None
	if isinstance(payload, dict):
		checkpoint_path = str(payload.get("pair_checkpoint_path", "")).strip()
		if checkpoint_path and Path(checkpoint_path).exists():
			print(checkpoint_path)
			raise SystemExit(0)

learning_metric_dir = project_root / "data" / species / "learning_metric"
if not learning_metric_dir.is_dir():
	raise SystemExit(1)

latest_by_model: dict[str, tuple[float, str]] = {}
for path in learning_metric_dir.glob("*.train.json"):
	try:
		payload = json.loads(path.read_text(encoding="utf-8"))
	except Exception:
		continue
	model_name = str(payload.get("model", "")).strip()
	checkpoint_path = str(payload.get("pair_checkpoint_path", "")).strip()
	if model_name not in {"cnn_v2", "cnn_v2_pair"} or checkpoint_path == "":
		continue
	if not Path(checkpoint_path).exists():
		continue
	mtime = path.stat().st_mtime
	previous = latest_by_model.get(model_name)
	if previous is None or mtime > previous[0]:
		latest_by_model[model_name] = (mtime, checkpoint_path)

resolved = _dedupe_keep_order(
	[checkpoint for _mtime, checkpoint in sorted(latest_by_model.values(), reverse=True)]
)
if not resolved:
	raise SystemExit(1)
print(",".join(resolved))
PY
}

resolve_train_species_list() {
	local data_root="$1"
	shift
	local -a resolved_species=()
	local raw_species=""
	local trimmed_species=""
	local canonical_species=""
	declare -A seen=()
	for raw_species in "$@"; do
		trimmed_species="$(echo "${raw_species}" | xargs)"
		if [[ -z "${trimmed_species}" ]]; then
			continue
		fi
		canonical_species="$(
			resolve_species_case "${trimmed_species}" "${data_root}"
		)" || return 1
		if [[ -n "${seen[${canonical_species}]:-}" ]]; then
			continue
		fi
		seen["${canonical_species}"]="1"
		resolved_species+=("${canonical_species}")
	done
	if [[ ${#resolved_species[@]} -eq 0 ]]; then
		return 1
	fi
	printf '%s\n' "${resolved_species[@]}"
}

resolve_artifact_species_name() {
	local configured_value="$1"
	shift
	if [[ -n "${configured_value}" && "${configured_value}" != "auto" ]]; then
		printf '%s\n' "${configured_value}"
		return 0
	fi
	local joined=""
	joined="$(
		printf '%s\n' "$@" | awk 'NF' | LC_ALL=C sort -u | paste -sd '_' -
	)"
	if [[ -z "${joined}" ]]; then
		return 1
	fi
	printf 'cross/%s\n' "${joined}"
}

resolve_cross_base_pair_checkpoints() {
	local python_bin="$1"
	local project_root="$2"
	local explicit_value="$3"
	shift 3
	local -a train_species_list=("$@")
	if [[ ${#train_species_list[@]} -eq 0 ]]; then
		return 1
	fi
	if [[ -n "${explicit_value}" ]]; then
		resolve_base_pair_checkpoints \
			"${python_bin}" \
			"${project_root}" \
			"${train_species_list[0]}" \
			"${explicit_value}"
		return $?
	fi
	local -a merged=()
	local species_name=""
	local resolved_csv=""
	local token=""
	declare -A seen=()
	for species_name in "${train_species_list[@]}"; do
		resolved_csv="$(
			resolve_base_pair_checkpoints \
				"${python_bin}" \
				"${project_root}" \
				"${species_name}" \
				""
		)" || return 1
		IFS=',' read -r -a resolved_tokens <<< "${resolved_csv}"
		for token in "${resolved_tokens[@]}"; do
			token="$(echo "${token}" | xargs)"
			if [[ -z "${token}" ]]; then
				continue
			fi
			if [[ -n "${seen[${token}]:-}" ]]; then
				continue
			fi
			seen["${token}"]="1"
			merged+=("${token}")
		done
	done
	if [[ ${#merged[@]} -eq 0 ]]; then
		return 1
	fi
	local merged_csv=""
	merged_csv="$(IFS=','; echo "${merged[*]}")"
	printf '%s\n' "${merged_csv}"
}

resolve_cross_train_paths() {
	local python_bin="$1"
	local project_root="$2"
	local data_root="$3"
	local artifact_species="$4"
	local donor_len="$5"
	local acceptor_len="$6"
	local train_pos_template="$7"
	local train_neg_template="$8"
	local train_species_csv="$9"
	"${python_bin}" - \
		"${project_root}" \
		"${data_root}" \
		"${artifact_species}" \
		"${donor_len}" \
		"${acceptor_len}" \
		"${train_pos_template}" \
		"${train_neg_template}" \
		"${train_species_csv}" <<'PY'
from __future__ import annotations

import os
import sys
from pathlib import Path

project_root = Path(sys.argv[1]).resolve()
data_root = Path(sys.argv[2]).resolve()
artifact_species = sys.argv[3].strip()
donor_len_raw = sys.argv[4].strip()
acceptor_len_raw = sys.argv[5].strip()
train_pos_template = sys.argv[6].strip()
train_neg_template = sys.argv[7].strip()
species_csv = sys.argv[8].strip()

if artifact_species == "":
    print(
        "[tune_cnn_v3_time.sh] ARTIFACT_SPECIES must not be empty.",
        file=sys.stderr,
    )
    raise SystemExit(2)

species_list = [token.strip() for token in species_csv.split(",") if token.strip()]
if not species_list:
    print(
        "[tune_cnn_v3_time.sh] TRAIN_SPECIES must contain at least one value.",
        file=sys.stderr,
    )
    raise SystemExit(2)

if (train_pos_template == "") != (train_neg_template == ""):
    print(
        "[tune_cnn_v3_time.sh] TRAIN_POS_PATH and TRAIN_NEG_PATH must be set "
        "together for cross-species mode.",
        file=sys.stderr,
    )
    raise SystemExit(2)


def _parse_optional_len(raw_value: str) -> int | None:
    text = raw_value.strip().lower()
    if text in {"", "none", "null"}:
        return None
    value = int(text)
    if value <= 0:
        raise ValueError("window length must be positive")
    return value


def _resolve_species_template(template: str, species: str) -> str:
    return (
        template.replace("${SPECIES}", species)
        .replace("{SPECIES}", species)
        .replace("{species}", species)
    )


def _copy_concat(inputs: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_file:
        for source_path in inputs:
            with Path(source_path).open("r", encoding="utf-8") as in_file:
                saw_content = False
                for line in in_file:
                    saw_content = True
                    out_file.write(line)
            if saw_content:
                with Path(source_path).open("rb") as raw_file:
                    raw_file.seek(0, os.SEEK_END)
                    if raw_file.tell() > 0:
                        raw_file.seek(-1, os.SEEK_END)
                        if raw_file.read(1) != b"\n":
                            out_file.write("\n")


donor_len = _parse_optional_len(donor_len_raw)
acceptor_len = _parse_optional_len(acceptor_len_raw)

sys.path.insert(0, str(project_root / "src"))
from util.data_proc import resolve_train_paths  # noqa: E402

resolved_pos_paths: list[str] = []
resolved_neg_paths: list[str] = []
for species in species_list:
    if train_pos_template != "":
        pos_path = _resolve_species_template(train_pos_template, species)
        neg_path = _resolve_species_template(train_neg_template, species)
    else:
        pos_path, neg_path, _ = resolve_train_paths(
            species=species,
            train_pos_path=None,
            train_neg_path=None,
            donor_len=donor_len,
            acceptor_len=acceptor_len,
        )
    if not Path(pos_path).is_file():
        print(
            f"[tune_cnn_v3_time.sh] TRAIN_POS_PATH not found for species={species}: "
            f"{pos_path}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if not Path(neg_path).is_file():
        print(
            f"[tune_cnn_v3_time.sh] TRAIN_NEG_PATH not found for species={species}: "
            f"{neg_path}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    resolved_pos_paths.append(pos_path)
    resolved_neg_paths.append(neg_path)

species_token = "_".join(sorted(species_list, key=str.casefold))
donor_token = "auto" if donor_len is None else str(donor_len)
acceptor_token = "auto" if acceptor_len is None else str(acceptor_len)
prefix = f"cross_{species_token}_d{donor_token}_a{acceptor_token}"
output_dir = data_root / artifact_species / "train"
merged_pos_path = output_dir / f"{prefix}.err"
merged_neg_path = output_dir / f"{prefix}.neg.err"
_copy_concat(resolved_pos_paths, merged_pos_path)
_copy_concat(resolved_neg_paths, merged_neg_path)
print(f"{merged_pos_path}\t{merged_neg_path}")
PY
}

resolve_search_space_file() {
	local explicit_file="$1"
	local species="$2"
	local tuning_model_name="$3"

	if [[ -n "${explicit_file}" && "${explicit_file}" != "auto" ]]; then
		if [[ -f "${explicit_file}" ]]; then
			printf '%s\n' "${explicit_file}"
			return 0
		fi
		echo "[tune_cnn_v3_time.sh] SEARCH_SPACE_FILE not found: ${explicit_file}" >&2
		return 2
	fi

	local target_file="${DATA_ROOT}/${species}/tuning/${tuning_model_name}/pair/search_space.json"
	if [[ -f "${target_file}" ]]; then
		printf '%s\n' "${target_file}"
		return 0
	fi

	local species_file="${DATA_ROOT}/${species}/tuning/${tuning_model_name}/search_space.json"
	if [[ -f "${species_file}" ]]; then
		printf '%s\n' "${species_file}"
		return 0
	fi
	if [[ "${tuning_model_name}" != "cnn_pair" ]]; then
		local base_target_file="${DATA_ROOT}/${species}/tuning/cnn_pair/pair/search_space.json"
		if [[ -f "${base_target_file}" ]]; then
			printf '%s\n' "${base_target_file}"
			return 0
		fi
		local base_species_file="${DATA_ROOT}/${species}/tuning/cnn_pair/search_space.json"
		if [[ -f "${base_species_file}" ]]; then
			printf '%s\n' "${base_species_file}"
			return 0
		fi
	fi

	return 1
}

normalize_json_object_file() {
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

run_double_descent_plot() {
	local python_bin="$1"
	local project_root="$2"
	local species_name="$3"
	local model_name="$4"

	"${python_bin}" "${project_root}/src/tools/plot_tuning_double_descent.py" \
		--project_root "${project_root}" \
		--species "${species_name}" \
		--target "pair" \
		--model "${model_name}" || true
}

if ! [[ "${TIME_BUDGET_MINUTES}" =~ ^[0-9]+$ ]] \
	|| [[ "${TIME_BUDGET_MINUTES}" -le 0 ]]; then
	echo "[tune_cnn_v3_time.sh] TIME_BUDGET_MINUTES must be a positive integer." >&2
	exit 1
fi
if ! [[ "${QUICK_TRIALS}" =~ ^[0-9]+$ ]] || [[ "${QUICK_TRIALS}" -le 0 ]]; then
	echo "[tune_cnn_v3_time.sh] QUICK_TRIALS must be a positive integer." >&2
	exit 1
fi
if ! [[ "${QUICK_EPOCHS}" =~ ^[0-9]+$ ]] || [[ "${QUICK_EPOCHS}" -le 0 ]]; then
	echo "[tune_cnn_v3_time.sh] QUICK_EPOCHS must be a positive integer." >&2
	exit 1
fi
if ! [[ "${TOP_K}" =~ ^[0-9]+$ ]] || [[ "${TOP_K}" -le 0 ]]; then
	echo "[tune_cnn_v3_time.sh] TOP_K must be a positive integer." >&2
	exit 1
fi
if ! [[ "${FULL_EPOCHS}" =~ ^[0-9]+$ ]] || [[ "${FULL_EPOCHS}" -le 0 ]]; then
	echo "[tune_cnn_v3_time.sh] FULL_EPOCHS must be a positive integer." >&2
	exit 1
fi
if [[ "${QUICK_COMPILE_MODE}" != "off" \
	&& "${QUICK_COMPILE_MODE}" != "on" \
	&& "${QUICK_COMPILE_MODE}" != "auto" ]]; then
	echo "[tune_cnn_v3_time.sh] QUICK_COMPILE_MODE must be off|on|auto." >&2
	exit 1
fi
if [[ "${FULL_COMPILE_MODE}" != "off" \
	&& "${FULL_COMPILE_MODE}" != "on" \
	&& "${FULL_COMPILE_MODE}" != "auto" ]]; then
	echo "[tune_cnn_v3_time.sh] FULL_COMPILE_MODE must be off|on|auto." >&2
	exit 1
fi
if [[ "${SEARCH_ALGO}" != "random" && "${SEARCH_ALGO}" != "history_guided" ]]; then
	echo "[tune_cnn_v3_time.sh] SEARCH_ALGO must be random|history_guided." >&2
	exit 1
fi
if ! [[ "${HISTORY_TOP_N}" =~ ^[0-9]+$ ]] || [[ "${HISTORY_TOP_N}" -le 0 ]]; then
	echo "[tune_cnn_v3_time.sh] HISTORY_TOP_N must be a positive integer." >&2
	exit 1
fi
if ! [[ "${GUIDED_RANDOM_FRACTION}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
	echo "[tune_cnn_v3_time.sh] GUIDED_RANDOM_FRACTION must be numeric in [0,1]." >&2
	exit 1
fi
if ! awk -v x="${GUIDED_RANDOM_FRACTION}" 'BEGIN{exit !(x>=0 && x<=1)}'; then
	echo "[tune_cnn_v3_time.sh] GUIDED_RANDOM_FRACTION must be in [0,1]." >&2
	exit 1
fi
if ! [[ "${GUIDED_MUTATION_RATE}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
	echo "[tune_cnn_v3_time.sh] GUIDED_MUTATION_RATE must be numeric in [0,1]." >&2
	exit 1
fi
if ! awk -v x="${GUIDED_MUTATION_RATE}" 'BEGIN{exit !(x>=0 && x<=1)}'; then
	echo "[tune_cnn_v3_time.sh] GUIDED_MUTATION_RATE must be in [0,1]." >&2
	exit 1
fi
if [[ ${#TRAIN_SPECIES[@]} -eq 0 ]]; then
	echo "[tune_cnn_v3_time.sh] TRAIN_SPECIES must contain at least one species." >&2
	exit 1
fi
if [[ "${UPDATE_DOUBLE_DESCENT_PLOT}" != "0" \
	&& "${UPDATE_DOUBLE_DESCENT_PLOT}" != "1" ]]; then
	echo "[tune_cnn_v3_time.sh] UPDATE_DOUBLE_DESCENT_PLOT must be 0 or 1." >&2
	exit 1
fi
if [[ "${MASK_MODE}" != "off" && "${MASK_MODE}" != "on" ]]; then
	echo "[tune_cnn_v3_time.sh] MASK_MODE must be off|on." >&2
	exit 1
fi
if [[ "${CHEAT_MODE}" != "off" && "${CHEAT_MODE}" != "on" ]]; then
	echo "[tune_cnn_v3_time.sh] CHEAT_MODE must be off|on." >&2
	exit 1
fi
TUNING_MODEL_NAME="cnn_v3"

PYTHON_BIN="$(resolve_python_bin)"
mapfile -t SEED_VALUES < <(resolve_seed_list)
mapfile -t TRAIN_SPECIES_RESOLVED < <(
	resolve_train_species_list "${DATA_ROOT}" "${TRAIN_SPECIES[@]}"
) || {
	echo "[tune_cnn_v3_time.sh] Failed to resolve TRAIN_SPECIES." >&2
	exit 1
}
if [[ ${#TRAIN_SPECIES_RESOLVED[@]} -eq 0 ]]; then
	echo "[tune_cnn_v3_time.sh] TRAIN_SPECIES resolved to an empty list." >&2
	exit 1
fi
ARTIFACT_SPECIES_RESOLVED="$(
	resolve_artifact_species_name \
		"${ARTIFACT_SPECIES}" \
		"${TRAIN_SPECIES_RESOLVED[@]}"
)"
if [[ -z "${ARTIFACT_SPECIES_RESOLVED}" ]]; then
	echo "[tune_cnn_v3_time.sh] ARTIFACT_SPECIES resolved to empty." >&2
	exit 1
fi
TRAIN_SPECIES_CSV="$(IFS=','; echo "${TRAIN_SPECIES_RESOLVED[*]}")"
if ! RESOLVED_BASE_PAIR_CHECKPOINTS="$(
	resolve_cross_base_pair_checkpoints \
		"${PYTHON_BIN}" \
		"${PROJECT_ROOT}" \
		"${BASE_PAIR_CHECKPOINTS}" \
		"${TRAIN_SPECIES_RESOLVED[@]}"
)"; then
	echo "[tune_cnn_v3_time.sh] Failed to resolve BASE_PAIR_CHECKPOINTS for TRAIN_SPECIES=${TRAIN_SPECIES_CSV}. Set BASE_PAIR_CHECKPOINTS explicitly." >&2
	exit 1
fi
if ! RESOLVED_CROSS_TRAIN_PATHS="$(
	resolve_cross_train_paths \
		"${PYTHON_BIN}" \
		"${PROJECT_ROOT}" \
		"${DATA_ROOT}" \
		"${ARTIFACT_SPECIES_RESOLVED}" \
		"${DONOR_LEN}" \
		"${ACCEPTOR_LEN}" \
		"${TRAIN_POS_PATH}" \
		"${TRAIN_NEG_PATH}" \
		"${TRAIN_SPECIES_CSV}"
)"; then
	exit 1
fi
IFS=$'\t' read -r CROSS_TRAIN_POS_PATH CROSS_TRAIN_NEG_PATH <<< \
	"${RESOLVED_CROSS_TRAIN_PATHS}"
if [[ -z "${CROSS_TRAIN_POS_PATH}" || -z "${CROSS_TRAIN_NEG_PATH}" ]]; then
	echo "[tune_cnn_v3_time.sh] Failed to build cross-species train files." >&2
	exit 1
fi
CROSS_TRAIN_POS_PATH_JSON="$(
	intronmodel_json_string_or_null \
		"${PYTHON_BIN}" \
		"${CROSS_TRAIN_POS_PATH}"
)"
CROSS_TRAIN_NEG_PATH_JSON="$(
	intronmodel_json_string_or_null \
		"${PYTHON_BIN}" \
		"${CROSS_TRAIN_NEG_PATH}"
)"
RESOLVED_MAX_MODEL_PARAMS="$(
	intronmodel_resolve_max_model_params \
		"tune_cnn_v3_time.sh" \
		"${MAX_MODEL_PARAMS}" \
		"${GPU_IDS}" \
		"${MAX_MODEL_PARAMS_FALLBACK}" \
		"${MAX_MODEL_PARAMS_MEM_FRACTION}" \
		"${MAX_MODEL_PARAMS_RESERVE_MIB}" \
		"${MAX_MODEL_PARAMS_BYTES_PER_PARAM}" \
		"${MAX_MODEL_PARAMS_MODEL_FACTOR}" \
		"${PYTHON_BIN}"
)"
START_SECONDS="${SECONDS}"
START_UNIX_SECONDS="$(date +%s)"
BUDGET_SECONDS=$((TIME_BUDGET_MINUTES * 60))
ETA_DEADLINE_EPOCH=$((START_UNIX_SECONDS + BUDGET_SECONDS))
ETA_DEADLINE_LABEL="$(format_eta "${ETA_DEADLINE_EPOCH}")"
RUNTIME_PROCESS_TITLE="$(build_eta_process_title "${ETA_DEADLINE_LABEL}")"
START_EPOCH="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TOTAL_CYCLE_SECONDS=0
COMPLETED_CYCLES=0

echo "[tune_cnn_v3_time.sh] start=${START_EPOCH} budget=${TIME_BUDGET_MINUTES}min"
echo "[tune_cnn_v3_time.sh] quick+full cycles: "\
	"quick_trials=${QUICK_TRIALS} quick_epochs=${QUICK_EPOCHS} "\
	"top_k=${TOP_K} full_epochs=${FULL_EPOCHS}"
echo "[tune_cnn_v3_time.sh] train_species=${TRAIN_SPECIES_CSV}"
echo "[tune_cnn_v3_time.sh] artifact_species=${ARTIFACT_SPECIES_RESOLVED}"
echo "[tune_cnn_v3_time.sh] seeds=${SEED_VALUES[*]}"
echo "[tune_cnn_v3_time.sh] base_pair_checkpoints=${RESOLVED_BASE_PAIR_CHECKPOINTS}"
echo "[tune_cnn_v3_time.sh] cross_train_pos=${CROSS_TRAIN_POS_PATH}"
echo "[tune_cnn_v3_time.sh] cross_train_neg=${CROSS_TRAIN_NEG_PATH}"

job_index=0
while [[ $((SECONDS - START_SECONDS)) -lt "${BUDGET_SECONDS}" ]]; do
	elapsed_seconds=$((SECONDS - START_SECONDS))
	remaining_seconds=$((BUDGET_SECONDS - elapsed_seconds))
	if [[ "${COMPLETED_CYCLES}" -gt 0 ]]; then
		avg_cycle_seconds_guard=$((TOTAL_CYCLE_SECONDS / COMPLETED_CYCLES))
		if [[ "${avg_cycle_seconds_guard}" -gt 0 ]] \
			&& [[ "${remaining_seconds}" -lt "${avg_cycle_seconds_guard}" ]]; then
			echo "[tune_cnn_v3_time.sh] stop before next cycle: "\
				"remaining=$(format_elapsed "${remaining_seconds}") "\
				"< avg_cycle=$(format_elapsed "${avg_cycle_seconds_guard}")"
			break
		fi
	fi
	remaining_hms="$(format_elapsed "${remaining_seconds}")"

	seed_index=$((job_index % ${#SEED_VALUES[@]}))
	species="${ARTIFACT_SPECIES_RESOLVED}"
	base_seed="${SEED_VALUES[${seed_index}]}"
	run_stamp="$(date +%Y%m%d_%H%M%S)"
	run_id="${run_stamp}_seed${base_seed}_c$(printf '%03d' "${job_index}")"
	output_dir="${DATA_ROOT}/${species}/tuning/${TUNING_MODEL_NAME}/pair/${run_id}"
	global_best_path="${DATA_ROOT}/${species}/tuning/${TUNING_MODEL_NAME}/pair/best_config.json"
	SEED_BEST_CONFIG_PATH=""
	if ! SEED_BEST_CONFIG_PATH="$(
		resolve_cross_species_best_seed \
			"tune_cnn_v3_time.sh" \
			"${PYTHON_BIN}" \
			"${DATA_ROOT}" \
			"${TUNING_MODEL_NAME}" \
			"${species}" \
			"pair" \
			"${global_best_path}" \
			"${CROSS_SPECIES_BEST_MODE}" \
			"${CROSS_SPECIES_BEST_OVERRIDE}" \
			"${CROSS_SPECIES_BEST_PREFERRED_SPECIES}"
	)"; then
		exit 1
	fi
	SEED_BEST_CONFIG_JSON="null"
	if [[ -n "${SEED_BEST_CONFIG_PATH}" ]]; then
		SEED_BEST_CONFIG_JSON="\"${SEED_BEST_CONFIG_PATH}\""
	fi

	objective_metric="pair_pr_auc"
	if [[ "${CHEAT_MODE}" == "on" ]]; then
		objective_metric="test_pr_auc"
	fi
	config_path="${output_dir}/hparam_search_config.json"
	mkdir -p "${output_dir}"
	TAG_JSON="$(intronmodel_json_string_or_null "${PYTHON_BIN}" "${TAG}")"
	target_search_space_json="${DEFAULT_SEARCH_SPACE_JSON_PAIR}"
	search_space_path=""
	if search_space_resolved="$(
		resolve_search_space_file \
			"${SEARCH_SPACE_FILE}" \
			"${species}" \
			"${TUNING_MODEL_NAME}"
	)"; then
		search_space_path="${search_space_resolved}"
		if ! target_space_json="$(
			normalize_json_object_file \
				"${PYTHON_BIN}" \
				"${search_space_path}" 2>&1
		)"; then
			echo "[tune_cnn_v3_time.sh] failed to parse search-space file: "\
				"${search_space_path}" >&2
			echo "[tune_cnn_v3_time.sh] parse detail: ${target_space_json}" >&2
			exit 1
		fi
		target_search_space_json="${target_space_json}"
		echo "[tune_cnn_v3_time.sh] using search space: ${search_space_path}"
	else
		search_space_status=$?
		if [[ "${search_space_status}" -eq 2 ]]; then
			exit 1
		fi
		echo "[tune_cnn_v3_time.sh] using embedded pair search space."
	fi

	cat > "${config_path}" <<JSON
{
  "project_root": "${PROJECT_ROOT}",
  "species": "${species}",
  "output_dir": "${output_dir}",
  "quick_trials": ${QUICK_TRIALS},
  "quick_epochs": ${QUICK_EPOCHS},
  "top_k": ${TOP_K},
  "full_epochs": ${FULL_EPOCHS},
  "base_seed": ${base_seed},
  "gpu_ids": "${GPU_IDS}",
  "max_parallel_trials": "${MAX_PARALLEL_TRIALS}",
  "objective_metric": "${objective_metric}",
  "global_best_config_path": "${global_best_path}",
  "seed_best_config_path": ${SEED_BEST_CONFIG_JSON},
  "search_algo": "${SEARCH_ALGO}",
  "history_top_n": ${HISTORY_TOP_N},
  "guided_random_fraction": ${GUIDED_RANDOM_FRACTION},
  "guided_mutation_rate": ${GUIDED_MUTATION_RATE},
  "min_batch_size": ${MIN_BATCH_SIZE},
  "max_oom_retries": ${MAX_OOM_RETRIES},
  "max_model_params": ${RESOLVED_MAX_MODEL_PARAMS},
	"base_args": {
		"model": "cnn_v3",
	    "species": "${species}",
		"train_target": "pair",
		"base_pair_checkpoints": "${RESOLVED_BASE_PAIR_CHECKPOINTS}",
		"meta_hidden_dim": 256,
		"meta_dropout": 0.2,
    "seed": ${base_seed},
    "donor_len": ${DONOR_LEN},
    "acceptor_len": ${ACCEPTOR_LEN},
    "val_frac": ${VAL_FRAC},
	"input_mode": "onehot",
	"pair_mode": "pair",
	"embedding_dim": 32,
	"bpe_pretrained_model_name": "zhihan1996/DNABERT-2-117M",
	"bpe_trust_remote_code": 0,
    "device": "${DEVICE}",
    "visualize": "${VISUALIZE}",
    "name_fields": "${NAME_FIELDS}",
    "tag": ${TAG_JSON},
    "sequence_transform": "${SEQUENCE_TRANSFORM}",
    "use_amp": ${USE_AMP},
    "amp_dtype": "${AMP_DTYPE}",
    "allow_tf32": ${ALLOW_TF32},
    "cudnn_benchmark": ${CUDNN_BENCHMARK},
    "deterministic": ${DETERMINISTIC},
    "num_workers": "${NUM_WORKERS}",
    "prefetch_factor": ${PREFETCH_FACTOR},
	    "persistent_workers": ${PERSISTENT_WORKERS},
	    "pin_memory": ${PIN_MEMORY},
	    "min_batch_size": ${MIN_BATCH_SIZE},
	    "max_oom_retries": ${MAX_OOM_RETRIES},
	    "train_pos_path": ${CROSS_TRAIN_POS_PATH_JSON},
	    "train_neg_path": ${CROSS_TRAIN_NEG_PATH_JSON}
	  },
  "quick_overrides": {
    "epochs": ${QUICK_EPOCHS},
    "compile_mode": "${QUICK_COMPILE_MODE}"
  },
  "full_overrides": {
    "epochs": ${FULL_EPOCHS},
    "compile_mode": "${FULL_COMPILE_MODE}"
  },
  "search_space": ${target_search_space_json}
}
JSON

	job_start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
	job_start_seconds="${SECONDS}"
	job_elapsed_hms="$(format_elapsed "${elapsed_seconds}")"
	printf '[tune_cnn_v3_time.sh] cycle=%s elapsed=%s start=%s ' \
		"${job_index}" "${job_elapsed_hms}" "${job_start}"
	printf 'ETA_remaining=%s species=%s target=pair seed=%s\n' \
		"${remaining_hms}" "${species}" "${base_seed}"
	echo "[tune_cnn_v3_time.sh] train_species=${TRAIN_SPECIES_CSV}"
	run_status=0
	intronmodel_run_with_process_title \
		"${RUNTIME_PROCESS_TITLE}" \
		"${PYTHON_BIN}" \
		"${PROJECT_ROOT}/src/tools/hparam_search.py" \
		--config "${config_path}" || run_status=$?
	if [[ "${run_status}" -eq 130 ]]; then
		echo "[tune_cnn_v3_time.sh] interrupted by user; stopping." >&2
		exit 130
	fi
	if [[ "${run_status}" -ne 0 ]]; then
		echo "[tune_cnn_v3_time.sh] cycle=${job_index} failed "\
			"species=${species} target=pair seed=${base_seed} "\
			"(exit=${run_status})" >&2
	fi
	if [[ "${UPDATE_DOUBLE_DESCENT_PLOT}" == "1" ]]; then
		run_double_descent_plot \
			"${PYTHON_BIN}" \
			"${PROJECT_ROOT}" \
			"${species}" \
			"${TUNING_MODEL_NAME}"
	fi
	cycle_duration_seconds=$((SECONDS - job_start_seconds))
	TOTAL_CYCLE_SECONDS=$((TOTAL_CYCLE_SECONDS + cycle_duration_seconds))
	COMPLETED_CYCLES=$((COMPLETED_CYCLES + 1))
	avg_cycle_seconds=$((TOTAL_CYCLE_SECONDS / COMPLETED_CYCLES))
	remaining_seconds=$((BUDGET_SECONDS - (SECONDS - START_SECONDS)))
	if [[ "${remaining_seconds}" -lt 0 ]]; then
		remaining_seconds=0
	fi
	estimated_cycles_left=0
	if [[ "${avg_cycle_seconds}" -gt 0 ]]; then
		estimated_cycles_left=$((remaining_seconds / avg_cycle_seconds))
	fi
	printf '[tune_cnn_v3_time.sh] cycle_done=%s cycle_time=%s avg_cycle=%s ' \
		"${job_index}" \
		"$(format_elapsed "${cycle_duration_seconds}")" \
		"$(format_elapsed "${avg_cycle_seconds}")"
	printf 'ETA_cycles_left=%s\n' "${estimated_cycles_left}"

	job_index=$((job_index + 1))
done

if [[ "${UPDATE_DOUBLE_DESCENT_PLOT}" == "1" ]]; then
	final_plot_species=("${ARTIFACT_SPECIES_RESOLVED}")
	for final_species in "${final_plot_species[@]}"; do
		run_double_descent_plot \
			"${PYTHON_BIN}" \
			"${PROJECT_ROOT}" \
			"${final_species}" \
			"${TUNING_MODEL_NAME}"
	done
fi

END_EPOCH="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TOTAL_SECONDS=$((SECONDS - START_SECONDS))
TOTAL_HMS="$(format_elapsed "${TOTAL_SECONDS}")"
echo "[tune_cnn_v3_time.sh] done start=${START_EPOCH} end=${END_EPOCH} "\
	"elapsed=${TOTAL_HMS} cycles=${job_index}"
