#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOT'
Usage: bash run/eval_intron_pr_auc.sh [options]

Options:
  --species <csv>              Species list (default: Dmel,Mmus,Athal)
  --data-root <path>           Data root (default: <repo>/data)
  --labeled-tsv <path>         Override labeled intron TSV path
  --labeled-name <filename>    Labeled TSV under data/<species>/raw
                               (default: intron_eval_flank10.tsv)
  --site-score-tsv <path>      Evaluate only this site_score TSV
  --site-score-pattern <glob>  Pattern under data/<species>/site_score
                               (default: *.tsv)
  --intron-score-op <op>       + | * | harmonic | min (default: *)
  --score-source <mode>        auto | donor_acceptor | pair (default: auto)
  --strict-missing             Fail if labeled introns lack usable scores
  --output-dir <path>          Base output dir
                               (default: data/<species>/eval_score/intron_pr_auc)
  --summary-name <filename>    Summary TSV name (default: summary.tsv)
  --write-rows                 Also write per-intron scored rows TSV
  -h, --help                   Show this help
EOT
}

# --------------------------
# USER DEFAULTS (optional edit)
# --------------------------
CONDA_ENV="intronmodel"
SPECIES="Dmel,Mmus,Athal,Hsap"
DATA_ROOT=""
LABELED_TSV=""
LABELED_NAME="intron_eval_flank10.tsv"
SITE_SCORE_TSV=""
SITE_SCORE_PATTERN="*.tsv"
INTRON_SCORE_OP="*"
SCORE_SOURCE="auto"
STRICT_MISSING="0"
OUTPUT_DIR=""
SUMMARY_NAME="summary.tsv"
WRITE_ROWS="0"

# --------------------------
# Runtime implementation
# --------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"
intronmodel_activate_conda "${CONDA_ENV}"
intronmodel_init_paths "${BASH_SOURCE[0]}"

# Auto-run inside tmux on SSH so jobs survive disconnects.
# Set INTRONMODEL_AUTO_TMUX=off|on|auto (default: auto).
intronmodel_enable_auto_tmux "${PROJECT_ROOT}" "$0" "${BASH_SOURCE[0]##*/}"

while [[ $# -gt 0 ]]; do
	case "$1" in
	--species)
		SPECIES="$2"
		shift 2
		;;
	--data-root)
		DATA_ROOT="$2"
		shift 2
		;;
	--labeled-tsv)
		LABELED_TSV="$2"
		shift 2
		;;
	--labeled-name)
		LABELED_NAME="$2"
		shift 2
		;;
	--site-score-tsv)
		SITE_SCORE_TSV="$2"
		shift 2
		;;
	--site-score-pattern)
		SITE_SCORE_PATTERN="$2"
		shift 2
		;;
	--intron-score-op)
		INTRON_SCORE_OP="$2"
		shift 2
		;;
	--score-source)
		SCORE_SOURCE="$2"
		shift 2
		;;
	--strict-missing)
		STRICT_MISSING="1"
		shift
		;;
	--output-dir)
		OUTPUT_DIR="$2"
		shift 2
		;;
	--summary-name)
		SUMMARY_NAME="$2"
		shift 2
		;;
	--write-rows)
		WRITE_ROWS="1"
		shift
		;;
	-h | --help)
		usage
		exit 0
		;;
	*)
		echo "Unknown argument: $1" >&2
		usage
		exit 1
		;;
	esac
done

if [[ -z "${DATA_ROOT}" ]]; then
	DATA_ROOT="${PROJECT_ROOT}/data"
fi

python_bin="$(intronmodel_resolve_python_bin "eval_intron_pr_auc.sh")"

IFS=',' read -r -a species_tokens <<< "${SPECIES}"
for raw_species in "${species_tokens[@]}"; do
	token="$(printf '%s' "${raw_species}" | tr -d '[:space:]')"
	if [[ -z "${token}" ]]; then
		continue
	fi

	species="$(intronmodel_resolve_species_case \
		"${token}" "${DATA_ROOT}" "eval_intron_pr_auc.sh")"
	raw_dir="${DATA_ROOT}/${species}/raw"
	site_score_dir="${DATA_ROOT}/${species}/site_score"
	if [[ ! -d "${raw_dir}" ]]; then
		echo "Raw directory not found: ${raw_dir}" >&2
		exit 2
	fi
	if [[ ! -d "${site_score_dir}" ]]; then
		echo "site_score directory not found: ${site_score_dir}" >&2
		exit 2
	fi

	labeled_tsv="${LABELED_TSV}"
	if [[ -z "${labeled_tsv}" ]]; then
		labeled_tsv="${raw_dir}/${LABELED_NAME}"
	fi
	if [[ ! -f "${labeled_tsv}" ]]; then
		echo "Labeled intron TSV not found for species=${species}: ${labeled_tsv}" >&2
		exit 3
	fi

	summary_dir="${OUTPUT_DIR}"
	if [[ -z "${summary_dir}" ]]; then
		summary_dir="${DATA_ROOT}/${species}/eval_score/intron_pr_auc"
	else
		summary_dir="${OUTPUT_DIR}/${species}"
	fi
	mkdir -p "${summary_dir}"
	summary_tsv="${summary_dir}/${SUMMARY_NAME}"

	printf '%s\n' \
		"species	site_score_tsv	summary_json	used_introns	positive_count	negative_count	positive_fraction	pr_auc	roc_auc	skipped_missing_score_introns	unlabeled_site_score_introns	intron_score_op	score_source" \
		> "${summary_tsv}"

	site_files=()
	if [[ -n "${SITE_SCORE_TSV}" ]]; then
		if [[ ! -f "${SITE_SCORE_TSV}" ]]; then
			echo "site_score TSV not found: ${SITE_SCORE_TSV}" >&2
			exit 4
		fi
		site_files=("${SITE_SCORE_TSV}")
	else
		shopt -s nullglob
		site_files=("${site_score_dir}"/${SITE_SCORE_PATTERN})
		shopt -u nullglob
		if [[ ${#site_files[@]} -eq 0 ]]; then
			echo "No site_score TSV matched for species=${species}" >&2
			exit 4
		fi
	fi

	echo "[eval_intron_pr_auc.sh] species=${species}"
	echo "[eval_intron_pr_auc.sh] labeled_tsv=${labeled_tsv}"
	echo "[eval_intron_pr_auc.sh] files=${#site_files[@]}"

	for site_score_tsv in "${site_files[@]}"; do
		stem="$(basename "${site_score_tsv}")"
		stem="${stem%.tsv}"
		out_json="${summary_dir}/${stem}.json"
		out_rows=""
		if [[ "${WRITE_ROWS}" == "1" ]]; then
			out_rows="${summary_dir}/${stem}.rows.tsv"
		fi

		run_args=(
			"${python_bin}" "${PROJECT_ROOT}/src/evaluate_intron_pr_auc.py"
			--labeled-intron-tsv "${labeled_tsv}"
			--site-score-tsv "${site_score_tsv}"
			--intron-score-op "${INTRON_SCORE_OP}"
			--score-source "${SCORE_SOURCE}"
			--output-json "${out_json}"
		)
		if [[ "${STRICT_MISSING}" == "1" ]]; then
			run_args+=(--strict-missing)
		fi
		if [[ -n "${out_rows}" ]]; then
			run_args+=(--output-tsv "${out_rows}")
		fi

		echo "[eval_intron_pr_auc.sh] site_score_tsv=${site_score_tsv}"
		"${run_args[@]}"

		"${python_bin}" - "${out_json}" "${summary_tsv}" "${species}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

json_path = Path(sys.argv[1])
summary_tsv = Path(sys.argv[2])
species = sys.argv[3]
payload = json.loads(json_path.read_text(encoding="utf-8"))

row = [
    species,
    str(payload["site_score_tsv"]),
    str(json_path),
    str(payload["used_introns"]),
    str(payload["positive_count"]),
    str(payload["negative_count"]),
    str(payload["positive_fraction"]),
    str(payload["pr_auc"]),
    str(payload["roc_auc"]),
    str(payload["skipped_missing_score_introns"]),
    str(payload["unlabeled_site_score_introns"]),
    str(payload["intron_score_op"]),
    str(payload["score_source"]),
]
with summary_tsv.open("a", encoding="utf-8") as handle:
    handle.write("\t".join(row))
    handle.write("\n")
PY
	done

	echo "[eval_intron_pr_auc.sh] wrote: ${summary_tsv}"
done
