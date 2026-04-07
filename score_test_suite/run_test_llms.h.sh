#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd "${script_dir}/.." && pwd)
cd "$script_dir"

# shellcheck source=/dev/null
source "${project_root}/run/lib/common.sh"
intronmodel_activate_conda "${CONDA_ENV:-intronmodel}"

pair_model="${PAIR_MODEL:-cnn_pair_v2}"
pair_device="${PAIR_DEVICE:-auto}"
pair_batch_size="${PAIR_BATCH_SIZE:-512}"
pair_min_score="${PAIR_MIN_SCORE:--2.0}"
pair_inactive_score="${PAIR_INACTIVE_SCORE:--1000}"
pair_filter_mode="${PAIR_FILTER_MODE:-skip}"
min_intron_length="${MIN_INTRON_LENGTH:-30}"
pair_best_config_path="${PAIR_BEST_CONFIG_PATH:-}"
pair_checkpoint_path="${PAIR_CHECKPOINT_PATH:-}"

rm -f {rna,cds}-*/*.fa.gff
for d in {rna,cds}-*; do
  [[ -d "${d}" ]] || continue
  pair_args=(
    --fasta "$d/$d.fa"
    --donor-input "$d/out.gt.txt"
    --acceptor-input "$d/out.ag.txt"
    --donor-output "$d/out.gt.txt"
    --acceptor-output "$d/out.ag.txt"
    --species Dmel
    --model-name "${pair_model}"
    --device "${pair_device}"
    --batch-size "${pair_batch_size}"
    --inactive-score "${pair_inactive_score}"
    --pair-min-score "${pair_min_score}"
    --min-intron-length "${min_intron_length}"
    --missing-pair-model-mode "${pair_filter_mode}"
  )
  if [[ -n "${pair_best_config_path}" ]]; then
    pair_args+=(--best-config-path "${pair_best_config_path}")
  fi
  if [[ -n "${pair_checkpoint_path}" ]]; then
    pair_args+=(--pair-checkpoint-path "${pair_checkpoint_path}")
  fi
  perl -ane '$score=$F[1]*1151.292546497023+396.5735902799727;$score=$score>-150?$score:-1000;print "$F[0]\t$score\n"' Students/out.gt.$d.h.txt > $d/out.gt.txt && \
  perl -ane '$score=$F[1]*1151.292546497023+396.5735902799727;$score=$score>-150?$score:-1000;print "$F[0]\t$score\n"' Students/out.ag.$d.h.txt > $d/out.ag.txt && \
  PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    python3 "${project_root}/src/tools/filter_score_test_suite_pairs.py" \
      "${pair_args[@]}" && \
  (cd $d && ../run_gene_finder_viterbi_nn.sh $d.fa)
done
gffread {rna,cds}-*/*.fa.gff > viterbi.gff
gffread {rna,cds}-*/*.gtf > viterbi_ref.gff
gffcompare -r viterbi_ref.gff viterbi.gff
cat gffcmp.stats
