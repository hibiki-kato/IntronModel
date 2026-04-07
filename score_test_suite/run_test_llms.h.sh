#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd "${script_dir}/.." && pwd)
cd "$script_dir"

# shellcheck source=/dev/null
source "${project_root}/run/lib/common.sh"
intronmodel_activate_conda "${CONDA_ENV:-intronmodel}"

input_tag="${INPUT_TAG:-h}"
score_mode="${SCORE_MODE:-h}"
use_pair_filter="${USE_PAIR_FILTER:-0}"
pair_model="${PAIR_MODEL:-cnn_pair_v2}"
pair_device="${PAIR_DEVICE:-auto}"
pair_batch_size="${PAIR_BATCH_SIZE:-512}"
pair_score_mode="${PAIR_SCORE_MODE:-additive}"
pair_min_score="${PAIR_MIN_SCORE:--2.0}"
pair_score_center="${PAIR_SCORE_CENTER:--2.0}"
pair_score_scale="${PAIR_SCORE_SCALE:-50}"
pair_delta_min="${PAIR_DELTA_MIN:--150}"
pair_delta_max="${PAIR_DELTA_MAX:-100}"
pair_no_pair_penalty="${PAIR_NO_PAIR_PENALTY:--150}"
pair_inactive_score="${PAIR_INACTIVE_SCORE:--1000}"
pair_filter_mode="${PAIR_FILTER_MODE:-error}"
min_intron_length="${MIN_INTRON_LENGTH:-30}"
pair_best_config_path="${PAIR_BEST_CONFIG_PATH:-}"
pair_checkpoint_path="${PAIR_CHECKPOINT_PATH:-}"

case "${score_mode}" in
  c)
    donor_transform='perl -ane '\''{$score=log($F[1]*2)*500+50;$score=$score>-150?$score:-1000;print "$F[0]\t$score\n"}'\'''
    acceptor_transform="${donor_transform}"
    ;;
  exp10_c)
    donor_transform='perl -ane '\''{$p=10**$F[1];$score=log($p*2)*500+50;$score=$score>-150?$score:-1000;print "$F[0]\t$score\n"}'\'''
    acceptor_transform="${donor_transform}"
    ;;
  h)
    donor_transform='perl -ane '\''$score=$F[1]*1151.292546497023+396.5735902799727;$score=$score>-150?$score:-1000;print "$F[0]\t$score\n"'\'''
    acceptor_transform="${donor_transform}"
    ;;
  none)
    donor_transform='perl -ane '\''print "$F[0]\t$F[1]\n"'\'''
    acceptor_transform="${donor_transform}"
    ;;
  *)
    echo "[run_test_llms.h.sh] unsupported SCORE_MODE: ${score_mode}" >&2
    exit 1
    ;;
esac

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
    --site-score-mode "${pair_score_mode}"
    --pair-min-score "${pair_min_score}"
    --pair-score-center "${pair_score_center}"
    --pair-score-scale "${pair_score_scale}"
    --pair-delta-min "${pair_delta_min}"
    --pair-delta-max "${pair_delta_max}"
    --no-pair-penalty "${pair_no_pair_penalty}"
    --min-intron-length "${min_intron_length}"
    --missing-pair-model-mode "${pair_filter_mode}"
  )
  if [[ -n "${pair_best_config_path}" ]]; then
    pair_args+=(--best-config-path "${pair_best_config_path}")
  fi
  if [[ -n "${pair_checkpoint_path}" ]]; then
    pair_args+=(--pair-checkpoint-path "${pair_checkpoint_path}")
  fi
  eval "${donor_transform}" "Students/out.gt.${d}.${input_tag}.txt" > "${d}/out.gt.txt" && \
  eval "${acceptor_transform}" "Students/out.ag.${d}.${input_tag}.txt" > "${d}/out.ag.txt" && \
  if [[ "${use_pair_filter}" == "1" ]]; then \
    PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
      python3 "${project_root}/src/tools/filter_score_test_suite_pairs.py" \
        "${pair_args[@]}" ; \
  fi && \
  (cd $d && ../run_gene_finder_viterbi_nn.sh $d.fa)
done
gffread {rna,cds}-*/*.fa.gff > viterbi.gff
gffread {rna,cds}-*/*.gtf > viterbi_ref.gff
gffcompare -r viterbi_ref.gff viterbi.gff
cat gffcmp.stats
