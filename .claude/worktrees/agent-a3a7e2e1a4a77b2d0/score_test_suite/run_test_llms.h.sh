#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd "${script_dir}/.." && pwd)
cd "$script_dir"

# shellcheck source=/dev/null
source "${project_root}/run/lib/common.sh"
intronmodel_activate_conda "${CONDA_ENV:-intronmodel}"

rm -f {rna,cds}-*/*.fa.gff
for d in {rna,cds}-*; do
  [[ -d "${d}" ]] || continue
  perl -ane '$thr=0.5; $p=$F[1]+0; $score=500*log($p/$thr); $score=$score>-150?$score:-1000; print "$F[0]\t$score\n"' Students/out.gt.$d.h.txt > $d/out.gt.txt && \
  perl -ane '$thr=0.5; $p=$F[1]+0; $score=500*log($p/$thr); $score=$score>-150?$score:-1000; print "$F[0]\t$score\n"' Students/out.ag.$d.h.txt > $d/out.ag.txt && \
  (cd $d && ../run_gene_finder_viterbi_nn.sh $d.fa)
done
gffread {rna,cds}-*/*.fa.gff > viterbi.gff
gffread {rna,cds}-*/*.gtf > viterbi_ref.gff
gffcompare -r viterbi_ref.gff viterbi.gff
cat gffcmp.stats
