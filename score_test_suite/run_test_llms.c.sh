#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir"

rm -f {rna,cds}-*/*.fa.gff
for d in $(ls -d {rna,cds}-*);do
  perl -ane '{$score=log($F[1]*2)*500+50;$score=$score>-150?$score:-1000;print "$F[0]\t$score\n"}' Students/out.gt.$d.c.txt > $d/out.gt.txt && \
  perl -ane '{$score=log($F[1]*2)*500+50;$score=$score>-150?$score:-1000;print "$F[0]\t$score\n"}' Students/out.ag.$d.c.txt > $d/out.ag.txt && \
  (cd $d && ../run_gene_finder_viterbi_nn.sh $d.fa)
done
gffread {rna,cds}-*/*.fa.gff > viterbi.gff
gffread {rna,cds}-*/*.gtf > viterbi_ref.gff
gffcompare -r viterbi_ref.gff viterbi.gff
cat gffcmp.stats

