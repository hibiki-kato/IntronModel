#!/bin/bash
rm -f {rna,cds}-*/*.fa.gff
for d in $(echo cds-NP_477286.2 cds-NP_525044.1 rna-NM_130502.3);do
  perl -ane '{$score=log($F[1]*5)*500;$score=$score>-150?$score:-1000;print "$F[0]\t$score\n"}' Students/out.gt.$d.b.txt > $d/out.gt.txt && \
  perl -ane '{$score=log($F[1]*5)*500;$score=$score>-150?$score:-1000;print "$F[0]\t$score\n"}' Students/out.ag.$d.b.txt > $d/out.ag.txt && \
  (cd $d && ../run_gene_finder_viterbi_nn.sh $d.fa)
done
gffread {rna,cds}-*/*.fa.gff > viterbi.gff
gffread {rna,cds}-*/*.gtf > viterbi_ref.gff
gffcompare -r viterbi_ref.gff viterbi.gff
cat gffcmp.stats

