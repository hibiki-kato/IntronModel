#!/bin/bash
rm -f {rna,cds}-*/*.fa.gff
for d in $(ls -d {rna,cds}-*);do
  (cd $d && ../run_gene_finder_viterbi1.sh $d.fa)
done
gffread {rna,cds}-*/*.fa.gff > viterbi.gff
gffread {rna,cds}-*/*.gtf > viterbi_ref.gff
gffcompare -r viterbi_ref.gff viterbi.gff
cat gffcmp.stats

