#!/bin/bash
FASTA=$1
MYPATH="`dirname \"$0\"`"
MYPATH="`( cd \"$MYPATH\" && pwd )`"
echo "This code only works for Drosophila"
#psauron -i $FASTA -a 1>psauron.out 2>&1 && \
#/ccb/salz4-3/alekseyz/annotation1/psauron_test/preprocess_scores1.pl $FASTA /ccb/salz4-3/alekseyz/annotation1/Dmel/eviann/GCF_000001215.4_Release_6_plus_ISO1_MT_genomic.fna.pwm /ccb/salz4-3/alekseyz/annotation1/Dmel/eviann/GCF_000001215.4_Release_6_plus_ISO1_MT_genomic.fna.neg.pwm psauron_score.csv && \
$MYPATH/gene_finder_viterbi1.pl $FASTA out.ps.txt out.gt.txt out.ag.txt 2>out.err | tee >( grep -v region|gffread -F >$FASTA.gff)
