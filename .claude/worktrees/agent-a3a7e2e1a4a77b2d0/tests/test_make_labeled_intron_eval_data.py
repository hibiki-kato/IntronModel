from __future__ import annotations

import csv
from pathlib import Path

import pytest

from util.make_labeled_intron_eval_data import (
    build_labeled_intron_dataset,
    reverse_complement,
)


def _write_text(path: Path, text: str) -> None:
    """Write UTF-8 text to a file."""
    path.write_text(text, encoding="utf-8")


def _repeat_sequence(length: int) -> str:
    """Create deterministic DNA sequence with period 4."""
    alphabet = "ACGT"
    return "".join(alphabet[idx % 4] for idx in range(length))


def _fetch(seq: str, start: int, end: int) -> str:
    """Fetch one 1-based inclusive interval from an in-memory sequence."""
    return seq[start - 1 : end]


def _prepare_inputs(
    tmp_path: Path,
    seq: str,
    query_lines: list[str],
    reference_lines: list[str],
) -> tuple[Path, Path, Path]:
    """Prepare FASTA, query GTF, and reference GFF test files."""
    fasta_path = tmp_path / "ref.fna"
    query_path = tmp_path / "query.gtf"
    reference_path = tmp_path / "reference.gff"

    wrapped = "\n".join(seq[idx : idx + 60] for idx in range(0, len(seq), 60))
    _write_text(fasta_path, f">chr1\n{wrapped}\n")
    _write_text(query_path, "\n".join(query_lines) + "\n")
    _write_text(reference_path, "\n".join(reference_lines) + "\n")
    return fasta_path, query_path, reference_path


def test_build_labeled_dataset_plus_strand_positive_label(tmp_path: Path) -> None:
    """Label query intron as true when it exists in reference introns."""
    seq = _repeat_sequence(500)
    intron_start = 181
    intron_end = 240

    fasta_path, query_path, reference_path = _prepare_inputs(
        tmp_path=tmp_path,
        seq=seq,
        query_lines=[
            (
                'chr1\ttest\texon\t151\t180\t.\t+\t.\t'
                'transcript_id "QTX1"; gene_id "QG1";'
            ),
            (
                'chr1\ttest\texon\t241\t270\t.\t+\t.\t'
                'transcript_id "QTX1"; gene_id "QG1";'
            ),
        ],
        reference_lines=[
            "chr1\tref\texon\t151\t180\t.\t+\t.\ttranscript_id=RTX1;gene=RG1",
            "chr1\tref\texon\t241\t270\t.\t+\t.\ttranscript_id=RTX1;gene=RG1",
        ],
    )

    out_tsv = tmp_path / "labeled.tsv"
    stats = build_labeled_intron_dataset(
        species="SpecA",
        fasta_path=fasta_path,
        query_gtf_path=query_path,
        reference_annotation_path=reference_path,
        output_tsv_path=out_tsv,
        donor_len=100,
        acceptor_len=100,
        flank_bp=10,
    )

    assert stats.written_rows == 1
    assert stats.positive_labels == 1
    assert stats.negative_labels == 0

    with out_tsv.open("r", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle, delimiter="\t"))

    assert row["label"] == "1"
    assert row["donor_label"] == "1"
    assert row["acceptor_label"] == "1"
    assert row["intron_start"] == str(intron_start)
    assert row["intron_end"] == str(intron_end)
    assert row["donor_seq_100bp"] == _fetch(seq, intron_start - 3, intron_start + 96)
    assert row["acceptor_seq_100bp"] == _fetch(seq, 241 - 97, 241 + 2)
    assert row["intron_flank_seq"] == _fetch(seq, intron_start - 10, intron_end + 10)


def test_build_labeled_dataset_plus_strand_negative_label(tmp_path: Path) -> None:
    """Label query intron as false when reference does not contain it."""
    seq = _repeat_sequence(520)
    fasta_path, query_path, reference_path = _prepare_inputs(
        tmp_path=tmp_path,
        seq=seq,
        query_lines=[
            (
                'chr1\ttest\texon\t151\t180\t.\t+\t.\t'
                'transcript_id "QTX2"; gene_id "QG2";'
            ),
            (
                'chr1\ttest\texon\t241\t270\t.\t+\t.\t'
                'transcript_id "QTX2"; gene_id "QG2";'
            ),
        ],
        reference_lines=[
            "chr1\tref\texon\t151\t190\t.\t+\t.\ttranscript_id=RTX2;gene=RG2",
            "chr1\tref\texon\t251\t270\t.\t+\t.\ttranscript_id=RTX2;gene=RG2",
        ],
    )

    out_tsv = tmp_path / "labeled.tsv"
    stats = build_labeled_intron_dataset(
        species="SpecB",
        fasta_path=fasta_path,
        query_gtf_path=query_path,
        reference_annotation_path=reference_path,
        output_tsv_path=out_tsv,
    )

    assert stats.written_rows == 1
    assert stats.positive_labels == 0
    assert stats.negative_labels == 1

    with out_tsv.open("r", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle, delimiter="\t"))
    assert row["label"] == "0"
    assert row["donor_label"] == "0"
    assert row["acceptor_label"] == "0"


def test_build_labeled_dataset_minus_strand_orientation(tmp_path: Path) -> None:
    """Write donor/acceptor and intron+flank in transcript orientation on ``-``."""
    seq = _repeat_sequence(600)
    intron_start = 211
    intron_end = 259

    fasta_path, query_path, reference_path = _prepare_inputs(
        tmp_path=tmp_path,
        seq=seq,
        query_lines=[
            (
                'chr1\ttest\texon\t260\t290\t.\t-\t.\t'
                'transcript_id "QTX3"; gene_id "QG3";'
            ),
            (
                'chr1\ttest\texon\t180\t210\t.\t-\t.\t'
                'transcript_id "QTX3"; gene_id "QG3";'
            ),
        ],
        reference_lines=[
            "chr1\tref\texon\t260\t290\t.\t-\t.\ttranscript_id=RTX3;gene=RG3",
            "chr1\tref\texon\t180\t210\t.\t-\t.\ttranscript_id=RTX3;gene=RG3",
        ],
    )

    out_tsv = tmp_path / "labeled.tsv"
    stats = build_labeled_intron_dataset(
        species="SpecC",
        fasta_path=fasta_path,
        query_gtf_path=query_path,
        reference_annotation_path=reference_path,
        output_tsv_path=out_tsv,
        donor_len=100,
        acceptor_len=100,
        flank_bp=10,
    )

    assert stats.written_rows == 1
    assert stats.positive_labels == 1

    donor_seq = reverse_complement(_fetch(seq, 163, 262))
    acceptor_seq = reverse_complement(_fetch(seq, 208, 307))
    flank_seq = reverse_complement(_fetch(seq, intron_start - 10, intron_end + 10))

    with out_tsv.open("r", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle, delimiter="\t"))

    assert row["strand"] == "-"
    assert row["label"] == "1"
    assert row["donor_label"] == "1"
    assert row["acceptor_label"] == "1"
    assert row["donor_seq_100bp"] == donor_seq
    assert row["acceptor_seq_100bp"] == acceptor_seq
    assert row["intron_flank_seq"] == flank_seq


def test_build_labeled_dataset_writes_donor_acceptor_site_labels(
    tmp_path: Path,
) -> None:
    """Write donor/acceptor labels independently of full intron label."""
    seq = _repeat_sequence(520)
    fasta_path, query_path, reference_path = _prepare_inputs(
        tmp_path=tmp_path,
        seq=seq,
        query_lines=[
            (
                'chr1\ttest\texon\t151\t180\t.\t+\t.\t'
                'transcript_id "QTXS1"; gene_id "QGS1";'
            ),
            (
                'chr1\ttest\texon\t241\t270\t.\t+\t.\t'
                'transcript_id "QTXS1"; gene_id "QGS1";'
            ),
        ],
        reference_lines=[
            "chr1\tref\texon\t151\t180\t.\t+\t.\ttranscript_id=RTXS1;gene=RGS1",
            "chr1\tref\texon\t251\t270\t.\t+\t.\ttranscript_id=RTXS1;gene=RGS1",
        ],
    )

    out_tsv = tmp_path / "labeled.tsv"
    stats = build_labeled_intron_dataset(
        species="SpecS",
        fasta_path=fasta_path,
        query_gtf_path=query_path,
        reference_annotation_path=reference_path,
        output_tsv_path=out_tsv,
    )

    assert stats.written_rows == 1
    assert stats.positive_labels == 0
    assert stats.negative_labels == 1

    with out_tsv.open("r", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle, delimiter="\t"))
    assert row["label"] == "0"
    assert row["donor_label"] == "1"
    assert row["acceptor_label"] == "0"


def test_build_labeled_dataset_validates_lengths(tmp_path: Path) -> None:
    """Reject invalid donor or acceptor lengths."""
    seq = _repeat_sequence(320)
    fasta_path, query_path, reference_path = _prepare_inputs(
        tmp_path=tmp_path,
        seq=seq,
        query_lines=[
            (
                'chr1\ttest\texon\t100\t120\t.\t+\t.\t'
                'transcript_id "QTX4"; gene_id "QG4";'
            ),
            (
                'chr1\ttest\texon\t150\t180\t.\t+\t.\t'
                'transcript_id "QTX4"; gene_id "QG4";'
            ),
        ],
        reference_lines=[
            "chr1\tref\texon\t100\t120\t.\t+\t.\ttranscript_id=RTX4",
            "chr1\tref\texon\t150\t180\t.\t+\t.\ttranscript_id=RTX4",
        ],
    )

    with pytest.raises(ValueError, match="donor-len"):
        _ = build_labeled_intron_dataset(
            species="SpecD",
            fasta_path=fasta_path,
            query_gtf_path=query_path,
            reference_annotation_path=reference_path,
            output_tsv_path=tmp_path / "x.tsv",
            donor_len=2,
            acceptor_len=100,
            flank_bp=10,
        )

    with pytest.raises(ValueError, match="acceptor-len"):
        _ = build_labeled_intron_dataset(
            species="SpecD",
            fasta_path=fasta_path,
            query_gtf_path=query_path,
            reference_annotation_path=reference_path,
            output_tsv_path=tmp_path / "y.tsv",
            donor_len=100,
            acceptor_len=2,
            flank_bp=10,
        )


def test_build_labeled_dataset_fails_without_query_transcript_id(
    tmp_path: Path,
) -> None:
    """Fail early when query exon rows do not include transcript IDs."""
    seq = _repeat_sequence(320)
    fasta_path, query_path, reference_path = _prepare_inputs(
        tmp_path=tmp_path,
        seq=seq,
        query_lines=[
            'chr1\ttest\texon\t100\t120\t.\t+\t.\tgene_id "QG5";',
            'chr1\ttest\texon\t150\t180\t.\t+\t.\tgene_id "QG5";',
        ],
        reference_lines=[
            "chr1\tref\texon\t100\t120\t.\t+\t.\ttranscript_id=RTX5",
            "chr1\tref\texon\t150\t180\t.\t+\t.\ttranscript_id=RTX5",
        ],
    )

    with pytest.raises(ValueError, match="Missing transcript ID"):
        _ = build_labeled_intron_dataset(
            species="SpecE",
            fasta_path=fasta_path,
            query_gtf_path=query_path,
            reference_annotation_path=reference_path,
            output_tsv_path=tmp_path / "z.tsv",
        )
