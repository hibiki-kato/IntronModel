from __future__ import annotations

import csv
from pathlib import Path

from util.make_test_data_from_gtf import main as make_test_data_main
from util.make_test_data_from_gtf import revcomp


def _write_text(path: Path, text: str) -> None:
    """Write UTF-8 text to a file path."""
    path.write_text(text, encoding="utf-8")


def _repeat_sequence(length: int) -> str:
    """Create deterministic DNA sequence with period 4."""
    alphabet = "ACGT"
    return "".join(alphabet[idx % 4] for idx in range(length))


def _fetch(seq: str, start: int, end: int) -> str:
    """Fetch one 1-based inclusive interval from a plain sequence."""
    return seq[start - 1 : end]


def _prepare_inputs(
    tmp_path: Path,
    seq: str,
    gtf_lines: list[str],
) -> tuple[Path, Path, Path]:
    """Create FASTA/GTF inputs and return paths with output TSV path."""
    fasta_path = tmp_path / "genome.fna"
    gtf_path = tmp_path / "anno.gtf"
    out_tsv = tmp_path / "transcripts.tsv"

    wrapped = "\n".join(seq[idx : idx + 60] for idx in range(0, len(seq), 60))
    _write_text(fasta_path, f">chr1\n{wrapped}\n")
    _write_text(gtf_path, "\n".join(gtf_lines) + "\n")
    return fasta_path, gtf_path, out_tsv


def _read_rows(path: Path) -> list[dict[str, str]]:
    """Read TSV rows as dictionaries."""
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_short_intron_default_keeps_requested_window_length(tmp_path: Path) -> None:
    """Without clipping, short introns still produce fixed donor/acceptor length."""
    seq = _repeat_sequence(1200)
    fasta_path, gtf_path, out_tsv = _prepare_inputs(
        tmp_path=tmp_path,
        seq=seq,
        gtf_lines=[
            (
                'chr1\ttest\texon\t100\t199\t.\t+\t.\t'
                'transcript_id "TX1"; gene_id "G1";'
            ),
            (
                'chr1\ttest\texon\t210\t309\t.\t+\t.\t'
                'transcript_id "TX1"; gene_id "G1";'
            ),
        ],
    )

    make_test_data_main(
        [
            "--fasta",
            str(fasta_path),
            "--gtf",
            str(gtf_path),
            "--out_tsv",
            str(out_tsv),
            "--donor_len",
            "100",
            "--acceptor_len",
            "100",
        ]
    )

    rows = _read_rows(out_tsv)
    assert len(rows) == 2
    assert len(rows[0]["seq"]) == 100
    assert len(rows[1]["seq"]) == 100


def test_short_intron_clip_short_intron_writes_variable_length(tmp_path: Path) -> None:
    """With clipping enabled, short introns are not extended into opposite exon."""
    seq = _repeat_sequence(1200)
    fasta_path, gtf_path, out_tsv = _prepare_inputs(
        tmp_path=tmp_path,
        seq=seq,
        gtf_lines=[
            (
                'chr1\ttest\texon\t100\t199\t.\t+\t.\t'
                'transcript_id "TX1"; gene_id "G1";'
            ),
            (
                'chr1\ttest\texon\t210\t309\t.\t+\t.\t'
                'transcript_id "TX1"; gene_id "G1";'
            ),
        ],
    )

    make_test_data_main(
        [
            "--fasta",
            str(fasta_path),
            "--gtf",
            str(gtf_path),
            "--out_tsv",
            str(out_tsv),
            "--donor_len",
            "100",
            "--acceptor_len",
            "100",
            "--clip-short-intron",
        ]
    )

    rows = _read_rows(out_tsv)
    assert len(rows) == 2
    donor_row = next(row for row in rows if row["site_type"] == "donor")
    acceptor_row = next(row for row in rows if row["site_type"] == "acceptor")

    # intron length is 10 bp, so kept length is 3 + 10.
    assert len(donor_row["seq"]) == 13
    assert len(acceptor_row["seq"]) == 13

    expected_donor = _fetch(seq, 197, 209)
    expected_acceptor = _fetch(seq, 200, 212)
    assert donor_row["seq"] == expected_donor
    assert acceptor_row["seq"] == expected_acceptor


def test_clip_short_intron_minus_strand_keeps_transcript_orientation(
    tmp_path: Path,
) -> None:
    """Clip mode keeps minus-strand windows in transcript orientation."""
    seq = _repeat_sequence(1200)
    fasta_path, gtf_path, out_tsv = _prepare_inputs(
        tmp_path=tmp_path,
        seq=seq,
        gtf_lines=[
            (
                'chr1\ttest\texon\t260\t300\t.\t-\t.\t'
                'transcript_id "TXM"; gene_id "GM";'
            ),
            (
                'chr1\ttest\texon\t200\t220\t.\t-\t.\t'
                'transcript_id "TXM"; gene_id "GM";'
            ),
        ],
    )

    make_test_data_main(
        [
            "--fasta",
            str(fasta_path),
            "--gtf",
            str(gtf_path),
            "--out_tsv",
            str(out_tsv),
            "--donor_len",
            "100",
            "--acceptor_len",
            "100",
            "--clip-short-intron",
        ]
    )

    rows = _read_rows(out_tsv)
    assert len(rows) == 2
    donor_row = next(row for row in rows if row["site_type"] == "donor")
    acceptor_row = next(row for row in rows if row["site_type"] == "acceptor")

    # intron length is 39 bp, so kept length is 3 + 39.
    assert len(donor_row["seq"]) == 42
    assert len(acceptor_row["seq"]) == 42

    donor_expected = revcomp(_fetch(seq, 221, 262))
    acceptor_expected = revcomp(_fetch(seq, 218, 259))
    assert donor_row["seq"] == donor_expected
    assert acceptor_row["seq"] == acceptor_expected
