from __future__ import annotations

import csv
from pathlib import Path

import pytest

from util.make_intron_training_data_from_err import (
    process_species,
    reverse_complement,
)


def _write_text(path: Path, text: str) -> None:
    """Write UTF-8 text to a file path."""
    path.write_text(text, encoding="utf-8")


def _repeat_sequence(length: int) -> str:
    """Create deterministic sequence with period 4."""
    alphabet = "ACGT"
    return "".join(alphabet[idx % 4] for idx in range(length))


def _fetch(seq: str, start: int, end: int) -> str:
    """Fetch 1-based inclusive interval from one in-memory sequence."""
    return seq[start - 1 : end]


def _prepare_species_dir(
    tmp_path: Path,
    species: str,
    fasta_name: str,
    seq: str,
    gtf_lines: list[str],
    pos_lines: list[str],
    neg_lines: list[str],
) -> Path:
    """Create one minimal species/raw test dataset."""
    raw_dir = tmp_path / "data" / species / "raw"
    raw_dir.mkdir(parents=True)

    fasta_path = raw_dir / fasta_name
    wrapped = "\n".join(seq[idx : idx + 60] for idx in range(0, len(seq), 60))
    _write_text(fasta_path, f">chr1\n{wrapped}\n")

    gtf_path = Path(f"{fasta_path}.gtf")
    _write_text(gtf_path, "\n".join(gtf_lines) + "\n")

    _write_text(raw_dir / "100bp.err", "\n".join(pos_lines) + "\n")
    _write_text(raw_dir / "100bp.neg.err", "\n".join(neg_lines) + "\n")

    return raw_dir


def test_plus_strand_sequence_extraction_with_flank(tmp_path: Path) -> None:
    """Extract intron+flank sequence correctly on ``+`` strand."""
    seq = _repeat_sequence(420)
    transcript_id = "TXP"
    ex1_start, ex1_end = 151, 180
    ex2_start, ex2_end = 241, 270
    intron_start = ex1_end + 1
    intron_end = ex2_start - 1

    donor_seq = _fetch(seq, intron_start - 3, intron_start + 96)
    acceptor_seq = _fetch(seq, ex2_start - 97, ex2_start + 2)
    intron_length = intron_end - intron_start + 1

    raw_dir = _prepare_species_dir(
        tmp_path=tmp_path,
        species="SpecPlus",
        fasta_name="ref.fna",
        seq=seq,
        gtf_lines=[
            (
                'chr1\ttest\texon\t151\t180\t.\t+\t.\t'
                'gene_id "g1"; transcript_id "TXP";'
            ),
            (
                'chr1\ttest\texon\t241\t270\t.\t+\t.\t'
                'gene_id "g1"; transcript_id "TXP";'
            ),
        ],
        pos_lines=[
            f"DEBUG donor {donor_seq} acceptor {acceptor_seq} + "
            f"{transcript_id} {intron_length // 2}"
        ],
        neg_lines=["DEBUG pair AAAA CCCC + 5"],
    )

    stats = process_species(
        species="SpecPlus",
        data_root=tmp_path / "data",
        flank_bp=10,
        pos_input_name="100bp.err",
        out_pos_name="intron_flank10.pos.tsv",
        out_qc_name="intron_flank10.pos.qc.tsv",
        out_neg_request_name="intron_flank10.neg_coordinate_request.tsv",
        strict=True,
    )

    assert stats.matched_rows == 1

    out_path = raw_dir / "intron_flank10.pos.tsv"
    with out_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)

    assert len(rows) == 1
    expected_seq = _fetch(seq, intron_start - 10, intron_end + 10)
    assert rows[0]["sequence"] == expected_seq
    assert rows[0]["strand"] == "+"


def test_minus_strand_sequence_is_transcript_oriented(tmp_path: Path) -> None:
    """Extract intron+flank sequence and reverse-complement on ``-`` strand."""
    seq = _repeat_sequence(430)
    transcript_id = "TXM"
    ex1_start, ex1_end = 260, 290
    ex2_start, ex2_end = 180, 210
    intron_start = ex2_end + 1
    intron_end = ex1_start - 1

    donor_start = (ex1_start - 1) - 96
    donor_end = (ex1_start - 1) + 3
    acceptor_start = ex2_end - 2
    acceptor_end = ex2_end + 97

    donor_seq = reverse_complement(_fetch(seq, donor_start, donor_end))
    acceptor_seq = reverse_complement(_fetch(seq, acceptor_start, acceptor_end))
    intron_length = intron_end - intron_start + 1

    raw_dir = _prepare_species_dir(
        tmp_path=tmp_path,
        species="SpecMinus",
        fasta_name="ref.fna",
        seq=seq,
        gtf_lines=[
            (
                'chr1\ttest\texon\t260\t290\t.\t-\t.\t'
                'gene_id "g2"; transcript_id "TXM";'
            ),
            (
                'chr1\ttest\texon\t180\t210\t.\t-\t.\t'
                'gene_id "g2"; transcript_id "TXM";'
            ),
        ],
        pos_lines=[
            f"DEBUG donor {donor_seq} acceptor {acceptor_seq} - "
            f"{transcript_id} {intron_length // 2}"
        ],
        neg_lines=["DEBUG pair AAAA CCCC - 5"],
    )

    stats = process_species(
        species="SpecMinus",
        data_root=tmp_path / "data",
        flank_bp=10,
        pos_input_name="100bp.err",
        out_pos_name="intron_flank10.pos.tsv",
        out_qc_name="intron_flank10.pos.qc.tsv",
        out_neg_request_name="intron_flank10.neg_coordinate_request.tsv",
        strict=True,
    )

    assert stats.matched_rows == 1

    out_path = raw_dir / "intron_flank10.pos.tsv"
    with out_path.open("r", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle, delimiter="\t"))

    genomic_seq = _fetch(seq, intron_start - 10, intron_end + 10)
    assert row["sequence"] == reverse_complement(genomic_seq)
    assert row["strand"] == "-"


def test_multiple_introns_selects_correct_candidate(tmp_path: Path) -> None:
    """Match the correct intron when one transcript has multiple introns."""
    seq = _repeat_sequence(500)
    transcript_id = "TXMULTI"
    exons = [(151, 180), (241, 270), (331, 360)]

    first_donor = _fetch(seq, 181 - 3, 181 + 96)
    first_acceptor = _fetch(seq, 241 - 97, 241 + 2)
    second_donor = _fetch(seq, 271 - 3, 271 + 96)
    second_acceptor = _fetch(seq, 331 - 97, 331 + 2)

    pos_lines = [
        f"DEBUG donor {second_donor} acceptor {second_acceptor} + "
        f"{transcript_id} {((330 - 271 + 1) // 2)}"
    ]

    gtf_lines = [
        (
            f'chr1\ttest\texon\t{start}\t{end}\t.\t+\t.\t'
            f'gene_id "g3"; transcript_id "{transcript_id}";'
        )
        for start, end in exons
    ]

    raw_dir = _prepare_species_dir(
        tmp_path=tmp_path,
        species="SpecMulti",
        fasta_name="ref.fna",
        seq=seq,
        gtf_lines=gtf_lines,
        pos_lines=pos_lines,
        neg_lines=["DEBUG pair AAAA CCCC + 5"],
    )

    _ = first_donor
    _ = first_acceptor

    stats = process_species(
        species="SpecMulti",
        data_root=tmp_path / "data",
        flank_bp=10,
        pos_input_name="100bp.err",
        out_pos_name="intron_flank10.pos.tsv",
        out_qc_name="intron_flank10.pos.qc.tsv",
        out_neg_request_name="intron_flank10.neg_coordinate_request.tsv",
        strict=True,
    )

    assert stats.matched_rows == 1
    out_path = raw_dir / "intron_flank10.pos.tsv"
    with out_path.open("r", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle, delimiter="\t"))

    assert row["intron_index"] == "2"
    assert row["intron_start"] == "271"
    assert row["intron_end"] == "330"


def test_strict_fails_when_no_candidate_matches(tmp_path: Path) -> None:
    """Fail in strict mode when donor/acceptor key is unmatched."""
    seq = _repeat_sequence(420)
    transcript_id = "TXNONE"

    donor_seq = _fetch(seq, 178, 277)
    acceptor_seq = _fetch(seq, 144, 243)
    mutated_donor = "N" + donor_seq[1:]

    _prepare_species_dir(
        tmp_path=tmp_path,
        species="SpecNoMatch",
        fasta_name="ref.fna",
        seq=seq,
        gtf_lines=[
            (
                'chr1\ttest\texon\t151\t180\t.\t+\t.\t'
                'gene_id "g4"; transcript_id "TXNONE";'
            ),
            (
                'chr1\ttest\texon\t241\t270\t.\t+\t.\t'
                'gene_id "g4"; transcript_id "TXNONE";'
            ),
        ],
        pos_lines=[
            f"DEBUG donor {mutated_donor} acceptor {acceptor_seq} + "
            f"{transcript_id} 30"
        ],
        neg_lines=["DEBUG pair AAAA CCCC + 5"],
    )

    with pytest.raises(ValueError, match="unmatched_rows=1"):
        _ = process_species(
            species="SpecNoMatch",
            data_root=tmp_path / "data",
            flank_bp=10,
            pos_input_name="100bp.err",
            out_pos_name="intron_flank10.pos.tsv",
            out_qc_name="intron_flank10.pos.qc.tsv",
            out_neg_request_name="intron_flank10.neg_coordinate_request.tsv",
            strict=True,
        )


def test_strict_fails_when_candidate_is_ambiguous(tmp_path: Path) -> None:
    """Fail in strict mode when one key maps to multiple introns."""
    seq = "A" * 500
    transcript_id = "TXAMB"

    donor_seq = "A" * 100
    acceptor_seq = "A" * 100

    _prepare_species_dir(
        tmp_path=tmp_path,
        species="SpecAmb",
        fasta_name="ref.fna",
        seq=seq,
        gtf_lines=[
            (
                'chr1\ttest\texon\t151\t180\t.\t+\t.\t'
                'gene_id "g5"; transcript_id "TXAMB";'
            ),
            (
                'chr1\ttest\texon\t241\t270\t.\t+\t.\t'
                'gene_id "g5"; transcript_id "TXAMB";'
            ),
            (
                'chr1\ttest\texon\t331\t360\t.\t+\t.\t'
                'gene_id "g5"; transcript_id "TXAMB";'
            ),
        ],
        pos_lines=[
            f"DEBUG donor {donor_seq} acceptor {acceptor_seq} + "
            f"{transcript_id} 30"
        ],
        neg_lines=["DEBUG pair AAAA CCCC + 5"],
    )

    with pytest.raises(ValueError, match="ambiguous_rows=1"):
        _ = process_species(
            species="SpecAmb",
            data_root=tmp_path / "data",
            flank_bp=10,
            pos_input_name="100bp.err",
            out_pos_name="intron_flank10.pos.tsv",
            out_qc_name="intron_flank10.pos.qc.tsv",
            out_neg_request_name="intron_flank10.neg_coordinate_request.tsv",
            strict=True,
        )


def test_strict_fails_on_half_length_mismatch(tmp_path: Path) -> None:
    """Fail in strict mode when half-length check does not pass."""
    seq = _repeat_sequence(420)
    transcript_id = "TXHALF"

    donor_seq = _fetch(seq, 178, 277)
    acceptor_seq = _fetch(seq, 144, 243)

    _prepare_species_dir(
        tmp_path=tmp_path,
        species="SpecHalf",
        fasta_name="ref.fna",
        seq=seq,
        gtf_lines=[
            (
                'chr1\ttest\texon\t151\t180\t.\t+\t.\t'
                'gene_id "g6"; transcript_id "TXHALF";'
            ),
            (
                'chr1\ttest\texon\t241\t270\t.\t+\t.\t'
                'gene_id "g6"; transcript_id "TXHALF";'
            ),
        ],
        pos_lines=[
            f"DEBUG donor {donor_seq} acceptor {acceptor_seq} + "
            f"{transcript_id} 999"
        ],
        neg_lines=["DEBUG pair AAAA CCCC + 5"],
    )

    with pytest.raises(ValueError, match="half_length_mismatch_rows=1"):
        _ = process_species(
            species="SpecHalf",
            data_root=tmp_path / "data",
            flank_bp=10,
            pos_input_name="100bp.err",
            out_pos_name="intron_flank10.pos.tsv",
            out_qc_name="intron_flank10.pos.qc.tsv",
            out_neg_request_name="intron_flank10.neg_coordinate_request.tsv",
            strict=True,
        )


def test_qc_schema_and_counts_with_non_strict_mode(tmp_path: Path) -> None:
    """Write QC summary with expected columns and counts in non-strict mode."""
    seq = _repeat_sequence(450)
    transcript_id = "TXQC"

    donor_seq = _fetch(seq, 178, 277)
    acceptor_seq = _fetch(seq, 144, 243)

    raw_dir = _prepare_species_dir(
        tmp_path=tmp_path,
        species="SpecQc",
        fasta_name="ref.fna",
        seq=seq,
        gtf_lines=[
            (
                'chr1\ttest\texon\t151\t180\t.\t+\t.\t'
                'gene_id "g7"; transcript_id "TXQC";'
            ),
            (
                'chr1\ttest\texon\t241\t270\t.\t+\t.\t'
                'gene_id "g7"; transcript_id "TXQC";'
            ),
        ],
        pos_lines=[
            f"DEBUG donor {donor_seq} acceptor {acceptor_seq} + TXQC 30",
            f"DEBUG donor {'N' + donor_seq[1:]} acceptor {acceptor_seq} + TXQC 30",
        ],
        neg_lines=["DEBUG pair AAAA CCCC + 5"],
    )

    stats = process_species(
        species="SpecQc",
        data_root=tmp_path / "data",
        flank_bp=10,
        pos_input_name="100bp.err",
        out_pos_name="intron_flank10.pos.tsv",
        out_qc_name="intron_flank10.pos.qc.tsv",
        out_neg_request_name="intron_flank10.neg_coordinate_request.tsv",
        strict=False,
    )

    assert stats.total_input_rows == 2
    assert stats.matched_rows == 1
    assert stats.unmatched_rows == 1

    qc_path = raw_dir / "intron_flank10.pos.qc.tsv"
    with qc_path.open("r", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle, delimiter="\t"))

    expected_columns = {
        "species",
        "total_input_rows",
        "matched_rows",
        "unmatched_rows",
        "ambiguous_rows",
        "half_length_mismatch_rows",
        "min_seq_len",
        "median_seq_len",
        "p95_seq_len",
        "p99_seq_len",
        "max_seq_len",
    }
    assert set(row.keys()) == expected_columns


def test_negative_request_file_contains_only_pair_rows(tmp_path: Path) -> None:
    """Extract only ``DEBUG pair`` rows from negative input."""
    seq = _repeat_sequence(420)

    raw_dir = _prepare_species_dir(
        tmp_path=tmp_path,
        species="SpecNeg",
        fasta_name="ref.fna",
        seq=seq,
        gtf_lines=[
            (
                'chr1\ttest\texon\t151\t180\t.\t+\t.\t'
                'gene_id "g8"; transcript_id "TXNEG";'
            ),
            (
                'chr1\ttest\texon\t241\t270\t.\t+\t.\t'
                'gene_id "g8"; transcript_id "TXNEG";'
            ),
        ],
        pos_lines=[
            (
                "DEBUG donor "
                f"{_fetch(seq, 178, 277)} acceptor {_fetch(seq, 144, 243)} "
                "+ TXNEG 30"
            )
        ],
        neg_lines=[
            "DEBUG pair ACGT TGCA + 31",
            "DEBUG donor AAAA +",
            "DEBUG acceptor CCCC -",
            "DEBUG pair TTTT GGGG - 12",
        ],
    )

    stats = process_species(
        species="SpecNeg",
        data_root=tmp_path / "data",
        flank_bp=10,
        pos_input_name="100bp.err",
        out_pos_name="intron_flank10.pos.tsv",
        out_qc_name="intron_flank10.pos.qc.tsv",
        out_neg_request_name="intron_flank10.neg_coordinate_request.tsv",
        strict=True,
    )
    assert stats.matched_rows == 1

    neg_req_path = raw_dir / "intron_flank10.neg_coordinate_request.tsv"
    with neg_req_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    assert len(rows) == 2
    assert rows[0]["source_line_no"] == "1"
    assert rows[0]["donor_seq_100bp"] == "ACGT"
    assert rows[1]["source_line_no"] == "4"
    assert rows[1]["acceptor_seq_100bp"] == "GGGG"
