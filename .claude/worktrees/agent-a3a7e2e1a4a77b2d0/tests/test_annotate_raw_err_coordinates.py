from __future__ import annotations

import csv
import random
from pathlib import Path

import pytest

from util.annotate_raw_err_coordinates import (
    InMemoryFastaSearchIndex,
    discover_species,
    process_species,
    reverse_complement,
)


def _write_text(path: Path, text: str) -> None:
    """Write UTF-8 text to a file path."""
    path.write_text(text, encoding="utf-8")


def _random_dna(length: int, seed: int) -> str:
    """Create deterministic DNA text."""
    generator = random.Random(seed)
    alphabet = "ACGT"
    return "".join(generator.choice(alphabet) for _ in range(length))


def _fetch(seq: str, start: int, end: int) -> str:
    """Fetch a 1-based inclusive interval from a string."""
    return seq[start - 1 : end]


def _write_fasta(path: Path, contigs: dict[str, str]) -> None:
    """Write one FASTA file with wrapped sequence lines."""
    lines: list[str] = []
    for name, seq in contigs.items():
        lines.append(f">{name}")
        lines.extend(seq[index : index + 60] for index in range(0, len(seq), 60))
    _write_text(path, "\n".join(lines) + "\n")


def _prepare_species_dir(
    tmp_path: Path,
    species: str,
    contigs: dict[str, str],
    gtf_lines: list[str],
    pos_lines: list[str],
    neg_lines: list[str],
) -> Path:
    """Create one temporary species/raw directory."""
    raw_dir = tmp_path / "data" / species / "raw"
    raw_dir.mkdir(parents=True)

    fasta_path = raw_dir / "ref.fna"
    _write_fasta(fasta_path, contigs)
    _write_text(Path(f"{fasta_path}.gtf"), "\n".join(gtf_lines) + "\n")
    _write_text(raw_dir / "100bp.err", "\n".join(pos_lines) + "\n")
    _write_text(raw_dir / "100bp.neg.err", "\n".join(neg_lines) + "\n")
    return raw_dir


def _read_rows(path: Path) -> list[dict[str, str]]:
    """Read TSV rows."""
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _assert_compact_columns(rows: list[dict[str, str]]) -> None:
    """Assert the compact output schema."""
    assert rows
    assert list(rows[0].keys()) == [
        "donor_seq",
        "acceptor_seq",
        "hit_count",
        "hit_coords",
    ]


def test_search_index_finds_reverse_complement_and_n_fallback(
    tmp_path: Path,
) -> None:
    """Search exact 100bp queries on both strands and with ``N`` fallback."""
    query = "ACGTN" * 20
    raw_dir = tmp_path / "data" / "SpecSearch" / "raw"
    raw_dir.mkdir(parents=True)
    fasta_path = raw_dir / "ref.fna"
    _write_fasta(
        fasta_path,
        {
            "chr1": query + "AAAA",
            "chr2": reverse_complement(query) + "CCCC",
        },
    )

    index = InMemoryFastaSearchIndex(fasta_path=fasta_path, seed_len=16)
    hits = index.search_queries([query, query])

    assert [f"{hit.contig}:{hit.start}-{hit.end}:{hit.strand}" for hit in hits[query]] == [
        "chr1:1-100:+",
        "chr2:1-100:-",
    ]


def test_process_species_outputs_positive_and_negative_plus(tmp_path: Path) -> None:
    """Annotate positive and negative plus-strand rows."""
    seq = _random_dna(length=700, seed=11)
    donor_seq = _fetch(seq, 178, 277)
    acceptor_seq = _fetch(seq, 144, 243)

    raw_dir = _prepare_species_dir(
        tmp_path=tmp_path,
        species="SpecPlus",
        contigs={"chr1": seq},
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
            f"DEBUG donor {donor_seq} acceptor {acceptor_seq} + TXP 30",
        ],
        neg_lines=[
            f"DEBUG donor {donor_seq} +",
            f"DEBUG acceptor {acceptor_seq} +",
            f"DEBUG pair {donor_seq} {acceptor_seq} + 30",
        ],
    )

    pos_out, neg_out = process_species(
        species="SpecPlus",
        data_root=tmp_path / "data",
        output_subdir="processed",
        pos_input_name="100bp.err",
        neg_input_name="100bp.neg.err",
        seed_len=16,
    )

    assert pos_out == raw_dir.parent / "processed" / "100bp.err.coords.tsv"
    assert neg_out == raw_dir.parent / "processed" / "100bp.neg.err.coords.tsv"

    pos_rows = _read_rows(pos_out)
    neg_rows = _read_rows(neg_out)

    _assert_compact_columns(pos_rows)
    _assert_compact_columns(neg_rows)

    assert pos_rows[0]["donor_seq"] == donor_seq
    assert pos_rows[0]["acceptor_seq"] == acceptor_seq
    assert pos_rows[0]["hit_count"] == "1"
    assert pos_rows[0]["hit_coords"] == "181 240"

    assert neg_rows[0]["donor_seq"] == donor_seq
    assert neg_rows[0]["acceptor_seq"] == ""
    assert neg_rows[0]["hit_count"] == "1"
    assert neg_rows[0]["hit_coords"] == "181"

    assert neg_rows[1]["donor_seq"] == ""
    assert neg_rows[1]["acceptor_seq"] == acceptor_seq
    assert neg_rows[1]["hit_count"] == "1"
    assert neg_rows[1]["hit_coords"] == "241"

    assert neg_rows[2]["donor_seq"] == donor_seq
    assert neg_rows[2]["acceptor_seq"] == acceptor_seq
    assert neg_rows[2]["hit_count"] == "1"
    assert neg_rows[2]["hit_coords"] == "181 240"


def test_process_species_outputs_negative_minus_and_partial_pair(
    tmp_path: Path,
) -> None:
    """Annotate minus-strand rows and keep partial pair hits."""
    seq = _random_dna(length=800, seed=13)
    donor_seq = reverse_complement(_fetch(seq, 163, 262))
    acceptor_seq = reverse_complement(_fetch(seq, 208, 307))
    missing_acceptor = "A" * 100

    pos_line = f"DEBUG donor {donor_seq} acceptor {acceptor_seq} - TXM 24"
    raw_dir = _prepare_species_dir(
        tmp_path=tmp_path,
        species="SpecMinus",
        contigs={"chr1": seq},
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
        pos_lines=[pos_line],
        neg_lines=[
            f"DEBUG donor {donor_seq} -",
            f"DEBUG acceptor {acceptor_seq} -",
            f"DEBUG pair {donor_seq} {acceptor_seq} - 24",
            f"DEBUG pair {donor_seq} {missing_acceptor} - 24",
        ],
    )

    _, neg_out = process_species(
        species="SpecMinus",
        data_root=tmp_path / "data",
        output_subdir="processed",
        pos_input_name="100bp.err",
        neg_input_name="100bp.neg.err",
        seed_len=16,
    )

    assert neg_out == raw_dir.parent / "processed" / "100bp.neg.err.coords.tsv"
    neg_rows = _read_rows(neg_out)

    _assert_compact_columns(neg_rows)

    assert neg_rows[0]["hit_coords"] == "259"
    assert neg_rows[1]["hit_coords"] == "210"
    assert neg_rows[2]["hit_count"] == "1"
    assert neg_rows[2]["hit_coords"] == "211 259"
    assert neg_rows[3]["hit_count"] == "0"
    assert neg_rows[3]["hit_coords"] == ""


def test_process_species_reports_multiple_hits_across_contigs(
    tmp_path: Path,
) -> None:
    """Report stable multi-hit coordinates for donor and pair rows."""
    seq = _random_dna(length=700, seed=17)
    donor_seq = _fetch(seq, 178, 277)
    acceptor_seq = _fetch(seq, 144, 243)

    _prepare_species_dir(
        tmp_path=tmp_path,
        species="SpecMultiHit",
        contigs={"chr1": seq, "chr2": seq},
        gtf_lines=[
            (
                'chr1\ttest\texon\t151\t180\t.\t+\t.\t'
                'gene_id "g3"; transcript_id "TXP";'
            ),
            (
                'chr1\ttest\texon\t241\t270\t.\t+\t.\t'
                'gene_id "g3"; transcript_id "TXP";'
            ),
        ],
        pos_lines=[f"DEBUG donor {donor_seq} acceptor {acceptor_seq} + TXP 30"],
        neg_lines=[
            f"DEBUG donor {donor_seq} +",
            f"DEBUG pair {donor_seq} {acceptor_seq} + 30",
        ],
    )

    _, neg_out = process_species(
        species="SpecMultiHit",
        data_root=tmp_path / "data",
        output_subdir="processed",
        pos_input_name="100bp.err",
        neg_input_name="100bp.neg.err",
        seed_len=16,
    )

    neg_rows = _read_rows(neg_out)
    _assert_compact_columns(neg_rows)

    assert neg_rows[0]["hit_count"] == "2"
    assert neg_rows[0]["hit_coords"] == "181;181"

    assert neg_rows[1]["hit_count"] == "2"
    assert neg_rows[1]["hit_coords"] == "181 240;181 240"


def test_process_species_marks_half_length_mismatch(tmp_path: Path) -> None:
    """Keep positive coordinates while flagging half-length mismatch."""
    seq = _random_dna(length=700, seed=19)
    donor_seq = _fetch(seq, 178, 277)
    acceptor_seq = _fetch(seq, 144, 243)

    _prepare_species_dir(
        tmp_path=tmp_path,
        species="SpecMismatch",
        contigs={"chr1": seq},
        gtf_lines=[
            (
                'chr1\ttest\texon\t151\t180\t.\t+\t.\t'
                'gene_id "g4"; transcript_id "TXP";'
            ),
            (
                'chr1\ttest\texon\t241\t270\t.\t+\t.\t'
                'gene_id "g4"; transcript_id "TXP";'
            ),
        ],
        pos_lines=[f"DEBUG donor {donor_seq} acceptor {acceptor_seq} + TXP 999"],
        neg_lines=[f"DEBUG donor {donor_seq} +"],
    )

    pos_out, _ = process_species(
        species="SpecMismatch",
        data_root=tmp_path / "data",
        output_subdir="processed",
        pos_input_name="100bp.err",
        neg_input_name="100bp.neg.err",
        seed_len=16,
    )

    pos_rows = _read_rows(pos_out)
    _assert_compact_columns(pos_rows)
    assert pos_rows[0]["hit_count"] == "1"
    assert pos_rows[0]["hit_coords"] == "181 240"


def test_process_species_accepts_truncated_negative_windows(
    tmp_path: Path,
) -> None:
    """Keep splice-site placement for negative windows truncated by contig ends."""
    seq = _random_dna(length=1090, seed=37)
    pos_donor_seq = _fetch(seq, 178, 277)
    pos_acceptor_seq = _fetch(seq, 144, 243)
    neg_donor_seq = reverse_complement(_fetch(seq, 973, 1072))
    neg_acceptor_seq = reverse_complement(_fetch(seq, 1004, 1090))

    _prepare_species_dir(
        tmp_path=tmp_path,
        species="SpecShortNeg",
        contigs={"chr1": seq},
        gtf_lines=[
            (
                'chr1\ttest\texon\t151\t180\t.\t+\t.\t'
                'gene_id "g7"; transcript_id "TXP";'
            ),
            (
                'chr1\ttest\texon\t241\t270\t.\t+\t.\t'
                'gene_id "g7"; transcript_id "TXP";'
            ),
        ],
        pos_lines=[
            f"DEBUG donor {pos_donor_seq} acceptor {pos_acceptor_seq} + TXP 30",
        ],
        neg_lines=[
            f"DEBUG acceptor {neg_acceptor_seq} -",
            f"DEBUG pair {neg_donor_seq} {neg_acceptor_seq} - 31",
        ],
    )

    _, neg_out = process_species(
        species="SpecShortNeg",
        data_root=tmp_path / "data",
        output_subdir="processed",
        pos_input_name="100bp.err",
        neg_input_name="100bp.neg.err",
        seed_len=16,
    )

    neg_rows = _read_rows(neg_out)
    assert len(neg_rows) == 2

    acceptor_row = neg_rows[0]
    _assert_compact_columns(neg_rows)
    assert acceptor_row["donor_seq"] == ""
    assert acceptor_row["acceptor_seq"] == neg_acceptor_seq
    assert acceptor_row["hit_count"] == "1"
    assert acceptor_row["hit_coords"] == "1006"

    pair_row = neg_rows[1]
    assert pair_row["donor_seq"] == neg_donor_seq
    assert pair_row["acceptor_seq"] == neg_acceptor_seq
    assert pair_row["hit_count"] == "1"
    assert pair_row["hit_coords"] == "1007 1069"


def test_process_species_fails_with_multiple_fasta_files(tmp_path: Path) -> None:
    """Reject ambiguous reference FASTA layouts."""
    raw_dir = _prepare_species_dir(
        tmp_path=tmp_path,
        species="SpecBadRef",
        contigs={"chr1": _random_dna(700, seed=23)},
        gtf_lines=[
            (
                'chr1\ttest\texon\t151\t180\t.\t+\t.\t'
                'gene_id "g5"; transcript_id "TX";'
            ),
            (
                'chr1\ttest\texon\t241\t270\t.\t+\t.\t'
                'gene_id "g5"; transcript_id "TX";'
            ),
        ],
        pos_lines=[
            (
                "DEBUG donor "
                + _random_dna(100, seed=1)
                + " acceptor "
                + _random_dna(100, seed=2)
                + " + TX 10"
            )
        ],
        neg_lines=[f"DEBUG donor {_random_dna(100, seed=3)} +"],
    )
    extra_fasta = raw_dir / "extra.fna"
    _write_fasta(extra_fasta, {"chrX": _random_dna(200, seed=29)})
    _write_text(Path(f"{extra_fasta}.gtf"), "")

    with pytest.raises(ValueError, match="Multiple \\.fna files"):
        process_species(
            species="SpecBadRef",
            data_root=tmp_path / "data",
            output_subdir="processed",
            pos_input_name="100bp.err",
            neg_input_name="100bp.neg.err",
            seed_len=16,
        )


def test_discover_species_finds_raw_inputs(tmp_path: Path) -> None:
    """Discover species names from raw ERR files."""
    _prepare_species_dir(
        tmp_path=tmp_path,
        species="SpecOne",
        contigs={"chr1": _random_dna(700, seed=31)},
        gtf_lines=[
            (
                'chr1\ttest\texon\t151\t180\t.\t+\t.\t'
                'gene_id "g6"; transcript_id "TX";'
            ),
            (
                'chr1\ttest\texon\t241\t270\t.\t+\t.\t'
                'gene_id "g6"; transcript_id "TX";'
            ),
        ],
        pos_lines=[
            (
                "DEBUG donor "
                + _random_dna(100, seed=4)
                + " acceptor "
                + _random_dna(100, seed=5)
                + " + TX 10"
            )
        ],
        neg_lines=[f"DEBUG donor {_random_dna(100, seed=6)} +"],
    )

    species = discover_species(
        data_root=tmp_path / "data",
        pos_input_name="100bp.err",
        neg_input_name="100bp.neg.err",
    )
    assert species == ["SpecOne"]
