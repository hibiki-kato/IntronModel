from __future__ import annotations

import csv
from pathlib import Path

from tools.build_unique_intron_assets import main as build_unique_intron_assets_main


def _write_tsv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    """Write one UTF-8 TSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def test_build_unique_intron_assets_merges_duplicate_coordinates(
    tmp_path: Path,
) -> None:
    """Merge transcript introns with identical coordinates into one unique intron."""
    species_dir = tmp_path / "data" / "Dmel"
    processed_dir = species_dir / "processed"
    raw_dir = species_dir / "raw"
    processed_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)

    donor_seq = "A" * 100
    acceptor_seq = "C" * 100

    _write_tsv(
        processed_dir / "transcripts.tsv",
        [
            "transcript_id",
            "gene_id",
            "site_type",
            "intron_index",
            "chrom",
            "strand",
            "boundary_pos",
            "seq",
        ],
        [
            ["tx1", "g1", "donor", "1", "chr1", "+", "100", donor_seq],
            ["tx1", "g1", "acceptor", "1", "chr1", "+", "161", acceptor_seq],
            ["tx2", "g2", "donor", "7", "chr1", "+", "100", donor_seq],
            ["tx2", "g2", "acceptor", "7", "chr1", "+", "161", acceptor_seq],
        ],
    )
    _write_tsv(
        processed_dir / "intron_eval_flank10.tsv",
        [
            "species",
            "transcript_id",
            "intron_index",
            "chrom",
            "strand",
            "intron_start",
            "intron_end",
            "label",
            "donor_label",
            "acceptor_label",
        ],
        [
            ["Dmel", "tx1", "1", "chr1", "+", "100", "160", "1", "1", "1"],
            ["Dmel", "tx2", "7", "chr1", "+", "100", "160", "1", "1", "1"],
        ],
    )
    _write_tsv(
        processed_dir / "intron_full_flank10.pos.tsv",
        ["chrom", "strand", "intron_start", "intron_end"],
        [["chr1", "+", "100", "160"]],
    )
    (raw_dir / "100bp.neg.err").write_text(
        f"DEBUG pair {donor_seq} {acceptor_seq} + 30\n",
        encoding="utf-8",
    )

    _ = build_unique_intron_assets_main(
        [
            "--species",
            "Dmel",
            "--data-root",
            str(tmp_path / "data"),
            "--overwrite",
        ]
    )

    unique_transcripts = processed_dir / "transcripts.unique.tsv"
    map_tsv = processed_dir / "transcripts.unique.map.tsv"
    unique_labels = processed_dir / "intron_eval_flank10.unique.tsv"
    catalog_tsv = processed_dir / "intron_unique_catalog.tsv"

    assert unique_transcripts.is_file()
    assert map_tsv.is_file()
    assert unique_labels.is_file()
    assert catalog_tsv.is_file()

    with unique_transcripts.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 2
    assert {row["site_type"] for row in rows} == {"donor", "acceptor"}
    assert {row["transcript_id"] for row in rows} == {"uintron_00000001"}
    assert {row["intron_index"] for row in rows} == {"1"}

    with map_tsv.open("r", encoding="utf-8", newline="") as handle:
        map_rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(map_rows) == 2
    assert {row["transcript_id"] for row in map_rows} == {"tx1", "tx2"}

    with unique_labels.open("r", encoding="utf-8", newline="") as handle:
        label_rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(label_rows) == 1
    assert label_rows[0]["label"] == "1"
    assert label_rows[0]["member_count"] == "2"
    assert label_rows[0]["seen_train_pos_coord"] == "1"
    assert label_rows[0]["seen_train_neg_seq"] == "1"
    assert label_rows[0]["train_leak"] == "1"

    with catalog_tsv.open("r", encoding="utf-8", newline="") as handle:
        catalog_rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(catalog_rows) == 1
    assert catalog_rows[0]["member_count"] == "2"
    assert catalog_rows[0]["intron_start"] == "100"
    assert catalog_rows[0]["intron_end"] == "160"


def test_build_unique_intron_assets_normalizes_minus_strand_coordinates(
    tmp_path: Path,
) -> None:
    """Normalize minus-strand coordinates from donor/acceptor boundary columns."""
    species_dir = tmp_path / "data" / "Mmus"
    processed_dir = species_dir / "processed"
    raw_dir = species_dir / "raw"
    processed_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)

    donor_seq = "G" * 100
    acceptor_seq = "T" * 100

    _write_tsv(
        processed_dir / "transcripts.tsv",
        [
            "transcript_id",
            "gene_id",
            "site_type",
            "intron_index",
            "chrom",
            "strand",
            "boundary_pos",
            "seq",
        ],
        [
            ["txm", "gm", "donor", "3", "chrM", "-", "300", donor_seq],
            ["txm", "gm", "acceptor", "3", "chrM", "-", "250", acceptor_seq],
        ],
    )
    _write_tsv(
        processed_dir / "intron_eval_flank10.tsv",
        [
            "species",
            "transcript_id",
            "intron_index",
            "chrom",
            "strand",
            "intron_start",
            "intron_end",
            "label",
        ],
        [
            ["Mmus", "txm", "3", "chrM", "-", "251", "300", "0"],
        ],
    )
    _write_tsv(
        processed_dir / "intron_full_flank10.pos.tsv",
        ["chrom", "strand", "intron_start", "intron_end"],
        [],
    )
    (raw_dir / "100bp.neg.err").write_text("", encoding="utf-8")

    _ = build_unique_intron_assets_main(
        [
            "--species",
            "Mmus",
            "--data-root",
            str(tmp_path / "data"),
            "--overwrite",
        ]
    )

    catalog_tsv = processed_dir / "intron_unique_catalog.tsv"
    with catalog_tsv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 1
    assert rows[0]["strand"] == "-"
    assert rows[0]["intron_start"] == "251"
    assert rows[0]["intron_end"] == "300"
