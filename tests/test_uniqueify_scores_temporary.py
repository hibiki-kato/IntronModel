from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tools.uniqueify_scores_temporary import main


def _write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _prepare_assets(data_root: Path, species: str) -> Path:
    species_dir = data_root / species
    processed_dir = species_dir / "processed"
    _write_tsv(
        processed_dir / "transcripts.unique.map.tsv",
        fieldnames=[
            "unique_transcript_id",
            "unique_intron_index",
            "transcript_id",
            "intron_index",
            "chrom",
            "strand",
            "intron_start",
            "intron_end",
        ],
        rows=[
            {
                "unique_transcript_id": "uintron_00000001",
                "unique_intron_index": "1",
                "transcript_id": "txA",
                "intron_index": "1",
                "chrom": "chr1",
                "strand": "+",
                "intron_start": "10",
                "intron_end": "20",
            },
            {
                "unique_transcript_id": "uintron_00000001",
                "unique_intron_index": "1",
                "transcript_id": "txB",
                "intron_index": "2",
                "chrom": "chr1",
                "strand": "+",
                "intron_start": "10",
                "intron_end": "20",
            },
            {
                "unique_transcript_id": "uintron_00000002",
                "unique_intron_index": "1",
                "transcript_id": "txC",
                "intron_index": "1",
                "chrom": "chr2",
                "strand": "-",
                "intron_start": "30",
                "intron_end": "40",
            },
        ],
    )
    _write_tsv(
        processed_dir / "intron_unique_catalog.tsv",
        fieldnames=[
            "unique_transcript_id",
            "unique_intron_index",
            "chrom",
            "strand",
            "intron_start",
            "intron_end",
            "member_count",
            "seen_train_pos_coord",
            "seen_train_neg_seq",
            "seen_train_any",
        ],
        rows=[
            {
                "unique_transcript_id": "uintron_00000001",
                "unique_intron_index": "1",
                "chrom": "chr1",
                "strand": "+",
                "intron_start": "10",
                "intron_end": "20",
                "member_count": "2",
                "seen_train_pos_coord": "1",
                "seen_train_neg_seq": "0",
                "seen_train_any": "1",
            },
            {
                "unique_transcript_id": "uintron_00000002",
                "unique_intron_index": "1",
                "chrom": "chr2",
                "strand": "-",
                "intron_start": "30",
                "intron_end": "40",
                "member_count": "1",
                "seen_train_pos_coord": "0",
                "seen_train_neg_seq": "0",
                "seen_train_any": "0",
            },
        ],
    )
    return species_dir


def test_uniqueify_scores_temporary_rewrites_site_and_intron(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    species_dir = _prepare_assets(data_root, "Spec")
    site_path = species_dir / "site_score" / "model.tsv"
    intron_path = species_dir / "intron_score" / "model.tsv"
    _write_tsv(
        site_path,
        fieldnames=[
            "transcript_id",
            "intron_index",
            "donor_score",
            "acceptor_score",
            "label",
        ],
        rows=[
            {
                "transcript_id": "txB",
                "intron_index": "2",
                "donor_score": "0.900100",
                "acceptor_score": "0.800000",
                "label": "1",
            },
            {
                "transcript_id": "txA",
                "intron_index": "1",
                "donor_score": "0.900000",
                "acceptor_score": "0.800000",
                "label": "1",
            },
            {
                "transcript_id": "txC",
                "intron_index": "1",
                "donor_score": "0.100000",
                "acceptor_score": "0.200000",
                "label": "0",
            },
        ],
    )
    _write_tsv(
        intron_path,
        fieldnames=["transcript_id", "intron_index", "score", "label"],
        rows=[
            {
                "transcript_id": "txA",
                "intron_index": "1",
                "score": "0.720000",
                "label": "1",
            },
            {
                "transcript_id": "txB",
                "intron_index": "2",
                "score": "0.720050",
                "label": "1",
            },
            {
                "transcript_id": "txC",
                "intron_index": "1",
                "score": "0.050000",
                "label": "0",
            },
        ],
    )

    exit_code = main(
        [
            "--data-root",
            str(data_root),
            "--species",
            "Spec",
            "--tolerance",
            "1e-4",
        ]
    )
    assert exit_code == 0

    site_rows = _read_tsv(site_path)
    intron_rows = _read_tsv(intron_path)
    assert len(site_rows) == 2
    assert len(intron_rows) == 2

    first_site = site_rows[0]
    assert first_site["transcript_id"] == "uintron_00000001"
    assert first_site["intron_index"] == "1"
    assert first_site["source_transcript_id"] == "txA"
    assert first_site["source_intron_index"] == "1"
    assert first_site["member_count"] == "2"
    assert first_site["seen_train_any"] == "1"
    assert first_site["label"] == "1"

    first_intron = intron_rows[0]
    assert first_intron["transcript_id"] == "uintron_00000001"
    assert first_intron["intron_index"] == "1"
    assert first_intron["source_transcript_id"] == "txA"
    assert first_intron["source_intron_index"] == "1"
    assert first_intron["member_count"] == "2"
    assert first_intron["seen_train_any"] == "1"
    assert first_intron["label"] == "1"


def test_uniqueify_scores_temporary_tolerance_exceeded_raises(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    species_dir = _prepare_assets(data_root, "Spec")
    site_path = species_dir / "site_score" / "model.tsv"
    _write_tsv(
        site_path,
        fieldnames=[
            "transcript_id",
            "intron_index",
            "donor_score",
            "acceptor_score",
            "label",
        ],
        rows=[
            {
                "transcript_id": "txA",
                "intron_index": "1",
                "donor_score": "0.1",
                "acceptor_score": "0.2",
                "label": "1",
            },
            {
                "transcript_id": "txB",
                "intron_index": "2",
                "donor_score": "0.2",
                "acceptor_score": "0.2",
                "label": "1",
            },
        ],
    )

    with pytest.raises(ValueError, match="Score drift exceeds tolerance"):
        _ = main(
            [
                "--data-root",
                str(data_root),
                "--species",
                "Spec",
                "--site-pattern",
                "model.tsv",
                "--intron-pattern",
                "__none__.tsv",
                "--tolerance",
                "1e-4",
            ]
        )


def test_uniqueify_scores_temporary_missing_map_or_catalog_raises(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    species_dir = data_root / "Spec"
    site_path = species_dir / "site_score" / "model.tsv"
    _write_tsv(
        site_path,
        fieldnames=[
            "transcript_id",
            "intron_index",
            "donor_score",
            "acceptor_score",
            "label",
        ],
        rows=[
            {
                "transcript_id": "txA",
                "intron_index": "1",
                "donor_score": "0.1",
                "acceptor_score": "0.2",
                "label": "1",
            }
        ],
    )

    with pytest.raises(FileNotFoundError, match="Unique intron map TSV not found"):
        _ = main(["--data-root", str(data_root), "--species", "Spec"])

    _write_tsv(
        species_dir / "processed" / "transcripts.unique.map.tsv",
        fieldnames=[
            "unique_transcript_id",
            "unique_intron_index",
            "transcript_id",
            "intron_index",
        ],
        rows=[
            {
                "unique_transcript_id": "uintron_1",
                "unique_intron_index": "1",
                "transcript_id": "txA",
                "intron_index": "1",
            }
        ],
    )
    with pytest.raises(
        FileNotFoundError, match="Unique intron catalog TSV not found"
    ):
        _ = main(["--data-root", str(data_root), "--species", "Spec"])


def test_uniqueify_scores_temporary_unmapped_key_raises(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    species_dir = _prepare_assets(data_root, "Spec")
    site_path = species_dir / "site_score" / "model.tsv"
    _write_tsv(
        site_path,
        fieldnames=[
            "transcript_id",
            "intron_index",
            "donor_score",
            "acceptor_score",
            "label",
        ],
        rows=[
            {
                "transcript_id": "not_in_map",
                "intron_index": "99",
                "donor_score": "0.1",
                "acceptor_score": "0.2",
                "label": "1",
            }
        ],
    )
    with pytest.raises(ValueError, match="Unmapped original intron key"):
        _ = main(
            [
                "--data-root",
                str(data_root),
                "--species",
                "Spec",
                "--site-pattern",
                "model.tsv",
                "--intron-pattern",
                "__none__.tsv",
            ]
        )


def test_uniqueify_scores_temporary_label_conflict_raises(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    species_dir = _prepare_assets(data_root, "Spec")
    intron_path = species_dir / "intron_score" / "model.tsv"
    _write_tsv(
        intron_path,
        fieldnames=["transcript_id", "intron_index", "score", "label"],
        rows=[
            {
                "transcript_id": "txA",
                "intron_index": "1",
                "score": "0.7",
                "label": "1",
            },
            {
                "transcript_id": "txB",
                "intron_index": "2",
                "score": "0.7",
                "label": "0",
            },
        ],
    )
    with pytest.raises(ValueError, match="Conflicting labels"):
        _ = main(
            [
                "--data-root",
                str(data_root),
                "--species",
                "Spec",
                "--site-pattern",
                "__none__.tsv",
                "--intron-pattern",
                "model.tsv",
            ]
        )


def test_uniqueify_scores_temporary_dry_run_does_not_overwrite(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    species_dir = _prepare_assets(data_root, "Spec")
    site_path = species_dir / "site_score" / "model.tsv"
    _write_tsv(
        site_path,
        fieldnames=[
            "transcript_id",
            "intron_index",
            "donor_score",
            "acceptor_score",
            "label",
        ],
        rows=[
            {
                "transcript_id": "txA",
                "intron_index": "1",
                "donor_score": "0.5000",
                "acceptor_score": "0.6000",
                "label": "1",
            }
        ],
    )
    before = site_path.read_text(encoding="utf-8")

    exit_code = main(
        [
            "--data-root",
            str(data_root),
            "--species",
            "Spec",
            "--site-pattern",
            "model.tsv",
            "--intron-pattern",
            "__none__.tsv",
            "--dry-run",
            "1",
        ]
    )
    assert exit_code == 0
    after = site_path.read_text(encoding="utf-8")
    assert before == after
