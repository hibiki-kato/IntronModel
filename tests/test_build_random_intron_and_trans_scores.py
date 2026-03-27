from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

from tools.build_random_intron_and_trans_scores import _expand_unique_rows
from tools.build_random_intron_and_trans_scores import main
from util.unique_intron import UniqueMapMember


def _write_tsv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    """Write one TSV file for a test fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def test_expand_unique_rows_maps_members() -> None:
    """Expand one unique intron row to all original transcript members."""
    unique_rows = [
        {"transcript_id": "uintron_00000001", "intron_index": 1, "score": 0.25},
    ]
    unique_map = {
        ("uintron_00000001", 1): [
            UniqueMapMember(transcript_id="tx1", intron_index=1),
            UniqueMapMember(transcript_id="tx2", intron_index=4),
        ]
    }

    expanded = _expand_unique_rows(unique_rows=unique_rows, unique_map=unique_map)

    assert expanded == [
        {"transcript_id": "tx1", "intron_index": 1, "score": 0.25},
        {"transcript_id": "tx2", "intron_index": 4, "score": 0.25},
    ]


def test_main_writes_random_intron_and_trans_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Write deterministic random intron and transcript score TSV files."""
    data_root = tmp_path / "data"
    species_dir = data_root / "SpX"

    _write_tsv(
        species_dir / "processed" / "intron_eval_flank10.unique.tsv",
        [
            "species",
            "transcript_id",
            "intron_index",
            "chrom",
            "strand",
            "intron_start",
            "intron_end",
            "intron_length",
            "label",
            "donor_label",
            "acceptor_label",
            "member_count",
            "seen_train_pos_coord",
            "seen_train_neg_seq",
            "train_leak",
        ],
        [
            [
                "SpX",
                "uintron_00000001",
                "1",
                "chr1",
                "+",
                "10",
                "20",
                "11",
                "1",
                "1",
                "1",
                "1",
                "0",
                "0",
                "0",
            ],
            [
                "SpX",
                "uintron_00000002",
                "1",
                "chr1",
                "+",
                "30",
                "40",
                "11",
                "0",
                "0",
                "0",
                "2",
                "0",
                "0",
                "0",
            ],
        ],
    )
    _write_tsv(
        species_dir / "processed" / "transcripts.unique.map.tsv",
        [
            "unique_transcript_id",
            "unique_intron_index",
            "transcript_id",
            "intron_index",
        ],
        [
            ["uintron_00000001", "1", "tx1", "1"],
            ["uintron_00000002", "1", "tx1", "2"],
            ["uintron_00000002", "1", "tx2", "1"],
        ],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_random_intron_and_trans_scores.py",
            "--data-root",
            str(data_root),
            "--species",
            "SpX",
            "--output-stem",
            "random",
            "--seed",
            "7",
        ],
    )

    assert main() == 0

    intron_path = species_dir / "intron_score" / "random.tsv"
    trans_path = species_dir / "trans_score" / "random.tsv"
    assert intron_path.is_file()
    assert trans_path.is_file()

    with intron_path.open("r", encoding="utf-8", newline="") as handle:
        intron_rows = list(csv.DictReader(handle, delimiter="\t"))
    with trans_path.open("r", encoding="utf-8", newline="") as handle:
        trans_rows = list(csv.DictReader(handle, delimiter="\t"))

    assert len(intron_rows) == 2
    assert len(trans_rows) == 2
    assert {row["transcript_id"] for row in trans_rows} == {"tx1", "tx2"}

    intron_scores = {row["intron_id"]: float(row["score"]) for row in intron_rows}
    tx1_score = next(
        float(row["trans_score"])
        for row in trans_rows
        if row["transcript_id"] == "tx1"
    )
    tx2_score = next(
        float(row["trans_score"])
        for row in trans_rows
        if row["transcript_id"] == "tx2"
    )

    assert tx1_score == pytest.approx(
        min(intron_scores["uintron_00000001"], intron_scores["uintron_00000002"])
    )
    assert tx2_score == pytest.approx(intron_scores["uintron_00000002"])
