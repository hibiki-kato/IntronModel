from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tools.export_site_scores_for_jim import _convert_one_file, main


def _write_tsv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [dict(row) for row in reader]


def test_convert_one_file_assigns_sequential_id_and_drops_intron_index(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.tsv"
    output_path = tmp_path / "output.tsv"
    _write_tsv(
        input_path,
        ["transcript_id", "intron_index", "donor_score", "acceptor_score", "label"],
        [
            ["tx1", "1", "0.1", "0.2", "1"],
            ["tx1", "2", "0.3", "0.4", "0"],
        ],
    )

    written = _convert_one_file(input_path, output_path)

    assert written == 2
    assert _read_tsv(output_path) == [
        {"ID": "1", "donor_score": "0.1", "acceptor_score": "0.2", "label": "1"},
        {"ID": "2", "donor_score": "0.3", "acceptor_score": "0.4", "label": "0"},
    ]


def test_convert_one_file_raises_on_missing_required_columns(tmp_path: Path) -> None:
    input_path = tmp_path / "broken.tsv"
    output_path = tmp_path / "out.tsv"
    _write_tsv(
        input_path,
        ["transcript_id", "donor_score", "acceptor_score"],
        [["tx1", "0.1", "0.2"]],
    )

    with pytest.raises(ValueError, match="transcript_id and intron_index"):
        _convert_one_file(input_path, output_path)


def test_main_writes_jim_files_for_species(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    input_path = data_root / "Dmel" / "site_score" / "cnn.tsv"
    _write_tsv(
        input_path,
        ["transcript_id", "intron_index", "donor_score", "acceptor_score", "label"],
        [["tx1", "1", "0.7", "0.8", "1"]],
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "export_site_scores_for_jim.py",
            "--data-root",
            str(data_root),
            "--species",
            "Dmel",
        ],
    )

    exit_code = main()
    output_path = data_root / "Dmel" / "jim" / "cnn.tsv"

    assert exit_code == 0
    assert _read_tsv(output_path) == [
        {"ID": "1", "donor_score": "0.7", "acceptor_score": "0.8", "label": "1"}
    ]
