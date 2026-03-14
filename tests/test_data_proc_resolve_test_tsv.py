from __future__ import annotations

from pathlib import Path

import pytest

from util import data_proc


def test_resolve_test_tsv_prefers_processed_over_raw(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Return processed/transcripts.unique.tsv when both files exist."""
    base_dir = tmp_path / "Dmel"
    raw_dir = base_dir / "raw"
    processed_dir = base_dir / "processed"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    (raw_dir / "transcripts.tsv").write_text("raw\n", encoding="utf-8")
    (processed_dir / "transcripts.unique.tsv").write_text(
        "processed\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        data_proc,
        "species_data_dirs",
        lambda _species: {
            "base": str(base_dir),
            "raw": str(raw_dir),
            "train": str(base_dir / "train"),
            "site_score": str(base_dir / "site_score"),
            "intron_score": str(base_dir / "intron_score"),
            "learning_metric": str(base_dir / "learning_metric"),
            "trans_score": str(base_dir / "trans_score"),
            "eval_score": str(base_dir / "eval_score"),
        },
    )

    resolved = data_proc.resolve_test_tsv("Dmel", None)
    assert resolved == str(processed_dir / "transcripts.unique.tsv")


def test_resolve_test_tsv_raises_when_processed_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Raise when processed/transcripts.unique.tsv does not exist."""
    base_dir = tmp_path / "Dmel"
    raw_dir = base_dir / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "transcripts.tsv").write_text("raw\n", encoding="utf-8")

    monkeypatch.setattr(
        data_proc,
        "species_data_dirs",
        lambda _species: {
            "base": str(base_dir),
            "raw": str(raw_dir),
            "train": str(base_dir / "train"),
            "site_score": str(base_dir / "site_score"),
            "intron_score": str(base_dir / "intron_score"),
            "learning_metric": str(base_dir / "learning_metric"),
            "trans_score": str(base_dir / "trans_score"),
            "eval_score": str(base_dir / "eval_score"),
        },
    )

    with pytest.raises(
        FileNotFoundError,
        match="Missing required processed unique transcript TSV",
    ):
        data_proc.resolve_test_tsv("Dmel", None)
