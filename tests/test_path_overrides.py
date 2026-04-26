from __future__ import annotations

from pathlib import Path

import pytest

import run_model
from util import data_proc


def test_species_data_dirs_respects_data_root_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTRONMODEL_DATA_ROOT", "/tmp/intron-data")

    dirs = data_proc.species_data_dirs("Dmel")

    assert dirs["base"] == "/tmp/intron-data/Dmel"
    assert dirs["raw"] == "/tmp/intron-data/Dmel/raw"
    assert dirs["processed"] == "/tmp/intron-data/Dmel/processed"
    assert dirs["train"] == "/tmp/intron-data/Dmel/train"
    assert dirs["learning_metric"] == "/tmp/intron-data/Dmel/learning_metric"


def test_checkpoint_paths_respect_model_root_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTRONMODEL_MODEL_ROOT", "/tmp/intron-model")

    paths = run_model._build_checkpoint_paths("Dmel", "cnn_dlen100_alen100")

    assert paths["donor"] == "/tmp/intron-model/Dmel/donor/cnn_dlen100_alen100.pt"
    assert paths["acceptor"] == "/tmp/intron-model/Dmel/acceptor/cnn_dlen100_alen100.pt"


def test_data_root_relative_override_is_resolved_from_project_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTRONMODEL_DATA_ROOT", "external_data")

    expected_root = Path(data_proc.project_root()) / "external_data" / "Athal"
    dirs = data_proc.species_data_dirs("Athal")

    assert Path(dirs["base"]) == expected_root


def test_resolve_train_paths_prefers_raw_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Infer default training .err files from species raw directory."""
    monkeypatch.setenv("INTRONMODEL_DATA_ROOT", str(tmp_path))
    species_root = tmp_path / "Dmel" / "raw"
    species_root.mkdir(parents=True, exist_ok=True)
    (species_root / "100bp.err").write_text("DEBUG donor AAAA +\n", encoding="utf-8")
    (species_root / "100bp.neg.err").write_text(
        "DEBUG donor CCCC +\n",
        encoding="utf-8",
    )

    pos_path, neg_path, inferred_len = data_proc.resolve_train_paths(
        species="Dmel",
        train_pos_path=None,
        train_neg_path=None,
        donor_len=100,
        acceptor_len=100,
    )

    assert Path(pos_path) == species_root / "100bp.err"
    assert Path(neg_path) == species_root / "100bp.neg.err"
    assert inferred_len == 100


def test_resolve_train_paths_uses_four_flank_required_length(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("INTRONMODEL_DATA_ROOT", str(tmp_path))
    species_root = tmp_path / "Dmel" / "raw"
    species_root.mkdir(parents=True, exist_ok=True)
    (species_root / "100bp.err").write_text("DEBUG donor AAAA +\n", encoding="utf-8")
    (species_root / "100bp.neg.err").write_text(
        "DEBUG donor CCCC +\n",
        encoding="utf-8",
    )
    (species_root / "150bp.err").write_text("DEBUG donor GGGG +\n", encoding="utf-8")
    (species_root / "150bp.neg.err").write_text(
        "DEBUG donor TTTT +\n",
        encoding="utf-8",
    )

    pos_path, neg_path, inferred_len = data_proc.resolve_train_paths(
        species="Dmel",
        train_pos_path=None,
        train_neg_path=None,
        donor_len=100,
        acceptor_len=100,
        acceptor_upstream=40,
        acceptor_downstream=90,
    )

    assert Path(pos_path) == species_root / "150bp.err"
    assert Path(neg_path) == species_root / "150bp.neg.err"
    assert inferred_len == 150
