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
    assert dirs["train"] == "/tmp/intron-data/Dmel/train"


def test_checkpoint_paths_respect_model_root_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTRONMODEL_MODEL_ROOT", "/tmp/intron-model")

    paths = run_model._build_checkpoint_paths("Dmel", "cnn_dlen100_alen100")

    assert paths["donor"] == "/tmp/intron-model/Dmel/donor/cnn_dlen100_alen100.pt"
    assert (
        paths["acceptor"]
        == "/tmp/intron-model/Dmel/acceptor/cnn_dlen100_alen100.pt"
    )


def test_data_root_relative_override_is_resolved_from_project_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTRONMODEL_DATA_ROOT", "external_data")

    expected_root = Path(data_proc.project_root()) / "external_data" / "Athal"
    dirs = data_proc.species_data_dirs("Athal")

    assert Path(dirs["base"]) == expected_root
