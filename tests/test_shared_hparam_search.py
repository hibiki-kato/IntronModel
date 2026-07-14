from __future__ import annotations

import json
from pathlib import Path

import pytest

from dev.IntronModel.src.tools import shared_hparam_search


def _config(tmp_path: Path) -> shared_hparam_search.SharedSearchConfig:
    return shared_hparam_search.SharedSearchConfig(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        output_dir=tmp_path / "output",
        species=("Hsap", "Dmel"),
        task="donor",
        trials=2,
        epochs=1,
        seed=7,
        objective_metric="pr_auc",
        base_args={"model": "cnn_v4", "batch_size": 8},
        search_space={"deformable_groups": {"type": "categorical", "values": [1, 2]}},
    )


def test_aggregate_species_objectives_uses_unweighted_mean() -> None:
    assert shared_hparam_search.aggregate_species_objectives({"Hsap": 0.8, "Dmel": 0.6}) == pytest.approx(0.7)


def test_shared_best_config_path_is_not_species_scoped(tmp_path: Path) -> None:
    path = shared_hparam_search.shared_best_config_path(tmp_path, "acceptor")
    assert path == tmp_path / "tuning" / "cnn_v4_shared" / "acceptor" / "best_config.json"
    assert "Hsap" not in str(path)


def test_run_search_aggregates_all_species_and_writes_provenance(tmp_path: Path) -> None:
    config = _config(tmp_path)
    calls: list[tuple[int, str]] = []

    def fake_runner(**kwargs: object) -> dict[str, object]:
        trial_id = int(kwargs["trial_id"])
        species = str(kwargs["species"])
        calls.append((trial_id, species))
        score = {0: {"Hsap": 0.6, "Dmel": 0.8}, 1: {"Hsap": 0.9, "Dmel": 0.7}}[trial_id][species]
        return {"species": species, "status": "success", "objective_score": score}

    assert shared_hparam_search.run_search(config, species_runner=fake_runner) == 0
    payload = json.loads(shared_hparam_search.shared_best_config_path(tmp_path / "data", "donor").read_text())

    assert calls == [(0, "Hsap"), (0, "Dmel"), (1, "Hsap"), (1, "Dmel")]
    assert payload["aggregation"] == "mean"
    assert payload["species"] == ["Hsap", "Dmel"]
    assert payload["objective_score"] == pytest.approx(0.8)
    assert payload["task"] == "donor"


def test_run_search_does_not_publish_partial_species_trial(tmp_path: Path) -> None:
    config = _config(tmp_path)

    def fake_runner(**kwargs: object) -> dict[str, object]:
        species = str(kwargs["species"])
        return {"species": species, "status": "success", "objective_score": 0.7} if species == "Hsap" else {"species": species, "status": "failed"}

    assert shared_hparam_search.run_search(config, species_runner=fake_runner) == 1
    payload = json.loads(shared_hparam_search.shared_best_config_path(tmp_path / "data", "donor").read_text())
    assert payload["status"] == "no_successful_trial"
