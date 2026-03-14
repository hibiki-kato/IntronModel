from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ANALYSIS_SCRIPT = Path(__file__).resolve().parents[1] / "analysis" / "script"
if str(ANALYSIS_SCRIPT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SCRIPT))

from score.best_hparam_latex import (  # noqa: E402
    build_hparam_table_rows,
    collect_best_hparams,
    generate_best_hparam_latex,
)


def _write_best_config(
    path: Path,
    *,
    objective_score: float,
    sampled_params: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "objective_score": objective_score,
        "sampled_params": sampled_params,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_collect_best_hparams_keeps_highest_objective(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_best_config(
        data_root / "SpA" / "tuning" / "cnn_pair" / "pair" / "run1" / "best_config.json",
        objective_score=0.91,
        sampled_params={"lr": 0.001, "batch_size": 128},
    )
    _write_best_config(
        data_root / "SpA" / "tuning" / "cnn_pair" / "pair" / "run2" / "best_config.json",
        objective_score=0.95,
        sampled_params={"lr": 0.002, "batch_size": 256},
    )
    _write_best_config(
        data_root / "SpB" / "tuning" / "cnn_pair" / "pair" / "run1" / "best_config.json",
        objective_score=0.90,
        sampled_params={"lr": 0.003},
    )

    summaries = collect_best_hparams(model_name="cnn_pair", data_root=data_root)

    assert [summary.species for summary in summaries] == ["SpA", "SpB"]
    spa_summary = summaries[0]
    assert spa_summary.objective_score == 0.95
    assert spa_summary.sampled_params["batch_size"] == "256"


def test_build_hparam_table_rows_uses_species_columns(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_best_config(
        data_root / "SpA" / "tuning" / "cnn_pair" / "pair" / "run1" / "best_config.json",
        objective_score=0.80,
        sampled_params={"lr": 0.001, "dropout": 0.2},
    )
    _write_best_config(
        data_root / "SpB" / "tuning" / "cnn_pair" / "pair" / "run1" / "best_config.json",
        objective_score=0.85,
        sampled_params={"lr": 0.002},
    )

    summaries = collect_best_hparams(model_name="cnn_pair", data_root=data_root)
    hparam_names, species_names, rows = build_hparam_table_rows(summaries)

    assert species_names == ["SpA", "SpB"]
    assert hparam_names == ["dropout", "lr"]
    assert rows == [["0.2", ""], ["0.001", "0.002"]]


def test_generate_best_hparam_latex_rejects_unknown_species(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_best_config(
        data_root / "SpA" / "tuning" / "cnn_pair" / "pair" / "run1" / "best_config.json",
        objective_score=0.81,
        sampled_params={"lr": 0.001},
    )

    with pytest.raises(ValueError, match="Species 'SpX' is not available"):
        generate_best_hparam_latex(
            model_name="cnn_pair",
            target_species="SpX",
            data_root=data_root,
        )


def test_generate_best_hparam_latex_uses_description_json(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_best_config(
        data_root / "SpA" / "tuning" / "cnn_pair" / "pair" / "run1" / "best_config.json",
        objective_score=0.81,
        sampled_params={"lr": 0.001, "fusion_mode": "mid"},
    )
    description_path = data_root / "tuning" / "hparam_descriptions.json"
    description_path.parent.mkdir(parents=True, exist_ok=True)
    description_path.write_text(
        json.dumps(
            {
                "common": {"lr": "Learning rate."},
                "models": {"cnn_pair": {"fusion_mode": "Feature merge stage."}},
            }
        ),
        encoding="utf-8",
    )

    output = generate_best_hparam_latex(
        model_name="cnn_pair",
        target_species="SpA",
        data_root=data_root,
    )

    assert output.description_source_path == description_path
    assert "lr: Learning rate. (best=0.001)" in output.itemize_latex
    assert "fusion\\_mode: Feature merge stage. (best=mid)" in output.itemize_latex
