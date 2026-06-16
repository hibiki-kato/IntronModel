from __future__ import annotations

import json
from pathlib import Path
import pickle
import sys

import numpy as np

ANALYSIS_SRC = Path(__file__).resolve().parents[1] / "analysis" / "src"
if str(ANALYSIS_SRC) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SRC))

from score.model_performance_comparison import (  # noqa: E402
    compute_max_f1_over_threshold,
    count_checkpoint_parameters,
    infer_model_family,
    resolve_family_parameter_count,
)


def test_infer_model_family_for_prefixed_names() -> None:
    assert infer_model_family("cnn_pair") == "cnn_pair"
    assert infer_model_family("markov_xgboost_dlen100") == "markov_xgboost"
    assert infer_model_family("cnn_resdil") == "cnn_resdil"
    assert infer_model_family("cnn_iopx_taggsoftmin") == "cnn"
    assert infer_model_family("dnabert6") == "dnabert6"
    assert infer_model_family("dnabert2_variant") == "dnabert2"


def test_count_checkpoint_parameters_from_pickle_model_state(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pkl"
    payload = {
        "model_state": {
            "weight": np.zeros((2, 3), dtype=np.float32),
            "bias": np.zeros((4,), dtype=np.float32),
        }
    }
    with checkpoint_path.open("wb") as handle:
        pickle.dump(payload, handle)

    parameter_count, source = count_checkpoint_parameters(checkpoint_path)

    assert parameter_count == 10
    assert source == "model_state"


def test_resolve_family_parameter_count_sums_donor_and_acceptor(
    tmp_path: Path,
) -> None:
    species_dir = tmp_path / "data" / "SpX"
    donor_ckpt = tmp_path / "model" / "SpX" / "donor.pt"
    acceptor_ckpt = tmp_path / "model" / "SpX" / "acceptor.pt"
    donor_ckpt.parent.mkdir(parents=True, exist_ok=True)

    donor_payload = {
        "model_state": {
            "w": np.zeros((3, 4), dtype=np.float32),
            "b": np.zeros((2,), dtype=np.float32),
        }
    }
    acceptor_payload = {
        "model_state": {
            "w": np.zeros((5, 2), dtype=np.float32),
        }
    }
    with donor_ckpt.open("wb") as handle:
        pickle.dump(donor_payload, handle)
    with acceptor_ckpt.open("wb") as handle:
        pickle.dump(acceptor_payload, handle)

    donor_best = species_dir / "tuning" / "cnn" / "donor" / "best_config.json"
    acceptor_best = (
        species_dir / "tuning" / "cnn" / "acceptor" / "best_config.json"
    )
    donor_best.parent.mkdir(parents=True, exist_ok=True)
    acceptor_best.parent.mkdir(parents=True, exist_ok=True)

    donor_best.write_text(
        json.dumps({"donor_checkpoint_path": str(donor_ckpt)}),
        encoding="utf-8",
    )
    acceptor_best.write_text(
        json.dumps({"acceptor_checkpoint_path": str(acceptor_ckpt)}),
        encoding="utf-8",
    )

    result = resolve_family_parameter_count(species_dir=species_dir, model_family="cnn")

    assert result.total_parameters == 24
    assert result.source.startswith("checkpoint:")
    assert len(result.checkpoint_paths) == 2


def test_compute_max_f1_over_threshold() -> None:
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.6, 0.4, 0.1]

    max_f1 = compute_max_f1_over_threshold(
        labels=labels,
        scores=scores,
    )

    assert max_f1 == 1.0
