from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from models import markov_xgboost
import run_model


class _FakeXGBClassifier:
    """Minimal deterministic binary classifier used for unit tests."""

    def __init__(self, **kwargs: object) -> None:
        del kwargs
        self._weights: np.ndarray | None = None
        self._bias: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> object:
        if x.ndim != 2:
            raise ValueError("x must be 2-D.")
        if y.ndim != 1:
            raise ValueError("y must be 1-D.")
        pos = x[y == 1]
        neg = x[y == 0]
        if pos.size == 0 or neg.size == 0:
            raise ValueError("Both classes are required.")

        pos_mean = np.mean(pos, axis=0)
        neg_mean = np.mean(neg, axis=0)
        self._weights = pos_mean - neg_mean
        self._bias = float(-0.5 * np.dot(pos_mean + neg_mean, self._weights))
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("Model is not fitted.")
        logits = x @ self._weights + self._bias
        probs = 1.0 / (1.0 + np.exp(-logits))
        probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
        return np.column_stack([1.0 - probs, probs])


def test_markov_log_odds_prefers_matching_class_sequences() -> None:
    positive_sequences = ["AAAAT", "AAAAG", "AAAAC", "AAAAT"]
    negative_sequences = ["CCCCA", "CCCCG", "CCCCT", "CCCCA"]

    pos_model = markov_xgboost._fit_markov_pwm_model(
        positive_sequences,
        order=2,
        alpha=0.5,
    )
    neg_model = markov_xgboost._fit_markov_pwm_model(
        negative_sequences,
        order=2,
        alpha=0.5,
    )

    pos_like_score = markov_xgboost._markov_log_odds_score(
        "AAAAT",
        positive_model=pos_model,
        negative_model=neg_model,
    )
    neg_like_score = markov_xgboost._markov_log_odds_score(
        "CCCCA",
        positive_model=pos_model,
        negative_model=neg_model,
    )

    assert pos_like_score > 0.0
    assert neg_like_score < 0.0


def test_run_model_parser_accepts_markov_order_argument() -> None:
    parser = run_model._build_parser(
        selected_model="markov_xgboost",
        skip_model_import_error=False,
    )
    args = parser.parse_args(
        [
            "--model",
            "markov_xgboost",
            "--markov_order",
            "3",
        ]
    )

    assert args.model == "markov_xgboost"
    assert int(args.markov_order) == 3
    assert args.markov_feature_mode == "per_base"
    assert args.markov_cache_mode == "auto"
    assert args.markov_cache_dir is None
    assert int(args.batch_size) == 1
    assert args.train_target == "pair"


def test_compute_pair_markov_features_supports_per_base_mode() -> None:
    donor_sequences = ["AAAAA", "AAAAT", "CCCCC", "CCCCG"]
    acceptor_sequences = ["TTTTT", "TTTTA", "GGGGG", "GGGGA"]
    labels = np.asarray([1, 1, 0, 0], dtype=np.int32)
    pair_models = markov_xgboost._build_pair_markov_models(
        donor_sequences=donor_sequences,
        acceptor_sequences=acceptor_sequences,
        labels=labels,
        order=2,
        alpha=0.5,
    )

    per_base_features = markov_xgboost._compute_pair_markov_features(
        donor_sequences,
        acceptor_sequences,
        pair_models=pair_models,
        feature_mode="per_base",
    )
    summary_features = markov_xgboost._compute_pair_markov_features(
        donor_sequences,
        acceptor_sequences,
        pair_models=pair_models,
        feature_mode="pair_summary",
    )

    assert per_base_features.shape == (4, 10)
    assert summary_features.shape == (4, 2)


def test_markov_xgboost_train_and_infer_with_fake_classifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(markov_xgboost, "_ImportedXGBClassifier", _FakeXGBClassifier)

    train_pos = tmp_path / "pos.err"
    train_neg = tmp_path / "neg.err"
    train_pos.write_text(
        "\n".join(
            [
                "DEBUG pair AAAAAT TTTAAG + 10",
                "DEBUG pair AAAAAG TTTAAG + 10",
                "DEBUG pair AAAAAC TTTAAA + 10",
                "DEBUG pair AAAATA TTTAAG + 10",
                "DEBUG pair AATAAT TTAAAG + 10",
                "DEBUG pair AAATAT TTTAAC + 10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    train_neg.write_text(
        "\n".join(
            [
                "DEBUG pair CCCCCA GGGGGA + 10",
                "DEBUG pair CCCCGC GGGGGC + 10",
                "DEBUG pair CCGCCC GGGCGC + 10",
                "DEBUG pair CGCCCC GGGCGG + 10",
                "DEBUG pair CCCCCG GGGGGT + 10",
                "DEBUG pair CCCCCC GGGGGG + 10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    checkpoint_path = tmp_path / "pair.pt"
    common_train_args = argparse.Namespace(
        species="Dmel",
        train_pos_path=str(train_pos),
        train_neg_path=str(train_neg),
        donor_len=6,
        acceptor_len=6,
        seed=1337,
        pair_checkpoint_path=str(checkpoint_path),
    )
    model_train_args = argparse.Namespace(
        train_target="pair",
        sequence_transform="none",
        markov_order=2,
        markov_alpha=0.5,
        markov_feature_mode="per_base",
        markov_cache_mode="off",
        markov_cache_dir=None,
        val_frac=0.25,
        xgb_n_estimators=50,
        xgb_max_depth=3,
        xgb_learning_rate=0.1,
        xgb_subsample=1.0,
        xgb_colsample_bytree=1.0,
        xgb_min_child_weight=1.0,
        xgb_reg_lambda=1.0,
        xgb_reg_alpha=0.0,
        xgb_tree_method="hist",
        xgb_n_jobs=-1,
        tag=None,
    )

    summary = markov_xgboost.train(common_train_args, model_train_args)

    assert checkpoint_path.exists()
    assert summary["model"] == "markov_xgboost"
    pair_payload = summary["pair"]
    assert isinstance(pair_payload, dict)
    assert pair_payload["checkpoint"] == str(checkpoint_path)
    assert float(pair_payload["best_score"]) >= 0.0

    test_tsv = tmp_path / "transcripts.tsv"
    test_tsv.write_text(
        "\t".join(
            [
                "transcript_id",
                "site_type",
                "intron_index",
                "seq",
                "intron_half_length",
            ]
        )
        + "\n"
        + "\n".join(
            [
                "tx1\tdonor\t1\tAAAAAT\t10",
                "tx1\tacceptor\t1\tTTTAAG\t10",
                "tx1\tdonor\t2\tCCCCCA\t10",
                "tx1\tacceptor\t2\tGGGGGA\t10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    common_infer_args = SimpleNamespace(
        species="Dmel",
        donor_len=6,
        acceptor_len=6,
        test_tsv=str(test_tsv),
        pair_checkpoint_path=str(checkpoint_path),
    )
    model_infer_args = SimpleNamespace(sequence_transform="none")

    rows = markov_xgboost.infer_site(common_infer_args, model_infer_args)

    assert len(rows) == 2
    assert all(str(row["site_type"]) == "pair" for row in rows)
    for row in rows:
        score = float(row["score"])
        assert score <= 0.0
        assert row["_score_space"] == "log10"
    assert float(rows[0]["score"]) > float(rows[1]["score"])


def test_load_or_build_markov_feature_bundle_uses_cache_when_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_pos = tmp_path / "pos.err"
    train_neg = tmp_path / "neg.err"
    train_pos.write_text("DEBUG pair AAAA TTTT + 10\n", encoding="utf-8")
    train_neg.write_text("DEBUG pair CCCC GGGG + 10\n", encoding="utf-8")

    donor_sequences = ["AAAA", "AAAT", "CCCC", "CCCG"]
    acceptor_sequences = ["TTTT", "TTTA", "GGGG", "GGGA"]
    labels = np.asarray([1, 1, 0, 0], dtype=np.int32)
    pair_models = markov_xgboost._build_pair_markov_models(
        donor_sequences=donor_sequences,
        acceptor_sequences=acceptor_sequences,
        labels=labels,
        order=2,
        alpha=0.5,
    )
    cached_bundle = markov_xgboost.MarkovFeatureBundle(
        train_features=np.asarray([[1.0, 0.0], [0.5, -0.5]], dtype=np.float64),
        val_features=np.asarray([[0.0, 1.0], [-0.5, 0.5]], dtype=np.float64),
        labels_train=np.asarray([1, 0], dtype=np.int32),
        labels_val=np.asarray([1, 0], dtype=np.int32),
        final_features=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        labels_all=np.asarray([1, 0], dtype=np.int32),
        final_pair_models=pair_models,
        feature_names=("donor_markov_log_odds", "acceptor_markov_log_odds"),
        n_total=4,
        n_train=2,
        n_val=2,
    )

    compute_call_count = 0

    def _fake_compute_bundle(**kwargs: object) -> markov_xgboost.MarkovFeatureBundle:
        nonlocal compute_call_count
        compute_call_count += 1
        del kwargs
        return cached_bundle

    monkeypatch.setattr(
        markov_xgboost,
        "_compute_markov_feature_bundle",
        _fake_compute_bundle,
    )

    common_kwargs = {
        "pos_path": str(train_pos),
        "neg_path": str(train_neg),
        "donor_len": 4,
        "acceptor_len": 4,
        "markov_order": 2,
        "markov_alpha": 0.5,
        "markov_feature_mode": "pair_summary",
        "val_frac": 0.25,
        "seed": 1337,
        "sequence_transform": "none",
        "markov_cache_mode": "auto",
        "markov_cache_dir": str(tmp_path / "cache"),
    }

    first_bundle, first_hit, first_cache_path = (
        markov_xgboost._load_or_build_markov_feature_bundle(**common_kwargs)
    )
    second_bundle, second_hit, second_cache_path = (
        markov_xgboost._load_or_build_markov_feature_bundle(**common_kwargs)
    )

    assert compute_call_count == 1
    assert first_hit is False
    assert second_hit is True
    assert first_cache_path == second_cache_path
    assert first_bundle.feature_names == second_bundle.feature_names


def test_resolve_markov_cache_dir_defaults_to_species_train_dir() -> None:
    resolved = markov_xgboost._resolve_markov_cache_dir(
        species="Dmel",
        cache_mode="auto",
        cache_dir=None,
    )

    assert isinstance(resolved, str)
    assert resolved.endswith("Dmel/train/markov_xgboost_cache")
