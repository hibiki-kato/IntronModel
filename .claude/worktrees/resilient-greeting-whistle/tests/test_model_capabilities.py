from __future__ import annotations

from util.model_capabilities import supports_quick_full_warm_start


def test_supports_quick_full_warm_start_covers_resumable_trainers() -> None:
    assert supports_quick_full_warm_start("cnn_pair_v3") is True
    assert supports_quick_full_warm_start("bert") is True
    assert supports_quick_full_warm_start("cnn_v3_meta") is True


def test_supports_quick_full_warm_start_rejects_non_resumable_backends() -> None:
    assert supports_quick_full_warm_start("reservoir") is False
    assert supports_quick_full_warm_start("markov_xgboost") is False
    assert supports_quick_full_warm_start("") is False
