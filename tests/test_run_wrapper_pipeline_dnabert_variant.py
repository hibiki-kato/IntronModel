from __future__ import annotations

from pathlib import Path

import pytest

from tools import run_wrapper_pipeline


def test_validate_dnabert_specific_accepts_variant_s() -> None:
    env = {
        "DNABERT_VARIANT": "s",
        "TRUST_REMOTE_CODE": "1",
    }
    run_wrapper_pipeline._validate_dnabert_specific(env)


def test_validate_dnabert_specific_rejects_unknown_variant() -> None:
    env = {
        "DNABERT_VARIANT": "x",
        "TRUST_REMOTE_CODE": "1",
    }
    with pytest.raises(ValueError):
        run_wrapper_pipeline._validate_dnabert_specific(env)


def test_resolve_dnabert_model_uses_dnaberts_for_variant_s(tmp_path: Path) -> None:
    env = {
        "DNABERT_VARIANT": "S",
        "PRETRAINED_MODEL_RELATIVE_PATH": "pretrained/dnabert-s",
        "PRETRAINED_MODEL_NAME": "",
    }
    run_wrapper_pipeline._resolve_dnabert_model(env, tmp_path)
    assert env["MODEL"] == "dnaberts"
    assert env["PRETRAINED_MODEL_NAME"] == str(tmp_path / "pretrained/dnabert-s")


def test_resolve_dnabert_model_uses_pair_suffix_for_pair_wrapper(
    tmp_path: Path,
) -> None:
    env = {
        "DNABERT_VARIANT": "2",
        "PRETRAINED_MODEL_RELATIVE_PATH": "pretrained/dnabert2",
        "PRETRAINED_MODEL_NAME": "",
    }
    run_wrapper_pipeline._resolve_dnabert_model(
        env,
        tmp_path,
        pair_mode=True,
    )
    assert env["MODEL"] == "dnabert2_pair"
    assert env["PRETRAINED_MODEL_NAME"] == str(tmp_path / "pretrained/dnabert2")
