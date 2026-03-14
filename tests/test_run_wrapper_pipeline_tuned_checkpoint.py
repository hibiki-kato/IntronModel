from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.run_wrapper_pipeline import (
    SPECS,
    _apply_cheat_mode_defaults,
    _harmonize_min_batch_size,
    _apply_tuned_overrides,
    _resolve_tuned_model_name,
    _resolve_tuned_checkpoint_path,
    _resolve_tuned_config_path,
)


def test_resolve_tuned_checkpoint_path_prefers_direct_field(tmp_path: Path) -> None:
    donor_ckpt = tmp_path / "donor_best.pt"
    donor_ckpt.write_bytes(b"donor")
    config_path = tmp_path / "best_config.json"
    config_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "donor_checkpoint_path": str(donor_ckpt),
            }
        ),
        encoding="utf-8",
    )

    resolved = _resolve_tuned_checkpoint_path(
        task="donor",
        tuned_config_path=config_path,
    )
    assert resolved == donor_ckpt.resolve()


def test_resolve_tuned_checkpoint_path_falls_back_to_trial_metrics(
    tmp_path: Path,
) -> None:
    donor_ckpt = tmp_path / "donor_full.pt"
    donor_ckpt.write_bytes(b"donor")
    config_path = tmp_path / "best_config.json"
    config_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "phase": "full",
                "trial_id": 7,
            }
        ),
        encoding="utf-8",
    )
    metrics_path = tmp_path / "full_trial_0007.metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "donor_checkpoint_path": str(donor_ckpt),
            }
        ),
        encoding="utf-8",
    )

    resolved = _resolve_tuned_checkpoint_path(
        task="donor",
        tuned_config_path=config_path,
    )
    assert resolved == donor_ckpt.resolve()


def test_resolve_tuned_checkpoint_path_uses_metrics_json_field(tmp_path: Path) -> None:
    donor_ckpt = tmp_path / "donor_metrics_json.pt"
    donor_ckpt.write_bytes(b"donor")
    metrics_path = tmp_path / "best.metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "donor_checkpoint_path": str(donor_ckpt),
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "best_config.json"
    config_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "metrics_json": str(metrics_path),
            }
        ),
        encoding="utf-8",
    )

    resolved = _resolve_tuned_checkpoint_path(
        task="donor",
        tuned_config_path=config_path,
    )
    assert resolved == donor_ckpt.resolve()


def test_resolve_tuned_config_path_does_not_fallback_to_other_model(
    tmp_path: Path,
) -> None:
    species = "Dmel"
    fallback_config = (
        tmp_path / species / "tuning" / "dnabert" / "acceptor" / "best_config.json"
    )
    fallback_config.parent.mkdir(parents=True, exist_ok=True)
    fallback_config.write_text(
        json.dumps({"status": "ok", "sampled_params": {"lr": 1e-5}}),
        encoding="utf-8",
    )

    resolved = _resolve_tuned_config_path(
        task="acceptor",
        explicit_path="",
        species=species,
        data_root=tmp_path,
        model_name="dnabert6",
        shared_path="",
    )
    assert resolved is None


def test_apply_tuned_overrides_uses_dnabert_variant_model_name(
    tmp_path: Path,
) -> None:
    species = "Dmel"
    tuned_config = (
        tmp_path / species / "tuning" / "dnabert6" / "acceptor" / "best_config.json"
    )
    tuned_config.parent.mkdir(parents=True, exist_ok=True)
    tuned_config.write_text(
        json.dumps(
            {
                "status": "ok",
                "sampled_params": {
                    "lr": 2.5e-5,
                    "batch_size": 12,
                    "loss": "focal",
                },
            }
        ),
        encoding="utf-8",
    )

    env: dict[str, str] = {
        "MODEL": "dnabert6",
        "SPECIES": species,
        "TRAIN_TARGET": "acceptor",
        "USE_TUNED_HPARAMS": "required",
        "ACCEPTOR_TUNED_CONFIG_PATH": "",
        "SHARED_TUNED_CONFIG_PATH": "",
        "ACCEPTOR_LR": "",
        "ACCEPTOR_BATCH_SIZE": "",
        "ACCEPTOR_LOSS": "",
    }
    resolved = _apply_tuned_overrides(SPECS["dnabert.sh"], env, tmp_path)

    assert resolved["acceptor"] == tuned_config.resolve()
    assert env["ACCEPTOR_LR"] == "2.5e-05"
    assert env["ACCEPTOR_BATCH_SIZE"] == "12"
    assert env["ACCEPTOR_LOSS"] == "focal"


def test_resolve_tuned_model_name_appends_mask_suffix_for_mask_mode() -> None:
    resolved = _resolve_tuned_model_name(
        spec=SPECS["cnn.sh"],
        model_name="cnn",
        mask_mode="on",
    )
    assert resolved == "cnn_mask"


def test_resolve_tuned_model_name_appends_cheat_suffix_for_cheat_mode() -> None:
    resolved = _resolve_tuned_model_name(
        spec=SPECS["cnn_pair.sh"],
        model_name="cnn_pair",
        mask_mode="off",
        tuned_hparams_mode="cheat",
    )
    assert resolved == "cnn_pair_cheat"


def test_resolve_tuned_model_name_combines_mask_and_cheat_suffixes() -> None:
    resolved = _resolve_tuned_model_name(
        spec=SPECS["cnn.sh"],
        model_name="cnn",
        mask_mode="on",
        tuned_hparams_mode="cheat",
    )
    assert resolved == "cnn_mask_cheat"


def test_resolve_tuned_model_name_appends_trunc_suffix_for_dnabert_pair() -> None:
    resolved = _resolve_tuned_model_name(
        spec=SPECS["dnabert_pair.sh"],
        model_name="dnabert2_pair",
        mask_mode="on",
    )
    assert resolved == "dnabert2_pair_trunc"


def test_apply_cheat_mode_defaults_appends_cheat_to_tag_and_name_fields() -> None:
    env: dict[str, str] = {
        "TUNED_HPARAMS_MODE": "cheat",
        "TAG": "exp1",
        "NAME_FIELDS": "bp_avg",
    }
    _apply_cheat_mode_defaults(env)

    assert env["TAG"] == "exp1_cheat"
    assert env["NAME_FIELDS"] == "bp_avg,tag"


def test_apply_tuned_overrides_reads_mask_tuning_dir_for_cnn(
    tmp_path: Path,
) -> None:
    species = "Dmel"
    tuned_config = (
        tmp_path / species / "tuning" / "cnn_mask" / "donor" / "best_config.json"
    )
    tuned_config.parent.mkdir(parents=True, exist_ok=True)
    tuned_config.write_text(
        json.dumps(
            {
                "status": "ok",
                "sampled_params": {
                    "lr": 0.0007,
                    "batch_size": 1024,
                },
            }
        ),
        encoding="utf-8",
    )

    env: dict[str, str] = {
        "MODEL": "cnn",
        "SPECIES": species,
        "MASK_MODE": "on",
        "TRAIN_TARGET": "donor",
        "USE_TUNED_HPARAMS": "required",
        "DONOR_TUNED_CONFIG_PATH": "",
        "SHARED_TUNED_CONFIG_PATH": "",
        "DONOR_LR": "",
        "DONOR_BATCH_SIZE": "",
    }
    resolved = _apply_tuned_overrides(SPECS["cnn.sh"], env, tmp_path)

    assert resolved["donor"] == tuned_config.resolve()
    assert env["DONOR_LR"] == "0.0007"
    assert env["DONOR_BATCH_SIZE"] == "1024"


def test_apply_tuned_overrides_reads_cheat_tuning_dir_for_cnn_pair(
    tmp_path: Path,
) -> None:
    species = "Dmel"
    tuned_config = (
        tmp_path
        / species
        / "tuning"
        / "cnn_pair_cheat"
        / "pair"
        / "best_config.json"
    )
    tuned_config.parent.mkdir(parents=True, exist_ok=True)
    tuned_config.write_text(
        json.dumps(
            {
                "status": "ok",
                "sampled_params": {
                    "lr": 0.0006,
                    "batch_size": 768,
                },
            }
        ),
        encoding="utf-8",
    )

    env: dict[str, str] = {
        "MODEL": "cnn_pair",
        "SPECIES": species,
        "TRAIN_TARGET": "pair",
        "USE_TUNED_HPARAMS": "required",
        "TUNED_HPARAMS_MODE": "cheat",
        "PAIR_TUNED_CONFIG_PATH": "",
        "SHARED_TUNED_CONFIG_PATH": "",
        "LR": "",
        "BATCH_SIZE": "",
    }
    resolved = _apply_tuned_overrides(SPECS["cnn_pair.sh"], env, tmp_path)

    assert resolved["pair"] == tuned_config.resolve()
    assert env["LR"] == "0.0006"
    assert env["BATCH_SIZE"] == "768"


def test_apply_tuned_overrides_loads_target_specific_window_len(
    tmp_path: Path,
) -> None:
    species = "Dmel"
    donor_config = (
        tmp_path / species / "tuning" / "cnn" / "donor" / "best_config.json"
    )
    donor_config.parent.mkdir(parents=True, exist_ok=True)
    donor_config.write_text(
        json.dumps(
            {
                "status": "ok",
                "sampled_params": {
                    "donor_len": 80,
                    "lr": 0.0007,
                },
            }
        ),
        encoding="utf-8",
    )

    env: dict[str, str] = {
        "MODEL": "cnn",
        "SPECIES": species,
        "TRAIN_TARGET": "donor",
        "USE_TUNED_HPARAMS": "required",
        "DONOR_TUNED_CONFIG_PATH": "",
        "SHARED_TUNED_CONFIG_PATH": "",
        "DONOR_LEN": "100",
        "ACCEPTOR_LEN": "100",
        "DONOR_LR": "0.1",
    }
    _ = _apply_tuned_overrides(SPECS["cnn.sh"], env, tmp_path)

    assert env["DONOR_LEN"] == "80"
    assert env["ACCEPTOR_LEN"] == "100"
    assert env["DONOR_LR"] == "0.0007"


def test_apply_tuned_overrides_loads_both_window_lens_for_pair_target(
    tmp_path: Path,
) -> None:
    species = "Dmel"
    pair_config = (
        tmp_path / species / "tuning" / "cnn_pair" / "pair" / "best_config.json"
    )
    pair_config.parent.mkdir(parents=True, exist_ok=True)
    pair_config.write_text(
        json.dumps(
            {
                "status": "ok",
                "sampled_params": {
                    "donor_len": 90,
                    "acceptor_len": 70,
                    "lr": 0.001,
                },
            }
        ),
        encoding="utf-8",
    )

    env: dict[str, str] = {
        "MODEL": "cnn_pair",
        "SPECIES": species,
        "TRAIN_TARGET": "pair",
        "USE_TUNED_HPARAMS": "required",
        "PAIR_TUNED_CONFIG_PATH": "",
        "SHARED_TUNED_CONFIG_PATH": "",
        "DONOR_LEN": "",
        "ACCEPTOR_LEN": "",
        "LR": "",
    }
    _ = _apply_tuned_overrides(SPECS["cnn_pair.sh"], env, tmp_path)

    assert env["DONOR_LEN"] == "90"
    assert env["ACCEPTOR_LEN"] == "70"
    assert env["LR"] == "0.001"


def test_harmonize_min_batch_size_clamps_to_task_override(
    capsys: pytest.CaptureFixture[str],
) -> None:
    env: dict[str, str] = {
        "MODEL": "dnabert2",
        "BATCH_SIZE": "64",
        "DONOR_BATCH_SIZE": "64",
        "ACCEPTOR_BATCH_SIZE": "24",
        "MIN_BATCH_SIZE": "64",
    }

    _harmonize_min_batch_size(env, script_name="dnabert.sh")

    assert env["MIN_BATCH_SIZE"] == "24"
    captured = capsys.readouterr()
    assert "MIN_BATCH_SIZE adjusted: 64 -> 24" in captured.err


def test_harmonize_min_batch_size_keeps_valid_value(
    capsys: pytest.CaptureFixture[str],
) -> None:
    env: dict[str, str] = {
        "MODEL": "dnabert2",
        "BATCH_SIZE": "64",
        "DONOR_BATCH_SIZE": "64",
        "ACCEPTOR_BATCH_SIZE": "24",
        "MIN_BATCH_SIZE": "16",
    }

    _harmonize_min_batch_size(env, script_name="dnabert.sh")

    assert env["MIN_BATCH_SIZE"] == "16"
    captured = capsys.readouterr()
    assert captured.err == ""


def test_harmonize_min_batch_size_clamps_single_task_model(
    capsys: pytest.CaptureFixture[str],
) -> None:
    env: dict[str, str] = {
        "MODEL": "cnn_pair",
        "BATCH_SIZE": "32",
        "MIN_BATCH_SIZE": "64",
    }

    _harmonize_min_batch_size(env, script_name="cnn_pair.sh")

    assert env["MIN_BATCH_SIZE"] == "32"
    captured = capsys.readouterr()
    assert "MIN_BATCH_SIZE adjusted: 64 -> 32" in captured.err


def test_harmonize_min_batch_size_ignores_missing_min_batch_size() -> None:
    env: dict[str, str] = {
        "MODEL": "markov_xgboost",
    }

    _harmonize_min_batch_size(env, script_name="markov_xgboost.sh")

    assert "MIN_BATCH_SIZE" not in env
