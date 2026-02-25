from __future__ import annotations

import json
from pathlib import Path

from tools.run_wrapper_pipeline import (
    SPECS,
    _apply_tuned_overrides,
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
