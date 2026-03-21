from __future__ import annotations

import json
from pathlib import Path

from tools import split_cnn_v2_both_to_site_best as splitter


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_build_target_payload_converts_branch_specific_params() -> None:
    source_payload: dict[str, object] = {
        "status": "ok",
        "donor_pr_auc": 0.91,
        "acceptor_pr_auc": 0.82,
        "mean_pr_auc": 0.865,
        "objective_metric": "mean_pr_auc",
        "objective_score": 0.865,
        "selection_score": 0.865,
        "sampled_params": {
            "train_target": "both",
            "pair_mode": "independent",
            "donor_len": 100,
            "acceptor_len": 80,
            "donor_conv_channels": "256,512,256",
            "donor_kernel_sizes": "15,11,7",
            "acceptor_conv_channels": "128,256,128",
            "acceptor_kernel_sizes": "11,7,5",
            "batch_size": 256,
            "lr": 5e-4,
        },
        "hparam_context": {
            "objective_metric": "mean_pr_auc",
            "fixed_run_args": {"train_target": "both"},
        },
    }

    donor_payload = splitter._build_target_payload(
        source_payload=source_payload,
        source_path=Path("/tmp/source.json"),
        target="donor",
    )

    sampled = donor_payload["sampled_params"]
    assert isinstance(sampled, dict)
    assert donor_payload["objective_metric"] == "donor_pr_auc"
    assert donor_payload["objective_score"] == 0.91
    assert donor_payload["donor_pr_auc"] == 0.91
    assert donor_payload["acceptor_pr_auc"] is None
    assert donor_payload["mean_pr_auc"] is None
    assert sampled["train_target"] == "donor"
    assert sampled["conv_channels"] == "256,512,256"
    assert sampled["kernel_sizes"] == "15,11,7"
    assert sampled["pair_mode"] == "independent"
    assert "donor_conv_channels" not in sampled
    assert "acceptor_conv_channels" not in sampled


def test_split_species_writes_donor_and_acceptor_best_configs(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    species = "Hsap"
    source_path = data_root / species / "tuning" / "cnn_v2" / "both" / "best_config.json"
    _write_json(
        source_path,
        {
            "status": "ok",
            "donor_pr_auc": 0.77,
            "acceptor_pr_auc": 0.81,
            "mean_pr_auc": 0.79,
            "objective_metric": "mean_pr_auc",
            "objective_score": 0.79,
            "selection_score": 0.79,
            "sampled_params": {
                "train_target": "both",
                "pair_mode": "independent",
                "donor_conv_channels": "64,128,256",
                "acceptor_conv_channels": "96,192,384",
                "batch_size": 256,
            },
            "hparam_context": {
                "objective_metric": "mean_pr_auc",
                "fixed_run_args": {"train_target": "both"},
            },
        },
    )

    splitter._split_species(data_root=data_root, species=species, dry_run=False)

    donor_path = (
        data_root / species / "tuning" / "cnn_v2" / "donor" / "best_config.json"
    )
    acceptor_path = (
        data_root / species / "tuning" / "cnn_v2" / "acceptor" / "best_config.json"
    )
    donor_payload = json.loads(donor_path.read_text(encoding="utf-8"))
    acceptor_payload = json.loads(acceptor_path.read_text(encoding="utf-8"))

    assert donor_payload["objective_metric"] == "donor_pr_auc"
    assert donor_payload["objective_score"] == 0.77
    assert donor_payload["sampled_params"]["train_target"] == "donor"
    assert donor_payload["sampled_params"]["conv_channels"] == "64,128,256"

    assert acceptor_payload["objective_metric"] == "acceptor_pr_auc"
    assert acceptor_payload["objective_score"] == 0.81
    assert acceptor_payload["sampled_params"]["train_target"] == "acceptor"
    assert acceptor_payload["sampled_params"]["conv_channels"] == "96,192,384"
