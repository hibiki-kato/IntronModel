from __future__ import annotations

import json
from pathlib import Path

from tools import split_cnn_v2_best


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write one JSON object to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_split_one_species_prefers_resolvable_source_best(tmp_path: Path) -> None:
    """Prefer one source best whose checkpoint resolves locally."""
    data_root = tmp_path / "data"
    model_root = tmp_path / "model"

    donor_checkpoint = model_root / "Dmel" / "donor" / "cnn_v2_demo_h123456789abc.pt"
    acceptor_checkpoint = (
        model_root / "Dmel" / "acceptor" / "cnn_v2_demo_h123456789abc.pt"
    )
    pair_checkpoint = (
        model_root / "Dmel" / "pair" / "cnn_v2_pair_demo_h123456789abc.pt"
    )
    donor_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    acceptor_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    pair_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    donor_checkpoint.write_bytes(b"donor")
    acceptor_checkpoint.write_bytes(b"acceptor")
    pair_checkpoint.write_bytes(b"pair")

    donor_source = data_root / "Dmel" / "tuning" / "cnn_v2" / "donor" / "best_config.json"
    acceptor_source = (
        data_root / "Dmel" / "tuning" / "cnn_v2" / "acceptor" / "best_config.json"
    )
    acceptor_unresolvable = (
        data_root / "Dmel" / "tuning" / "cnn_mask" / "acceptor" / "best_config.json"
    )
    pair_source = (
        data_root / "Dmel" / "tuning" / "cnn_v2_pair" / "pair" / "best_config.json"
    )

    _write_json(
        donor_source,
        {
            "status": "ok",
            "objective_score": 0.8,
            "donor_pr_auc": 0.8,
            "acceptor_pr_auc": 0.1,
                "sampled_params": {"train_target": "donor"},
                "donor_checkpoint_path": (
                    "/export/hibiki/intronmodel/model/Dmel/donor/"
                    "cnn_v2_demo_habcdef123456.pt"
                ),
            },
        )
    _write_json(
        acceptor_source,
        {
            "status": "ok",
            "objective_score": 0.7,
            "donor_pr_auc": 0.1,
            "acceptor_pr_auc": 0.7,
                "sampled_params": {"train_target": "acceptor"},
                "acceptor_checkpoint_path": (
                    "/export/hibiki/intronmodel/model/Dmel/acceptor/"
                    "cnn_v2_demo_habcdef123456.pt"
                ),
            },
        )
    _write_json(
        acceptor_unresolvable,
        {
            "status": "ok",
            "objective_score": 0.95,
            "donor_pr_auc": 0.1,
            "acceptor_pr_auc": 0.95,
            "sampled_params": {"train_target": "acceptor"},
            "acceptor_checkpoint_path": (
                "/export/hibiki/intronmodel/model/Dmel/acceptor/missing_hash.pt"
            ),
        },
    )
    _write_json(
        pair_source,
        {
            "status": "ok",
            "objective_score": 0.6,
            "sampled_params": {"train_target": "pair"},
            "pair_checkpoint_path": str(pair_checkpoint),
        },
    )

    _ = split_cnn_v2_best._split_one_species(
        data_root=data_root,
        model_root_dir=model_root,
        species="Dmel",
        dry_run=False,
    )

    donor_best = data_root / "Dmel" / "tuning" / "cnn_v2" / "donor" / "best_config.json"
    acceptor_best = (
        data_root / "Dmel" / "tuning" / "cnn_v2" / "acceptor" / "best_config.json"
    )
    pair_best = data_root / "Dmel" / "tuning" / "cnn_v2_pair" / "pair" / "best_config.json"

    donor_payload = json.loads(donor_best.read_text(encoding="utf-8"))
    acceptor_payload = json.loads(acceptor_best.read_text(encoding="utf-8"))
    pair_payload = json.loads(pair_best.read_text(encoding="utf-8"))

    assert donor_payload["source_best_config"] == str(donor_source)
    assert donor_payload["donor_checkpoint_path"] == str(donor_checkpoint.resolve())
    assert acceptor_payload["source_best_config"] == str(acceptor_source)
    assert acceptor_payload["acceptor_checkpoint_path"] == str(
        acceptor_checkpoint.resolve()
    )
    assert pair_payload["pair_checkpoint_path"] == str(pair_checkpoint.resolve())
