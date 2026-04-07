from __future__ import annotations

import json
from pathlib import Path

from tools.scan_splice_candidate_sites import (
    build_candidate_windows,
    load_resolved_best_model_paths,
)


def _write_checkpoint(path: Path, *, window_len: int) -> None:
    """Write one minimal checkpoint payload for scan-resolution tests."""

    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"window_len": window_len}, path)


def test_build_candidate_windows_keeps_donor_layout() -> None:
    """Donor windows should remain anchored on the G of one GT motif."""
    sequence = "CCCGTATAGCCA"

    donor_candidates, _ = build_candidate_windows(
        sequence,
        donor_window_len=6,
        acceptor_window_len=6,
    )

    assert [(candidate.coordinate, candidate.window) for candidate in donor_candidates] == [
        (3, "CCCGTA"),
    ]


def test_build_candidate_windows_uses_exon_start_acceptor_layout() -> None:
    """Acceptor windows should match the exon-start layout used in training."""
    sequence = "CCCGTATAGCCA"

    _, acceptor_candidates = build_candidate_windows(
        sequence,
        donor_window_len=6,
        acceptor_window_len=6,
    )

    assert [
        (candidate.coordinate, candidate.window)
        for candidate in acceptor_candidates
    ] == [
        (7, "TAGCCA"),
    ]


def test_load_resolved_best_model_paths_uses_task_best_configs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Scan should resolve donor and acceptor checkpoints from task best configs."""

    data_root = tmp_path / "data"
    model_root = tmp_path / "model"
    donor_checkpoint = model_root / "Dmel" / "donor" / "cnn_v2.99.pt"
    acceptor_checkpoint = model_root / "Dmel" / "acceptor" / "cnn_v2.98.pt"
    _write_checkpoint(donor_checkpoint, window_len=50)
    _write_checkpoint(acceptor_checkpoint, window_len=100)

    donor_best = data_root / "Dmel" / "tuning" / "cnn_v2" / "donor" / "best_config.json"
    acceptor_best = (
        data_root / "Dmel" / "tuning" / "cnn_v2" / "acceptor" / "best_config.json"
    )
    donor_best.parent.mkdir(parents=True, exist_ok=True)
    acceptor_best.parent.mkdir(parents=True, exist_ok=True)
    donor_best.write_text(
        json.dumps(
            {
                "status": "ok",
                "published_name": "cnn_v2.01",
                "donor_checkpoint_path": str(donor_checkpoint),
            }
        ),
        encoding="utf-8",
    )
    acceptor_best.write_text(
        json.dumps(
            {
                "status": "ok",
                "published_name": "cnn_v2.01",
                "acceptor_checkpoint_path": str(acceptor_checkpoint),
            }
        ),
        encoding="utf-8",
    )
    version_history = data_root / "Dmel" / "tuning" / "cnn_v2" / "version_history.tsv"
    version_history.write_text(
        "\t".join(
            [
                "version",
                "published_name",
                "published_at",
                "source_best_config",
                "objective_metric",
                "objective_score",
                "updated_side",
                "carry_forward_side",
                "donor_checkpoint_path",
                "acceptor_checkpoint_path",
                "pair_checkpoint_path",
                "metrics_json",
                "archive_status",
            ]
        )
        + "\n"
        + "\t".join(
            [
                "1",
                "cnn_v2.01",
                "2026-04-06T00:00:00Z",
                "data/Dmel/tuning/cnn_v2/donor/best_config.json",
                "pr_auc",
                "0.81",
                "donor",
                "acceptor",
                "model/Dmel/donor/cnn_v2.01.pt",
                "model/Dmel/acceptor/cnn_v2.01.pt",
                "",
                "data/Dmel/learning_metric/cnn_v2.01.train.json",
                "live",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("INTRONMODEL_MODEL_ROOT", str(model_root))

    resolved = load_resolved_best_model_paths(
        data_root=data_root,
        species="Dmel",
        model_name="cnn_v2",
        device="cpu",
    )

    assert resolved.donor_checkpoint_path == donor_checkpoint.resolve()
    assert resolved.acceptor_checkpoint_path == acceptor_checkpoint.resolve()
    assert resolved.donor_window_len == 50
    assert resolved.acceptor_window_len == 100
