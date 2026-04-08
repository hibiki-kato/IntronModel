from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import tools.scan_splice_candidate_sites as scan_splice_candidate_sites

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

    assert [
        (candidate.coordinate, candidate.window) for candidate in donor_candidates
    ] == [
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
        (candidate.coordinate, candidate.window) for candidate in acceptor_candidates
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


def test_load_site_model_uses_dnabert_loader_for_dnabert2(monkeypatch) -> None:
    """DNABERT-2 site scoring should load through the DNABERT module."""
    fake_model = object()
    fake_tokenizer = object()

    def fake_load_task_model(
        checkpoint_path: str,
        device: str,
    ) -> tuple[object, dict[str, object], object]:
        assert checkpoint_path == "checkpoint.pt"
        assert device == "cpu"
        return fake_model, {"max_tokens": 128, "input_kmer": 6}, fake_tokenizer

    monkeypatch.setattr(
        scan_splice_candidate_sites.dnabert_model,
        "load_task_model",
        fake_load_task_model,
    )

    model, metadata = scan_splice_candidate_sites.load_site_model(
        "checkpoint.pt",
        "cpu",
        "dnabert2",
    )

    assert model is fake_model
    assert metadata["tokenizer"] is fake_tokenizer
    assert metadata["max_tokens"] == 128
    assert metadata["input_kmer"] == 6


def test_score_site_sequences_uses_dnabert_metadata(monkeypatch) -> None:
    """DNABERT-2 scoring should forward tokenizer and max-token metadata."""
    captured: dict[str, object] = {}

    def fake_score_sequences(
        *,
        model: object,
        sequences: list[str],
        tokenizer: object,
        max_tokens: int,
        device: str,
        batch_size: int,
        task_name: str,
        input_kmer: int | None,
        use_amp: bool,
        amp_dtype: object,
    ) -> np.ndarray:
        captured["model"] = model
        captured["sequences"] = sequences
        captured["tokenizer"] = tokenizer
        captured["max_tokens"] = max_tokens
        captured["device"] = device
        captured["batch_size"] = batch_size
        captured["task_name"] = task_name
        captured["input_kmer"] = input_kmer
        captured["use_amp"] = use_amp
        captured["amp_dtype"] = amp_dtype
        return np.asarray([1.25, 2.5], dtype=np.float64)

    monkeypatch.setattr(
        scan_splice_candidate_sites.dnabert_model,
        "score_sequences",
        fake_score_sequences,
    )

    fake_model = object()
    fake_tokenizer = object()
    scores = scan_splice_candidate_sites.score_site_sequences(
        model=fake_model,
        sequences=["AAAA", "CCCC"],
        window_len=50,
        device="cpu",
        batch_size=32,
        model_name="dnabert2",
        model_metadata={
            "tokenizer": fake_tokenizer,
            "max_tokens": "128",
            "input_kmer": 6,
        },
    )

    assert scores.tolist() == [1.25, 2.5]
    assert captured == {
        "model": fake_model,
        "sequences": ["AAAA", "CCCC"],
        "tokenizer": fake_tokenizer,
        "max_tokens": 128,
        "device": "cpu",
        "batch_size": 32,
        "task_name": "score_test_suite",
        "input_kmer": 6,
        "use_amp": False,
        "amp_dtype": None,
    }
