from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pytest

from models import cnn_v3


def test_parse_checkpoint_csv_deduplicates_preserving_order(
    tmp_path: Path,
) -> None:
    ckpt_a = tmp_path / "a.pt"
    ckpt_b = tmp_path / "b.pt"
    ckpt_a.write_bytes(b"a")
    ckpt_b.write_bytes(b"b")
    parsed = cnn_v3._parse_checkpoint_csv(
        f" {ckpt_a}, {ckpt_b}, {ckpt_a} ",
        arg_name="--base_pair_checkpoints",
    )
    assert parsed == [str(ckpt_a.resolve()), str(ckpt_b.resolve())]


def test_parse_checkpoint_csv_rejects_empty() -> None:
    with pytest.raises(ValueError, match="--base_pair_checkpoints"):
        _ = cnn_v3._parse_checkpoint_csv(" , , ", arg_name="--base_pair_checkpoints")


def test_parse_checkpoint_csv_rejects_missing_path(
    tmp_path: Path,
) -> None:
    ckpt_a = tmp_path / "a.pt"
    ckpt_a.write_bytes(b"a")
    missing = tmp_path / "missing.pt"
    with pytest.raises(FileNotFoundError, match="Base checkpoint not found"):
        _ = cnn_v3._parse_checkpoint_csv(
            f"{ckpt_a},{missing}",
            arg_name="--base_pair_checkpoints",
        )


def test_resolve_base_pair_checkpoints_from_args(tmp_path: Path) -> None:
    ckpt_a = tmp_path / "a.pt"
    ckpt_b = tmp_path / "b.pt"
    ckpt_a.write_bytes(b"a")
    ckpt_b.write_bytes(b"b")
    args = argparse.Namespace(
        base_pair_checkpoints=f"{ckpt_a},{ckpt_b}",
    )
    resolved = cnn_v3._resolve_base_pair_checkpoints(args)
    assert resolved == [str(ckpt_a.resolve()), str(ckpt_b.resolve())]


def test_stratified_split_indices_has_both_classes() -> None:
    labels = np.asarray([1, 1, 1, 0, 0, 0], dtype=np.float32)
    train_idx, val_idx = cnn_v3._stratified_split_indices(
        labels,
        val_frac=0.33,
        seed=1337,
    )
    assert train_idx.size > 0
    assert val_idx.size > 0
    assert np.any(labels[train_idx] > 0.5)
    assert np.any(labels[train_idx] <= 0.5)
    assert np.any(labels[val_idx] > 0.5)
    assert np.any(labels[val_idx] <= 0.5)
