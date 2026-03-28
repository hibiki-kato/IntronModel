from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from models import reservoir


class _DummyReadout:
    """Lightweight readout stub that returns 2-class logits."""

    def __init__(self) -> None:
        self._bias: float = 0.0

    def fit(self, input_repr: np.ndarray, y_data: np.ndarray) -> None:
        del y_data
        self._bias = float(np.mean(input_repr))

    def predict(self, input_repr: np.ndarray) -> np.ndarray:
        base = np.mean(input_repr, axis=1, keepdims=True) + self._bias
        return np.concatenate((-base, base), axis=1)


class _DummyReservoirCore:
    """Reservoir-state stub compatible with ``_compute_input_representation``."""

    def get_states(self, x_data: np.ndarray, n_drop: int, bidir: bool) -> np.ndarray:
        states = x_data[:, n_drop:, :]
        if states.shape[1] == 0:
            raise ValueError("n_drop removed all steps")
        if bidir:
            return np.concatenate((states, states[:, ::-1, :]), axis=2)
        return states


class _DummyRCModel:
    """Minimal RC model protocol implementation used by tests."""

    def __init__(self, mts_rep: str, n_drop: int) -> None:
        self.n_drop: int = n_drop
        self.bidir: bool = False
        self.dimred_method: str | None = None
        self.mts_rep: str = mts_rep
        self._reservoir: _DummyReservoirCore = _DummyReservoirCore()
        self.readout: _DummyReadout = _DummyReadout()

    def _repr(self, x_data: np.ndarray) -> np.ndarray:
        states = self._reservoir.get_states(x_data, n_drop=self.n_drop, bidir=False)
        if self.mts_rep == "last":
            return states[:, -1, :]
        return np.mean(states, axis=1)

    def fit(self, X: np.ndarray, Y: np.ndarray, verbose: bool = True) -> None:
        del verbose, Y
        input_repr = self._repr(X)
        # Deterministic fit only records a simple offset from train data.
        self.readout.fit(
            input_repr,
            np.zeros((input_repr.shape[0], 2), dtype=np.float32),
        )

    def predict(self, Xte: np.ndarray) -> np.ndarray:
        logits = self.readout.predict(self._repr(Xte))
        return np.argmax(logits, axis=1)


def _write_training_file(
    path: Path,
    donor_sequences: list[str],
    acceptor_sequences: list[str],
) -> None:
    """Write debug-format examples consumed by ``read_examples_single_task``."""
    rows: list[str] = []
    for seq in donor_sequences:
        rows.append(f"DEBUG donor {seq}")
    for seq in acceptor_sequences:
        rows.append(f"DEBUG acceptor {seq}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _install_dummy_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch RC model builder to avoid external dependency in tests."""

    def _builder(**kwargs: object) -> _DummyRCModel:
        mts_rep = str(kwargs["mts_rep"])
        washout = int(kwargs["washout"])
        return _DummyRCModel(mts_rep=mts_rep, n_drop=washout)

    monkeypatch.setattr(reservoir, "_build_rc_model", _builder)


def test_train_and_infer_with_dummy_rc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_dummy_builder(monkeypatch)

    pos_path = tmp_path / "pos.err"
    neg_path = tmp_path / "neg.err"
    _write_training_file(
        pos_path,
        donor_sequences=["AAAAAAAAAA", "AAAACCCCCC", "GGGGAAAAAA"],
        acceptor_sequences=["CCCCCCCCCC", "CCCGGGGGGG", "TTTTCCCCCC"],
    )
    _write_training_file(
        neg_path,
        donor_sequences=["TTTTTTTTTT", "TTTTGGGGGG", "CCCCGGGGGG"],
        acceptor_sequences=["GGGGGGGGGG", "GGGGTTTTTT", "AAAATTTTTT"],
    )

    donor_ckpt = tmp_path / "donor.ckpt"
    acceptor_ckpt = tmp_path / "acceptor.ckpt"

    donor_summary = reservoir.train_task_model(
        task="donor",
        pos_path=str(pos_path),
        neg_path=str(neg_path),
        checkpoint_path=str(donor_ckpt),
        window_len=10,
        donor_len=10,
        acceptor_len=10,
        epochs=2,
        batch_size=8,
        input_mode="onehot",
        kmer_k=3,
        max_tokens="auto",
        input_dim=6,
        reservoir_size=32,
        spectral_radius=0.95,
        leak=0.3,
        sparsity=0.1,
        input_scale=0.5,
        pooling="mean",
        mts_rep="auto",
        read_order="forward",
        val_frac=0.34,
    )
    acceptor_summary = reservoir.train_task_model(
        task="acceptor",
        pos_path=str(pos_path),
        neg_path=str(neg_path),
        checkpoint_path=str(acceptor_ckpt),
        window_len=10,
        donor_len=10,
        acceptor_len=10,
        epochs=2,
        batch_size=8,
        input_mode="onehot",
        kmer_k=3,
        max_tokens="auto",
        input_dim=6,
        reservoir_size=32,
        spectral_radius=0.95,
        leak=0.3,
        sparsity=0.1,
        input_scale=0.5,
        pooling="mean",
        mts_rep="auto",
        read_order="forward",
        val_frac=0.34,
    )

    assert donor_ckpt.exists()
    assert acceptor_ckpt.exists()
    assert donor_summary["best_metric"] in {"pr_auc", "roc_auc", "acc@0.5"}
    assert acceptor_summary["best_metric"] in {"pr_auc", "roc_auc", "acc@0.5"}

    rows = [
        {
            "transcript_id": "tx1",
            "intron_index": 0,
            "site_type": "donor",
            "seq": "AAAAAAAACC",
        },
        {
            "transcript_id": "tx1",
            "intron_index": 0,
            "site_type": "acceptor",
            "seq": "CCCCCCCCAA",
        },
        {
            "transcript_id": "tx2",
            "intron_index": 1,
            "site_type": "donor",
            "seq": "TTTTTTTTAA",
        },
        {
            "transcript_id": "tx2",
            "intron_index": 1,
            "site_type": "acceptor",
            "seq": "GGGGGGGGAA",
        },
    ]

    inferred = reservoir.infer_site_scores(
        site_rows=rows,
        donor_model_path=str(donor_ckpt),
        acceptor_model_path=str(acceptor_ckpt),
        batch_size=2,
    )

    assert len(inferred) == len(rows)
    for row in inferred:
        assert float(row["score"]) <= 0.0
        assert row["_score_space"] == "log10"


def test_train_task_model_rejects_large_washout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_dummy_builder(monkeypatch)

    pos_path = tmp_path / "pos.err"
    neg_path = tmp_path / "neg.err"
    _write_training_file(
        pos_path,
        donor_sequences=["AAAAAAAA", "AAAACCCC"],
        acceptor_sequences=["CCCCCCCC", "CCCCGGGG"],
    )
    _write_training_file(
        neg_path,
        donor_sequences=["TTTTTTTT", "TTTTGGGG"],
        acceptor_sequences=["GGGGGGGG", "GGGGTTTT"],
    )

    with pytest.raises(ValueError, match="washout"):
        reservoir.train_task_model(
            task="donor",
            pos_path=str(pos_path),
            neg_path=str(neg_path),
            checkpoint_path=str(tmp_path / "donor.ckpt"),
            window_len=8,
            donor_len=8,
            acceptor_len=8,
            epochs=2,
            input_mode="onehot",
            max_tokens=4,
            washout=4,
            pooling="mean",
        )


def test_resolve_mts_rep_maps_legacy_pooling() -> None:
    assert (
        reservoir._resolve_mts_rep(pooling="mean_max", mts_rep_arg="auto")
        == "mean"
    )
    assert (
        reservoir._resolve_mts_rep(pooling="logit_sum", mts_rep_arg="auto")
        == "output"
    )

    with pytest.raises(ValueError, match="pooling"):
        reservoir._resolve_mts_rep(pooling="invalid", mts_rep_arg="auto")


def test_resolve_state_budget_gb_parses_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTRONMODEL_RC_STATE_BUDGET_GB", "3.5")
    resolved = reservoir._resolve_state_budget_gb()
    assert resolved == pytest.approx(3.5)


def test_resolve_state_budget_gb_auto_uses_detected_ram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTRONMODEL_RC_STATE_BUDGET_GB", "auto")
    monkeypatch.setattr(
        reservoir,
        "_detect_system_total_memory_bytes",
        lambda: 64 * 1024**3,
    )

    expected = min(
        reservoir.AUTO_STATE_BUDGET_MAX_GB,
        max(
            reservoir.AUTO_STATE_BUDGET_MIN_GB,
            64.0 * reservoir.AUTO_STATE_BUDGET_FRACTION,
        ),
    )
    resolved = reservoir._resolve_state_budget_gb()
    assert resolved == pytest.approx(expected)


def test_train_task_model_caps_examples_by_state_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_dummy_builder(monkeypatch)
    monkeypatch.setenv("INTRONMODEL_RC_STATE_BUDGET_GB", "0.0001")

    pos_path = tmp_path / "pos.err"
    neg_path = tmp_path / "neg.err"
    _write_training_file(
        pos_path,
        donor_sequences=[
            "AAAAAAAAAA",
            "AAAACCCCCC",
            "GGGGAAAAAA",
            "CCCCAAAAGG",
            "TTTTAAAACC",
            "AAAAGGGGCC",
        ],
        acceptor_sequences=[
            "CCCCCCCCCC",
            "CCCGGGGGGG",
            "TTTTCCCCCC",
            "GGGGCCCCAA",
            "AAAAACCCCC",
            "CCCCAAAATT",
        ],
    )
    _write_training_file(
        neg_path,
        donor_sequences=[
            "TTTTTTTTTT",
            "TTTTGGGGGG",
            "CCCCGGGGGG",
            "GGGGTTTTCC",
            "CCCCCTTTTT",
            "GGGGGAAAAA",
        ],
        acceptor_sequences=[
            "GGGGGGGGGG",
            "GGGGTTTTTT",
            "AAAATTTTTT",
            "TTTTGGGGAA",
            "CCCCGGGGTT",
            "AAAACCCCTT",
        ],
    )

    summary = reservoir.train_task_model(
        task="donor",
        pos_path=str(pos_path),
        neg_path=str(neg_path),
        checkpoint_path=str(tmp_path / "donor.ckpt"),
        window_len=10,
        donor_len=10,
        acceptor_len=10,
        epochs=2,
        input_mode="onehot",
        max_tokens=10,
        reservoir_size=512,
        pooling="mean",
    )

    assert float(summary["state_budget_gib"]) == pytest.approx(0.0001)
    assert int(summary["train_examples_after_cap"]) < int(
        summary["train_examples_before_cap"]
    )
