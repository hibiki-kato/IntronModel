from __future__ import annotations

from run_model import _build_checkpoint_stem_from_params


def test_checkpoint_stem_uses_model_params_not_name_fields() -> None:
    base_params: dict[str, object] = {
        "epochs": 20,
        "batch_size": 512,
        "lr": 5e-4,
        "loss": "focal",
        "seed": 1337,
        "name_fields": "bp_avg,loss",
    }
    changed_name_field_params: dict[str, object] = {
        **base_params,
        "name_fields": "bp_window,tag",
    }

    stem_a = _build_checkpoint_stem_from_params(
        model_name="cnn",
        donor_len=100,
        acceptor_len=100,
        inferred_train_len=None,
        raw_params=base_params,
    )
    stem_b = _build_checkpoint_stem_from_params(
        model_name="cnn",
        donor_len=100,
        acceptor_len=100,
        inferred_train_len=None,
        raw_params=changed_name_field_params,
    )

    assert stem_a == stem_b
    assert "name_fields" not in stem_a


def test_checkpoint_stem_excludes_eval_and_transcript_args() -> None:
    raw_params: dict[str, object] = {
        "epochs": 20,
        "loss": "focal",
        "train_only": True,
        "softmin_tau": 0.5,
        "good": 15169,
        "total": 38235,
        "ref": 32288,
        "intron_score_op": "*",
        "transcript_score_agg": "min",
    }

    stem = _build_checkpoint_stem_from_params(
        model_name="cnn",
        donor_len=100,
        acceptor_len=120,
        inferred_train_len=None,
        raw_params=raw_params,
    )

    assert stem.startswith("cnn_dlen100_alen120")
    assert "good" not in stem
    assert "total" not in stem
    assert "ref" not in stem
    assert "iop" not in stem
    assert "tagg" not in stem
    assert "stau" not in stem
    assert "train_only" not in stem


def test_checkpoint_stem_changes_when_model_hyperparameter_changes() -> None:
    raw_params_lr_low: dict[str, object] = {"lr": 5e-4, "epochs": 20}
    raw_params_lr_high: dict[str, object] = {"lr": 1e-3, "epochs": 20}

    stem_low = _build_checkpoint_stem_from_params(
        model_name="cnn",
        donor_len=100,
        acceptor_len=100,
        inferred_train_len=None,
        raw_params=raw_params_lr_low,
    )
    stem_high = _build_checkpoint_stem_from_params(
        model_name="cnn",
        donor_len=100,
        acceptor_len=100,
        inferred_train_len=None,
        raw_params=raw_params_lr_high,
    )

    assert stem_low != stem_high


def test_checkpoint_stem_falls_back_to_inferred_window_length() -> None:
    stem = _build_checkpoint_stem_from_params(
        model_name="cnn",
        donor_len=None,
        acceptor_len=None,
        inferred_train_len=80,
        raw_params={"epochs": 20},
    )

    assert "dlen80" in stem
    assert "alen80" in stem
