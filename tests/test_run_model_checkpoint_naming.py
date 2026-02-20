from __future__ import annotations

import re

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
        "compile_mode": "auto",
        "use_amp": 1,
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
    assert "compile_mode" not in stem
    assert "use_amp" not in stem


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


def test_checkpoint_stem_ignores_train_target_when_both() -> None:
    stem_default = _build_checkpoint_stem_from_params(
        model_name="cnn",
        donor_len=100,
        acceptor_len=100,
        inferred_train_len=None,
        raw_params={"epochs": 20},
    )
    stem_both = _build_checkpoint_stem_from_params(
        model_name="cnn",
        donor_len=100,
        acceptor_len=100,
        inferred_train_len=None,
        raw_params={"epochs": 20, "train_target": "both"},
    )

    assert stem_default == stem_both


def test_checkpoint_stem_includes_single_task_train_target() -> None:
    stem = _build_checkpoint_stem_from_params(
        model_name="cnn",
        donor_len=100,
        acceptor_len=100,
        inferred_train_len=None,
        raw_params={"epochs": 20, "train_target": "donor"},
    )

    assert "train_targetdonor" in stem


def test_checkpoint_stem_is_shortened_with_hash_when_too_long() -> None:
    raw_params: dict[str, object] = {
        "epochs": 20,
        "batch_size": 512,
        "lr": 5e-4,
        "loss": "focal",
        "donor_batch_size": 256,
        "acceptor_batch_size": 2048,
        "donor_lr": 0.0016758706451374796,
        "acceptor_lr": 0.00012123312331233123,
        "donor_conv_channels": "64,128,256,512",
        "acceptor_conv_channels": "128,256,512,1024",
        "donor_weight_decay": 0.0017901308021123963,
        "acceptor_weight_decay": 0.000010123456789123,
        "donor_kernel_size": 11,
        "acceptor_kernel_size": 9,
        "donor_dropout": 0.251930575023598,
        "acceptor_dropout": 0.4123456789123456,
        "donor_fc_hidden": 2048,
        "acceptor_fc_hidden": 1024,
    }
    stem = _build_checkpoint_stem_from_params(
        model_name="cnn",
        donor_len=100,
        acceptor_len=100,
        inferred_train_len=None,
        raw_params=raw_params,
    )
    stem_repeat = _build_checkpoint_stem_from_params(
        model_name="cnn",
        donor_len=100,
        acceptor_len=100,
        inferred_train_len=None,
        raw_params=raw_params,
    )

    assert stem == stem_repeat
    assert len(stem) <= 200
    assert re.search(r"_h[0-9a-f]{12}$", stem) is not None
