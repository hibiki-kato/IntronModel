from __future__ import annotations

from util.data_proc import build_run_name
from util.data_proc import build_output_stem


def test_build_output_stem_uses_raw_tag_value_without_tag_prefix() -> None:
    stem = build_output_stem(
        model_name="cnn",
        donor_len=100,
        acceptor_len=100,
        fallback_train_len=None,
        name_fields=["tag"],
        name_params={"tag": "cnn_pad"},
    )

    assert stem == "cnn_pad"


def test_build_output_stem_appends_short_tag_after_model_name() -> None:
    stem = build_output_stem(
        model_name="cnn_pair",
        donor_len=100,
        acceptor_len=100,
        fallback_train_len=None,
        name_fields=["tag"],
        name_params={"tag": "pad"},
    )

    assert stem == "cnn_pair_pad"


def test_build_output_stem_keeps_pair_before_trunc_tag_for_dnabert() -> None:
    stem = build_output_stem(
        model_name="dnabert2_pair",
        donor_len=100,
        acceptor_len=100,
        fallback_train_len=None,
        name_fields=["tag"],
        name_params={"tag": "trunc"},
    )

    assert stem == "dnabert2_pair_trunc"


def test_build_run_name_does_not_append_bp_suffix_by_default() -> None:
    run_name = build_run_name(
        model_name="cnn_v2",
        donor_len=100,
        acceptor_len=100,
        lr=5e-4,
        batch_size=512,
        epochs=10,
        tag=None,
    )

    assert run_name == "cnn_v2_lr0.0005_bs512_ep10"


def test_build_run_name_appends_tag_without_bp_suffix() -> None:
    run_name = build_run_name(
        model_name="cnn_v2",
        donor_len=100,
        acceptor_len=100,
        lr=5e-4,
        batch_size=512,
        epochs=10,
        tag="expA",
    )

    assert run_name == "cnn_v2_lr0.0005_bs512_ep10_expA"
