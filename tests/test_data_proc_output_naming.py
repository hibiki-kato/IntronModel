from __future__ import annotations

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
