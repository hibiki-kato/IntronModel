from __future__ import annotations

import pytest

from util.sequence_transform import (
    PairSequenceRecord,
    apply_pair_sequence_transform,
    apply_site_sequence_transform,
    validate_sequence_transform,
)


def test_validate_sequence_transform_normalizes_mode() -> None:
    assert validate_sequence_transform(" NONE ") == "none"


def test_apply_site_sequence_transform_none_uppercases() -> None:
    transformed = apply_site_sequence_transform(
        "acgtn",
        site_type="donor",
        transform_mode="none",
        intron_half_length=None,
    )
    assert transformed == "ACGTN"


def test_apply_site_sequence_transform_masks_donor_and_acceptor() -> None:
    donor = apply_site_sequence_transform(
        "AAAACCCC",
        site_type="donor",
        transform_mode="mask_outside_intron_n",
        intron_half_length=2,
        exon_context_bp=1,
    )
    acceptor = apply_site_sequence_transform(
        "AAAACCCC",
        site_type="acceptor",
        transform_mode="mask_outside_intron_n",
        intron_half_length=2,
        exon_context_bp=1,
    )

    assert donor == "AAANNNNN"
    assert acceptor == "NNNNNCCC"


def test_apply_site_sequence_transform_requires_intron_half_length() -> None:
    with pytest.raises(ValueError, match="requires intron_half_length"):
        _ = apply_site_sequence_transform(
            "AAAACCCC",
            site_type="donor",
            transform_mode="mask_outside_intron_n",
            intron_half_length=None,
        )


def test_apply_pair_sequence_transform_applies_to_both_sides() -> None:
    transformed = apply_pair_sequence_transform(
        PairSequenceRecord(donor_seq="AAAACCCC", acceptor_seq="GGGGTTTT"),
        transform_mode="mask_outside_intron_n",
        intron_half_length=1,
        exon_context_bp=2,
    )

    assert transformed == PairSequenceRecord(
        donor_seq="AAANNNNN",
        acceptor_seq="NNNNNTTT",
    )
