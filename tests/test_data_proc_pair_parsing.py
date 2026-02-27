from __future__ import annotations

from pathlib import Path

import pytest

from util.data_proc import (
    read_examples_single_task,
    read_examples_single_task_with_metadata,
)


def _write_text(path: Path, text: str) -> None:
    """Write UTF-8 text to a file path."""
    path.write_text(text, encoding="utf-8")


def test_read_examples_single_task_supports_simple_pair_records(
    tmp_path: Path,
) -> None:
    """Use donor/acceptor subsequences from ``DEBUG pair`` records."""
    pos_path = tmp_path / "pos.err"
    neg_path = tmp_path / "neg.err"

    _write_text(
        pos_path,
        "\n".join(
            [
                "DEBUG pair ACGTAC GTTTAA +",
                "DEBUG donor GGGGGG +",
                "",
            ]
        ),
    )
    _write_text(
        neg_path,
        "\n".join(
            [
                "DEBUG pair TTTTTT CCCCCC -",
                "DEBUG acceptor AAAATT +",
                "",
            ]
        ),
    )

    donor_examples = read_examples_single_task(
        str(pos_path),
        str(neg_path),
        "donor",
        donor_len=4,
        acceptor_len=4,
    )
    acceptor_examples = read_examples_single_task(
        str(pos_path),
        str(neg_path),
        "acceptor",
        donor_len=4,
        acceptor_len=4,
    )

    assert donor_examples == [
        ("ACGT", 1),
        ("GGGG", 1),
        ("TTTT", 0),
    ]
    assert acceptor_examples == [
        ("TTAA", 1),
        ("CCCC", 0),
        ("AATT", 0),
    ]


def test_read_examples_single_task_keeps_labeled_pair_compatibility(
    tmp_path: Path,
) -> None:
    """Keep compatibility with ``DEBUG donor ... acceptor ...`` pair records."""
    pos_path = tmp_path / "pos.err"
    neg_path = tmp_path / "neg.err"

    _write_text(
        pos_path,
        "\n".join(
            [
                "DEBUG donor AACCAA acceptor TTGGTT +",
                "",
            ]
        ),
    )
    _write_text(
        neg_path,
        "\n".join(
            [
                "DEBUG donor GGGGTT acceptor CCCCAA -",
                "",
            ]
        ),
    )

    donor_examples = read_examples_single_task(
        str(pos_path),
        str(neg_path),
        "donor",
        donor_len=4,
        acceptor_len=4,
    )
    acceptor_examples = read_examples_single_task(
        str(pos_path),
        str(neg_path),
        "acceptor",
        donor_len=4,
        acceptor_len=4,
    )

    assert donor_examples == [("AACC", 1), ("GGGG", 0)]
    assert acceptor_examples == [("GGTT", 1), ("CCAA", 0)]


def test_read_examples_single_task_supports_new_pair_extensions(
    tmp_path: Path,
) -> None:
    """Support extra metadata tokens in new positive/negative pair records."""
    pos_path = tmp_path / "pos.err"
    neg_path = tmp_path / "neg.err"

    _write_text(
        pos_path,
        "\n".join(
            [
                "DEBUG donor AACCAA acceptor TTGGTT + TX001 42",
                "",
            ]
        ),
    )
    _write_text(
        neg_path,
        "\n".join(
            [
                "DEBUG pair GGGGTT CCCCAA - 31",
                "",
            ]
        ),
    )

    donor_examples = read_examples_single_task(
        str(pos_path),
        str(neg_path),
        "donor",
        donor_len=4,
        acceptor_len=4,
    )
    acceptor_examples = read_examples_single_task(
        str(pos_path),
        str(neg_path),
        "acceptor",
        donor_len=4,
        acceptor_len=4,
    )

    assert donor_examples == [("AACC", 1), ("GGGG", 0)]
    assert acceptor_examples == [("GGTT", 1), ("CCAA", 0)]


def test_read_examples_single_task_with_metadata_exposes_new_fields(
    tmp_path: Path,
) -> None:
    """Expose transcript and intron metadata without affecting sequence labels."""
    pos_path = tmp_path / "pos.err"
    neg_path = tmp_path / "neg.err"

    _write_text(
        pos_path,
        "\n".join(
            [
                "DEBUG donor AACCAA acceptor TTGGTT + TX777 56",
                "",
            ]
        ),
    )
    _write_text(
        neg_path,
        "\n".join(
            [
                "DEBUG pair GGGGTT CCCCAA - 31",
                "DEBUG donor TTTTAA +",
                "",
            ]
        ),
    )

    donor_examples = read_examples_single_task_with_metadata(
        str(pos_path),
        str(neg_path),
        "donor",
        donor_len=4,
        acceptor_len=4,
    )

    assert [item.sequence for item in donor_examples] == ["AACC", "GGGG", "TTTT"]
    assert [item.label for item in donor_examples] == [1, 0, 0]
    assert donor_examples[0].transcript_id == "TX777"
    assert donor_examples[0].intron_half_length == 56
    assert donor_examples[1].transcript_id is None
    assert donor_examples[1].intron_half_length == 31
    assert donor_examples[2].transcript_id is None
    assert donor_examples[2].intron_half_length is None


def test_read_examples_single_task_with_metadata_rejects_unknown_task(
    tmp_path: Path,
) -> None:
    """Reject invalid task names early with a clear error."""
    pos_path = tmp_path / "pos.err"
    neg_path = tmp_path / "neg.err"
    _write_text(pos_path, "DEBUG donor AAAA +\n")
    _write_text(neg_path, "DEBUG donor CCCC +\n")

    with pytest.raises(ValueError, match="task must be either"):
        _ = read_examples_single_task_with_metadata(
            str(pos_path),
            str(neg_path),
            "pair",
            donor_len=4,
            acceptor_len=4,
        )
