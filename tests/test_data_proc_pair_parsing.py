from __future__ import annotations

from pathlib import Path

import pytest

from util.data_proc import (
    read_examples_pair_task_with_metadata,
    read_examples_single_task,
    read_examples_single_task_with_metadata,
    read_test_pair_rows,
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


def test_read_examples_pair_task_negative_pair_only_filter(tmp_path: Path) -> None:
    """When enabled, keep negative rows that start with ``DEBUG pair`` only."""
    pos_path = tmp_path / "pos.err"
    neg_path = tmp_path / "neg.err"

    _write_text(
        pos_path,
        "\n".join(
            [
                "DEBUG pair AACCAA TTGGTT + 56",
                "",
            ]
        ),
    )
    _write_text(
        neg_path,
        "\n".join(
            [
                "DEBUG pair GGGGTT CCCCAA - 31",
                "DEBUG donor TTTTAA acceptor AAAACC - TX123 31",
                "",
            ]
        ),
    )

    filtered = read_examples_pair_task_with_metadata(
        str(pos_path),
        str(neg_path),
        donor_len=4,
        acceptor_len=4,
        negative_pair_only=True,
    )
    unfiltered = read_examples_pair_task_with_metadata(
        str(pos_path),
        str(neg_path),
        donor_len=4,
        acceptor_len=4,
        negative_pair_only=False,
    )

    assert len(filtered) == 2
    assert len(unfiltered) == 3
    assert [item.label for item in filtered] == [1, 0]
    assert [item.label for item in unfiltered] == [1, 0, 0]


def test_read_test_pair_rows_pairs_and_skips_rows(tmp_path: Path) -> None:
    """Pair donor/acceptor rows by transcript and intron index."""
    tsv_path = tmp_path / "transcripts.tsv"
    _write_text(
        tsv_path,
        "\n".join(
            [
                "transcript_id\tsite_type\tintron_index\tseq\tintron_half_length",
                "tx1\tdonor\t1\tAAAACCCC\t5",
                "tx1\tacceptor\t1\tGGGGTTTT\t5",
                "tx2\tdonor\t1\tAA\t3",
                "tx2\tacceptor\t1\tTTTTTT\t3",
                "tx3\tdonor\t2\tCCCCAAAA\t4",
                "",
            ]
        ),
    )

    rows, skipped_short, skipped_unpaired = read_test_pair_rows(
        test_tsv=str(tsv_path),
        donor_len=4,
        acceptor_len=4,
    )

    assert rows == [
        {
            "transcript_id": "tx1",
            "intron_index": 1,
            "donor_seq": "AAAA",
            "acceptor_seq": "TTTT",
            "intron_half_length": 5,
        }
    ]
    assert skipped_short == 1
    assert skipped_unpaired == 2
