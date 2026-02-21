from __future__ import annotations

from pathlib import Path

from util.data_proc import read_examples_single_task


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
