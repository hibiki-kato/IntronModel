from __future__ import annotations

from pathlib import Path

import pytest

from util.make_trimmed_pair_data import (
    _read_negative_pairs,
    _trim_pair,
    process_species,
)


def _write_text(path: Path, text: str) -> None:
    """Write UTF-8 text to a file path."""
    path.write_text(text, encoding="utf-8")


def _prepare_species_raw(tmp_path: Path, species: str) -> Path:
    """Create species raw directory and return path."""
    raw_dir = tmp_path / "data" / species / "raw"
    raw_dir.mkdir(parents=True)
    return raw_dir


def test_trim_pair_uses_half_length_plus_context() -> None:
    """Trim donor prefix and acceptor suffix with half-length rule."""
    donor = "A" * 100
    acceptor = "C" * 100

    trimmed = _trim_pair(
        donor_seq=donor,
        acceptor_seq=acceptor,
        intron_half_length=12,
        exon_context_bp=3,
    )

    assert len(trimmed.donor_seq) == 15
    assert len(trimmed.acceptor_seq) == 15
    assert trimmed.donor_seq == "A" * 15
    assert trimmed.acceptor_seq == "C" * 15


def test_read_negative_pairs_ignores_non_pair_lines(tmp_path: Path) -> None:
    """Parse only ``DEBUG pair`` rows from negative source file."""
    neg_path = tmp_path / "neg.err"
    _write_text(
        neg_path,
        "\n".join(
            [
                "DEBUG pair ACGT TGCA + 12",
                "DEBUG donor AAAA +",
                "DEBUG acceptor CCCC -",
                "DEBUG pair TTTT GGGG - 9",
                "",
            ]
        ),
    )

    rows = _read_negative_pairs(neg_path, strict=True)
    assert len(rows) == 2
    assert rows[0].source_line_no == 1
    assert rows[1].source_line_no == 4


def test_process_species_writes_trimmed_files(tmp_path: Path) -> None:
    """Generate trimmed positive/negative pair files for one species."""
    raw_dir = _prepare_species_raw(tmp_path, "SpecTrim")

    _write_text(
        raw_dir / "100bp.err",
        "\n".join(
            [
                "DEBUG donor "
                + ("A" * 100)
                + " acceptor "
                + ("C" * 100)
                + " + TX1 10",
                "",
            ]
        ),
    )
    _write_text(
        raw_dir / "100bp.neg.err",
        "\n".join(
            [
                "DEBUG pair " + ("G" * 100) + " " + ("T" * 100) + " - 8",
                "DEBUG donor AAAA +",
                "",
            ]
        ),
    )

    stats = process_species(
        species="SpecTrim",
        data_root=tmp_path / "data",
        pos_input_name="100bp.err",
        neg_input_name="100bp.neg.err",
        out_pos_name="100bp_trimmed.err",
        out_neg_name="100bp_trimmed.neg.err",
        exon_context_bp=3,
        pad_with_n=False,
        strict=True,
    )

    assert stats.positive_rows == 1
    assert stats.negative_pair_rows == 1

    pos_text = (raw_dir / "100bp_trimmed.err").read_text(encoding="utf-8").strip()
    neg_text = (raw_dir / "100bp_trimmed.neg.err").read_text(
        encoding="utf-8"
    ).strip()

    # half=10 -> keep 13
    assert "DEBUG donor " + ("A" * 13) in pos_text
    assert "acceptor " + ("C" * 13) in pos_text
    # half=8 -> keep 11
    assert "DEBUG pair " + ("G" * 11) in neg_text
    assert " " + ("T" * 11) + " - 8" in neg_text


def test_trim_pair_rejects_negative_half_length() -> None:
    """Reject negative intron-half values."""
    with pytest.raises(ValueError, match="must be >= 0"):
        _ = _trim_pair(
            donor_seq="AAAA",
            acceptor_seq="CCCC",
            intron_half_length=-1,
            exon_context_bp=3,
        )


def test_trim_pair_with_n_padding_preserves_original_length() -> None:
    """Pad trimmed-out regions with N while keeping fixed length."""
    donor = "A" * 12
    acceptor = "C" * 12

    trimmed = _trim_pair(
        donor_seq=donor,
        acceptor_seq=acceptor,
        intron_half_length=4,
        exon_context_bp=2,
        pad_with_n=True,
    )

    assert len(trimmed.donor_seq) == 12
    assert len(trimmed.acceptor_seq) == 12
    assert trimmed.donor_seq == ("A" * 6) + ("N" * 6)
    assert trimmed.acceptor_seq == ("N" * 6) + ("C" * 6)


def test_process_species_with_n_padding_writes_fixed_length(tmp_path: Path) -> None:
    """Write N-padded outputs that preserve source sequence lengths."""
    raw_dir = _prepare_species_raw(tmp_path, "SpecPad")

    _write_text(
        raw_dir / "100bp.err",
        "DEBUG donor " + ("A" * 10) + " acceptor " + ("C" * 10) + " + TX1 3\n",
    )
    _write_text(
        raw_dir / "100bp.neg.err",
        "DEBUG pair " + ("G" * 10) + " " + ("T" * 10) + " - 3\n",
    )

    _ = process_species(
        species="SpecPad",
        data_root=tmp_path / "data",
        pos_input_name="100bp.err",
        neg_input_name="100bp.neg.err",
        out_pos_name="100bp_trimmed.err",
        out_neg_name="100bp_trimmed.neg.err",
        exon_context_bp=2,
        pad_with_n=True,
        strict=True,
    )

    pos_tokens = (
        (raw_dir / "100bp_trimmed.err").read_text(encoding="utf-8").strip().split()
    )
    neg_tokens = (
        (raw_dir / "100bp_trimmed.neg.err")
        .read_text(encoding="utf-8")
        .strip()
        .split()
    )

    # keep_len=5 -> donor keeps prefix and pads tail; acceptor pads head.
    assert pos_tokens[2] == ("A" * 5) + ("N" * 5)
    assert pos_tokens[4] == ("N" * 5) + ("C" * 5)
    assert neg_tokens[2] == ("G" * 5) + ("N" * 5)
    assert neg_tokens[3] == ("N" * 5) + ("T" * 5)
