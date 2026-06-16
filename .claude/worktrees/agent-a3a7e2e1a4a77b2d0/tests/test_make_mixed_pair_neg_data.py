from __future__ import annotations

from pathlib import Path

import pytest

from util.data_proc import parse_debug_training_record
from util.make_mixed_pair_neg_data import process_species


def _write_text(path: Path, text: str) -> None:
    """Write UTF-8 text to a file path."""
    path.write_text(text, encoding="utf-8")


def _prepare_species_raw(tmp_path: Path, species: str) -> Path:
    """Create species raw directory and return it."""
    raw_dir = tmp_path / "data" / species / "raw"
    raw_dir.mkdir(parents=True)
    return raw_dir


def test_process_species_writes_one_side_mixed_pairs(tmp_path: Path) -> None:
    """Generate mixed negatives and write them under processed directory."""
    raw_dir = _prepare_species_raw(tmp_path, "SpecMix")
    _write_text(
        raw_dir / "100bp.err",
        "\n".join(
            [
                "DEBUG donor AAAA acceptor CCCC + TX1 10",
                "DEBUG donor TTTT acceptor GGGG - TX2 8",
                "",
            ]
        ),
    )
    _write_text(
        raw_dir / "100bp.neg.err",
        "\n".join(
            [
                "DEBUG donor NNNN +",
                "DEBUG donor MMMM -",
                "DEBUG acceptor QQQQ +",
                "DEBUG acceptor PPPP -",
                "DEBUG pair ZZZZ YYYY + 2",
                "",
            ]
        ),
    )

    stats = process_species(
        species="SpecMix",
        data_root=tmp_path / "data",
        pos_input_name="100bp.err",
        neg_input_name="100bp.neg.err",
        output_name="100bp_mixed_one_side.neg.err",
        mix_mode="both",
        samples_per_negative=1,
        seed=7,
        shuffle=False,
        strict=True,
    )

    output_path = (
        tmp_path
        / "data"
        / "SpecMix"
        / "processed"
        / "100bp_mixed_one_side.neg.err"
    )
    rows = [
        line
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert stats.generated_pairs == 4
    assert len(rows) == 4

    known_positive = {("AAAA", "CCCC"), ("TTTT", "GGGG")}
    for row in rows:
        parsed = parse_debug_training_record(row)
        assert parsed is not None
        assert parsed.record_type == "pair"
        assert parsed.donor_seq is not None
        assert parsed.acceptor_seq is not None
        pair_key = (parsed.donor_seq, parsed.acceptor_seq)
        assert pair_key not in known_positive
        assert row.startswith("DEBUG pair ")


def test_process_species_generates_per_negative_count(tmp_path: Path) -> None:
    """Generate counts anchored on negatives rather than positives."""
    raw_dir = _prepare_species_raw(tmp_path, "SpecNegCount")
    _write_text(
        raw_dir / "100bp.err",
        "\n".join(
            [
                "DEBUG donor AAAA acceptor CCCC + TX1 10",
                "",
            ]
        ),
    )
    _write_text(
        raw_dir / "100bp.neg.err",
        "\n".join(
            [
                "DEBUG donor NNNN +",
                "DEBUG donor MMMM +",
                "DEBUG donor LLLL +",
                "DEBUG acceptor QQQQ +",
                "DEBUG acceptor PPPP +",
                "",
            ]
        ),
    )

    stats = process_species(
        species="SpecNegCount",
        data_root=tmp_path / "data",
        pos_input_name="100bp.err",
        neg_input_name="100bp.neg.err",
        output_name="100bp_mixed_one_side.neg.err",
        mix_mode="both",
        samples_per_negative=1,
        seed=3,
        shuffle=False,
        strict=True,
    )
    assert stats.generated_pairs == 5


def test_process_species_strict_mode_rejects_malformed_debug_lines(
    tmp_path: Path,
) -> None:
    """Raise clear error when strict parsing hits malformed DEBUG lines."""
    raw_dir = _prepare_species_raw(tmp_path, "SpecStrict")
    _write_text(raw_dir / "100bp.err", "DEBUG donor AAAA acceptor CCCC + TX1 10\n")
    _write_text(
        raw_dir / "100bp.neg.err",
        "\n".join(
            [
                "DEBUG donor NNNN +",
                "DEBUG pair AAAA",
                "",
            ]
        ),
    )

    with pytest.raises(ValueError, match="Malformed DEBUG line"):
        _ = process_species(
            species="SpecStrict",
            data_root=tmp_path / "data",
            pos_input_name="100bp.err",
            neg_input_name="100bp.neg.err",
            output_name="100bp_mixed_one_side.neg.err",
            mix_mode="both",
            samples_per_negative=1,
            seed=1,
            shuffle=True,
            strict=True,
        )


def test_process_species_requires_negative_acceptor_pool(tmp_path: Path) -> None:
    """Fail early when chosen mode needs missing negative site pools."""
    raw_dir = _prepare_species_raw(tmp_path, "SpecPool")
    _write_text(raw_dir / "100bp.err", "DEBUG donor AAAA acceptor CCCC + TX1 10\n")
    _write_text(raw_dir / "100bp.neg.err", "DEBUG donor NNNN +\n")

    with pytest.raises(ValueError, match="Negative acceptor pool is empty"):
        _ = process_species(
            species="SpecPool",
            data_root=tmp_path / "data",
            pos_input_name="100bp.err",
            neg_input_name="100bp.neg.err",
            output_name="100bp_mixed_one_side.neg.err",
            mix_mode="donor_pos",
            samples_per_negative=1,
            seed=11,
            shuffle=True,
            strict=True,
        )
