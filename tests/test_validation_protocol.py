from __future__ import annotations

from pathlib import Path

from util.path_format import resolve_path_string
from util.validation_protocol import build_validation_protocol


def test_build_validation_protocol_includes_train_source_signatures(
    tmp_path: Path,
) -> None:
    pos_path = tmp_path / "train_pos.err"
    neg_path = tmp_path / "train_neg.err"
    pos_path.write_text("DEBUG pair 0 AA GT\n", encoding="utf-8")
    neg_path.write_text("DEBUG pair 0 CC AG\n", encoding="utf-8")

    protocol = build_validation_protocol(
        val_frac=0.1,
        seed=1337,
        train_pos_path=str(pos_path),
        train_neg_path=str(neg_path),
        metric_primary="pair_pr_auc",
    )

    assert protocol["include_pair_mixed_negatives"] is False
    signature_obj = protocol["train_source_signature"]
    assert isinstance(signature_obj, dict)
    pos_signature = signature_obj["train_pos"]
    neg_signature = signature_obj["train_neg"]
    assert isinstance(pos_signature, dict)
    assert isinstance(neg_signature, dict)
    assert pos_signature["exists"] is True
    assert neg_signature["exists"] is True
    assert not Path(str(pos_signature["path"])).is_absolute()
    assert not Path(str(neg_signature["path"])).is_absolute()
    assert int(pos_signature["size_bytes"]) == pos_path.stat().st_size
    assert int(neg_signature["size_bytes"]) == neg_path.stat().st_size


def test_build_validation_protocol_tracks_pair_mixed_negative_signatures(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "Dmel" / "raw"
    processed_dir = tmp_path / "Dmel" / "processed"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    pos_path = raw_dir / "100bp.err"
    neg_path = raw_dir / "100bp.neg.err"
    mixed_neg_path = processed_dir / "100bp_mixed_one_side.neg.err"
    pos_path.write_text("DEBUG pair tx1 100 AA GT +\n", encoding="utf-8")
    neg_path.write_text("DEBUG pair tx2 120 CC AG -\n", encoding="utf-8")
    mixed_neg_path.write_text("DEBUG pair tx3 150 GG TT +\n", encoding="utf-8")

    protocol = build_validation_protocol(
        val_frac=0.1,
        seed=1337,
        train_pos_path=str(pos_path),
        train_neg_path=str(neg_path),
        metric_primary="pair_pr_auc",
        include_pair_mixed_negatives=True,
    )

    assert protocol["include_pair_mixed_negatives"] is True
    signature_obj = protocol["train_source_signature"]
    assert isinstance(signature_obj, dict)
    extra_negatives_obj = signature_obj["pair_extra_negatives"]
    assert isinstance(extra_negatives_obj, list)
    assert len(extra_negatives_obj) == 1
    extra_signature = extra_negatives_obj[0]
    assert isinstance(extra_signature, dict)
    assert extra_signature["exists"] is True
    assert not Path(str(extra_signature["path"])).is_absolute()
    assert resolve_path_string(
        str(extra_signature["path"]),
        base_dir=Path.cwd(),
    ) == mixed_neg_path.resolve()


def test_build_validation_protocol_changes_when_train_source_changes(
    tmp_path: Path,
) -> None:
    pos_path = tmp_path / "train_pos.err"
    neg_path = tmp_path / "train_neg.err"
    pos_path.write_text("DEBUG pair 0 AA GT\n", encoding="utf-8")
    neg_path.write_text("DEBUG pair 0 CC AG\n", encoding="utf-8")

    first_protocol = build_validation_protocol(
        val_frac=0.1,
        seed=7,
        train_pos_path=str(pos_path),
        train_neg_path=str(neg_path),
        metric_primary="pair_pr_auc",
    )
    pos_path.write_text(
        "DEBUG pair 0 AA GT\nDEBUG pair 1 TT GG\n",
        encoding="utf-8",
    )
    second_protocol = build_validation_protocol(
        val_frac=0.1,
        seed=7,
        train_pos_path=str(pos_path),
        train_neg_path=str(neg_path),
        metric_primary="pair_pr_auc",
    )

    first_signature = first_protocol["train_source_signature"]
    second_signature = second_protocol["train_source_signature"]
    assert isinstance(first_signature, dict)
    assert isinstance(second_signature, dict)
    first_pos = first_signature["train_pos"]
    second_pos = second_signature["train_pos"]
    assert isinstance(first_pos, dict)
    assert isinstance(second_pos, dict)
    assert int(first_pos["size_bytes"]) != int(second_pos["size_bytes"])
