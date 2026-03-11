from __future__ import annotations

from pathlib import Path

import pytest

from evaluate_intron_pr_auc import evaluate_labeled_introns


def _write_text(path: Path, text: str) -> None:
    """Write UTF-8 text to a file."""
    path.write_text(text, encoding="utf-8")


def _write_labeled_tsv(path: Path, rows: list[tuple[str, int, int]]) -> None:
    """Write labeled intron TSV."""
    lines = ["transcript_id\tintron_index\tlabel"]
    for transcript_id, intron_index, label in rows:
        lines.append(f"{transcript_id}\t{intron_index}\t{label}")
    _write_text(path, "\n".join(lines) + "\n")


def _write_site_score_tsv(
    path: Path,
    rows: list[tuple[str, int, str, float]],
) -> None:
    """Write site score TSV."""
    lines = ["transcript_id\tintron_index\tsite_type\tscore"]
    for transcript_id, intron_index, site_type, score in rows:
        lines.append(f"{transcript_id}\t{intron_index}\t{site_type}\t{score}")
    _write_text(path, "\n".join(lines) + "\n")


def _write_site_score_wide_tsv(
    path: Path,
    rows: list[tuple[str, int, float, float, int]],
) -> None:
    """Write site score TSV in wide format."""
    lines = ["transcript_id\tintron_index\tdonor_score\tacceptor_score\tlabel"]
    for transcript_id, intron_index, donor_score, acceptor_score, label in rows:
        lines.append(
            f"{transcript_id}\t{intron_index}\t"
            f"{donor_score:.6f}\t{acceptor_score:.6f}\t{label}"
        )
    _write_text(path, "\n".join(lines) + "\n")


def test_evaluate_labeled_introns_donor_acceptor_perfect_rank(
    tmp_path: Path,
) -> None:
    """Compute perfect PR/ROC AUC from donor+acceptor intron scores."""
    labeled_tsv = tmp_path / "labeled.tsv"
    site_score_tsv = tmp_path / "site_score.tsv"

    _write_labeled_tsv(
        labeled_tsv,
        rows=[
            ("tx1", 1, 1),
            ("tx2", 1, 0),
            ("tx3", 1, 1),
            ("tx4", 1, 0),
        ],
    )
    _write_site_score_tsv(
        site_score_tsv,
        rows=[
            ("tx1", 1, "donor", 0.95),
            ("tx1", 1, "acceptor", 0.90),
            ("tx2", 1, "donor", 0.20),
            ("tx2", 1, "acceptor", 0.20),
            ("tx3", 1, "donor", 0.90),
            ("tx3", 1, "acceptor", 0.85),
            ("tx4", 1, "donor", 0.25),
            ("tx4", 1, "acceptor", 0.30),
        ],
    )

    summary, rows = evaluate_labeled_introns(
        labeled_tsv=labeled_tsv,
        site_score_tsv=site_score_tsv,
        intron_score_op="*",
        score_source="donor_acceptor",
    )

    assert summary.used_introns == 4
    assert summary.positive_count == 2
    assert summary.negative_count == 2
    assert summary.skipped_missing_score_introns == 0
    assert summary.pr_auc == pytest.approx(1.0)
    assert summary.roc_auc == pytest.approx(1.0)
    assert len(rows) == 4


def test_evaluate_labeled_introns_auto_mode_uses_pair_scores(tmp_path: Path) -> None:
    """Use pair scores in auto mode when donor/acceptor scores are absent."""
    labeled_tsv = tmp_path / "labeled.tsv"
    site_score_tsv = tmp_path / "site_score.tsv"

    _write_labeled_tsv(
        labeled_tsv,
        rows=[
            ("tx1", 1, 1),
            ("tx2", 1, 0),
            ("tx3", 1, 1),
        ],
    )
    _write_site_score_tsv(
        site_score_tsv,
        rows=[
            ("tx1", 1, "pair", 0.90),
            ("tx2", 1, "pair", 0.10),
            ("tx3", 1, "pair", 0.80),
        ],
    )

    summary, _ = evaluate_labeled_introns(
        labeled_tsv=labeled_tsv,
        site_score_tsv=site_score_tsv,
        intron_score_op="*",
        score_source="auto",
    )

    assert summary.used_introns == 3
    assert summary.skipped_missing_score_introns == 0
    assert summary.pr_auc == pytest.approx(1.0)
    assert summary.roc_auc == pytest.approx(1.0)


def test_evaluate_labeled_introns_strict_missing_raises(tmp_path: Path) -> None:
    """Fail when strict missing mode is enabled and scores are missing."""
    labeled_tsv = tmp_path / "labeled.tsv"
    site_score_tsv = tmp_path / "site_score.tsv"

    _write_labeled_tsv(
        labeled_tsv,
        rows=[
            ("tx1", 1, 1),
            ("tx2", 1, 0),
        ],
    )
    _write_site_score_tsv(
        site_score_tsv,
        rows=[
            ("tx1", 1, "donor", 0.9),
            ("tx1", 1, "acceptor", 0.8),
        ],
    )

    with pytest.raises(ValueError, match="Missing intron scores"):
        _ = evaluate_labeled_introns(
            labeled_tsv=labeled_tsv,
            site_score_tsv=site_score_tsv,
            intron_score_op="*",
            score_source="donor_acceptor",
            strict_missing=True,
        )


def test_evaluate_labeled_introns_rejects_single_class_labels(
    tmp_path: Path,
) -> None:
    """Fail when joined labels do not contain both positive and negative rows."""
    labeled_tsv = tmp_path / "labeled.tsv"
    site_score_tsv = tmp_path / "site_score.tsv"

    _write_labeled_tsv(
        labeled_tsv,
        rows=[
            ("tx1", 1, 1),
            ("tx2", 1, 1),
        ],
    )
    _write_site_score_tsv(
        site_score_tsv,
        rows=[
            ("tx1", 1, "donor", 0.9),
            ("tx1", 1, "acceptor", 0.8),
            ("tx2", 1, "donor", 0.7),
            ("tx2", 1, "acceptor", 0.6),
        ],
    )

    with pytest.raises(ValueError, match="Both positive and negative labels"):
        _ = evaluate_labeled_introns(
            labeled_tsv=labeled_tsv,
            site_score_tsv=site_score_tsv,
            intron_score_op="*",
            score_source="donor_acceptor",
        )


def test_evaluate_labeled_introns_reads_wide_site_score_format(
    tmp_path: Path,
) -> None:
    """Compute metrics from the new wide site_score format."""
    labeled_tsv = tmp_path / "labeled.tsv"
    site_score_tsv = tmp_path / "site_score.tsv"

    _write_labeled_tsv(
        labeled_tsv,
        rows=[
            ("tx1", 1, 1),
            ("tx2", 1, 0),
            ("tx3", 1, 1),
            ("tx4", 1, 0),
        ],
    )
    _write_site_score_wide_tsv(
        site_score_tsv,
        rows=[
            ("tx1", 1, 0.95, 0.90, 1),
            ("tx2", 1, 0.20, 0.20, 0),
            ("tx3", 1, 0.90, 0.85, 1),
            ("tx4", 1, 0.25, 0.30, 0),
        ],
    )

    summary, rows = evaluate_labeled_introns(
        labeled_tsv=labeled_tsv,
        site_score_tsv=site_score_tsv,
        intron_score_op="*",
        score_source="donor_acceptor",
    )

    assert summary.used_introns == 4
    assert summary.skipped_missing_score_introns == 0
    assert summary.pr_auc == pytest.approx(1.0)
    assert summary.roc_auc == pytest.approx(1.0)
    assert len(rows) == 4
