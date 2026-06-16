from __future__ import annotations

from pathlib import Path
import sys

import pytest
import matplotlib

matplotlib.use("Agg")

ANALYSIS_SCRIPT = Path(__file__).resolve().parents[1] / "analysis" / "script"
if str(ANALYSIS_SCRIPT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SCRIPT))

from score.snpr_curves import (  # noqa: E402
    build_intron_agreement_heatmap,
    build_intron_score_snpr_figure,
    build_site_score_snpr_figure,
    build_transcript_agreement_heatmap,
    build_transcript_score_roc_figure,
    build_transcript_score_snpr_figure,
    intron_score_agreement_at_sensitivity,
    resolve_transcript_class_file,
    transcript_score_agreement_at_sensitivity,
)


def _write_transcript_inputs(base_dir: Path) -> None:
    raw_dir = base_dir / "data" / "SpX" / "raw"
    score_dir = base_dir / "data" / "SpX" / "trans_score"
    raw_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)

    (raw_dir / "transcript_class.txt").write_text(
        "\n".join(
            [
                "tx1 =",
                "tx2 j",
                "tx3 =",
                "tx4 c",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (score_dir / "model.tsv").write_text(
        "\n".join(
            [
                "transcript_id\tmin_intron_index\tScore_donor\tScore_acceptor"
                "\tmin_donor_plus_acceptor",
                "tx1\t1\t0.1\t0.1\t0.1",
                "tx2\t1\t0.2\t0.2\t0.2",
                "tx3\t1\t0.9\t0.9\t0.9",
                "tx4\t1\t0.8\t0.8\t0.8",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_resolve_transcript_class_file_prefers_raw_default(tmp_path: Path) -> None:
    _write_transcript_inputs(tmp_path)

    resolved = resolve_transcript_class_file(
        repo_root=tmp_path,
        species="SpX",
    )

    assert resolved == tmp_path / "data" / "SpX" / "raw" / "transcript_class.txt"


def test_build_transcript_score_snpr_uses_test_positive_denominator(
    tmp_path: Path,
) -> None:
    _write_transcript_inputs(tmp_path)

    figure, curves, skipped = build_transcript_score_snpr_figure(
        repo_root=tmp_path,
        species="SpX",
        pattern="*.tsv",
    )

    assert len(skipped) == 0
    assert len(curves) == 1
    curve = curves[0]

    assert curve.positive_count == 2
    assert curve.used_row_count == 4
    assert curve.point_count == 3
    assert curve.sensitivities == (50.0, 50.0, 50.0)
    assert curve.precisions == (33.33, 50.0, 100.0)

    figure.clf()


def test_build_transcript_score_snpr_places_legend_outside_right(
    tmp_path: Path,
) -> None:
    """The SN-PR legend should be anchored outside the plotting area."""

    _write_transcript_inputs(tmp_path)

    figure, _, _ = build_transcript_score_snpr_figure(
        repo_root=tmp_path,
        species="SpX",
        pattern="*.tsv",
    )

    legend = figure.axes[0].get_legend()
    assert legend is not None
    assert legend._loc == 2

    bbox = legend.get_bbox_to_anchor()._bbox
    assert bbox.x0 == pytest.approx(1.02)
    assert bbox.y0 == pytest.approx(1.0)

    figure.clf()


def test_build_transcript_score_roc_places_legend_outside_right(
    tmp_path: Path,
) -> None:
    """The ROC legend should be anchored outside the plotting area."""

    _write_transcript_inputs(tmp_path)

    figure, _, _ = build_transcript_score_roc_figure(
        repo_root=tmp_path,
        species="SpX",
        pattern="*.tsv",
    )

    legend = figure.axes[0].get_legend()
    assert legend is not None
    assert legend._loc == 2

    bbox = legend.get_bbox_to_anchor()._bbox
    assert bbox.x0 == pytest.approx(1.02)
    assert bbox.y0 == pytest.approx(1.0)

    figure.clf()


def test_build_transcript_score_roc_curve(tmp_path: Path) -> None:
    _write_transcript_inputs(tmp_path)

    figure, curves, skipped = build_transcript_score_roc_figure(
        repo_root=tmp_path,
        species="SpX",
        pattern="*.tsv",
    )

    assert len(skipped) == 0
    assert len(curves) == 1
    curve = curves[0]
    assert curve.positive_count == 2
    assert curve.negative_count == 2
    assert curve.point_count == 5
    assert 0.0 <= curve.roc_auc <= 1.0
    assert curve.fprs[0] == 0.0
    assert curve.tprs[0] == 0.0

    figure.clf()


def _write_common_intron_assets(base_dir: Path) -> None:
    processed_dir = base_dir / "data" / "SpX" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "transcripts.unique.map.tsv").write_text(
        "\n".join(
            [
                "unique_transcript_id\tunique_intron_index\ttranscript_id"
                "\tintron_index",
                "tx1\t1\ttx1\t1",
                "tx2\t1\ttx2\t1",
                "tx3\t1\ttx3\t1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (processed_dir / "intron_eval_flank10.unique.tsv").write_text(
        "\n".join(
            [
                "transcript_id\tintron_index\tlabel\tdonor_label\tacceptor_label"
                "\ttrain_leak",
                "tx1\t1\t1\t1\t1\t1",
                "tx2\t1\t0\t0\t0\t0",
                "tx3\t1\t1\t1\t1\t0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_intron_score_snpr_draws_train_positive_hline_only_when_included(
    tmp_path: Path,
) -> None:
    _write_common_intron_assets(tmp_path)
    intron_dir = tmp_path / "data" / "SpX" / "intron_score"
    intron_dir.mkdir(parents=True, exist_ok=True)
    (intron_dir / "model.tsv").write_text(
        "\n".join(
            [
                "intron_id\tscore\tlabel",
                "tx1\t0.9\t1",
                "tx2\t0.1\t0",
                "tx3\t0.8\t1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    figure, _, _ = build_intron_score_snpr_figure(
        repo_root=tmp_path,
        species="SpX",
        exclude_train_duplicates=False,
    )
    line_labels = [line.get_label() for line in figure.axes[0].lines]
    assert any("train+ in train data:" in label for label in line_labels)
    hline = next(
        line
        for line in figure.axes[0].lines
        if "train+ in train data:" in line.get_label()
    )
    assert list(hline.get_ydata()) == [33.33, 33.33]
    figure.clf()

    figure_excluded, _, _ = build_intron_score_snpr_figure(
        repo_root=tmp_path,
        species="SpX",
        exclude_train_duplicates=True,
    )
    excluded_labels = [line.get_label() for line in figure_excluded.axes[0].lines]
    assert not any("train+ in train data:" in label for label in excluded_labels)
    figure_excluded.clf()


def test_site_score_snpr_draws_train_positive_hline_when_included(
    tmp_path: Path,
) -> None:
    _write_common_intron_assets(tmp_path)
    site_dir = tmp_path / "data" / "SpX" / "site_score"
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "model.tsv").write_text(
        "\n".join(
            [
                "transcript_id\tintron_index\tdonor_score\tacceptor_score\tlabel",
                "tx1\t1\t0.8\t0.9\t1",
                "tx2\t1\t0.1\t0.2\t0",
                "tx3\t1\t0.7\t0.8\t1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    figure, _, _ = build_site_score_snpr_figure(
        repo_root=tmp_path,
        species="SpX",
        exclude_train_duplicates=False,
    )
    line_labels = [line.get_label() for line in figure.axes[0].lines]
    assert any("train+ in train data:" in label for label in line_labels)
    hline = next(
        line
        for line in figure.axes[0].lines
        if "train+ in train data:" in line.get_label()
    )
    assert list(hline.get_ydata()) == [33.33, 33.33]
    figure.clf()


def test_intron_score_agreement_at_sensitivity(tmp_path: Path) -> None:
    _write_common_intron_assets(tmp_path)
    intron_dir = tmp_path / "data" / "SpX" / "intron_score"
    intron_dir.mkdir(parents=True, exist_ok=True)

    (intron_dir / "model_a.tsv").write_text(
        "\n".join(
            [
                "intron_id\tscore\tlabel",
                "tx1\t0.90\t1",
                "tx2\t0.10\t0",
                "tx3\t0.80\t1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (intron_dir / "model_b.tsv").write_text(
        "\n".join(
            [
                "intron_id\tscore\tlabel",
                "tx1\t0.95\t1",
                "tx2\t0.70\t0",
                "tx3\t0.60\t1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    model_rows, pair_rows, skipped = intron_score_agreement_at_sensitivity(
        repo_root=tmp_path,
        species="SpX",
        target_sensitivity=0.5,
    )

    assert len(skipped) == 0
    assert len(model_rows) == 2
    assert len(pair_rows) == 1
    pair = pair_rows[0]
    assert pair["model_a"] == "model_a"
    assert pair["model_b"] == "model_b"
    assert pair["common_rows"] == 3
    assert pair["agreement"] == 1.0
    assert pair["disagreement"] == 0.0
    assert pair["double_fault"] == (1.0 / 3.0)


def test_build_intron_agreement_heatmap() -> None:
    pair_rows = [
        {
            "model_a": "m1",
            "model_b": "m2",
            "common_rows": 10,
            "agreement": 0.7,
            "disagreement": 0.3,
            "kappa": 0.4,
            "both_positive_rate": 0.2,
            "both_negative_rate": 0.5,
            "double_fault": 0.1,
        }
    ]

    figure = build_intron_agreement_heatmap(
        pair_rows=pair_rows,
        metric="disagreement",
    )

    assert len(figure.axes) >= 1
    figure.clf()


def test_transcript_score_agreement_at_sensitivity(tmp_path: Path) -> None:
    _write_transcript_inputs(tmp_path)

    model_rows, pair_rows, skipped = transcript_score_agreement_at_sensitivity(
        repo_root=tmp_path,
        species="SpX",
        target_sensitivity=0.5,
        pattern="*.tsv",
    )

    assert len(skipped) == 0
    assert len(model_rows) == 1
    assert len(pair_rows) == 0
    row = model_rows[0]
    assert row["model_name"] == "model"
    assert row["used_row_count"] == 4
    assert row["positive_count"] == 2
    assert 0.0 < row["threshold"] <= 1.0
    assert row["achieved_sensitivity"] >= 0.5


def test_build_transcript_agreement_heatmap() -> None:
    pair_rows = [
        {
            "model_a": "m1",
            "model_b": "m2",
            "common_rows": 10,
            "agreement": 0.7,
            "disagreement": 0.3,
            "kappa": 0.4,
            "both_positive_rate": 0.2,
            "both_negative_rate": 0.5,
            "double_fault": 0.1,
        }
    ]

    figure = build_transcript_agreement_heatmap(
        pair_rows=pair_rows,
        metric="kappa",
    )

    assert len(figure.axes) >= 1
    figure.clf()
