from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ANALYSIS_SCRIPT = Path(__file__).resolve().parents[1] / "analysis" / "script"
if str(ANALYSIS_SCRIPT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SCRIPT))

from score.general_model_snpr import (  # noqa: E402
    _compute_species_curves,
    _build_arg_parser,
    _resolve_eval_species,
    _write_interactive_html,
    run_snpr_experiment,
    go,
)


def _create_species_like_directory(base_dir: Path, species: str) -> None:
    species_dir = base_dir / species
    (species_dir / "raw").mkdir(parents=True)
    (species_dir / "processed").mkdir(parents=True)
    (species_dir / "intron_score").mkdir(parents=True)
    (species_dir / "raw" / "transcript_class.txt").write_text(
        "tx1 =\n",
        encoding="utf-8",
    )
    (species_dir / "processed" / "transcripts.tsv").write_text(
        "transcript_id\tgene_id\tsite_type\tintron_index\n",
        encoding="utf-8",
    )
    (species_dir / "processed" / "transcripts.unique.map.tsv").write_text(
        "unique_transcript_id\tunique_intron_index\ttranscript_id\tintron_index"
        "\tchrom\tstrand\tintron_start\tintron_end\n",
        encoding="utf-8",
    )


def test_resolve_eval_species_all_filters_non_species_directories(
    tmp_path: Path,
) -> None:
    _create_species_like_directory(tmp_path, "Hsap")
    _create_species_like_directory(tmp_path, "Mmus")
    (tmp_path / "tuning").mkdir()

    resolved = _resolve_eval_species(tmp_path, "all")

    assert resolved == ["Hsap", "Mmus"]


def test_compute_species_curves_returns_species_aware_summary() -> None:
    labels = np.asarray([0, 1, 1, 0], dtype=np.int64)
    model_to_scores = {
        "baseline_min": np.asarray([0.1, 0.7, 0.8, 0.2], dtype=np.float64)
    }

    summaries, curves = _compute_species_curves(
        eval_species="Hsap",
        model_to_scores=model_to_scores,
        labels=labels,
        sensitivity_denominator=10,
    )

    assert len(summaries) == 1
    assert len(curves) == 1
    assert summaries[0].eval_species == "Hsap"
    assert summaries[0].point_count == 3
    assert summaries[0].sensitivity_denominator == 10
    assert curves[0].eval_species == "Hsap"
    assert curves[0].model_name == "baseline_min"


def test_write_interactive_html_requires_plotly_or_writes_file(
    tmp_path: Path,
) -> None:
    labels = np.asarray([0, 1, 1, 0], dtype=np.int64)
    model_to_scores = {
        "baseline_min": np.asarray([0.1, 0.7, 0.8, 0.2], dtype=np.float64)
    }
    summaries, curves = _compute_species_curves(
        eval_species="Hsap",
        model_to_scores=model_to_scores,
        labels=labels,
        sensitivity_denominator=10,
    )
    output_html = tmp_path / "snpr.html"

    if go is None:
        with pytest.raises(ImportError):
            _write_interactive_html(
                curves=curves,
                summaries=summaries,
                output_html=output_html,
                title="test",
            )
    else:
        _write_interactive_html(
            curves=curves,
            summaries=summaries,
            output_html=output_html,
            title="test",
        )
        assert output_html.is_file()
        content = output_html.read_text(encoding="utf-8")
        assert "plotly" in content.lower()


def test_write_interactive_html_places_legend_outside_right(
    tmp_path: Path,
) -> None:
    """The Plotly SN-PR legend should be rendered outside the plot area."""

    if go is None:
        pytest.skip("plotly is not available")

    labels = np.asarray([0, 1, 1, 0], dtype=np.int64)
    model_to_scores = {
        "baseline_min": np.asarray([0.1, 0.7, 0.8, 0.2], dtype=np.float64)
    }
    summaries, curves = _compute_species_curves(
        eval_species="Hsap",
        model_to_scores=model_to_scores,
        labels=labels,
        sensitivity_denominator=10,
    )
    output_html = tmp_path / "snpr.html"

    _write_interactive_html(
        curves=curves,
        summaries=summaries,
        output_html=output_html,
        title="test",
    )

    content = output_html.read_text(encoding="utf-8").replace(" ", "")
    assert '"legend":{"x":1.02' in content
    assert '"xanchor":"left"' in content
    assert '"yanchor":"top"' in content
    assert '"orientation":"v"' in content
    assert '"r":220' in content


def test_build_arg_parser_includes_logreg_c() -> None:
    """The SN-PR CLI should forward the L1 logistic regularization strength."""

    parser = _build_arg_parser()
    args = parser.parse_args(
        [
            "--train-species",
            "Mmus",
            "--eval-species",
            "Hsap",
            "--logreg-c",
            "0.25",
        ]
    )

    assert args.logreg_c == 0.25


def test_run_snpr_experiment_uses_l1_training_return_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SN-PR flow should accept the two-value L1 training return."""

    _create_species_like_directory(tmp_path, "SpX")

    train_rows = [object()] * 30
    features = np.asarray(
        [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
            [0.7, 0.8],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
    split = SimpleNamespace(
        train_indices=(0, 1),
        valid_indices=(2,),
        test_indices=(3,),
    )
    trained_run = SimpleNamespace(
        model_name="logreg",
        test_scores=np.asarray([0.2, 0.8], dtype=np.float64),
    )
    summary = SimpleNamespace(to_row=lambda: {"model_name": "logreg"})
    curve = SimpleNamespace(
        eval_species="SpX",
        model_name="logreg",
        sensitivities=np.asarray([10.0], dtype=np.float64),
        precisions=np.asarray([20.0], dtype=np.float64),
        f1_scores=np.asarray([30.0], dtype=np.float64),
    )

    monkeypatch.setattr(
        "score.general_model_snpr.build_species_feature_rows",
        lambda **kwargs: (
            train_rows,
            SimpleNamespace(transcript_count=30, nan_score_count=0),
        ),
    )
    monkeypatch.setattr(
        "score.general_model_snpr._to_model_arrays",
        lambda rows: (
            features,
            labels,
            np.asarray(["tx1", "tx2", "tx3", "tx4"], dtype=np.str_),
            np.asarray(["g1", "g1", "g2", "g2"], dtype=np.str_),
            np.asarray([1, 1, 1, 1], dtype=np.int64),
            np.asarray([10.0, 20.0, 30.0, 40.0], dtype=np.float64),
        ),
    )
    monkeypatch.setattr(
        "score.general_model_snpr.split_train_valid_test",
        lambda **kwargs: split,
    )
    monkeypatch.setattr(
        "score.general_model_snpr._train_model_suite",
        lambda *args, **kwargs: ([trained_run], object()),
    )
    monkeypatch.setattr(
        "score.general_model_snpr.evaluate_models_on_dataset",
        lambda **kwargs: [trained_run],
    )
    monkeypatch.setattr(
        "score.general_model_snpr._compute_species_curves",
        lambda **kwargs: ([summary], [curve]),
    )
    monkeypatch.setattr("score.general_model_snpr._plot_snpr", lambda **kwargs: None)
    monkeypatch.setattr("score.general_model_snpr.ensure_dir", lambda path: None)
    monkeypatch.setattr("score.general_model_snpr.write_tsv", lambda *args, **kwargs: None)

    args = argparse.Namespace(
        data_root=tmp_path,
        train_species="SpX",
        eval_species="SpX",
        score_model="cnn",
        max_transcripts=None,
        random_state=42,
        test_size=0.2,
        valid_size=0.2,
        precision_target=0.85,
        recall_target=0.85,
        logreg_c=1.0,
        output_png=tmp_path / "snpr.png",
        output_summary_tsv=tmp_path / "snpr.tsv",
        output_interactive_html=None,
        sn_denominator="eval_positive",
        reference_gff=None,
    )

    assert run_snpr_experiment(args) == 0
