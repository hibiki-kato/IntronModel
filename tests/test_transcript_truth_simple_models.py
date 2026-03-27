from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import matplotlib.pyplot as plt

ANALYSIS_SCRIPT = Path(__file__).resolve().parents[1] / "analysis" / "script"
if str(ANALYSIS_SCRIPT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SCRIPT))

from score.transcript_truth_simple_models import (  # noqa: E402
    DatasetSplit,
    MinScoreBaselineModel,
    FEATURE_COLUMNS,
    build_species_feature_rows,
    build_transcript_features,
    build_transcript_features_with_report,
    evaluate_model,
    plot_baseline_min_distribution,
    plot_feature_score_distributions,
    plot_logreg_coefficients_grid,
    plot_train_test_score_distributions,
    _select_best_logreg_run,
    _train_logreg_grid,
    run_l1_experiment,
    train_logistic_regression,
    split_train_valid_test,
)


def test_build_transcript_features_list_format_padding_rules() -> None:
    rows = [
        {
            "transcript_id": "tx1",
            "gene_id": "g1",
            "transcript_label": "1",
            "intron_scores": "[0.25]",
        },
        {
            "transcript_id": "tx2",
            "gene_id": "g2",
            "transcript_label": "0",
            "intron_scores": "0.10,0.30",
        },
    ]

    features = build_transcript_features(rows)
    by_tx = {row.transcript_id: row for row in features}

    tx1 = by_tx["tx1"]
    assert tx1.n_introns == 1
    assert tx1.min_score == 0.25
    assert tx1.second_smallest_score == 0.25
    assert tx1.third_smallest_score == 0.25
    assert np.isclose(tx1.log_score_sum, np.log(0.25))
    assert np.isclose(tx1.geometric_mean_score, 0.25)
    assert np.isclose(tx1.harmonic_mean_score, 0.25)
    assert tx1.variance_score == 0.0
    assert tx1.count_above_0_8 == 0

    tx2 = by_tx["tx2"]
    assert tx2.n_introns == 2
    assert tx2.min_score == 0.10
    assert tx2.second_smallest_score == 0.30
    assert tx2.third_smallest_score == 0.30
    assert np.isclose(tx2.variance_score, 0.01)
    assert np.isclose(tx2.coefficient_of_variation_score, 0.5)
    assert np.isclose(tx2.harmonic_mean_score, 0.15)
    assert np.isclose(tx2.lower_10_mean, 0.20)
    assert np.isclose(tx2.upper_2_mean, 0.20)
    assert np.isclose(tx2.mean_minus_min, 0.10)
    assert np.isclose(tx2.max_minus_median, 0.10)


def test_build_transcript_features_long_format_nan_report() -> None:
    rows = [
        {
            "transcript_id": "tx1",
            "gene_id": "g1",
            "transcript_label": "1",
            "intron_score": "0.4",
        },
        {
            "transcript_id": "tx1",
            "gene_id": "g1",
            "transcript_label": "1",
            "intron_score": "",
        },
        {
            "transcript_id": "tx2",
            "gene_id": "g2",
            "transcript_label": "0",
            "intron_score": "0.1",
        },
    ]

    features, report = build_transcript_features_with_report(rows)

    assert len(features) == 2
    assert report.nan_score_count == 1
    assert report.transcript_count == 2


def test_split_train_valid_test_gene_wise_no_overlap() -> None:
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    groups = np.asarray(["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4"])

    split = split_train_valid_test(
        labels=labels,
        groups=groups,
        test_size=0.25,
        valid_size=0.33,
        random_state=7,
    )

    train_groups = set(groups[list(split.train_indices)])
    valid_groups = set(groups[list(split.valid_indices)])
    test_groups = set(groups[list(split.test_indices)])

    assert train_groups.isdisjoint(valid_groups)
    assert train_groups.isdisjoint(test_groups)
    assert valid_groups.isdisjoint(test_groups)


def test_evaluate_model_returns_confusion_and_auc() -> None:
    model = MinScoreBaselineModel(min_feature_index=0)
    x = np.asarray([[0.1], [0.2], [0.8], [0.9]], dtype=np.float64)
    y = np.asarray([0, 0, 1, 1], dtype=np.int64)
    model.fit(x, y)

    result = evaluate_model(model, x, y, threshold=0.5)

    assert result.tn == 2
    assert result.tp == 2
    assert result.fp == 0
    assert result.fn == 0
    assert result.auroc == 1.0


def test_train_logistic_regression_uses_l1_with_standard_scaling() -> None:
    x = np.asarray(
        [
            [0.0, 1.0, 0.2],
            [1.0, 0.0, 0.8],
            [0.1, 0.9, 0.3],
            [0.9, 0.1, 0.7],
        ],
        dtype=np.float64,
    )
    y = np.asarray([0, 1, 0, 1], dtype=np.int64)

    model = train_logistic_regression(x, y, random_state=7, C=0.5)

    scaler = model.named_steps["scaler"]
    logreg = model.named_steps["logreg"]

    assert scaler.__class__.__name__ == "StandardScaler"
    assert logreg.solver == "saga"
    assert logreg.l1_ratio == 1.0
    assert logreg.C == 0.5
    assert logreg.coef_.shape == (1, x.shape[1])


def test_run_l1_experiment_selects_best_c_and_shows_plots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The L1 notebook path should compare C values and show figures."""

    train_x = np.asarray(
        [
            [0.1, 1.0, 0.2],
            [0.2, 0.9, 0.3],
            [0.8, 0.2, 0.7],
            [0.9, 0.1, 0.8],
        ],
        dtype=np.float64,
    )
    train_y = np.asarray([0, 0, 1, 1], dtype=np.int64)
    test_x = np.asarray(
        [
            [0.05, 0.9, 0.1],
            [0.95, 0.05, 0.9],
        ],
        dtype=np.float64,
    )
    test_y = np.asarray([0, 1], dtype=np.int64)

    class FakeModel:
        """Minimal fake model keyed by C value."""

        def __init__(self, c_value: float) -> None:
            self.c_value = c_value

        def predict_proba(self, features: np.ndarray) -> np.ndarray:
            if self.c_value == 1.0:
                scores = np.asarray([0.8, 0.2], dtype=np.float64)
            elif self.c_value == 10.0:
                scores = np.asarray([0.3, 0.7], dtype=np.float64)
            else:
                scores = np.asarray([0.05, 0.95], dtype=np.float64)
            return np.column_stack((1.0 - scores, scores))

    def fake_train_logistic_regression(*args, **kwargs):
        return FakeModel(float(kwargs["C"]))

    def fake_split_train_test(**kwargs):
        return (
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([2, 3], dtype=np.int64),
        )

    def fake_to_model_arrays(rows):
        _ = rows
        feature_matrix = np.zeros((4, len(FEATURE_COLUMNS)), dtype=np.float64)
        feature_matrix[:, FEATURE_COLUMNS.index("min_score")] = train_x[:, 0]
        feature_matrix[:, FEATURE_COLUMNS.index("mean_score")] = train_x[:, 1]
        feature_matrix[:, FEATURE_COLUMNS.index("max_score")] = train_x[:, 2]
        return (
            feature_matrix,
            np.asarray([0, 0, 1, 1], dtype=np.int64),
            np.asarray(["tx1", "tx2", "tx3", "tx4"], dtype=np.str_),
            np.asarray(["g1", "g2", "g3", "g4"], dtype=np.str_),
            np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
            np.asarray([100.0, 100.0, 100.0, 100.0], dtype=np.float64),
        )

    monkeypatch.setattr(
        "score.transcript_truth_simple_models.train_logistic_regression",
        fake_train_logistic_regression,
    )
    monkeypatch.setattr(
        "score.transcript_truth_simple_models.split_train_test",
        fake_split_train_test,
    )
    monkeypatch.setattr(
        "score.transcript_truth_simple_models._to_model_arrays",
        fake_to_model_arrays,
    )
    monkeypatch.setattr(
        "score.transcript_truth_simple_models._write_feature_table",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "score.transcript_truth_simple_models._write_split_table",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "score.transcript_truth_simple_models._extract_logreg_coefficients_grid",
        lambda **kwargs: [
            {
                "c_value": 1.0,
                "feature_name": "min_score",
                "standardized_coefficient": 0.1,
                "original_scale_coefficient": 0.2,
                "feature_std": 1.0,
                "selected": True,
                "abs_standardized_coefficient": 0.1,
            },
            {
                "c_value": 10.0,
                "feature_name": "min_score",
                "standardized_coefficient": 0.2,
                "original_scale_coefficient": 0.3,
                "feature_std": 1.0,
                "selected": True,
                "abs_standardized_coefficient": 0.2,
            },
            {
                "c_value": 100.0,
                "feature_name": "min_score",
                "standardized_coefficient": 0.3,
                "original_scale_coefficient": 0.4,
                "feature_std": 1.0,
                "selected": True,
                "abs_standardized_coefficient": 0.3,
            },
        ],
    )

    result = run_l1_experiment(
        feature_rows=[object()] * 30,
        output_dir=tmp_path,
        random_state=7,
        test_size=0.25,
        logreg_cs=(1.0, 10.0, 100.0),
    )

    assert [run.c_value for run in result.runs] == [1.0, 10.0, 100.0]
    assert result.selected_run.c_value == 100.0
    assert len(result.coefficient_rows) == 3
    assert result.selected_feature_names == ("min_score",)
    assert np.array_equal(result.train_labels, np.asarray([0, 0], dtype=np.int64))
    assert np.array_equal(result.test_labels, np.asarray([1, 1], dtype=np.int64))
    assert (tmp_path / "tables" / "logreg_coefficients.tsv").is_file()


def test_plot_baseline_min_distribution_shows_plots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The baseline helper should save and show the distribution plot."""

    feature_rows = [object()] * 30
    feature_matrix = np.asarray(
        np.column_stack(
            [
                np.linspace(0.0, 1.0, 30),
                np.linspace(0.1, 0.9, 30),
            ]
        ),
        dtype=np.float64,
    )
    labels = np.asarray([index % 2 for index in range(30)], dtype=np.int64)
    transcript_ids = np.asarray(
        [f"tx{i}" for i in range(30)],
        dtype=np.str_,
    )
    gene_ids = np.asarray(
        [f"g{i // 2}" for i in range(30)],
        dtype=np.str_,
    )
    intron_counts = np.arange(30, dtype=np.float64)
    lengths = np.full(30, 100.0, dtype=np.float64)

    monkeypatch.setattr(
        "score.transcript_truth_simple_models._to_model_arrays",
        lambda rows: (
            feature_matrix,
            labels,
            transcript_ids,
            gene_ids,
            intron_counts,
            lengths,
        ),
    )
    monkeypatch.setattr(
        "score.transcript_truth_simple_models.split_train_test",
        lambda **kwargs: (
            np.asarray(range(0, 20), dtype=np.int64),
            np.asarray(range(20, 30), dtype=np.int64),
        ),
    )

    show_calls: list[bool] = []
    monkeypatch.setattr(plt, "show", lambda: show_calls.append(True))

    plot_baseline_min_distribution(
        feature_rows=feature_rows,
        output_dir=tmp_path,
        random_state=7,
        show_plots=True,
    )

    assert (tmp_path / "figures" / "baseline_min_score_distribution.png").is_file()
    assert len(show_calls) == 1


def test_train_logreg_grid_selects_best_c_by_test_auroc() -> None:
    """The grid helper should preserve all Cs and select the best test AUROC."""

    train_x = np.asarray([[0.0], [1.0], [0.2], [0.8]], dtype=np.float64)
    train_y = np.asarray([0, 1, 0, 1], dtype=np.int64)
    test_x = np.asarray([[0.05], [0.95]], dtype=np.float64)
    test_y = np.asarray([0, 1], dtype=np.int64)

    scores_by_key: dict[tuple[float, int], np.ndarray] = {
        (1.0, id(train_x)): np.asarray([0.9, 0.1, 0.8, 0.2], dtype=np.float64),
        (1.0, id(test_x)): np.asarray([0.8, 0.2], dtype=np.float64),
        (10.0, id(train_x)): np.asarray([0.1, 0.9, 0.2, 0.8], dtype=np.float64),
        (10.0, id(test_x)): np.asarray([0.2, 0.8], dtype=np.float64),
        (100.0, id(train_x)): np.asarray([0.05, 0.95, 0.1, 0.9], dtype=np.float64),
        (100.0, id(test_x)): np.asarray([0.05, 0.95], dtype=np.float64),
    }

    class FakeModel:
        """Minimal fake model keyed by C value."""

        def __init__(self, c_value: float) -> None:
            self.c_value = c_value

        def predict_proba(self, features: np.ndarray) -> np.ndarray:
            scores = scores_by_key[(self.c_value, id(features))]
            return np.column_stack((1.0 - scores, scores))

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "score.transcript_truth_simple_models.train_logistic_regression",
        lambda *args, **kwargs: FakeModel(float(kwargs["C"])),
    )
    runs = _train_logreg_grid(
        train_x,
        train_y,
        test_x,
        test_y,
        random_state=7,
        logreg_cs=(1.0, 10.0, 100.0),
    )

    assert [run.c_value for run in runs] == [1.0, 10.0, 100.0]
    selected = _select_best_logreg_run(runs, test_y)
    assert selected.c_value == 100.0
    assert np.isclose(selected.test_scores[1], 0.95)


def test_plot_logreg_coefficients_grid_and_train_test_distribution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The point plots should save and display without using bar charts."""

    coefficient_rows = [
        {
            "c_value": 1.0,
            "feature_name": "min_score",
            "standardized_coefficient": 0.1,
            "original_scale_coefficient": 0.2,
            "feature_std": 1.0,
            "selected": True,
            "abs_standardized_coefficient": 0.1,
        },
        {
            "c_value": 10.0,
            "feature_name": "min_score",
            "standardized_coefficient": 0.2,
            "original_scale_coefficient": 0.3,
            "feature_std": 1.0,
            "selected": True,
            "abs_standardized_coefficient": 0.2,
        },
        {
            "c_value": 100.0,
            "feature_name": "min_score",
            "standardized_coefficient": 0.3,
            "original_scale_coefficient": 0.4,
            "feature_std": 1.0,
            "selected": True,
            "abs_standardized_coefficient": 0.3,
        },
        {
            "c_value": 1.0,
            "feature_name": "mean_score",
            "standardized_coefficient": -0.1,
            "original_scale_coefficient": -0.2,
            "feature_std": 1.0,
            "selected": True,
            "abs_standardized_coefficient": 0.1,
        },
        {
            "c_value": 10.0,
            "feature_name": "mean_score",
            "standardized_coefficient": -0.2,
            "original_scale_coefficient": -0.3,
            "feature_std": 1.0,
            "selected": True,
            "abs_standardized_coefficient": 0.2,
        },
        {
            "c_value": 100.0,
            "feature_name": "mean_score",
            "standardized_coefficient": -0.3,
            "original_scale_coefficient": -0.4,
            "feature_std": 1.0,
            "selected": True,
            "abs_standardized_coefficient": 0.3,
        },
    ]

    show_calls: list[bool] = []
    monkeypatch.setattr(plt, "show", lambda: show_calls.append(True))

    plot_logreg_coefficients_grid(
        tmp_path,
        coefficient_rows,
        show_plots=True,
    )
    plot_train_test_score_distributions(
        output_dir=tmp_path,
        train_scores=np.asarray([0.1, 0.9], dtype=np.float64),
        train_labels=np.asarray([0, 1], dtype=np.int64),
        test_scores=np.asarray([0.2, 0.8], dtype=np.float64),
        test_labels=np.asarray([0, 1], dtype=np.int64),
        selected_c=10.0,
        show_plots=True,
    )

    assert (tmp_path / "figures" / "logreg_coefficients.png").is_file()
    assert (
        tmp_path / "figures" / "logreg_balanced_score_distribution.png"
    ).is_file()
    assert len(show_calls) == 2


def test_plot_feature_score_distributions_shows_plots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The feature-score grid should save and show a compact distribution map."""

    feature_rows = [
        object(),
        object(),
        object(),
        object(),
    ]
    feature_matrix = np.asarray(
        np.zeros((4, len(FEATURE_COLUMNS)), dtype=np.float64),
        dtype=np.float64,
    )
    feature_matrix[:, FEATURE_COLUMNS.index("min_score")] = np.asarray(
        [0.1, 0.2, 0.8, 0.9],
        dtype=np.float64,
    )
    feature_matrix[:, FEATURE_COLUMNS.index("mean_score")] = np.asarray(
        [1.0, 0.9, 0.2, 0.1],
        dtype=np.float64,
    )
    feature_matrix[:, FEATURE_COLUMNS.index("max_score")] = np.asarray(
        [0.2, 0.3, 0.7, 0.8],
        dtype=np.float64,
    )
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    transcript_ids = np.asarray(["tx1", "tx2", "tx3", "tx4"], dtype=np.str_)
    gene_ids = np.asarray(["g1", "g2", "g3", "g4"], dtype=np.str_)
    intron_counts = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    lengths = np.asarray([100.0, 100.0, 100.0, 100.0], dtype=np.float64)

    monkeypatch.setattr(
        "score.transcript_truth_simple_models._to_model_arrays",
        lambda rows: (
            feature_matrix,
            labels,
            transcript_ids,
            gene_ids,
            intron_counts,
            lengths,
        ),
    )

    show_calls: list[bool] = []
    monkeypatch.setattr(plt, "show", lambda: show_calls.append(True))

    plot_feature_score_distributions(
        feature_rows=feature_rows,
        output_dir=tmp_path,
        selected_features=["min_score", "mean_score", "max_score"],
        show_plots=True,
    )

    assert (tmp_path / "figures" / "feature_score_distributions.png").is_file()
    assert len(show_calls) == 1


def test_build_species_feature_rows_from_unique_map(tmp_path: Path) -> None:
    species_dir = tmp_path / "SpX"
    (species_dir / "raw").mkdir(parents=True)
    (species_dir / "processed").mkdir(parents=True)
    (species_dir / "intron_score").mkdir(parents=True)

    (species_dir / "raw" / "transcript_class.txt").write_text(
        "\n".join(
            [
                "tx1 =",
                "tx2 j",
                "tx3 c",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (species_dir / "processed" / "transcripts.tsv").write_text(
        "\n".join(
            [
                "transcript_id\tgene_id\tsite_type\tintron_index",
                "tx1\tg1\tdonor\t1",
                "tx2\tg2\tdonor\t1",
                "tx3\tg3\tdonor\t1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (species_dir / "processed" / "transcripts.unique.map.tsv").write_text(
        "\n".join(
            [
                "unique_transcript_id\tunique_intron_index\ttranscript_id\t"
                "intron_index\tchrom\tstrand\tintron_start\tintron_end",
                "uintron_1\t1\ttx1\t1\tchr1\t+\t1\t2",
                "uintron_2\t1\ttx1\t2\tchr1\t+\t3\t4",
                "uintron_2\t1\ttx2\t1\tchr1\t+\t3\t4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (species_dir / "intron_score" / "cnn.tsv").write_text(
        "\n".join(
            [
                "intron_id\tscore\tlabel",
                "uintron_1\t0.2\t1",
                "uintron_2\t0.8\t1",
                "uintron_3\tnot_a_number\t1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    features, report = build_species_feature_rows(
        data_root=tmp_path,
        species="SpX",
        score_model="cnn",
    )

    assert report.total_input_rows == 3
    assert report.nan_score_count == 1
    assert report.transcript_count == 2
    by_tx = {row.transcript_id: row for row in features}
    assert by_tx["tx1"].n_introns == 2
    assert by_tx["tx1"].min_score == 0.2
    assert by_tx["tx2"].n_introns == 1
    assert "tx3" not in by_tx


def test_build_species_feature_rows_fallbacks_without_unique_map(
    tmp_path: Path,
) -> None:
    species_dir = tmp_path / "SpY"
    (species_dir / "raw").mkdir(parents=True)
    (species_dir / "processed").mkdir(parents=True)
    (species_dir / "intron_score").mkdir(parents=True)

    (species_dir / "raw" / "transcript_class.txt").write_text(
        "tx1 =\ntx2 j\n",
        encoding="utf-8",
    )
    (species_dir / "processed" / "transcripts.tsv").write_text(
        "\n".join(
            [
                "transcript_id\tgene_id\tsite_type\tintron_index",
                "tx1\tg1\tdonor\t1",
                "tx2\tg2\tdonor\t1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (species_dir / "intron_score" / "cnn.tsv").write_text(
        "\n".join(
            [
                "transcript_id\tintron_index\tscore\tlabel",
                "tx1\t1\t0.4\t1",
                "tx2\t1\t0.3\t1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    features, report = build_species_feature_rows(
        data_root=tmp_path,
        species="SpY",
        score_model="cnn.tsv",
    )

    assert report.transcript_count == 2
    assert {row.transcript_id for row in features} == {"tx1", "tx2"}
