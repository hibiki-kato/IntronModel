from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ANALYSIS_SCRIPT = Path(__file__).resolve().parents[1] / "analysis" / "script"
if str(ANALYSIS_SCRIPT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SCRIPT))

from score.transcript_truth_simple_models import (  # noqa: E402
    MinScoreBaselineModel,
    build_species_feature_rows,
    build_transcript_features,
    build_transcript_features_with_report,
    evaluate_model,
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
                "transcript_id\tintron_index\tscore\tlabel",
                "uintron_1\t1\t0.2\t1",
                "uintron_2\t1\t0.8\t1",
                "uintron_3\t1\tnot_a_number\t1",
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
