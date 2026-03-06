from __future__ import annotations

from pathlib import Path
import sys

import pytest


ANALYSIS_SRC = Path(__file__).resolve().parents[1] / "analysis" / "src"
if str(ANALYSIS_SRC) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SRC))

from raw.raw_data_notebook_lib import (  # noqa: E402
    AnnotationCoverageRow,
    EvaluationTranscriptIntronCountRow,
    EvaluationTranscriptGroupIntronCountRow,
    SiteLabelCountRow,
    build_annotation_coverage_rows,
    build_evaluation_transcript_intron_count_rows,
    build_evaluation_transcript_group_intron_count_rows,
    build_intron_count_comparison_rows,
    build_site_label_count_rows,
    build_noncanonical_ratio_rows,
    build_sequence_quality_rows,
    build_species_overlap_sets,
    build_duplicate_rate_rows,
    collect_species_intron_length_profiles,
    parse_final_score_intron_lengths,
    parse_negative_pair_count,
    parse_training_pair_records,
    parse_training_intron_lengths,
    plot_site_label_count_comparison,
)


def test_parse_training_intron_lengths_reads_tail_times_two(tmp_path: Path) -> None:
    path = tmp_path / "100bp.err"
    path.write_text(
        "\n".join(
            [
                "DEBUG donor AAAA acceptor CCCC + TX1 10",
                "DEBUG donor GGGG acceptor TTTT + TX2 3",
                "DEBUG donor NNNN +",
                "not-a-debug-line",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert parse_training_intron_lengths(path) == [20, 6]


def test_parse_final_score_intron_lengths_pairs_donor_acceptor(
    tmp_path: Path,
) -> None:
    content = "\n".join(
        [
            "transcript_id\tintron_index\tsite_type\tboundary_pos\tseq",
            "tx1\t1\tdonor\t100\tAAAA",
            "tx1\t1\tacceptor\t130\tCCCC",
            "tx2\t4\tdonor\t700\tAAAA",
            "tx2\t4\tacceptor\t655\tCCCC",
            "tx3\t1\tdonor\t10\tAAAA",
        ]
    )
    path = tmp_path / "transcripts.tsv"
    path.write_text(content + "\n", encoding="utf-8")

    try:
        assert parse_final_score_intron_lengths(path) == [30, 45]
    finally:
        path.unlink(missing_ok=True)


def test_build_noncanonical_ratio_rows_filters_requested_sources() -> None:
    rows = [
        {
            "species": "A",
            "dataset": "training",
            "subset": "pos",
            "pairs_total": "100",
            "pairs_non_gt_ag_fraction": "0.2",
        },
        {
            "species": "A",
            "dataset": "transcript",
            "subset": "all",
            "pairs_total": "50",
            "pairs_non_gt_ag_fraction": "0.1",
        },
        {
            "species": "A",
            "dataset": "training",
            "subset": "neg",
            "pairs_total": "75",
            "pairs_non_gt_ag_fraction": "0.3",
        },
        {
            "species": "A",
            "dataset": "training",
            "subset": "all",
            "pairs_total": "999",
            "pairs_non_gt_ag_fraction": "0.9",
        },
    ]

    out = build_noncanonical_ratio_rows(
        rows,
        sources=(
            ("training", "pos", "train_pos"),
            ("training", "neg", "train_neg"),
            ("transcript", "all", "final_score"),
        ),
    )

    assert len(out) == 3
    assert out[0].species == "A"
    assert out[0].source_label == "final_score"
    assert out[0].noncanonical_fraction == 0.1
    assert out[1].source_label == "train_neg"
    assert out[1].noncanonical_fraction == 0.3
    assert out[2].source_label == "train_pos"


def test_build_annotation_coverage_rows_counts_reference_query_and_inference(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    species_dir = data_root / "SpX"
    raw_dir = species_dir / "raw"
    raw_dir.mkdir(parents=True)

    (raw_dir / "reference.fix.gff").write_text(
        "\n".join(
            [
                "chr1\tref\tgene\t1\t100\t.\t+\t.\tID=gene-G1;gene=G1",
                (
                    "chr1\tref\texon\t1\t40\t.\t+\t.\tParent=tx1;"
                    "transcript_id=tx1;gene=G1"
                ),
                "chr1\tref\tgene\t200\t400\t.\t+\t.\tID=gene-G2;gene=G2",
                (
                    "chr1\tref\texon\t200\t280\t.\t+\t.\tParent=tx2;"
                    "transcript_id=tx2;gene=G2"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (raw_dir / "query.fna.gtf").write_text(
        "\n".join(
            [
                (
                    'chr1\tquery\ttranscript\t1\t100\t.\t+\t.\t'
                    'transcript_id "qtx1"; gene_id "QG1";'
                ),
                (
                    'chr1\tquery\texon\t1\t40\t.\t+\t.\t'
                    'transcript_id "qtx1"; gene_id "QG1";'
                ),
                (
                    'chr1\tquery\ttranscript\t200\t300\t.\t+\t.\t'
                    'transcript_id "qtx2"; gene_id "QG2";'
                ),
                (
                    'chr1\tquery\texon\t200\t240\t.\t+\t.\t'
                    'transcript_id "qtx2"; gene_id "QG2";'
                ),
                (
                    'chr1\tquery\ttranscript\t400\t500\t.\t+\t.\t'
                    'transcript_id "qtx3"; gene_id "QG2";'
                ),
                (
                    'chr1\tquery\texon\t400\t450\t.\t+\t.\t'
                    'transcript_id "qtx3"; gene_id "QG2";'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (raw_dir / "transcripts.tsv").write_text(
        "\n".join(
            [
                "transcript_id\tgene_id\tsite_type\tintron_index\tseq",
                "qtx1\tQG1\tdonor\t1\tAAAA",
                "qtx1\tQG1\tacceptor\t1\tCCCC",
                "qtx3\tQG2\tdonor\t1\tGGGG",
                "qtx3\tQG2\tacceptor\t1\tTTTT",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = build_annotation_coverage_rows(data_root)

    assert rows == [
        AnnotationCoverageRow(
            species="SpX",
            reference_gene_count=2,
            query_gene_count=2,
            query_transcript_count=3,
            inference_gene_count=2,
            inference_transcript_count=2,
        )
    ]


def test_build_evaluation_transcript_intron_count_rows(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    species_dir = data_root / "SpX"
    raw_dir = species_dir / "raw"
    raw_dir.mkdir(parents=True)

    (raw_dir / "transcripts.tsv").write_text(
        "\n".join(
            [
                "transcript_id\tintron_index\tsite_type\tseq",
                "tx1\t1\tdonor\tAAAA",
                "tx1\t1\tacceptor\tCCCC",
                "tx2\t1\tdonor\tGGGG",
                "tx2\t1\tacceptor\tTTTT",
                "tx2\t2\tdonor\tGGGG",
                "tx2\t2\tacceptor\tTTTT",
                "tx3\t3\tdonor\tACGT",
                "tx3\t3\tacceptor\tTGCA",
                "tx3\t7\tdonor\tACGT",
                "tx3\t7\tacceptor\tTGCA",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = build_evaluation_transcript_intron_count_rows(data_root)

    assert rows[0] == EvaluationTranscriptIntronCountRow(
        species="SpX",
        intron_count=1,
        transcript_count=1,
        transcript_fraction=rows[0].transcript_fraction,
        cumulative_transcript_count=1,
        cumulative_fraction=rows[0].cumulative_fraction,
    )
    assert rows[1] == EvaluationTranscriptIntronCountRow(
        species="SpX",
        intron_count=2,
        transcript_count=2,
        transcript_fraction=rows[1].transcript_fraction,
        cumulative_transcript_count=3,
        cumulative_fraction=rows[1].cumulative_fraction,
    )
    assert rows[0].transcript_fraction == pytest.approx(1.0 / 3.0)
    assert rows[0].cumulative_fraction == pytest.approx(1.0 / 3.0)
    assert rows[1].transcript_fraction == pytest.approx(2.0 / 3.0)
    assert rows[1].cumulative_fraction == pytest.approx(1.0)


def test_build_evaluation_transcript_group_intron_count_rows(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    species_dir = data_root / "SpX"
    raw_dir = species_dir / "raw"
    raw_dir.mkdir(parents=True)

    (raw_dir / "transcripts.tsv").write_text(
        "\n".join(
            [
                "transcript_id\tintron_index\tsite_type\tseq",
                "tx_true_1\t1\tdonor\tAAAA",
                "tx_true_1\t1\tacceptor\tCCCC",
                "tx_true_2\t1\tdonor\tGGGG",
                "tx_true_2\t1\tacceptor\tTTTT",
                "tx_true_2\t2\tdonor\tGGGG",
                "tx_true_2\t2\tacceptor\tTTTT",
                "tx_false_1\t3\tdonor\tACGT",
                "tx_false_1\t3\tacceptor\tTGCA",
                "tx_false_1\t7\tdonor\tACGT",
                "tx_false_1\t7\tacceptor\tTGCA",
                "tx_contained\t4\tdonor\tACGT",
                "tx_contained\t4\tacceptor\tTGCA",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (raw_dir / "transcript_class.txt").write_text(
        "\n".join(
            [
                "tx_true_1 =",
                "tx_true_2 =",
                "tx_false_1 j",
                "tx_contained c",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = build_evaluation_transcript_group_intron_count_rows(data_root)

    assert rows == [
        EvaluationTranscriptGroupIntronCountRow(
            species="SpX",
            transcript_group="false",
            intron_count=2,
            transcript_count=1,
            transcript_fraction_within_group=1.0,
            cumulative_transcript_count=1,
            cumulative_fraction_within_group=1.0,
        ),
        EvaluationTranscriptGroupIntronCountRow(
            species="SpX",
            transcript_group="true",
            intron_count=1,
            transcript_count=1,
            transcript_fraction_within_group=0.5,
            cumulative_transcript_count=1,
            cumulative_fraction_within_group=0.5,
        ),
        EvaluationTranscriptGroupIntronCountRow(
            species="SpX",
            transcript_group="true",
            intron_count=2,
            transcript_count=1,
            transcript_fraction_within_group=0.5,
            cumulative_transcript_count=2,
            cumulative_fraction_within_group=1.0,
        ),
    ]


def test_collect_species_intron_length_profiles(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    species_dir = data_root / "SpX"
    raw_dir = species_dir / "raw"
    raw_dir.mkdir(parents=True)

    (raw_dir / "transcripts.tsv").write_text(
        "\n".join(
            [
                "transcript_id\tintron_index\tsite_type\tboundary_pos\tseq",
                "tx1\t1\tdonor\t100\tAAAA",
                "tx1\t1\tacceptor\t120\tCCCC",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (raw_dir / "100bp.err").write_text(
        "DEBUG donor AAAA acceptor CCCC + TX1 10\n",
        encoding="utf-8",
    )

    profiles = collect_species_intron_length_profiles(data_root)

    assert len(profiles) == 1
    profile = profiles[0]
    assert profile.species == "SpX"
    assert profile.final_score_lengths == [20]
    assert profile.training_lengths == [20]


def test_parse_negative_pair_count_counts_only_debug_pair(tmp_path: Path) -> None:
    path = tmp_path / "100bp.neg.err"
    path.write_text(
        "\n".join(
            [
                "DEBUG pair AAAA CCCC + 4",
                "DEBUG donor TTTT +",
                "DEBUG pair GGGG TTTT - 7",
                "DEBUG acceptor CCCC +",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert parse_negative_pair_count(path) == 2


def test_build_intron_count_comparison_rows(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    species_dir = data_root / "SpX"
    raw_dir = species_dir / "raw"
    raw_dir.mkdir(parents=True)

    (raw_dir / "transcripts.tsv").write_text(
        "\n".join(
            [
                "transcript_id\tintron_index\tsite_type\tboundary_pos\tseq",
                "tx1\t1\tdonor\t100\tAAAA",
                "tx1\t1\tacceptor\t120\tCCCC",
                "tx1\t2\tdonor\t200\tAAAA",
                "tx1\t2\tacceptor\t230\tCCCC",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (raw_dir / "100bp.err").write_text(
        "\n".join(
            [
                "DEBUG donor AAAA acceptor CCCC + TX1 10",
                "DEBUG donor GGGG acceptor TTTT + TX1 15",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (raw_dir / "100bp.neg.err").write_text(
        "\n".join(
            [
                "DEBUG pair AAAA CCCC + 5",
                "DEBUG pair GGGG TTTT - 6",
                "DEBUG donor ACGT +",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = build_intron_count_comparison_rows(data_root)

    assert len(rows) == 3
    counts = {(row.species, row.source_label): row.intron_count for row in rows}
    assert counts[("SpX", "Final")] == 2
    assert counts[("SpX", "Train (positive)")] == 2
    assert counts[("SpX", "Train (negative_pair)")] == 2


def test_parse_training_pair_records_supports_pair_and_pair_like(
    tmp_path: Path,
) -> None:
    path = tmp_path / "train_pair_records.err"
    path.write_text(
        "\n".join(
            [
                "DEBUG pair AAAA CCCC + 5",
                "DEBUG donor GGGG acceptor TTTT - TX1 7",
                "DEBUG donor ACGT +",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        records = parse_training_pair_records(path)
    finally:
        path.unlink(missing_ok=True)

    assert len(records) == 2
    assert records[0].donor_seq == "AAAA"
    assert records[0].acceptor_seq == "CCCC"
    assert records[0].transcript_id is None
    assert records[1].transcript_id == "TX1"


def test_quality_duplicate_and_overlap_rows(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    species_dir = data_root / "SpX"
    raw_dir = species_dir / "raw"
    raw_dir.mkdir(parents=True)

    (raw_dir / "transcripts.tsv").write_text(
        "\n".join(
            [
                "transcript_id\tintron_index\tsite_type\tboundary_pos\tseq",
                "tx1\t1\tdonor\t100\tAAAAN",
                "tx1\t1\tacceptor\t120\tCCCCN",
                "tx2\t1\tdonor\t200\tGGGGG",
                "tx2\t1\tacceptor\t230\tTTTTT",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (raw_dir / "100bp.err").write_text(
        "\n".join(
            [
                "DEBUG donor AAAAN acceptor CCCCN + tx1 10",
                "DEBUG donor GGGGG acceptor TTTTT + tx2 15",
                "DEBUG donor GGGGG acceptor TTTTT + tx2 15",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (raw_dir / "100bp.neg.err").write_text(
        "\n".join(
            [
                "DEBUG pair AAAAN CCCCN + 5",
                "DEBUG pair NNNNN NNNNN - 6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    quality_rows = build_sequence_quality_rows(data_root)
    duplicate_rows = build_duplicate_rate_rows(data_root)
    overlap_sets = build_species_overlap_sets(data_root)

    assert len(quality_rows) == 3
    assert len(duplicate_rows) == 3
    assert len(overlap_sets) == 1

    dup_map = {
        (row.species, row.source_label): row.duplicate_fraction
        for row in duplicate_rows
    }
    assert dup_map[("SpX", "Final")] == 0.0
    assert dup_map[("SpX", "Train (positive)")] > 0.0


def test_build_site_label_count_rows_counts_train_and_test(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    species_dir = data_root / "SpX"
    raw_dir = species_dir / "raw"
    raw_dir.mkdir(parents=True)

    (raw_dir / "100bp.err").write_text(
        "\n".join(
            [
                "DEBUG donor AAAA acceptor CCCC + TX1 10",
                "DEBUG donor GGGG +",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (raw_dir / "100bp.neg.err").write_text(
        "\n".join(
            [
                "DEBUG pair AAAA CCCC + 10",
                "DEBUG donor TTTT +",
                "DEBUG acceptor GGGG -",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (raw_dir / "intron_eval_flank10.tsv").write_text(
        "\n".join(
            [
                "species\ttranscript_id\tintron_index\tdonor_label\tacceptor_label",
                "SpX\ttx1\t1\t1\t0",
                "SpX\ttx2\t2\t0\t1",
                "SpX\ttx3\t3\t1\t1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = build_site_label_count_rows(data_root)
    result = {
        (row.species, row.split, row.site_type): (
            row.positive_count,
            row.negative_count,
        )
        for row in rows
    }

    assert result[("SpX", "train", "donor")] == (2, 2)
    assert result[("SpX", "train", "acceptor")] == (1, 2)
    assert result[("SpX", "test", "donor")] == (2, 1)
    assert result[("SpX", "test", "acceptor")] == (2, 1)


def test_build_site_label_count_rows_handles_large_tsv_fields(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    species_dir = data_root / "SpX"
    raw_dir = species_dir / "raw"
    raw_dir.mkdir(parents=True)

    large_context = "A" * 150_000
    (raw_dir / "intron_eval_flank10.tsv").write_text(
        "\n".join(
            [
                (
                    "species\ttranscript_id\tintron_index\tdonor_label\t"
                    "acceptor_label\tcontext"
                ),
                f"SpX\ttx1\t1\t1\t0\t{large_context}",
                f"SpX\ttx2\t2\t0\t1\t{large_context}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = build_site_label_count_rows(data_root)
    result = {
        (row.species, row.split, row.site_type): (
            row.positive_count,
            row.negative_count,
        )
        for row in rows
    }

    assert result[("SpX", "test", "donor")] == (1, 1)
    assert result[("SpX", "test", "acceptor")] == (1, 1)


def test_plot_site_label_count_comparison_raises_on_empty_rows() -> None:
    try:
        plot_site_label_count_comparison([], title="test")
        assert False, "Expected ValueError for empty site-label rows."
    except ValueError as exc:
        assert "No site-label rows" in str(exc)


def test_plot_site_label_count_comparison_runs_with_single_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)

    rows = [
        SiteLabelCountRow(
            species="SpX",
            split="train",
            site_type="donor",
            positive_count=3,
            negative_count=5,
        ),
        SiteLabelCountRow(
            species="SpY",
            split="train",
            site_type="donor",
            positive_count=7,
            negative_count=2,
        ),
    ]
    plot_site_label_count_comparison(rows, title="site-label counts")


def test_plot_site_label_count_comparison_saves_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)

    rows = [
        SiteLabelCountRow(
            species="SpX",
            split="train",
            site_type="donor",
            positive_count=3,
            negative_count=5,
        ),
        SiteLabelCountRow(
            species="SpY",
            split="train",
            site_type="donor",
            positive_count=7,
            negative_count=2,
        ),
    ]
    output_path = tmp_path / "site_label_counts.png"

    plot_site_label_count_comparison(
        rows,
        title="site-label counts",
        output_path=output_path,
    )

    assert output_path.is_file()
    assert output_path.stat().st_size > 0
