from __future__ import annotations

from pathlib import Path

import pytest

from evaluate_scores import count_reference_transcripts, evaluate_score_file


def test_count_reference_transcripts_counts_exon_runs(tmp_path: Path) -> None:
    ref_gff = tmp_path / "ref.gff"
    ref_gff.write_text(
        "chr\tsrc\tgene\t1\t100\t.\t+\t.\tID=g1\n"
        "chr\tsrc\texon\t1\t10\t.\t+\t.\tParent=t1\n"
        "chr\tsrc\texon\t11\t20\t.\t+\t.\tParent=t1\n"
        "chr\tsrc\tmRNA\t1\t50\t.\t+\t.\tID=t2\n"
        "chr\tsrc\texon\t21\t30\t.\t+\t.\tParent=t2\n"
        "chr\tsrc\texon\t31\t40\t.\t+\t.\tParent=t2\n"
        "chr\tsrc\texon\t41\t50\t.\t+\t.\tParent=t2\n",
        encoding="utf-8",
    )

    assert count_reference_transcripts(ref_gff) == 2


def test_count_reference_transcripts_raises_without_multi_exon_run(
    tmp_path: Path,
) -> None:
    ref_gff = tmp_path / "ref.gff"
    ref_gff.write_text(
        "chr\tsrc\tgene\t1\t100\t.\t+\t.\tID=g1\n"
        "chr\tsrc\texon\t1\t10\t.\t+\t.\tParent=t1\n"
        "chr\tsrc\tmRNA\t1\t50\t.\t+\t.\tID=t2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ref_gff"):
        _ = count_reference_transcripts(ref_gff)


def test_evaluate_score_file_uses_internal_counts_and_ref_gff(
    tmp_path: Path,
) -> None:
    class_file = tmp_path / "transcript_class.txt"
    class_file.write_text(
        "t1 =\n"
        "t2 c\n"
        "t3 j\n"
        "t4 =\n",
        encoding="utf-8",
    )

    score_file = tmp_path / "scores.tsv"
    score_file.write_text(
        "tid c1 c2 c3 score marker\n"
        "t1 x x x 1.0 0\n"
        "t2 x x x 0.5 0\n"
        "t3 x x x 2.0 0\n"
        "t4 x x x 3.0 10000\n",
        encoding="utf-8",
    )

    ref_gff = tmp_path / "ref.gff"
    ref_gff.write_text(
        "chr\tsrc\texon\t1\t10\t.\t+\t.\tParent=t1\n"
        "chr\tsrc\texon\t11\t20\t.\t+\t.\tParent=t1\n",
        encoding="utf-8",
    )

    lines = evaluate_score_file(
        class_file=class_file,
        score_file=score_file,
        ref_gff=ref_gff,
    )

    assert len(lines) == 1
    fields = lines[0].split()
    assert fields[0] == "t1"
    assert fields[2] == "="
    assert float(fields[3]) == pytest.approx(0.0)
    assert float(fields[4]) == pytest.approx(0.0)
    assert float(fields[5]) == pytest.approx(0.0)


def test_hsap_cnn_pair_v2_eval_matches_current_reference_files() -> None:
    """Verify Hsap cnn_pair_v2 evaluation uses the current class file."""

    repo_root = Path(__file__).resolve().parents[1]
    class_file = repo_root / "data" / "Hsap" / "raw" / "transcript_class.txt"
    score_file = repo_root / "data" / "Hsap" / "trans_score" / "cnn_pair_v2.tsv"
    ref_gff = (
        repo_root
        / "data"
        / "Hsap"
        / "raw"
        / "GCF_000001405.40_GRCh38.p14_genomic.noalpsnopatches.filtered.gff"
    )
    eval_file = repo_root / "data" / "Hsap" / "eval_score" / "cnn_pair_v2.txt"

    if not class_file.is_file() or not score_file.is_file() or not ref_gff.is_file():
        pytest.skip("Hsap reference evaluation files are required for this test.")
    if not eval_file.is_file():
        pytest.skip(f"Missing evaluation output: {eval_file}")

    expected_lines = evaluate_score_file(
        class_file=class_file,
        score_file=score_file,
        ref_gff=ref_gff,
    )
    observed_lines = eval_file.read_text(encoding="utf-8").splitlines()

    assert observed_lines == expected_lines
