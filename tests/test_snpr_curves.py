from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

ANALYSIS_SCRIPT = Path(__file__).resolve().parents[1] / "analysis" / "script"
if str(ANALYSIS_SCRIPT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SCRIPT))

from score.snpr_curves import (  # noqa: E402
    build_transcript_score_snpr_figure,
    resolve_transcript_class_file,
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
