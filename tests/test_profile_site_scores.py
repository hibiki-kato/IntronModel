from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

ANALYSIS_SCRIPT = Path(__file__).resolve().parents[1] / "analysis" / "script"
if str(ANALYSIS_SCRIPT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SCRIPT))

from score.profile_site_scores import build_donor_acceptor_figure  # noqa: E402


def _write_site_score_tsv(base_dir: Path) -> None:
    score_dir = base_dir / "data" / "Athal" / "site_score"
    score_dir.mkdir(parents=True, exist_ok=True)
    (score_dir / "cnn.tsv").write_text(
        "\n".join(
            [
                "transcript_id\tintron_index\tdonor_score\tacceptor_score\tlabel",
                "tx1\t1\t0.10\t0.20\t0",
                "tx2\t2\t0.20\t0.80\t1",
                "tx3\t3\t0.90\t0.30\t0",
                "tx4\t4\t0.70\t0.90\t1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_build_donor_acceptor_figure_adds_heatmap_axes_by_default(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    _write_site_score_tsv(tmp_path)

    figure, summaries = build_donor_acceptor_figure(
        repo_root=tmp_path,
        selected_pairs=[("Athal", "cnn")],
        split_by_label=True,
    )

    assert len(summaries) == 2
    assert summaries[0].plotted_label == 0
    assert summaries[1].plotted_label == 1
    assert len(figure.axes) == 6
    assert figure.axes[1].get_title() == "2D histogram"
    assert figure.axes[3].get_title() == "2D histogram"

    plt.close(figure)


def test_build_donor_acceptor_figure_rejects_non_positive_histogram_bins(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    _write_site_score_tsv(tmp_path)

    with pytest.raises(ValueError, match="histogram_bins"):
        build_donor_acceptor_figure(
            repo_root=tmp_path,
            selected_pairs=[("Athal", "cnn")],
            histogram_bins=0,
        )
