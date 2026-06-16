from __future__ import annotations

from pathlib import Path
import csv
import sys

import matplotlib
import pytest

matplotlib.use("Agg")

ANALYSIS_SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "analysis" / "script"
if str(ANALYSIS_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SCRIPT_ROOT))

from raw.inspect_intron_overlap_pairs import (  # noqa: E402
    IntronInterval,
    count_overlapping_pairs,
    main,
    plot_intron_overlap_pairs,
    resolve_species_with_catalog,
    summarize_species_overlap,
)


def _write_catalog(path: Path, rows: list[tuple[str, int, str, int, int]]) -> None:
    """Write one minimal intron catalog TSV for tests."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "unique_transcript_id\tunique_intron_index\tchrom\tstrand\t"
        "intron_start\tintron_end\tmember_count\tseen_train_pos_coord\t"
        "seen_train_neg_seq\ttrain_leak",
    ]
    for unique_transcript_id, unique_intron_index, chrom, start, end in rows:
        lines.append(
            f"{unique_transcript_id}\t{unique_intron_index}\t{chrom}\t+\t"
            f"{start}\t{end}\t1\t0\t0\t0"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_count_overlapping_pairs_deduplicates_and_groups_by_chromosome() -> None:
    """Distinct overlaps are counted once, duplicate coordinates are ignored."""

    intervals = [
        IntronInterval(chrom="chr1", start=1, end=5),
        IntronInterval(chrom="chr1", start=4, end=6),
        IntronInterval(chrom="chr1", start=10, end=12),
        IntronInterval(chrom="chr1", start=12, end=15),
        IntronInterval(chrom="chr2", start=1, end=100),
        IntronInterval(chrom="chr2", start=1, end=100),
    ]

    assert count_overlapping_pairs(intervals) == 2


def test_resolve_species_with_catalog_and_plot_intron_overlap_pairs(
    tmp_path: Path,
) -> None:
    """Resolve catalog species and render the overlap figure."""

    data_root = tmp_path / "data"
    _write_catalog(
        data_root / "SpeciesA" / "processed" / "intron_unique_catalog.tsv",
        [
            ("u1", 1, "chr1", 1, 5),
            ("u2", 1, "chr1", 4, 6),
        ],
    )
    _write_catalog(
        data_root / "SpeciesB" / "processed" / "intron_unique_catalog.tsv",
        [
            ("u1", 1, "chr2", 10, 15),
        ],
    )

    assert resolve_species_with_catalog(repo_root=tmp_path) == [
        "SpeciesA",
        "SpeciesB",
    ]
    assert resolve_species_with_catalog(
        repo_root=tmp_path,
        species="SpeciesB, SpeciesA",
    ) == ["SpeciesB", "SpeciesA"]

    figure, summaries = plot_intron_overlap_pairs(repo_root=tmp_path)

    assert [summary.species for summary in summaries] == ["SpeciesA", "SpeciesB"]
    assert [summary.overlap_pair_count for summary in summaries] == [1, 0]
    assert len(figure.axes) == 1
    figure.clf()


def test_main_writes_summary_and_plot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI output should summarize each species and render a bar plot."""

    data_root = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    _write_catalog(
        data_root / "SpeciesA" / "processed" / "intron_unique_catalog.tsv",
        [
            ("u1", 1, "chr1", 1, 5),
            ("u2", 1, "chr1", 4, 6),
            ("u3", 1, "chr1", 10, 12),
            ("u4", 1, "chr1", 12, 15),
        ],
    )
    _write_catalog(
        data_root / "SpeciesB" / "processed" / "intron_unique_catalog.tsv",
        [
            ("u1", 1, "chr1", 20, 25),
            ("u2", 1, "chr2", 30, 35),
        ],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inspect_intron_overlap_pairs.py",
            "--data-root",
            str(data_root),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert main() == 0

    summary_path = output_dir / "tables" / "intron_overlap_pairs_summary.tsv"
    output_svg = output_dir / "intron_overlap_pairs.svg"
    assert summary_path.is_file()
    assert output_svg.is_file()
    assert output_svg.stat().st_size > 0
    assert "<svg" in output_svg.read_text(encoding="utf-8")

    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    assert rows == [
        {
            "species": "SpeciesA",
            "unique_intron_count": "4",
            "possible_pair_count": "6",
            "overlap_pair_count": "2",
            "overlap_pair_rate": "0.3333333333333333",
        },
        {
            "species": "SpeciesB",
            "unique_intron_count": "2",
            "possible_pair_count": "1",
            "overlap_pair_count": "0",
            "overlap_pair_rate": "0.0",
        },
    ]

    summary = summarize_species_overlap(
        data_root=data_root,
        species="SpeciesA",
    )
    assert summary.overlap_pair_count == 2
    assert summary.overlap_pair_rate == pytest.approx(1 / 3)
