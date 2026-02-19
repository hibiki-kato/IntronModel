from __future__ import annotations

from pathlib import Path

import pytest

from util.gffcompare_util import (
    compute_good_total,
    parse_ref_count_from_stats,
    parse_tmap_classifications,
)


def test_parse_tmap_classifications_header_commented(tmp_path: Path) -> None:
    tmap = tmp_path / "x.tmap"
    tmap.write_text(
        "#ref_id\tclass_code\tqry_id\tref_gene_id\tqry_gene_id\n"
        "R1\t=\tQ1\tG1\tGq1\n"
        "R2\tc\tQ2\tG2\tGq2\n",
        encoding="utf-8",
    )
    mapping = parse_tmap_classifications(tmap)
    assert mapping == {"Q1": "=", "Q2": "c"}


def test_compute_good_total_excludes_c_by_default() -> None:
    mapping = {"Q1": "=", "Q2": "c", "Q3": "j"}
    good, total = compute_good_total(mapping)
    assert good == 1
    assert total == 2


def test_parse_ref_count_from_stats_reference_mrnas(tmp_path: Path) -> None:
    stats = tmp_path / "x.stats"
    stats.write_text(
        "# gffcompare output\nReference mRNAs : 32288\nQuery mRNAs : 38235\n",
        encoding="utf-8",
    )
    assert parse_ref_count_from_stats(stats) == 32288


def test_parse_ref_count_from_stats_missing_raises(tmp_path: Path) -> None:
    stats = tmp_path / "x.stats"
    stats.write_text("Query mRNAs : 10\n", encoding="utf-8")
    with pytest.raises(ValueError):
        _ = parse_ref_count_from_stats(stats)
