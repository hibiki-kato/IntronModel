"""Tests for src/util/make_transcript_class_from_tmap.py."""

from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from src.util.make_transcript_class_from_tmap import (
    parse_tmap,
    parse_tracking,
    write_transcript_class,
)

_GFFCOMPARE_AVAILABLE = shutil.which("gffcompare") is not None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_class_set(path: Path) -> set[str]:
    """Return the set of non-empty lines from a transcript_class.txt file."""
    return {line for line in path.read_text(encoding="utf-8").splitlines() if line}


def _write_tmap(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content), encoding="utf-8")


def _write_tracking(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content), encoding="utf-8")


def test_parse_tmap_returns_expected_records(tmp_path: Path) -> None:
    tmap = tmp_path / "test.tmap"
    _write_tmap(
        tmap,
        """\
        ref_gene_id\tref_id\tclass_code\tqry_gene_id\tqry_id\tnum_exons\tFPKM\tTPM\tcov\tlen\tmajor_iso_id\tref_match_len
        GeneA\trna-NM_001.1\t=\tXLOC_000001\tMSTRG_00000001:2:3.45\t3\t0.0\t0.0\t1.0\t500\tMSTRG_00000001:2:3.45\t500
        GeneB\trna-NM_002.1\tj\tXLOC_000002\tMSTRG_00000002:1:1.20\t2\t0.0\t0.0\t1.0\t300\tMSTRG_00000002:1:1.20\t-
        GeneC\t-\tu\tXLOC_000003\tMSTRG_00000003:3:0.80\t4\t0.0\t0.0\t1.0\t400\t-\t-
        """,
    )

    records = parse_tmap(tmap)

    assert records == [
        ("MSTRG_00000001:2:3.45", "="),
        ("MSTRG_00000002:1:1.20", "j"),
        ("MSTRG_00000003:3:0.80", "u"),
    ]


def test_parse_tmap_skips_empty_qry_id(tmp_path: Path) -> None:
    tmap = tmp_path / "test.tmap"
    _write_tmap(
        tmap,
        """\
        ref_gene_id\tref_id\tclass_code\tqry_gene_id\tqry_id\tnum_exons\tFPKM\tTPM\tcov\tlen\tmajor_iso_id\tref_match_len
        GeneA\trna-NM_001.1\t=\tXLOC_000001\tMSTRG_00000001:1:2.00\t2\t0.0\t0.0\t1.0\t200\tMSTRG_00000001:1:2.00\t200
        GeneB\t-\tj\tXLOC_000002\t\t1\t0.0\t0.0\t0.0\t0\t-\t-
        """,
    )

    records = parse_tmap(tmap)

    assert len(records) == 1
    assert records[0] == ("MSTRG_00000001:1:2.00", "=")


def test_parse_tmap_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        parse_tmap(tmp_path / "nonexistent.tmap")


def test_parse_tmap_raises_on_empty_file(tmp_path: Path) -> None:
    tmap = tmp_path / "empty.tmap"
    tmap.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        parse_tmap(tmap)


def test_parse_tmap_raises_on_wrong_header(tmp_path: Path) -> None:
    tmap = tmp_path / "bad_header.tmap"
    tmap.write_text("wrong_header\tcol2\tcol3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="header"):
        parse_tmap(tmap)


def test_parse_tmap_raises_on_too_few_columns(tmp_path: Path) -> None:
    tmap = tmp_path / "short.tmap"
    _write_tmap(
        tmap,
        """\
        ref_gene_id\tref_id\tclass_code\tqry_gene_id\tqry_id\tnum_exons
        GeneA\trna-NM_001.1\t=
        """,
    )

    with pytest.raises(ValueError, match="columns"):
        parse_tmap(tmap)


def test_write_transcript_class_creates_correct_file(tmp_path: Path) -> None:
    records = [
        ("MSTRG_00000001:2:3.45", "="),
        ("MSTRG_00000002:1:1.20", "j"),
        ("MSTRG_00000003:3:0.80", "u"),
    ]
    out = tmp_path / "transcript_class.txt"

    write_transcript_class(records, out)

    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines == [
        "MSTRG_00000001:2:3.45 =",
        "MSTRG_00000002:1:1.20 j",
        "MSTRG_00000003:3:0.80 u",
    ]


def test_write_transcript_class_creates_parent_dirs(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "dir" / "transcript_class.txt"
    records = [("MSTRG_00000001:1:1.00", "=")]

    write_transcript_class(records, out)

    assert out.is_file()


def test_write_transcript_class_raises_on_empty_records(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        write_transcript_class([], tmp_path / "transcript_class.txt")


def test_roundtrip_parse_and_write(tmp_path: Path) -> None:
    tmap = tmp_path / "test.tmap"
    _write_tmap(
        tmap,
        """\
        ref_gene_id\tref_id\tclass_code\tqry_gene_id\tqry_id\tnum_exons\tFPKM\tTPM\tcov\tlen\tmajor_iso_id\tref_match_len
        GeneA\trna-NM_001.1\t=\tXLOC_000001\tMSTRG_00000001:2:5.50\t3\t0.0\t0.0\t1.0\t500\tMSTRG_00000001:2:5.50\t500
        GeneB\trna-NM_002.1\tc\tXLOC_000002\tMSTRG_00000002:1:0.90\t2\t0.0\t0.0\t1.0\t200\tMSTRG_00000002:1:0.90\t100
        """,
    )
    out = tmp_path / "transcript_class.txt"

    records = parse_tmap(tmap)
    write_transcript_class(records, out)

    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines == [
        "MSTRG_00000001:2:5.50 =",
        "MSTRG_00000002:1:0.90 c",
    ]


# ---------------------------------------------------------------------------
# parse_tracking unit tests
# ---------------------------------------------------------------------------


def test_parse_tracking_returns_expected_records(tmp_path: Path) -> None:
    tracking = tmp_path / "test.tracking"
    _write_tracking(
        tracking,
        """\
        TCONS_00000001|3|500\tXLOC_000001\tGeneA|rna-NM_001.1\t=\tq1:XLOC_000001|MSTRG_00000001:2:3.45|3|0.0|0.0|1.0|500
        TCONS_00000002|2|300\tXLOC_000002\tGeneB|rna-NM_002.1\tj\tq1:XLOC_000002|MSTRG_00000002:1:1.20|2|0.0|0.0|1.0|300
        TCONS_00000003|4|400\tXLOC_000003\t-\tu\tq1:XLOC_000003|MSTRG_00000003:3:0.80|4|0.0|0.0|1.0|400
        """,
    )

    records = parse_tracking(tracking)

    assert records == [
        ("MSTRG_00000001:2:3.45", "="),
        ("MSTRG_00000002:1:1.20", "j"),
        ("MSTRG_00000003:3:0.80", "u"),
    ]


def test_parse_tracking_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        parse_tracking(tmp_path / "nonexistent.tracking")


def test_parse_tracking_raises_on_empty_file(tmp_path: Path) -> None:
    tracking = tmp_path / "empty.tracking"
    tracking.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        parse_tracking(tracking)


def test_parse_tracking_raises_on_too_few_columns(tmp_path: Path) -> None:
    tracking = tmp_path / "short.tracking"
    _write_tracking(
        tracking,
        """\
        TCONS_00000001|3|500\tXLOC_000001\tGeneA|rna-NM_001.1
        """,
    )

    with pytest.raises(ValueError, match="columns"):
        parse_tracking(tracking)


def test_parse_tracking_novel_transcript(tmp_path: Path) -> None:
    """Novel transcripts (class_code ``u``) with ref ``-`` are parsed correctly."""
    tracking = tmp_path / "novel.tracking"
    _write_tracking(
        tracking,
        """\
        TCONS_00000074|3|965\tXLOC_000008\t-\tu\tq1:XLOC_000008|MSTRG_00000286:1:0.82|3|0.0|0.0|1.0|965
        """,
    )

    records = parse_tracking(tracking)

    assert records == [("MSTRG_00000286:1:0.82", "u")]


# ---------------------------------------------------------------------------
# Integration tests: require gffcompare on PATH and existing species data
# ---------------------------------------------------------------------------


def _run_gffcompare(
    ref_annotation: Path,
    query_gtf: Path,
    out_prefix: Path,
) -> Path:
    """Run gffcompare and return the path to the generated ``.tracking`` file.

    Parameters
    ----------
    ref_annotation : Path
        Reference GFF/GFF3 annotation.
    query_gtf : Path
        Query GTF (e.g. StringTie output).
    out_prefix : Path
        Prefix for gffcompare output files.

    Returns
    -------
    Path
        Path to the ``.tracking`` file produced by gffcompare.

    Raises
    ------
    FileNotFoundError
        If the expected ``.tracking`` file is not created by gffcompare.
    """
    result = subprocess.run(
        [
            "gffcompare",
            "-r",
            str(ref_annotation),
            str(query_gtf),
            "-o",
            str(out_prefix),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"gffcompare failed (exit {result.returncode}):\n{result.stderr}"
        )
    tracking_path = out_prefix.parent / f"{out_prefix.name}.tracking"
    if not tracking_path.is_file():
        raise FileNotFoundError(
            f"Expected tracking not found after gffcompare: {tracking_path}"
        )
    return tracking_path


def _find_fix_gff(raw_dir: Path) -> Path | None:
    candidates = sorted(raw_dir.glob("*.fix.gff")) + sorted(raw_dir.glob("*.gff.fix"))
    if candidates:
        return candidates[0]
    candidates = sorted(raw_dir.glob("*.gff"))
    return candidates[0] if candidates else None


def _find_fna_gtf(raw_dir: Path) -> Path | None:
    candidates = sorted(raw_dir.glob("*.fna.gtf"))
    if candidates:
        return candidates[0]
    candidates = sorted(raw_dir.glob("*.gtf"))
    return candidates[0] if candidates else None


@pytest.mark.skipif(
    not _GFFCOMPARE_AVAILABLE,
    reason="gffcompare not found on PATH",
)
@pytest.mark.parametrize("species", ["Dmel", "Mmus", "Athal"])
def test_gffcompare_output_matches_existing_transcript_class(
    tmp_path: Path,
    species: str,
) -> None:
    """Re-run gffcompare on existing data and verify the parsed output matches
    the committed ``transcript_class.txt`` for that species.

    The existing file is never modified; all gffcompare output goes to
    ``tmp_path``.

    Parameters
    ----------
    tmp_path : Path
        Pytest-provided temporary directory.
    species : str
        Species folder name under ``data/``.
    """
    root = _project_root()
    raw_dir = root / "data" / species / "raw"
    existing_class_file = raw_dir / "transcript_class.txt"

    if not existing_class_file.is_file():
        pytest.skip(f"transcript_class.txt not found for {species}")

    ref_annotation = _find_fix_gff(raw_dir)
    query_gtf = _find_fna_gtf(raw_dir)

    if ref_annotation is None:
        pytest.skip(f"No reference GFF found for {species}")
    if query_gtf is None:
        pytest.skip(f"No query GTF found for {species}")

    tracking_path = _run_gffcompare(
        ref_annotation=ref_annotation,
        query_gtf=query_gtf,
        out_prefix=tmp_path / "gffcmp",
    )

    records = parse_tracking(tracking_path)
    out_path = tmp_path / "transcript_class.txt"
    write_transcript_class(records, out_path)

    generated = _read_class_set(out_path)
    existing = _read_class_set(existing_class_file)

    assert generated == existing, (
        f"[{species}] transcript_class.txt mismatch: "
        f"{len(generated - existing)} lines only in generated, "
        f"{len(existing - generated)} lines only in existing."
    )
