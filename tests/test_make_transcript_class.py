"""Tests for annotated GTF transcript class extraction."""

from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from src.util.make_transcript_class_from_annotated_gtf import (
    parse_annotated_gtf,
    write_transcript_class,
)

_GFFCOMPARE_AVAILABLE = shutil.which("gffcompare") is not None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_gtf(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content), encoding="utf-8")


def test_parse_annotated_gtf_returns_expected_records(tmp_path: Path) -> None:
    gtf = tmp_path / "sample.annotated.gtf"
    _write_gtf(
        gtf,
        """\
        chr1\tgffcompare\ttranscript\t100\t200\t.\t+\t.\tgene_id "g1"; transcript_id "tx1"; class_code "=";
        chr1\tgffcompare\texon\t100\t150\t.\t+\t.\tgene_id "g1"; transcript_id "tx1"; exon_number "1";
        chr1\tgffcompare\ttranscript\t300\t400\t.\t+\t.\tgene_id "g2"; transcript_id "tx2"; class_code "j";
        """,
    )

    records = parse_annotated_gtf(gtf)

    assert records == [("tx1", "="), ("tx2", "j")]


def test_parse_annotated_gtf_skips_rows_without_required_attrs(tmp_path: Path) -> None:
    gtf = tmp_path / "missing_attrs.annotated.gtf"
    _write_gtf(
        gtf,
        """\
        chr1\tgffcompare\ttranscript\t100\t200\t.\t+\t.\tgene_id "g1"; transcript_id "tx1";
        chr1\tgffcompare\ttranscript\t300\t400\t.\t+\t.\tgene_id "g2"; class_code "u";
        chr1\tgffcompare\ttranscript\t500\t600\t.\t+\t.\tgene_id "g3"; transcript_id "tx3"; class_code "c";
        """,
    )

    records = parse_annotated_gtf(gtf)

    assert records == [("tx3", "c")]


def test_parse_annotated_gtf_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        parse_annotated_gtf(tmp_path / "not_found.annotated.gtf")


def test_parse_annotated_gtf_raises_on_malformed_row(tmp_path: Path) -> None:
    gtf = tmp_path / "bad.annotated.gtf"
    _write_gtf(gtf, "chr1\tgffcompare\ttranscript\n")

    with pytest.raises(ValueError, match="columns"):
        parse_annotated_gtf(gtf)


def test_write_transcript_class_writes_expected_lines(tmp_path: Path) -> None:
    out_path = tmp_path / "transcript_class.txt"

    write_transcript_class([("tx1", "="), ("tx2", "u")], out_path)

    assert out_path.read_text(encoding="utf-8").splitlines() == ["tx1 =", "tx2 u"]


def test_write_transcript_class_raises_on_empty_records(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        write_transcript_class([], tmp_path / "transcript_class.txt")


def _find_reference_annotation(raw_dir: Path) -> Path | None:
    preferred = sorted(raw_dir.glob("*.fix.gff")) + sorted(raw_dir.glob("*.gff.fix"))
    if preferred:
        return preferred[0]

    generic = sorted(raw_dir.glob("*.gff")) + sorted(raw_dir.glob("*.gff3"))
    if generic:
        return generic[0]
    return None


def _find_query_gtf(raw_dir: Path) -> Path | None:
    fasta_candidates = sorted(raw_dir.glob("*.clean.fna")) + sorted(
        raw_dir.glob("*.fna")
    )
    for fasta in fasta_candidates:
        direct_gtf = Path(f"{fasta}.gtf")
        if direct_gtf.is_file():
            return direct_gtf

    fna_gtf = sorted(raw_dir.glob("*.fna.gtf"))
    if fna_gtf:
        return fna_gtf[0]

    gtf = sorted(raw_dir.glob("*.gtf"))
    if gtf:
        return gtf[0]
    return None


def _run_gffcompare(ref_annotation: Path, query_gtf: Path, out_prefix: Path) -> Path:
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
            "gffcompare failed "
            f"(exit {result.returncode}):\n{result.stdout}\n{result.stderr}"
        )

    annotated = out_prefix.parent / f"{out_prefix.name}.annotated.gtf"
    if not annotated.is_file():
        raise FileNotFoundError(
            f"Expected annotated gtf not found after gffcompare: {annotated}"
        )
    return annotated


@pytest.mark.skipif(not _GFFCOMPARE_AVAILABLE, reason="gffcompare not found on PATH")
@pytest.mark.parametrize("species", ["Dmel", "Mmus", "Athal", "Hsap"])
def test_gffcompare_annotated_matches_existing_transcript_class(
    tmp_path: Path,
    species: str,
) -> None:
    """Generated transcript_class must match committed transcript_class.txt."""
    root = _project_root()
    raw_dir = root / "data" / species / "raw"
    existing_class = raw_dir / "transcript_class.txt"

    if not existing_class.is_file():
        pytest.skip(f"transcript_class.txt not found for {species}")

    ref_annotation = _find_reference_annotation(raw_dir)
    if ref_annotation is None:
        pytest.skip(f"reference annotation not found for {species}")

    query_gtf = _find_query_gtf(raw_dir)
    if query_gtf is None:
        pytest.skip(f"query gtf not found for {species}")

    annotated_gtf = _run_gffcompare(
        ref_annotation=ref_annotation,
        query_gtf=query_gtf,
        out_prefix=tmp_path / "gffcompare_tmp",
    )

    records = parse_annotated_gtf(annotated_gtf)
    generated_class = tmp_path / "transcript_class.generated.txt"
    write_transcript_class(records, generated_class)

    generated_lines = generated_class.read_text(encoding="utf-8").splitlines()
    existing_lines = existing_class.read_text(encoding="utf-8").splitlines()

    assert generated_lines == existing_lines
