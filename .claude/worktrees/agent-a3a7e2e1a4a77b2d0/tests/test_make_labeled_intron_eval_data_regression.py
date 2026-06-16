from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest


def _set_csv_field_limit_max() -> None:
    """Set CSV field size limit to the largest supported value."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _parse_attributes(attr_text: str) -> dict[str, str]:
    """Parse GTF/GFF attributes into a key-value map."""
    attributes: dict[str, str] = {}
    for raw_token in attr_text.strip().strip(";").split(";"):
        token = raw_token.strip()
        if token == "":
            continue
        if "=" in token:
            key, value = token.split("=", 1)
        elif " " in token:
            key, value = token.split(" ", 1)
            value = value.strip().strip('"')
        else:
            continue
        key = key.strip()
        value = value.strip().strip('"')
        if key != "" and key not in attributes:
            attributes[key] = value
    return attributes


def _resolve_transcript_id(attributes: dict[str, str]) -> str | None:
    """Resolve transcript identifier from parsed attributes."""
    for key in ("transcript_id", "transcriptId"):
        value = attributes.get(key, "").strip()
        if value != "":
            return value
    parent = attributes.get("Parent", "").strip()
    if parent != "":
        return parent.split(",", 1)[0].strip()
    return None


def _build_reference_intron_set_from_gff(
    reference_gff: Path,
) -> set[tuple[str, str, int, int]]:
    """Build reference intron keys from exon rows in a GFF/GTF file.

    Returns
    -------
    set[tuple[str, str, int, int]]
        Keys are ``(chrom, strand, intron_start, intron_end)``.
    """
    exons_by_key: dict[tuple[str, str, str], list[tuple[int, int]]] = {}

    with reference_gff.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("#") or line.strip() == "":
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            chrom, _, feature, start_text, end_text, _, strand, _, attrs = fields
            if feature != "exon" or strand not in {"+", "-"}:
                continue
            attributes = _parse_attributes(attrs)
            transcript_id = _resolve_transcript_id(attributes)
            if transcript_id is None:
                continue
            start = int(start_text)
            end = int(end_text)
            key = (transcript_id, chrom, strand)
            exons_by_key.setdefault(key, []).append((start, end))

    introns: set[tuple[str, str, int, int]] = set()
    for transcript_id, chrom, strand in exons_by_key.keys():
        exons = exons_by_key[(transcript_id, chrom, strand)]
        ordered_exons = sorted(
            exons,
            key=lambda exon: (exon[0], exon[1]),
            reverse=(strand == "-"),
        )
        for index in range(len(ordered_exons) - 1):
            upstream = ordered_exons[index]
            downstream = ordered_exons[index + 1]
            if strand == "+":
                intron_start = upstream[1] + 1
                intron_end = downstream[0] - 1
            else:
                intron_start = downstream[1] + 1
                intron_end = upstream[0] - 1
            if intron_end >= intron_start:
                introns.add((chrom, strand, intron_start, intron_end))

    return introns


def _resolve_reference_annotation(raw_dir: Path) -> Path | None:
    """Resolve one reference annotation by runtime priority."""
    for pattern in ("*.fix.gff", "*.gff.fix", "*.gff", "*.gff3"):
        for candidate in sorted(raw_dir.glob(pattern)):
            if candidate.is_file():
                return candidate
    return None


@pytest.mark.parametrize("species", ["Dmel", "Mmus", "Athal", "Hsap"])
def test_processed_labels_match_reference_introns(species: str) -> None:
    """Verify processed labels match reference intron membership exactly."""
    repo_root = Path(__file__).resolve().parents[1]
    species_dir = repo_root / "data" / species
    processed_tsv = species_dir / "processed" / "intron_eval_flank10.tsv"
    raw_dir = species_dir / "raw"
    reference_gff = _resolve_reference_annotation(raw_dir)

    if not processed_tsv.is_file() or reference_gff is None:
        pytest.skip(
            f"{species} processed/reference files are required for regression test."
        )

    _set_csv_field_limit_max()
    reference_introns = _build_reference_intron_set_from_gff(reference_gff)

    mismatch_count = 0
    total_rows = 0
    with processed_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"chrom", "strand", "intron_start", "intron_end", "label"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"Missing required columns in {processed_tsv}")

        for row in reader:
            total_rows += 1
            key = (
                str(row["chrom"]).strip(),
                str(row["strand"]).strip(),
                int(str(row["intron_start"]).strip()),
                int(str(row["intron_end"]).strip()),
            )
            expected_label = 1 if key in reference_introns else 0
            observed_label = int(str(row["label"]).strip())
            if observed_label != expected_label:
                mismatch_count += 1

    assert total_rows > 0
    assert mismatch_count == 0
