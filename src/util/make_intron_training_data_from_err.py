from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_POSITIVE_RECORD_PATTERN = re.compile(
    r"^DEBUG donor ([ACGTNacgtn]+) acceptor ([ACGTNacgtn]+) ([+-]) (\S+) (-?\d+)$"
)
_NEGATIVE_PAIR_PATTERN = re.compile(
    r"^DEBUG pair ([ACGTNacgtn]+) ([ACGTNacgtn]+) ([+-]) (-?\d+)$"
)
_GTF_TRANSCRIPT_ID_PATTERN = re.compile(r'transcript_id "([^"]+)"')


@dataclass(frozen=True)
class Exon:
    """One exon in genomic coordinates.

    Attributes
    ----------
    chrom : str
        Contig name.
    start : int
        1-based inclusive genomic start.
    end : int
        1-based inclusive genomic end.
    """

    chrom: str
    start: int
    end: int


@dataclass(frozen=True)
class TranscriptModel:
    """Annotation model for one transcript.

    Attributes
    ----------
    transcript_id : str
        Transcript identifier from GTF.
    chrom : str
        Contig name.
    strand : str
        ``+`` or ``-``.
    exons : tuple[Exon, ...]
        Exons in transcript order (5' -> 3').
    """

    transcript_id: str
    chrom: str
    strand: str
    exons: tuple[Exon, ...]


@dataclass(frozen=True)
class IntronCandidate:
    """Intron candidate recovered from transcript exon structure.

    Attributes
    ----------
    transcript_id : str
        Transcript identifier.
    intron_index : int
        1-based intron index in transcript order.
    chrom : str
        Contig name.
    strand : str
        ``+`` or ``-``.
    intron_start : int
        1-based inclusive genomic start (lower coordinate).
    intron_end : int
        1-based inclusive genomic end (higher coordinate).
    intron_length : int
        Intron length in base pairs.
    donor_seq_100bp : str
        Donor-side 100 bp sequence in transcript orientation.
    acceptor_seq_100bp : str
        Acceptor-side 100 bp sequence in transcript orientation.
    """

    transcript_id: str
    intron_index: int
    chrom: str
    strand: str
    intron_start: int
    intron_end: int
    intron_length: int
    donor_seq_100bp: str
    acceptor_seq_100bp: str


@dataclass(frozen=True)
class PositiveErrRecord:
    """One strict-format record from ``100bp.err``.

    Attributes
    ----------
    source_line_no : int
        1-based source line number.
    donor_seq_100bp : str
        Donor-side 100 bp sequence.
    acceptor_seq_100bp : str
        Acceptor-side 100 bp sequence.
    strand : str
        ``+`` or ``-``.
    transcript_id : str
        Transcript identifier.
    intron_half_length : int
        Integer value from the last column of ``100bp.err``.
    """

    source_line_no: int
    donor_seq_100bp: str
    acceptor_seq_100bp: str
    strand: str
    transcript_id: str
    intron_half_length: int


@dataclass(frozen=True)
class PositiveOutputRow:
    """One row for ``intron_full_flank10.pos.tsv`` output."""

    species: str
    label: int
    transcript_id: str
    intron_index: int
    chrom: str
    strand: str
    intron_start: int
    intron_end: int
    intron_length: int
    intron_half_length: int
    flank_bp: int
    donor_seq_100bp: str
    acceptor_seq_100bp: str
    sequence: str
    source_line_no: int


@dataclass(frozen=True)
class NegativeRequestRow:
    """One row for negative coordinate request output."""

    species: str
    source_line_no: int
    donor_seq_100bp: str
    acceptor_seq_100bp: str
    strand: str
    intron_half_length: int


@dataclass(frozen=True)
class SpeciesStats:
    """Per-species quality-control summary."""

    species: str
    total_input_rows: int
    matched_rows: int
    unmatched_rows: int
    ambiguous_rows: int
    half_length_mismatch_rows: int
    min_seq_len: int
    median_seq_len: int
    p95_seq_len: int
    p99_seq_len: int
    max_seq_len: int


@dataclass(frozen=True)
class FastaContigIndex:
    """Random-access index metadata for one FASTA contig.

    Attributes
    ----------
    seq_offset : int
        Byte offset of the first sequence base.
    seq_length : int
        Sequence length in bases.
    line_bases : int
        Bases per wrapped FASTA line for this contig.
    line_bytes : int
        Bytes per wrapped FASTA line (including newline bytes).
    """

    seq_offset: int
    seq_length: int
    line_bases: int
    line_bytes: int


class FastaIndexedReader:
    """FASTA random-access reader without external dependencies.

    Notes
    -----
    The index builder is linear in FASTA file size: ``O(F)``.
    Interval fetch uses one seek and one read, and is linear in requested
    interval span: ``O(L)``.
    """

    def __init__(self, fasta_path: Path) -> None:
        """Initialize reader and build contig index.

        Parameters
        ----------
        fasta_path : Path
            FASTA file path.

        Raises
        ------
        FileNotFoundError
            If FASTA file does not exist.
        ValueError
            If FASTA structure is malformed.
        """
        if not fasta_path.exists():
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
        self._fasta_path = fasta_path
        self._index = self._build_index(fasta_path)
        self._handle = fasta_path.open("rb")

    def close(self) -> None:
        """Close underlying file handle."""
        self._handle.close()

    def __enter__(self) -> FastaIndexedReader:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Exit context manager and close file handle."""
        _ = exc_type
        _ = exc
        _ = tb
        self.close()

    def has_contig(self, contig: str) -> bool:
        """Return whether the contig exists in FASTA index."""
        return contig in self._index

    def contig_length(self, contig: str) -> int:
        """Return contig length in bases.

        Parameters
        ----------
        contig : str
            Contig name.

        Returns
        -------
        int
            Contig length.

        Raises
        ------
        KeyError
            If contig is missing.
        """
        return self._index[contig].seq_length

    def fetch_interval(self, contig: str, start: int, end: int) -> str:
        """Fetch one inclusive genomic interval.

        Parameters
        ----------
        contig : str
            Contig name.
        start : int
            1-based inclusive start.
        end : int
            1-based inclusive end.

        Returns
        -------
        str
            Uppercase DNA sequence.

        Raises
        ------
        KeyError
            If contig does not exist.
        ValueError
            If coordinates are out of bounds.
        """
        meta = self._index[contig]
        if start < 1 or end < start or end > meta.seq_length:
            raise ValueError(
                f"Invalid interval on {contig}: start={start}, end={end}, "
                f"length={meta.seq_length}"
            )

        byte_start = self._byte_offset(meta=meta, one_based_pos=start)
        byte_end = self._byte_offset(meta=meta, one_based_pos=end)
        span = byte_end - byte_start + 1

        self._handle.seek(byte_start)
        raw = self._handle.read(span)
        clean = raw.replace(b"\n", b"").replace(b"\r", b"").upper()
        expected = end - start + 1
        if len(clean) != expected:
            raise ValueError(
                f"Failed interval read for {contig}:{start}-{end}; "
                f"expected={expected}, got={len(clean)}"
            )
        return clean.decode("ascii")

    @staticmethod
    def _build_index(fasta_path: Path) -> dict[str, FastaContigIndex]:
        """Build byte-level FASTA index.

        Parameters
        ----------
        fasta_path : Path
            FASTA file path.

        Returns
        -------
        dict[str, FastaContigIndex]
            Per-contig index metadata.

        Raises
        ------
        ValueError
            If FASTA headers are duplicated or sequence lines are malformed.
        """
        index: dict[str, FastaContigIndex] = {}

        with fasta_path.open("rb") as handle:
            current_name: str | None = None
            seq_offset = 0
            seq_length = 0
            line_bases = 0
            line_bytes = 0

            while True:
                line_start = handle.tell()
                line = handle.readline()
                if not line:
                    break

                if line.startswith(b">"):
                    if current_name is not None:
                        index[current_name] = FastaContigIndex(
                            seq_offset=seq_offset,
                            seq_length=seq_length,
                            line_bases=line_bases,
                            line_bytes=line_bytes,
                        )
                    current_name = (
                        line[1:].strip().split(maxsplit=1)[0].decode("ascii")
                    )
                    if current_name in index:
                        raise ValueError(
                            f"Duplicate FASTA contig name: {current_name}"
                        )
                    seq_offset = handle.tell()
                    seq_length = 0
                    line_bases = 0
                    line_bytes = 0
                    continue

                if current_name is None:
                    raise ValueError("FASTA sequence line appeared before header")

                stripped = line.rstrip(b"\r\n")
                if len(stripped) == 0:
                    continue
                if line_bases == 0:
                    line_bases = len(stripped)
                    line_bytes = len(line)
                seq_length += len(stripped)
                _ = line_start

            if current_name is not None:
                index[current_name] = FastaContigIndex(
                    seq_offset=seq_offset,
                    seq_length=seq_length,
                    line_bases=line_bases,
                    line_bytes=line_bytes,
                )

        if not index:
            raise ValueError(f"No contigs found in FASTA: {fasta_path}")
        return index

    @staticmethod
    def _byte_offset(meta: FastaContigIndex, one_based_pos: int) -> int:
        """Map one-based base position to byte offset in FASTA file."""
        zero_based = one_based_pos - 1
        full_lines = zero_based // meta.line_bases
        within_line = zero_based % meta.line_bases
        return meta.seq_offset + full_lines * meta.line_bytes + within_line


def reverse_complement(seq: str) -> str:
    """Return reverse complement of DNA sequence."""
    table = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(table)[::-1].upper()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list[str] | None, optional
        CLI token list. ``None`` uses ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate positive intron full-length training data from "
            "100bp.err and GTF/FASTA annotations."
        )
    )
    parser.add_argument(
        "--species",
        required=True,
        help="Comma-separated species names (e.g., Dmel,Mmus,Athal).",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Data root directory containing <species>/raw.",
    )
    parser.add_argument(
        "--flank-bp",
        type=int,
        default=10,
        help="Flank size in bp added to both intron ends.",
    )
    parser.add_argument(
        "--pos-input-name",
        default="100bp.err",
        help="Positive input filename under raw directory.",
    )
    parser.add_argument(
        "--out-pos-name",
        default="intron_full_flank10.pos.tsv",
        help="Positive output TSV filename under raw directory.",
    )
    parser.add_argument(
        "--out-qc-name",
        default="intron_full_flank10.pos.qc.tsv",
        help="QC summary TSV filename under raw directory.",
    )
    parser.add_argument(
        "--out-neg-request-name",
        default="intron_full_flank10.neg_coordinate_request.tsv",
        help="Negative coordinate-request TSV filename under raw directory.",
    )
    parser.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        default=True,
        help=(
            "Fail when unmatched/ambiguous/half-length mismatch records are "
            "observed."
        ),
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Allow processing even when mismatches are found.",
    )
    args = parser.parse_args(argv)

    if args.flank_bp <= 0:
        raise ValueError("--flank-bp must be > 0")

    return args


def _species_tokens(species_arg: str) -> list[str]:
    """Split comma-separated species argument into non-empty names."""
    tokens = [item.strip() for item in species_arg.split(",")]
    return [item for item in tokens if item]


def _find_species_reference_files(raw_dir: Path) -> tuple[Path, Path]:
    """Resolve FASTA and corresponding ``.fna.gtf`` paths.

    Parameters
    ----------
    raw_dir : Path
        Species raw directory.

    Returns
    -------
    tuple[Path, Path]
        FASTA and GTF paths.

    Raises
    ------
    FileNotFoundError
        If required files are missing.
    ValueError
        If FASTA candidates are ambiguous.
    """
    fasta_candidates = sorted(raw_dir.glob("*.fna"))
    if not fasta_candidates:
        raise FileNotFoundError(f"No .fna file found under: {raw_dir}")
    if len(fasta_candidates) > 1:
        joined = ", ".join(str(path.name) for path in fasta_candidates)
        raise ValueError(
            f"Multiple .fna files found under {raw_dir}: {joined}. "
            "Please keep only one reference FASTA."
        )

    fasta_path = fasta_candidates[0]
    gtf_path = Path(f"{fasta_path}.gtf")
    if not gtf_path.exists():
        raise FileNotFoundError(
            f"Expected GTF not found for FASTA {fasta_path.name}: {gtf_path.name}"
        )
    return fasta_path, gtf_path


def _parse_gtf_transcripts(gtf_path: Path) -> dict[str, TranscriptModel]:
    """Parse exon-based transcript models from GTF.

    Parameters
    ----------
    gtf_path : Path
        GTF annotation path.

    Returns
    -------
    dict[str, TranscriptModel]
        Mapping from transcript ID to transcript model.

    Raises
    ------
    ValueError
        If transcript attributes are inconsistent.

    Notes
    -----
    Complexity is linear in number of GTF lines: ``O(N)``.
    """
    exons_by_tid: dict[str, list[Exon]] = {}
    chrom_by_tid: dict[str, str] = {}
    strand_by_tid: dict[str, str] = {}

    with gtf_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue

            chrom, _, feature, start_s, end_s, _, strand, _, attrs = parts
            if feature != "exon":
                continue
            if strand not in {"+", "-"}:
                continue

            match = _GTF_TRANSCRIPT_ID_PATTERN.search(attrs)
            if match is None:
                continue
            transcript_id = match.group(1)

            start = int(start_s)
            end = int(end_s)
            if start <= 0 or end < start:
                raise ValueError(
                    f"Invalid exon coordinates in {gtf_path}: "
                    f"{transcript_id} {start}-{end}"
                )

            if transcript_id in chrom_by_tid and chrom_by_tid[transcript_id] != chrom:
                raise ValueError(
                    f"Transcript {transcript_id} spans multiple contigs: "
                    f"{chrom_by_tid[transcript_id]} vs {chrom}"
                )
            if (
                transcript_id in strand_by_tid
                and strand_by_tid[transcript_id] != strand
            ):
                raise ValueError(
                    f"Transcript {transcript_id} has inconsistent strand: "
                    f"{strand_by_tid[transcript_id]} vs {strand}"
                )

            chrom_by_tid[transcript_id] = chrom
            strand_by_tid[transcript_id] = strand
            exons_by_tid.setdefault(transcript_id, []).append(
                Exon(chrom=chrom, start=start, end=end)
            )

    transcript_models: dict[str, TranscriptModel] = {}
    for transcript_id, exons in exons_by_tid.items():
        strand = strand_by_tid[transcript_id]
        ordered = sorted(
            exons,
            key=lambda exon: (exon.start, exon.end),
            reverse=(strand == "-"),
        )
        transcript_models[transcript_id] = TranscriptModel(
            transcript_id=transcript_id,
            chrom=chrom_by_tid[transcript_id],
            strand=strand,
            exons=tuple(ordered),
        )

    return transcript_models


def _donor_coords_plus(intron_start_tx: int) -> tuple[int, int]:
    """Return donor 100bp interval on ``+`` strand."""
    return intron_start_tx - 3, intron_start_tx + 96


def _acceptor_coords_plus(acceptor_boundary_tx: int) -> tuple[int, int]:
    """Return acceptor 100bp interval on ``+`` strand."""
    return acceptor_boundary_tx - 97, acceptor_boundary_tx + 2


def _coords_minus(boundary_tx: int, left_bp: int, right_bp: int) -> tuple[int, int]:
    """Return genomic interval for transcript-oriented window on ``-`` strand."""
    start = boundary_tx - (right_bp - 1)
    end = boundary_tx + left_bp
    return start, end


def _build_intron_candidate_index(
    transcripts: dict[str, TranscriptModel],
    fasta_reader: FastaIndexedReader,
) -> tuple[
    dict[tuple[str, str, str, str], list[IntronCandidate]],
    dict[str, str],
]:
    """Build key-indexed intron candidates from transcript exon structures.

    Parameters
    ----------
    transcripts : dict[str, TranscriptModel]
        Parsed transcript annotation models.
    fasta_reader : FastaIndexedReader
        Indexed FASTA reader.

    Returns
    -------
    tuple[dict[tuple[str, str, str, str], list[IntronCandidate]], dict[str, str]]
        Candidate map keyed by
        ``(transcript_id, strand, donor_seq_100bp, acceptor_seq_100bp)`` and
        transcript-to-strand map.

    Notes
    -----
    Let ``T`` be number of transcripts and ``I`` be number of recovered introns.
    Runtime is ``O(T + I)`` plus sequence fetch cost.
    """
    candidate_map: dict[tuple[str, str, str, str], list[IntronCandidate]] = {}
    strand_by_tid: dict[str, str] = {}

    for transcript_id, model in transcripts.items():
        strand_by_tid[transcript_id] = model.strand
        if len(model.exons) < 2:
            continue
        if not fasta_reader.has_contig(model.chrom):
            continue

        contig_len = fasta_reader.contig_length(model.chrom)

        for idx in range(len(model.exons) - 1):
            upstream = model.exons[idx]
            downstream = model.exons[idx + 1]

            if model.strand == "+":
                intron_start_tx = upstream.end + 1
                acceptor_boundary_tx = downstream.start
                intron_start = intron_start_tx
                intron_end = downstream.start - 1
                donor_start, donor_end = _donor_coords_plus(intron_start_tx)
                acceptor_start, acceptor_end = _acceptor_coords_plus(
                    acceptor_boundary_tx
                )
                if (
                    intron_start < 1
                    or intron_end < intron_start
                    or donor_start < 1
                    or donor_end > contig_len
                    or acceptor_start < 1
                    or acceptor_end > contig_len
                ):
                    continue
                donor_seq = fasta_reader.fetch_interval(
                    model.chrom,
                    donor_start,
                    donor_end,
                )
                acceptor_seq = fasta_reader.fetch_interval(
                    model.chrom,
                    acceptor_start,
                    acceptor_end,
                )
            else:
                intron_start_tx = upstream.start - 1
                acceptor_boundary_tx = downstream.end
                intron_start = downstream.end + 1
                intron_end = upstream.start - 1
                donor_start, donor_end = _coords_minus(
                    intron_start_tx,
                    left_bp=3,
                    right_bp=97,
                )
                acceptor_start, acceptor_end = _coords_minus(
                    acceptor_boundary_tx,
                    left_bp=97,
                    right_bp=3,
                )
                if (
                    intron_start < 1
                    or intron_end < intron_start
                    or donor_start < 1
                    or donor_end > contig_len
                    or acceptor_start < 1
                    or acceptor_end > contig_len
                ):
                    continue
                donor_seq = reverse_complement(
                    fasta_reader.fetch_interval(
                        model.chrom,
                        donor_start,
                        donor_end,
                    )
                )
                acceptor_seq = reverse_complement(
                    fasta_reader.fetch_interval(
                        model.chrom,
                        acceptor_start,
                        acceptor_end,
                    )
                )

            intron_length = intron_end - intron_start + 1
            if intron_length <= 0:
                continue

            candidate = IntronCandidate(
                transcript_id=transcript_id,
                intron_index=idx + 1,
                chrom=model.chrom,
                strand=model.strand,
                intron_start=intron_start,
                intron_end=intron_end,
                intron_length=intron_length,
                donor_seq_100bp=donor_seq,
                acceptor_seq_100bp=acceptor_seq,
            )
            key = (
                transcript_id,
                model.strand,
                donor_seq,
                acceptor_seq,
            )
            candidate_map.setdefault(key, []).append(candidate)

    return candidate_map, strand_by_tid


def _read_positive_err_records(pos_path: Path) -> list[PositiveErrRecord]:
    """Read strict positive records from ``100bp.err``.

    Parameters
    ----------
    pos_path : Path
        Positive training file path.

    Returns
    -------
    list[PositiveErrRecord]
        Parsed strict-format records.

    Raises
    ------
    ValueError
        If any line does not satisfy strict format requirements.
    """
    records: list[PositiveErrRecord] = []

    with pos_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            match = _POSITIVE_RECORD_PATTERN.match(line)
            if match is None:
                raise ValueError(
                    f"Invalid positive record at {pos_path}:{line_no}: {line}"
                )

            donor_seq = match.group(1).upper()
            acceptor_seq = match.group(2).upper()
            if len(donor_seq) != 100 or len(acceptor_seq) != 100:
                raise ValueError(
                    f"Positive record must contain 100bp donor/acceptor at "
                    f"{pos_path}:{line_no}"
                )
            records.append(
                PositiveErrRecord(
                    source_line_no=line_no,
                    donor_seq_100bp=donor_seq,
                    acceptor_seq_100bp=acceptor_seq,
                    strand=match.group(3),
                    transcript_id=match.group(4),
                    intron_half_length=int(match.group(5)),
                )
            )

    return records


def _extract_negative_request_rows(
    species: str,
    neg_path: Path,
) -> list[NegativeRequestRow]:
    """Extract ``DEBUG pair`` rows from negative file.

    Parameters
    ----------
    species : str
        Species name.
    neg_path : Path
        Negative input file path.

    Returns
    -------
    list[NegativeRequestRow]
        Rows for coordinate request TSV.
    """
    rows: list[NegativeRequestRow] = []
    with neg_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            match = _NEGATIVE_PAIR_PATTERN.match(line)
            if match is None:
                continue
            donor_seq = match.group(1).upper()
            acceptor_seq = match.group(2).upper()
            rows.append(
                NegativeRequestRow(
                    species=species,
                    source_line_no=line_no,
                    donor_seq_100bp=donor_seq,
                    acceptor_seq_100bp=acceptor_seq,
                    strand=match.group(3),
                    intron_half_length=int(match.group(4)),
                )
            )
    return rows


def _fetch_intron_with_flank(
    fasta_reader: FastaIndexedReader,
    candidate: IntronCandidate,
    flank_bp: int,
) -> str | None:
    """Fetch transcript-oriented intron sequence with flanks.

    Parameters
    ----------
    fasta_reader : FastaIndexedReader
        Indexed FASTA reader.
    candidate : IntronCandidate
        Matched intron candidate.
    flank_bp : int
        Flank size in bp on both ends.

    Returns
    -------
    str | None
        Transcript-oriented sequence, or ``None`` when out of contig bounds.
    """
    contig_len = fasta_reader.contig_length(candidate.chrom)
    start = candidate.intron_start - flank_bp
    end = candidate.intron_end + flank_bp
    if start < 1 or end > contig_len:
        return None
    seq = fasta_reader.fetch_interval(candidate.chrom, start, end)
    if candidate.strand == "-":
        return reverse_complement(seq)
    return seq


def _percentile_nearest_rank(values: list[int], p: float) -> int:
    """Return nearest-rank percentile for integer list.

    Parameters
    ----------
    values : list[int]
        Value list.
    p : float
        Percentile in [0, 1].

    Returns
    -------
    int
        Percentile value, or 0 for empty input.
    """
    if not values:
        return 0
    sorted_values = sorted(values)
    rank = max(1, math.ceil(p * len(sorted_values)))
    return sorted_values[rank - 1]


def _build_stats(
    species: str,
    total_input_rows: int,
    matched_rows: int,
    unmatched_rows: int,
    ambiguous_rows: int,
    half_length_mismatch_rows: int,
    seq_lengths: list[int],
) -> SpeciesStats:
    """Build one species QC summary object."""
    return SpeciesStats(
        species=species,
        total_input_rows=total_input_rows,
        matched_rows=matched_rows,
        unmatched_rows=unmatched_rows,
        ambiguous_rows=ambiguous_rows,
        half_length_mismatch_rows=half_length_mismatch_rows,
        min_seq_len=min(seq_lengths) if seq_lengths else 0,
        median_seq_len=_percentile_nearest_rank(seq_lengths, 0.5),
        p95_seq_len=_percentile_nearest_rank(seq_lengths, 0.95),
        p99_seq_len=_percentile_nearest_rank(seq_lengths, 0.99),
        max_seq_len=max(seq_lengths) if seq_lengths else 0,
    )


def _ensure_feasibility(
    records: list[PositiveErrRecord],
    transcript_strands: dict[str, str],
    species: str,
) -> None:
    """Run pre-match feasibility checks required by specification.

    Parameters
    ----------
    records : list[PositiveErrRecord]
        Positive input records.
    transcript_strands : dict[str, str]
        Transcript strand mapping parsed from GTF.
    species : str
        Species label for error messages.

    Raises
    ------
    ValueError
        If transcript IDs or strands are inconsistent with GTF.
    """
    missing_tid = 0
    strand_mismatch = 0

    for record in records:
        strand = transcript_strands.get(record.transcript_id)
        if strand is None:
            missing_tid += 1
            continue
        if strand != record.strand:
            strand_mismatch += 1

    if missing_tid > 0 or strand_mismatch > 0:
        raise ValueError(
            f"Feasibility check failed for {species}: "
            f"missing_tid_rows={missing_tid}, strand_mismatch_rows={strand_mismatch}"
        )


def _write_positive_output(path: Path, rows: Iterable[PositiveOutputRow]) -> None:
    """Write positive output rows as TSV."""
    fieldnames = [
        "species",
        "label",
        "transcript_id",
        "intron_index",
        "chrom",
        "strand",
        "intron_start",
        "intron_end",
        "intron_length",
        "intron_half_length",
        "flank_bp",
        "donor_seq_100bp",
        "acceptor_seq_100bp",
        "sequence",
        "source_line_no",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "species": row.species,
                    "label": row.label,
                    "transcript_id": row.transcript_id,
                    "intron_index": row.intron_index,
                    "chrom": row.chrom,
                    "strand": row.strand,
                    "intron_start": row.intron_start,
                    "intron_end": row.intron_end,
                    "intron_length": row.intron_length,
                    "intron_half_length": row.intron_half_length,
                    "flank_bp": row.flank_bp,
                    "donor_seq_100bp": row.donor_seq_100bp,
                    "acceptor_seq_100bp": row.acceptor_seq_100bp,
                    "sequence": row.sequence,
                    "source_line_no": row.source_line_no,
                }
            )


def _write_qc_summary(path: Path, stats: SpeciesStats) -> None:
    """Write one-row QC summary as TSV."""
    fieldnames = [
        "species",
        "total_input_rows",
        "matched_rows",
        "unmatched_rows",
        "ambiguous_rows",
        "half_length_mismatch_rows",
        "min_seq_len",
        "median_seq_len",
        "p95_seq_len",
        "p99_seq_len",
        "max_seq_len",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "species": stats.species,
                "total_input_rows": stats.total_input_rows,
                "matched_rows": stats.matched_rows,
                "unmatched_rows": stats.unmatched_rows,
                "ambiguous_rows": stats.ambiguous_rows,
                "half_length_mismatch_rows": stats.half_length_mismatch_rows,
                "min_seq_len": stats.min_seq_len,
                "median_seq_len": stats.median_seq_len,
                "p95_seq_len": stats.p95_seq_len,
                "p99_seq_len": stats.p99_seq_len,
                "max_seq_len": stats.max_seq_len,
            }
        )


def _write_negative_request_output(
    path: Path,
    rows: Iterable[NegativeRequestRow],
) -> None:
    """Write negative coordinate request rows as TSV."""
    fieldnames = [
        "species",
        "source_line_no",
        "donor_seq_100bp",
        "acceptor_seq_100bp",
        "strand",
        "intron_half_length",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "species": row.species,
                    "source_line_no": row.source_line_no,
                    "donor_seq_100bp": row.donor_seq_100bp,
                    "acceptor_seq_100bp": row.acceptor_seq_100bp,
                    "strand": row.strand,
                    "intron_half_length": row.intron_half_length,
                }
            )


def process_species(
    species: str,
    data_root: Path,
    flank_bp: int,
    pos_input_name: str,
    out_pos_name: str,
    out_qc_name: str,
    out_neg_request_name: str,
    strict: bool,
) -> SpeciesStats:
    """Process one species and write all required outputs.

    Parameters
    ----------
    species : str
        Species name under ``data_root``.
    data_root : Path
        Data root path.
    flank_bp : int
        Flank base pairs on both intron ends.
    pos_input_name : str
        Positive input filename under raw directory.
    out_pos_name : str
        Positive output TSV filename under raw directory.
    out_qc_name : str
        QC output TSV filename under raw directory.
    out_neg_request_name : str
        Negative request output TSV filename under raw directory.
    strict : bool
        Strict consistency mode.

    Returns
    -------
    SpeciesStats
        QC summary.

    Raises
    ------
    FileNotFoundError
        If required inputs are missing.
    ValueError
        If strict checks fail.
    """
    raw_dir = data_root / species / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found for {species}: {raw_dir}")

    pos_path = raw_dir / pos_input_name
    if not pos_path.exists():
        raise FileNotFoundError(f"Positive input not found: {pos_path}")

    neg_path = raw_dir / "100bp.neg.err"
    if not neg_path.exists():
        raise FileNotFoundError(f"Negative input not found: {neg_path}")

    fasta_path, gtf_path = _find_species_reference_files(raw_dir)
    positive_records = _read_positive_err_records(pos_path)
    transcript_models = _parse_gtf_transcripts(gtf_path)

    with FastaIndexedReader(fasta_path) as fasta_reader:
        candidate_map, transcript_strands = _build_intron_candidate_index(
            transcript_models,
            fasta_reader,
        )

        _ensure_feasibility(positive_records, transcript_strands, species)

        output_rows: list[PositiveOutputRow] = []
        seq_lengths: list[int] = []

        unmatched_rows = 0
        ambiguous_rows = 0
        half_mismatch_rows = 0

        for record in positive_records:
            key = (
                record.transcript_id,
                record.strand,
                record.donor_seq_100bp,
                record.acceptor_seq_100bp,
            )
            candidates = candidate_map.get(key, [])
            if not candidates:
                unmatched_rows += 1
                continue
            if len(candidates) != 1:
                ambiguous_rows += 1
                continue

            candidate = candidates[0]
            expected_half = candidate.intron_length // 2
            if record.intron_half_length != expected_half:
                half_mismatch_rows += 1
                continue

            sequence = _fetch_intron_with_flank(
                fasta_reader=fasta_reader,
                candidate=candidate,
                flank_bp=flank_bp,
            )
            if sequence is None:
                unmatched_rows += 1
                continue

            output_row = PositiveOutputRow(
                species=species,
                label=1,
                transcript_id=record.transcript_id,
                intron_index=candidate.intron_index,
                chrom=candidate.chrom,
                strand=candidate.strand,
                intron_start=candidate.intron_start,
                intron_end=candidate.intron_end,
                intron_length=candidate.intron_length,
                intron_half_length=record.intron_half_length,
                flank_bp=flank_bp,
                donor_seq_100bp=record.donor_seq_100bp,
                acceptor_seq_100bp=record.acceptor_seq_100bp,
                sequence=sequence,
                source_line_no=record.source_line_no,
            )
            output_rows.append(output_row)
            seq_lengths.append(len(sequence))

    stats = _build_stats(
        species=species,
        total_input_rows=len(positive_records),
        matched_rows=len(output_rows),
        unmatched_rows=unmatched_rows,
        ambiguous_rows=ambiguous_rows,
        half_length_mismatch_rows=half_mismatch_rows,
        seq_lengths=seq_lengths,
    )

    if strict:
        errors: list[str] = []
        if stats.unmatched_rows > 0:
            errors.append(f"unmatched_rows={stats.unmatched_rows}")
        if stats.ambiguous_rows > 0:
            errors.append(f"ambiguous_rows={stats.ambiguous_rows}")
        if stats.half_length_mismatch_rows > 0:
            errors.append(
                f"half_length_mismatch_rows={stats.half_length_mismatch_rows}"
            )
        if errors:
            joined = ", ".join(errors)
            raise ValueError(f"Strict check failed for {species}: {joined}")

    negative_rows = _extract_negative_request_rows(species=species, neg_path=neg_path)

    _write_positive_output(raw_dir / out_pos_name, output_rows)
    _write_qc_summary(raw_dir / out_qc_name, stats)
    _write_negative_request_output(raw_dir / out_neg_request_name, negative_rows)

    return stats


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv : list[str] | None, optional
        CLI token list. ``None`` uses ``sys.argv``.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    species_list = _species_tokens(args.species)
    if not species_list:
        raise ValueError("--species must contain at least one species name")

    data_root = Path(args.data_root)
    for species in species_list:
        stats = process_species(
            species=species,
            data_root=data_root,
            flank_bp=args.flank_bp,
            pos_input_name=args.pos_input_name,
            out_pos_name=args.out_pos_name,
            out_qc_name=args.out_qc_name,
            out_neg_request_name=args.out_neg_request_name,
            strict=args.strict,
        )
        print(
            f"[{species}] total={stats.total_input_rows} matched={stats.matched_rows} "
            f"unmatched={stats.unmatched_rows} ambiguous={stats.ambiguous_rows} "
            f"half_mismatch={stats.half_length_mismatch_rows}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
