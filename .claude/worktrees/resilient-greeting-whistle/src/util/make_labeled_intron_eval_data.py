from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote


@dataclass(frozen=True)
class Exon:
    """One exon in genomic coordinates.

    Attributes
    ----------
    chrom : str
        Contig name.
    start : int
        1-based inclusive start.
    end : int
        1-based inclusive end.
    """

    chrom: str
    start: int
    end: int


@dataclass(frozen=True)
class TranscriptModel:
    """One transcript annotation model.

    Attributes
    ----------
    transcript_id : str
        Transcript identifier.
    gene_id : str
        Gene identifier if present, otherwise empty string.
    chrom : str
        Contig name.
    strand : str
        Transcript strand, either ``+`` or ``-``.
    exons : tuple[Exon, ...]
        Exons in transcript order (5' -> 3').
    """

    transcript_id: str
    gene_id: str
    chrom: str
    strand: str
    exons: tuple[Exon, ...]


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
        Bases per wrapped FASTA line.
    line_bytes : int
        Bytes per wrapped FASTA line, including newline bytes.
    """

    seq_offset: int
    seq_length: int
    line_bases: int
    line_bytes: int


class FastaIndexedReader:
    """Random-access FASTA reader without external dependencies.

    Notes
    -----
    The index build phase is linear in FASTA size: ``O(F)``.
    Each interval fetch is linear in requested sequence span: ``O(L)``.
    """

    def __init__(self, fasta_path: Path) -> None:
        """Initialize the reader and build an in-memory contig index.

        Parameters
        ----------
        fasta_path : Path
            FASTA file path.

        Raises
        ------
        FileNotFoundError
            If ``fasta_path`` does not exist.
        ValueError
            If the FASTA structure is malformed.
        """
        if not fasta_path.exists():
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
        self._fasta_path = fasta_path
        self._index = self._build_index(fasta_path)
        self._handle = fasta_path.open("rb")

    def __enter__(self) -> FastaIndexedReader:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Exit context manager and close file handle."""
        _ = exc_type
        _ = exc
        _ = tb
        self.close()

    def close(self) -> None:
        """Close the underlying FASTA file handle."""
        self._handle.close()

    def has_contig(self, contig: str) -> bool:
        """Return whether a contig exists in the FASTA index.

        Parameters
        ----------
        contig : str
            Contig name.

        Returns
        -------
        bool
            ``True`` when the contig exists.
        """
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
            Contig length in bases.

        Raises
        ------
        KeyError
            If the contig is missing.
        """
        return self._index[contig].seq_length

    def fetch_interval(self, contig: str, start: int, end: int) -> str:
        """Fetch one 1-based inclusive genomic interval.

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
            If coordinates are invalid or out of bounds.
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
        """Build byte-level index for all FASTA contigs.

        Parameters
        ----------
        fasta_path : Path
            FASTA file path.

        Returns
        -------
        dict[str, FastaContigIndex]
            Contig name to index metadata.

        Raises
        ------
        ValueError
            If FASTA is malformed or contig names are duplicated.
        """
        index: dict[str, FastaContigIndex] = {}

        with fasta_path.open("rb") as handle:
            current_name: str | None = None
            seq_offset = 0
            seq_length = 0
            line_bases = 0
            line_bytes = 0

            while True:
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
        """Map 1-based base position to FASTA byte offset.

        Parameters
        ----------
        meta : FastaContigIndex
            Contig index metadata.
        one_based_pos : int
            1-based sequence position.

        Returns
        -------
        int
            Byte offset in FASTA file.
        """
        zero_based = one_based_pos - 1
        full_lines = zero_based // meta.line_bases
        within_line = zero_based % meta.line_bases
        return meta.seq_offset + full_lines * meta.line_bytes + within_line


@dataclass(frozen=True)
class BuildStats:
    """Summary statistics for dataset construction.

    Attributes
    ----------
    species : str
        Species label used for output rows.
    query_transcript_count : int
        Number of parsed query transcripts.
    reference_transcript_count : int
        Number of parsed reference transcripts.
    reference_intron_count : int
        Number of unique reference introns.
    written_rows : int
        Number of written intron candidate rows.
    positive_labels : int
        Number of rows labeled as true introns.
    negative_labels : int
        Number of rows labeled as false introns.
    skipped_short_transcripts : int
        Number of transcripts with fewer than two exons.
    skipped_missing_contig_transcripts : int
        Number of transcripts skipped because contig is absent in FASTA.
    skipped_out_of_bounds_introns : int
        Number of introns skipped due to out-of-bounds sequence windows.
    """

    species: str
    query_transcript_count: int
    reference_transcript_count: int
    reference_intron_count: int
    written_rows: int
    positive_labels: int
    negative_labels: int
    skipped_short_transcripts: int
    skipped_missing_contig_transcripts: int
    skipped_out_of_bounds_introns: int


def reverse_complement(seq: str) -> str:
    """Return reverse complement of a DNA sequence.

    Parameters
    ----------
    seq : str
        Input DNA sequence.

    Returns
    -------
    str
        Reverse-complement sequence in uppercase.
    """
    table = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(table)[::-1].upper()


def parse_attributes(attr_text: str) -> dict[str, str]:
    """Parse GTF/GFF attribute text into a key-value mapping.

    Parameters
    ----------
    attr_text : str
        Attribute field text (9th column).

    Returns
    -------
    dict[str, str]
        Parsed attributes. For repeated keys, the first value is kept.
    """
    attributes: dict[str, str] = {}
    for raw_token in attr_text.strip().strip(";").split(";"):
        token = raw_token.strip()
        if not token:
            continue

        key = ""
        value = ""
        if "=" in token:
            key, value = token.split("=", 1)
        elif " " in token:
            key, value = token.split(" ", 1)
            value = value.strip().strip('"')
        else:
            continue

        key = key.strip()
        value = unquote(value.strip().strip('"'))
        if key and key not in attributes:
            attributes[key] = value

    return attributes


def resolve_transcript_id(attributes: dict[str, str]) -> str | None:
    """Resolve transcript identifier from parsed attributes.

    Parameters
    ----------
    attributes : dict[str, str]
        Parsed attribute dictionary.

    Returns
    -------
    str | None
        Resolved transcript ID when available.
    """
    for key in ("transcript_id", "transcriptId"):
        value = attributes.get(key, "").strip()
        if value:
            return value

    parent = attributes.get("Parent", "").strip()
    if parent:
        return parent.split(",")[0].strip()

    return None


def parse_transcript_models(
    annotation_path: Path,
    feature_name: str,
    fail_on_missing_transcript_id: bool,
    allow_transcript_id_collisions: bool = False,
) -> dict[str, TranscriptModel]:
    """Parse exon-based transcript models from GTF/GFF annotation.

    Parameters
    ----------
    annotation_path : Path
        Annotation file path.
    feature_name : str
        Feature name used to build transcript exons.
    fail_on_missing_transcript_id : bool
        If ``True``, raise on exon rows with no transcript ID.
    allow_transcript_id_collisions : bool, default=False
        If ``True``, allow one transcript ID to appear on multiple
        contig/strand combinations by splitting them into separate models.

    Returns
    -------
    dict[str, TranscriptModel]
        Transcript ID to parsed transcript model.

    Raises
    ------
    FileNotFoundError
        If annotation file does not exist.
    ValueError
        If coordinates are invalid, transcript IDs are inconsistent, or
        required transcript IDs are missing.

    Notes
    -----
    Runtime is linear in annotation line count: ``O(N)``.
    """
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    exons_by_tid: dict[str, list[Exon]] = {}
    gene_by_tid: dict[str, str] = {}
    chrom_by_tid: dict[str, str] = {}
    strand_by_tid: dict[str, str] = {}

    with annotation_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith("#"):
                continue

            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue

            chrom, _, feature, start_s, end_s, _, strand, _, attrs = fields
            if feature != feature_name:
                continue
            if strand not in {"+", "-"}:
                continue

            attributes = parse_attributes(attrs)
            transcript_id = resolve_transcript_id(attributes)
            if transcript_id is None:
                if fail_on_missing_transcript_id:
                    raise ValueError(
                        "Missing transcript ID at "
                        f"{annotation_path}:{line_no}: {line.strip()}"
                    )
                continue

            start = int(start_s)
            end = int(end_s)
            if start <= 0 or end < start:
                raise ValueError(
                    f"Invalid exon coordinates at {annotation_path}:{line_no}: "
                    f"{start}-{end}"
                )

            transcript_key = transcript_id
            has_collision = (
                transcript_id in chrom_by_tid
                and (
                    chrom_by_tid[transcript_id] != chrom
                    or strand_by_tid[transcript_id] != strand
                )
            )
            if has_collision:
                if not allow_transcript_id_collisions:
                    if chrom_by_tid[transcript_id] != chrom:
                        raise ValueError(
                            f"Transcript {transcript_id} has multiple contigs: "
                            f"{chrom_by_tid[transcript_id]} vs {chrom}"
                        )
                    raise ValueError(
                        f"Transcript {transcript_id} has multiple strands: "
                        f"{strand_by_tid[transcript_id]} vs {strand}"
                    )
                transcript_key = f"{transcript_id}|{chrom}|{strand}"
                suffix = 2
                while (
                    transcript_key in chrom_by_tid
                    and (
                        chrom_by_tid[transcript_key] != chrom
                        or strand_by_tid[transcript_key] != strand
                    )
                ):
                    transcript_key = f"{transcript_id}|{chrom}|{strand}|{suffix}"
                    suffix += 1

            gene_id = (
                attributes.get("gene_id", "").strip()
                or attributes.get("gene", "").strip()
                or attributes.get("gene_name", "").strip()
            )

            chrom_by_tid[transcript_key] = chrom
            strand_by_tid[transcript_key] = strand
            if transcript_key not in gene_by_tid or not gene_by_tid[transcript_key]:
                gene_by_tid[transcript_key] = gene_id
            exons_by_tid.setdefault(transcript_key, []).append(
                Exon(chrom=chrom, start=start, end=end)
            )

    models: dict[str, TranscriptModel] = {}
    for transcript_key, exons in exons_by_tid.items():
        strand = strand_by_tid[transcript_key]
        ordered_exons = sorted(
            exons,
            key=lambda exon: (exon.start, exon.end),
            reverse=(strand == "-"),
        )
        models[transcript_key] = TranscriptModel(
            transcript_id=transcript_key,
            gene_id=gene_by_tid.get(transcript_key, ""),
            chrom=chrom_by_tid[transcript_key],
            strand=strand,
            exons=tuple(ordered_exons),
        )

    return models


def donor_coords_plus(intron_start: int, donor_len: int) -> tuple[int, int]:
    """Return donor 100bp-like interval on ``+`` strand.

    Parameters
    ----------
    intron_start : int
        1-based first intronic base.
    donor_len : int
        Total donor window length.

    Returns
    -------
    tuple[int, int]
        1-based inclusive interval.
    """
    left_bp = 3
    right_bp = donor_len - left_bp
    start = intron_start - left_bp
    end = intron_start + right_bp - 1
    return start, end


def acceptor_coords_plus(acceptor_boundary: int, acceptor_len: int) -> tuple[int, int]:
    """Return acceptor 100bp-like interval on ``+`` strand.

    Parameters
    ----------
    acceptor_boundary : int
        1-based first exonic base of downstream exon.
    acceptor_len : int
        Total acceptor window length.

    Returns
    -------
    tuple[int, int]
        1-based inclusive interval.
    """
    right_bp = 3
    left_bp = acceptor_len - right_bp
    start = acceptor_boundary - left_bp
    end = acceptor_boundary + right_bp - 1
    return start, end


def coords_minus(boundary: int, left_bp: int, right_bp: int) -> tuple[int, int]:
    """Return genomic interval for transcript-oriented window on ``-`` strand.

    Parameters
    ----------
    boundary : int
        Boundary coordinate in transcript orientation.
    left_bp : int
        Bases to keep on the left side after reverse complement.
    right_bp : int
        Bases to keep on the right side after reverse complement.

    Returns
    -------
    tuple[int, int]
        1-based inclusive genomic interval.
    """
    start = boundary - (right_bp - 1)
    end = boundary + left_bp
    return start, end


def build_reference_intron_set(
    transcripts: dict[str, TranscriptModel],
) -> set[tuple[str, str, int, int]]:
    """Build set of reference introns from transcript exon chains.

    Parameters
    ----------
    transcripts : dict[str, TranscriptModel]
        Reference transcript models.

    Returns
    -------
    set[tuple[str, str, int, int]]
        Unique intron keys:
        ``(chrom, strand, intron_start, intron_end)``.

    Notes
    -----
    Runtime is linear in number of transcript exons: ``O(E)``.
    """
    introns: set[tuple[str, str, int, int]] = set()
    for model in transcripts.values():
        if len(model.exons) < 2:
            continue
        for idx in range(len(model.exons) - 1):
            upstream = model.exons[idx]
            downstream = model.exons[idx + 1]
            if model.strand == "+":
                intron_start = upstream.end + 1
                intron_end = downstream.start - 1
            else:
                intron_start = downstream.end + 1
                intron_end = upstream.start - 1
            if intron_end >= intron_start:
                introns.add((model.chrom, model.strand, intron_start, intron_end))
    return introns


def build_reference_splice_site_sets(
    transcripts: dict[str, TranscriptModel],
) -> tuple[set[tuple[str, str, int]], set[tuple[str, str, int]]]:
    """Build reference donor/acceptor site sets from transcript exon chains.

    Parameters
    ----------
    transcripts : dict[str, TranscriptModel]
        Reference transcript models.

    Returns
    -------
    tuple[set[tuple[str, str, int]], set[tuple[str, str, int]]]
        ``(donor_sites, acceptor_sites)`` where each site key is
        ``(chrom, strand, boundary_pos)``.

    Notes
    -----
    Runtime is linear in number of transcript exons: ``O(E)``.
    """
    donor_sites: set[tuple[str, str, int]] = set()
    acceptor_sites: set[tuple[str, str, int]] = set()
    for model in transcripts.values():
        if len(model.exons) < 2:
            continue
        for idx in range(len(model.exons) - 1):
            upstream = model.exons[idx]
            downstream = model.exons[idx + 1]
            if model.strand == "+":
                donor_boundary = upstream.end + 1
                acceptor_boundary = downstream.start
                intron_start = donor_boundary
                intron_end = downstream.start - 1
            else:
                donor_boundary = upstream.start - 1
                acceptor_boundary = downstream.end
                intron_start = downstream.end + 1
                intron_end = upstream.start - 1
            if intron_end >= intron_start:
                donor_sites.add((model.chrom, model.strand, donor_boundary))
                acceptor_sites.add((model.chrom, model.strand, acceptor_boundary))
    return donor_sites, acceptor_sites


def build_labeled_intron_dataset(
    species: str,
    fasta_path: Path,
    query_gtf_path: Path,
    reference_annotation_path: Path,
    output_tsv_path: Path,
    donor_len: int = 100,
    acceptor_len: int = 100,
    flank_bp: int = 10,
    query_feature: str = "exon",
    reference_feature: str = "exon",
    limit: int = 0,
) -> BuildStats:
    """Build labeled intron-candidate evaluation dataset.

    Parameters
    ----------
    species : str
        Species label written to output rows.
    fasta_path : Path
        Reference genome FASTA path.
    query_gtf_path : Path
        Query transcript annotation (typically StringTie GTF).
    reference_annotation_path : Path
        Reference annotation path (GFF/GTF) for truth labels.
    output_tsv_path : Path
        Output TSV path.
    donor_len : int, default=100
        Donor window length.
    acceptor_len : int, default=100
        Acceptor window length.
    flank_bp : int, default=10
        Flank size added to both intron ends.
    query_feature : str, default="exon"
        Feature name for query intron reconstruction.
    reference_feature : str, default="exon"
        Feature name for reference intron reconstruction.
    limit : int, default=0
        Max intron rows to write. ``0`` means no limit.

    Returns
    -------
    BuildStats
        Dataset build summary.

    Raises
    ------
    ValueError
        If argument values are invalid.
    FileNotFoundError
        If required input files are missing.
    """
    if donor_len < 3:
        raise ValueError("--donor-len must be >= 3")
    if acceptor_len < 3:
        raise ValueError("--acceptor-len must be >= 3")
    if flank_bp <= 0:
        raise ValueError("--flank-bp must be > 0")
    if limit < 0:
        raise ValueError("--limit must be >= 0")

    query_transcripts = parse_transcript_models(
        annotation_path=query_gtf_path,
        feature_name=query_feature,
        fail_on_missing_transcript_id=True,
    )
    reference_transcripts = parse_transcript_models(
        annotation_path=reference_annotation_path,
        feature_name=reference_feature,
        fail_on_missing_transcript_id=False,
        allow_transcript_id_collisions=True,
    )
    reference_introns = build_reference_intron_set(reference_transcripts)
    (
        reference_donor_sites,
        reference_acceptor_sites,
    ) = build_reference_splice_site_sets(reference_transcripts)

    out_dir = output_tsv_path.parent
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    skipped_short_transcripts = 0
    skipped_missing_contig_transcripts = 0
    skipped_out_of_bounds_introns = 0
    written_rows = 0
    positive_labels = 0
    negative_labels = 0
    reached_limit = False

    fieldnames = [
        "species",
        "transcript_id",
        "gene_id",
        "intron_index",
        "chrom",
        "strand",
        "intron_start",
        "intron_end",
        "intron_length",
        "label",
        "donor_label",
        "acceptor_label",
        "donor_boundary_pos",
        "acceptor_boundary_pos",
        "donor_seq_100bp",
        "acceptor_seq_100bp",
        "intron_flank_bp",
        "intron_flank_seq",
    ]

    with (
        FastaIndexedReader(fasta_path) as fasta_reader,
        output_tsv_path.open("w", encoding="utf-8", newline="") as handle,
    ):
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for transcript_id in sorted(query_transcripts.keys()):
            model = query_transcripts[transcript_id]
            if len(model.exons) < 2:
                skipped_short_transcripts += 1
                continue

            if not fasta_reader.has_contig(model.chrom):
                skipped_missing_contig_transcripts += 1
                continue
            contig_len = fasta_reader.contig_length(model.chrom)

            for idx in range(len(model.exons) - 1):
                upstream = model.exons[idx]
                downstream = model.exons[idx + 1]

                if model.strand == "+":
                    donor_boundary = upstream.end + 1
                    acceptor_boundary = downstream.start
                    intron_start = donor_boundary
                    intron_end = downstream.start - 1
                    donor_start, donor_end = donor_coords_plus(
                        intron_start=donor_boundary,
                        donor_len=donor_len,
                    )
                    acceptor_start, acceptor_end = acceptor_coords_plus(
                        acceptor_boundary=acceptor_boundary,
                        acceptor_len=acceptor_len,
                    )
                else:
                    donor_boundary = upstream.start - 1
                    acceptor_boundary = downstream.end
                    intron_start = downstream.end + 1
                    intron_end = upstream.start - 1
                    donor_start, donor_end = coords_minus(
                        boundary=donor_boundary,
                        left_bp=3,
                        right_bp=donor_len - 3,
                    )
                    acceptor_start, acceptor_end = coords_minus(
                        boundary=acceptor_boundary,
                        left_bp=acceptor_len - 3,
                        right_bp=3,
                    )

                if intron_end < intron_start:
                    continue

                flank_start = intron_start - flank_bp
                flank_end = intron_end + flank_bp
                is_out_of_bounds = (
                    donor_start < 1
                    or donor_end > contig_len
                    or acceptor_start < 1
                    or acceptor_end > contig_len
                    or flank_start < 1
                    or flank_end > contig_len
                )
                if is_out_of_bounds:
                    skipped_out_of_bounds_introns += 1
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
                intron_flank_seq = fasta_reader.fetch_interval(
                    model.chrom,
                    flank_start,
                    flank_end,
                )
                if model.strand == "-":
                    donor_seq = reverse_complement(donor_seq)
                    acceptor_seq = reverse_complement(acceptor_seq)
                    intron_flank_seq = reverse_complement(intron_flank_seq)

                intron_length = intron_end - intron_start + 1
                label = int(
                    (
                        model.chrom,
                        model.strand,
                        intron_start,
                        intron_end,
                    )
                    in reference_introns
                )
                if label == 1:
                    positive_labels += 1
                else:
                    negative_labels += 1
                donor_label = int(
                    (
                        model.chrom,
                        model.strand,
                        donor_boundary,
                    )
                    in reference_donor_sites
                )
                acceptor_label = int(
                    (
                        model.chrom,
                        model.strand,
                        acceptor_boundary,
                    )
                    in reference_acceptor_sites
                )

                writer.writerow(
                    {
                        "species": species,
                        "transcript_id": model.transcript_id,
                        "gene_id": model.gene_id,
                        "intron_index": idx + 1,
                        "chrom": model.chrom,
                        "strand": model.strand,
                        "intron_start": intron_start,
                        "intron_end": intron_end,
                        "intron_length": intron_length,
                        "label": label,
                        "donor_label": donor_label,
                        "acceptor_label": acceptor_label,
                        "donor_boundary_pos": donor_boundary,
                        "acceptor_boundary_pos": acceptor_boundary,
                        "donor_seq_100bp": donor_seq,
                        "acceptor_seq_100bp": acceptor_seq,
                        "intron_flank_bp": flank_bp,
                        "intron_flank_seq": intron_flank_seq,
                    }
                )
                written_rows += 1

                if limit > 0 and written_rows >= limit:
                    reached_limit = True
                    break

            if reached_limit:
                break

    return BuildStats(
        species=species,
        query_transcript_count=len(query_transcripts),
        reference_transcript_count=len(reference_transcripts),
        reference_intron_count=len(reference_introns),
        written_rows=written_rows,
        positive_labels=positive_labels,
        negative_labels=negative_labels,
        skipped_short_transcripts=skipped_short_transcripts,
        skipped_missing_contig_transcripts=skipped_missing_contig_transcripts,
        skipped_out_of_bounds_introns=skipped_out_of_bounds_introns,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list[str] | None, optional
        CLI token list. ``None`` uses ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build intron-candidate test data with donor/acceptor windows, "
            "intron+flank sequence, and intron/site truth labels."
        )
    )
    parser.add_argument("--species", required=True, help="Species label.")
    parser.add_argument("--fasta", required=True, help="Reference FASTA path.")
    parser.add_argument("--query-gtf", required=True, help="Query GTF path.")
    parser.add_argument(
        "--reference-annotation",
        required=True,
        help="Reference GFF/GTF path used for intron truth labels.",
    )
    parser.add_argument("--out-tsv", required=True, help="Output TSV path.")
    parser.add_argument(
        "--donor-len",
        type=int,
        default=100,
        help="Donor window length (default: 100).",
    )
    parser.add_argument(
        "--acceptor-len",
        type=int,
        default=100,
        help="Acceptor window length (default: 100).",
    )
    parser.add_argument(
        "--flank-bp",
        type=int,
        default=10,
        help="Flank bases on both intron ends (default: 10).",
    )
    parser.add_argument(
        "--query-feature",
        default="exon",
        help="Feature name for query intron extraction (default: exon).",
    )
    parser.add_argument(
        "--reference-feature",
        default="exon",
        help="Feature name for reference intron extraction (default: exon).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional row limit (0 means no limit).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run CLI entry point.

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
    stats = build_labeled_intron_dataset(
        species=args.species,
        fasta_path=Path(args.fasta),
        query_gtf_path=Path(args.query_gtf),
        reference_annotation_path=Path(args.reference_annotation),
        output_tsv_path=Path(args.out_tsv),
        donor_len=args.donor_len,
        acceptor_len=args.acceptor_len,
        flank_bp=args.flank_bp,
        query_feature=args.query_feature,
        reference_feature=args.reference_feature,
        limit=args.limit,
    )
    print(
        f"[{stats.species}] rows={stats.written_rows} "
        f"label1={stats.positive_labels} label0={stats.negative_labels} "
        f"query_tx={stats.query_transcript_count} "
        f"ref_tx={stats.reference_transcript_count} "
        f"ref_introns={stats.reference_intron_count} "
        f"skip_short_tx={stats.skipped_short_transcripts} "
        f"skip_missing_contig_tx={stats.skipped_missing_contig_transcripts} "
        f"skip_oob_introns={stats.skipped_out_of_bounds_introns}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
