from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence, TextIO

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from util.make_intron_training_data_from_err import (  # noqa: E402
    IntronCandidate,
    _build_intron_candidate_index,
    _find_species_reference_files,
    _parse_gtf_transcripts,
    reverse_complement,
)


_POSITIVE_PATTERN = re.compile(
    r"^DEBUG donor ([ACGTNacgtn]+) acceptor ([ACGTNacgtn]+) ([+-]) (\S+) (-?\d+)$"
)
_NEGATIVE_DONOR_PATTERN = re.compile(r"^DEBUG donor ([ACGTNacgtn]+) ([+-])$")
_NEGATIVE_ACCEPTOR_PATTERN = re.compile(
    r"^DEBUG acceptor ([ACGTNacgtn]+) ([+-])$"
)
_NEGATIVE_PAIR_PATTERN = re.compile(
    r"^DEBUG pair ([ACGTNacgtn]+) ([ACGTNacgtn]+) ([+-]) (-?\d+)$"
)

_DEFAULT_OUTPUT_COLUMNS = [
    "donor_seq",
    "acceptor_seq",
    "hit_count",
    "hit_coords",
]

_BASE_TO_BITS: dict[int, int] = {
    ord("A"): 0,
    ord("C"): 1,
    ord("G"): 2,
    ord("T"): 3,
}


@dataclass(frozen=True)
class RawErrRecord:
    """One parsed raw ERR record.

    Attributes
    ----------
    source_file : str
        Source filename.
    source_line_no : int
        1-based source line number.
    raw_line : str
        Original line content without trailing newline.
    record_kind : str
        Parsed record kind.
    strand : str
        Transcript-oriented strand token from the source line.
    donor_seq_100bp : str | None
        Donor-side query sequence.
    acceptor_seq_100bp : str | None
        Acceptor-side query sequence.
    transcript_id : str | None
        Transcript identifier for positive rows.
    intron_half_length : int | None
        Intron half-length token when present.
    """

    source_file: str
    source_line_no: int
    raw_line: str
    record_kind: str
    strand: str
    donor_seq_100bp: str | None
    acceptor_seq_100bp: str | None
    transcript_id: str | None
    intron_half_length: int | None


@dataclass(frozen=True)
class WindowHit:
    """One exact sequence-window match in genomic coordinates.

    Attributes
    ----------
    contig : str
        Contig name.
    start : int
        1-based inclusive window start.
    end : int
        1-based inclusive window end.
    strand : str
        ``+`` when the query matched the forward genomic strand, ``-`` when
        the reverse complement matched.
    """

    contig: str
    start: int
    end: int
    strand: str


@dataclass(frozen=True)
class SiteHit:
    """One splice-site coordinate."""

    contig: str
    position: int
    strand: str


@dataclass(frozen=True)
class IntronHit:
    """One intron interval in genomic coordinates."""

    contig: str
    start: int
    end: int
    strand: str


@dataclass(frozen=True)
class SearchVariant:
    """One search variant for a query sequence."""

    query_seq: str
    search_seq: bytes
    strand: str
    seed_code: int | None
    seed_offset: int


@dataclass
class ProgressReporter:
    """Lightweight terminal progress reporter.

    Attributes
    ----------
    species : str
        Species label shown in progress messages.
    enabled : bool, default=True
        Whether progress output is enabled.
    stream : TextIO
        Output stream for progress messages.
    width : int, default=24
        Progress bar width in characters.
    """

    species: str
    enabled: bool = True
    stream: TextIO = field(default_factory=lambda: sys.stderr)
    width: int = 24
    _progress_active: bool = field(default=False, init=False, repr=False)

    def log(self, message: str) -> None:
        """Write one stage log line."""
        if not self.enabled:
            return
        self.finish()
        print(f"[{self.species}] {message}", file=self.stream, flush=True)

    def update(
        self,
        label: str,
        current: int,
        total: int,
        detail: str = "",
    ) -> None:
        """Render one in-place progress line."""
        if not self.enabled:
            return
        bounded_total = max(total, 1)
        bounded_current = max(0, min(current, bounded_total))
        filled = (self.width * bounded_current) // bounded_total
        bar = "#" * filled + "-" * (self.width - filled)
        percent = 100.0 * bounded_current / bounded_total
        suffix = f" {detail}" if detail else ""
        line = (
            f"\r[{self.species}] {label} [{bar}] "
            f"{percent:6.2f}% ({bounded_current}/{bounded_total}){suffix}"
        )
        self.stream.write(line)
        self.stream.flush()
        self._progress_active = True

    def finish(self) -> None:
        """Terminate the current progress line, if any."""
        if not self.enabled or not self._progress_active:
            return
        self.stream.write("\n")
        self.stream.flush()
        self._progress_active = False


class InMemoryFastaSearchIndex:
    """In-memory FASTA loader and exact-match search backend.

    Notes
    -----
    FASTA loading is linear in file size: ``O(F)``. Seed-based query search is
    linear in total contig length plus candidate verification work:
    ``O(F + C)`` where ``C`` is the number of seed candidate checks.
    """

    def __init__(
        self,
        fasta_path: Path,
        seed_len: int = 16,
        progress_reporter: ProgressReporter | None = None,
    ) -> None:
        """Load the FASTA into memory.

        Parameters
        ----------
        fasta_path : Path
            FASTA path.
        seed_len : int, default=16
            Exact seed length used for query indexing.

        Raises
        ------
        FileNotFoundError
            If the FASTA file is missing.
        ValueError
            If the FASTA is malformed or ``seed_len`` is invalid.
        """
        if seed_len <= 0:
            raise ValueError("seed_len must be > 0")
        if not fasta_path.exists():
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

        self._seed_len = seed_len
        self._progress_reporter = progress_reporter
        self._contig_order: list[str] = []
        self._contigs = self._load_fasta(fasta_path=fasta_path)
        self._contig_rank = {
            contig: index for index, contig in enumerate(self._contig_order)
        }
        self._seed_mask = (1 << (2 * seed_len)) - 1

    def has_contig(self, contig: str) -> bool:
        """Return whether the contig is available."""
        return contig in self._contigs

    def contig_length(self, contig: str) -> int:
        """Return contig length in bases."""
        return len(self._contigs[contig])

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
            If the contig does not exist.
        ValueError
            If coordinates are out of bounds.
        """
        if start < 1 or end < start or end > self.contig_length(contig):
            raise ValueError(
                f"Invalid interval on {contig}: start={start}, end={end}, "
                f"length={self.contig_length(contig)}"
            )
        return self._contigs[contig][start - 1 : end].decode("ascii")

    def contig_sort_key(self, contig: str) -> int:
        """Return stable contig order rank."""
        return self._contig_rank[contig]

    def search_queries(
        self,
        queries: Iterable[str],
    ) -> dict[str, tuple[WindowHit, ...]]:
        """Search exact query windows on both genomic strands.

        Parameters
        ----------
        queries : Iterable[str]
            Query sequences in transcript orientation.

        Returns
        -------
        dict[str, tuple[WindowHit, ...]]
            Exact matches for each unique query.
        """
        normalized_queries = sorted({query.upper() for query in queries if query})
        results: dict[str, list[WindowHit]] = {
            query: [] for query in normalized_queries
        }
        if not normalized_queries:
            return {}

        seed_map: dict[int, list[SearchVariant]] = {}
        fallback_variants: list[SearchVariant] = []
        for query in normalized_queries:
            for strand, search_seq_text in (
                ("+", query),
                ("-", reverse_complement(query)),
            ):
                search_seq = search_seq_text.encode("ascii")
                variant = self._build_variant(
                    query_seq=query,
                    search_seq=search_seq,
                    strand=strand,
                )
                if variant.seed_code is None:
                    fallback_variants.append(variant)
                else:
                    seed_map.setdefault(variant.seed_code, []).append(variant)

        total_contigs = len(self._contig_order)
        for index, contig in enumerate(self._contig_order, start=1):
            contig_seq = self._contigs[contig]
            self._scan_contig(
                contig=contig,
                contig_seq=contig_seq,
                seed_map=seed_map,
                fallback_variants=fallback_variants,
                results=results,
            )
            if self._progress_reporter is not None:
                self._progress_reporter.update(
                    label="search windows",
                    current=index,
                    total=total_contigs,
                    detail=contig,
                )

        deduped_results: dict[str, tuple[WindowHit, ...]] = {}
        for query, hits in results.items():
            unique_hits = {
                (hit.contig, hit.start, hit.end, hit.strand): hit for hit in hits
            }
            ordered_hits = tuple(
                unique_hits[key]
                for key in sorted(
                    unique_hits,
                    key=lambda item: (
                        self.contig_sort_key(item[0]),
                        item[1],
                        item[2],
                        item[3],
                    ),
                )
            )
            deduped_results[query] = ordered_hits
        return deduped_results

    def _scan_contig(
        self,
        contig: str,
        contig_seq: bytes,
        seed_map: dict[int, list[SearchVariant]],
        fallback_variants: Sequence[SearchVariant],
        results: dict[str, list[WindowHit]],
    ) -> None:
        """Scan one contig and collect query hits."""
        rolling_code = 0
        valid_run = 0

        for end_index, base in enumerate(contig_seq):
            encoded = _BASE_TO_BITS.get(base)
            if encoded is None:
                rolling_code = 0
                valid_run = 0
                continue

            rolling_code = ((rolling_code << 2) | encoded) & self._seed_mask
            valid_run += 1
            if valid_run < self._seed_len:
                continue

            for variant in seed_map.get(rolling_code, ()):
                seed_start = end_index - self._seed_len + 1
                window_start = seed_start - variant.seed_offset
                window_end = window_start + len(variant.search_seq)
                if window_start < 0 or window_end > len(contig_seq):
                    continue
                if contig_seq[window_start:window_end] != variant.search_seq:
                    continue
                results[variant.query_seq].append(
                    WindowHit(
                        contig=contig,
                        start=window_start + 1,
                        end=window_end,
                        strand=variant.strand,
                    )
                )

        for variant in fallback_variants:
            search_from = 0
            while True:
                index = contig_seq.find(variant.search_seq, search_from)
                if index < 0:
                    break
                results[variant.query_seq].append(
                    WindowHit(
                        contig=contig,
                        start=index + 1,
                        end=index + len(variant.search_seq),
                        strand=variant.strand,
                    )
                )
                search_from = index + 1

    def _build_variant(
        self,
        query_seq: str,
        search_seq: bytes,
        strand: str,
    ) -> SearchVariant:
        """Build one strand-specific search variant."""
        seed_info = _select_seed_code(search_seq=search_seq, seed_len=self._seed_len)
        if seed_info is None:
            return SearchVariant(
                query_seq=query_seq,
                search_seq=search_seq,
                strand=strand,
                seed_code=None,
                seed_offset=0,
            )
        seed_code, seed_offset = seed_info
        return SearchVariant(
            query_seq=query_seq,
            search_seq=search_seq,
            strand=strand,
            seed_code=seed_code,
            seed_offset=seed_offset,
        )

    def _load_fasta(self, fasta_path: Path) -> dict[str, bytes]:
        """Load all FASTA contigs into memory."""
        contigs: dict[str, bytes] = {}
        current_name: str | None = None
        sequence_parts: list[bytes] = []

        with fasta_path.open("rb") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith(b">"):
                    if current_name is not None:
                        contigs[current_name] = b"".join(sequence_parts).upper()
                    current_name = line[1:].split(maxsplit=1)[0].decode("ascii")
                    if current_name in contigs:
                        raise ValueError(
                            f"Duplicate FASTA contig name: {current_name}"
                        )
                    self._contig_order.append(current_name)
                    sequence_parts = []
                    continue

                if current_name is None:
                    raise ValueError("FASTA sequence line appeared before header")
                sequence_parts.append(line.upper())

        if current_name is not None:
            contigs[current_name] = b"".join(sequence_parts).upper()

        if not contigs:
            raise ValueError(f"No contigs found in FASTA: {fasta_path}")
        if self._progress_reporter is not None:
            self._progress_reporter.log(
                f"Loaded FASTA with {len(contigs)} contigs from {fasta_path.name}"
            )
        return contigs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Annotate raw 100bp.err and 100bp.neg.err rows with genomic "
            "splice-site and intron coordinates."
        )
    )
    parser.add_argument(
        "--species",
        default="",
        help=(
            "Comma-separated species list. When omitted, species are discovered "
            "from data/<species>/raw."
        ),
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Data root containing <species>/raw.",
    )
    parser.add_argument(
        "--output-subdir",
        default="processed",
        help="Output subdirectory under each species directory.",
    )
    parser.add_argument(
        "--pos-input-name",
        default="100bp.err",
        help="Positive input filename under raw.",
    )
    parser.add_argument(
        "--neg-input-name",
        default="100bp.neg.err",
        help="Negative input filename under raw.",
    )
    parser.add_argument(
        "--seed-len",
        type=int,
        default=16,
        help="Exact seed length used for search acceleration.",
    )
    parser.add_argument(
        "--no-progress",
        dest="show_progress",
        action="store_false",
        default=True,
        help="Disable progress-bar style terminal updates.",
    )
    args = parser.parse_args(argv)
    if args.seed_len <= 0:
        raise ValueError("--seed-len must be > 0")
    return args


def discover_species(
    data_root: Path,
    pos_input_name: str,
    neg_input_name: str,
) -> list[str]:
    """Discover species that expose raw ERR inputs."""
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    species: list[str] = []
    for species_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        raw_dir = species_dir / "raw"
        if not raw_dir.is_dir():
            continue
        if (raw_dir / pos_input_name).exists() or (raw_dir / neg_input_name).exists():
            species.append(species_dir.name)
    return species


def process_species(
    species: str,
    data_root: Path,
    output_subdir: str,
    pos_input_name: str,
    neg_input_name: str,
    seed_len: int,
    show_progress: bool = False,
) -> tuple[Path, Path]:
    """Process one species and write positive/negative coordinate tables.

    Parameters
    ----------
    species : str
        Species name.
    data_root : Path
        Data root directory.
    output_subdir : str
        Output subdirectory name under the species directory.
    pos_input_name : str
        Positive input filename.
    neg_input_name : str
        Negative input filename.
    seed_len : int
        Search seed length.

    Returns
    -------
    tuple[Path, Path]
        Positive and negative output paths.
    """
    raw_dir = data_root / species / "raw"
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw directory not found for {species}: {raw_dir}")

    pos_path = raw_dir / pos_input_name
    neg_path = raw_dir / neg_input_name
    if not pos_path.exists():
        raise FileNotFoundError(f"Positive input not found: {pos_path}")
    if not neg_path.exists():
        raise FileNotFoundError(f"Negative input not found: {neg_path}")

    output_dir = data_root / species / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    reporter = ProgressReporter(species=species, enabled=show_progress)
    fasta_path, gtf_path = _find_species_reference_files(raw_dir)
    reporter.log(f"Loading FASTA and GTF from {raw_dir}")
    search_index = InMemoryFastaSearchIndex(
        fasta_path=fasta_path,
        seed_len=seed_len,
        progress_reporter=reporter if show_progress else None,
    )
    transcripts = _parse_gtf_transcripts(gtf_path)
    reporter.log(f"Building positive intron candidate index from {gtf_path.name}")
    candidate_map, _ = _build_intron_candidate_index(
        transcripts=transcripts,
        fasta_reader=search_index,
    )

    reporter.log("Reading raw ERR inputs")
    positive_records = _read_positive_records(pos_path)
    negative_records = _read_negative_records(neg_path)
    negative_queries = _collect_negative_queries(negative_records)
    reporter.log(
        "Searching negative donor/acceptor windows "
        f"for {len(negative_queries)} unique queries"
    )
    negative_hits = search_index.search_queries(negative_queries)
    reporter.finish()

    reporter.log("Annotating positive rows")
    positive_rows = _build_positive_rows(
        species=species,
        records=positive_records,
        candidate_map=candidate_map,
        search_index=search_index,
        reporter=reporter,
    )
    reporter.finish()
    reporter.log("Annotating negative rows")
    negative_rows = _build_negative_rows(
        species=species,
        records=negative_records,
        query_hits=negative_hits,
        search_index=search_index,
        reporter=reporter,
    )
    reporter.finish()

    pos_output_path = output_dir / f"{pos_input_name}.coords.tsv"
    neg_output_path = output_dir / f"{neg_input_name}.coords.tsv"
    reporter.log(f"Writing outputs to {output_dir}")
    _write_output_rows(pos_output_path, positive_rows)
    _write_output_rows(neg_output_path, negative_rows)
    reporter.log("Done")
    return pos_output_path, neg_output_path


def main(argv: list[str] | None = None) -> int:
    """Run the coordinate annotation CLI."""
    args = parse_args(argv)
    data_root = Path(args.data_root)
    if args.species.strip():
        species_list = [
            token.strip() for token in args.species.split(",") if token.strip()
        ]
    else:
        species_list = discover_species(
            data_root=data_root,
            pos_input_name=args.pos_input_name,
            neg_input_name=args.neg_input_name,
        )
    if not species_list:
        raise ValueError("No species found to process")

    for species in species_list:
        process_species(
            species=species,
            data_root=data_root,
            output_subdir=args.output_subdir,
            pos_input_name=args.pos_input_name,
            neg_input_name=args.neg_input_name,
            seed_len=args.seed_len,
            show_progress=args.show_progress,
        )
    return 0


def _build_positive_rows(
    species: str,
    records: Sequence[RawErrRecord],
    candidate_map: dict[tuple[str, str, str, str], list[IntronCandidate]],
    search_index: InMemoryFastaSearchIndex,
    reporter: ProgressReporter,
) -> list[dict[str, str]]:
    """Build all positive output rows with progress updates."""
    rows: list[dict[str, str]] = []
    total = len(records)
    for index, record in enumerate(records, start=1):
        rows.append(
            _build_positive_output_row(
                species=species,
                record=record,
                candidates=candidate_map.get(
                    (
                        _require_str(record.transcript_id, "transcript_id"),
                        record.strand,
                        _require_str(record.donor_seq_100bp, "donor_seq_100bp"),
                        _require_str(
                            record.acceptor_seq_100bp,
                            "acceptor_seq_100bp",
                        ),
                    ),
                    [],
                ),
                search_index=search_index,
            )
        )
        _maybe_update_record_progress(
            reporter=reporter,
            label="annotate positive",
            current=index,
            total=total,
        )
    return rows


def _build_negative_rows(
    species: str,
    records: Sequence[RawErrRecord],
    query_hits: dict[str, tuple[WindowHit, ...]],
    search_index: InMemoryFastaSearchIndex,
    reporter: ProgressReporter,
) -> list[dict[str, str]]:
    """Build all negative output rows with progress updates."""
    rows: list[dict[str, str]] = []
    total = len(records)
    for index, record in enumerate(records, start=1):
        rows.append(
            _build_negative_output_row(
                species=species,
                record=record,
                query_hits=query_hits,
                search_index=search_index,
            )
        )
        _maybe_update_record_progress(
            reporter=reporter,
            label="annotate negative",
            current=index,
            total=total,
        )
    return rows


def _read_positive_records(path: Path) -> list[RawErrRecord]:
    """Read positive raw ERR rows."""
    records: list[RawErrRecord] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            match = _POSITIVE_PATTERN.match(line)
            if match is None:
                raise ValueError(f"Invalid positive record at {path}:{line_no}: {line}")
            donor_seq = _normalize_query_seq(
                seq=match.group(1),
                source_path=path,
                line_no=line_no,
                expected_length=100,
            )
            acceptor_seq = _normalize_query_seq(
                seq=match.group(2),
                source_path=path,
                line_no=line_no,
                expected_length=100,
            )
            records.append(
                RawErrRecord(
                    source_file=path.name,
                    source_line_no=line_no,
                    raw_line=line,
                    record_kind="positive_pair",
                    strand=match.group(3),
                    donor_seq_100bp=donor_seq,
                    acceptor_seq_100bp=acceptor_seq,
                    transcript_id=match.group(4),
                    intron_half_length=int(match.group(5)),
                )
            )
    return records


def _read_negative_records(path: Path) -> list[RawErrRecord]:
    """Read negative raw ERR rows."""
    records: list[RawErrRecord] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            donor_match = _NEGATIVE_DONOR_PATTERN.match(line)
            if donor_match is not None:
                records.append(
                    RawErrRecord(
                        source_file=path.name,
                        source_line_no=line_no,
                        raw_line=line,
                        record_kind="negative_donor",
                        strand=donor_match.group(2),
                        donor_seq_100bp=_normalize_query_seq(
                            seq=donor_match.group(1),
                            source_path=path,
                            line_no=line_no,
                            expected_length=None,
                        ),
                        acceptor_seq_100bp=None,
                        transcript_id=None,
                        intron_half_length=None,
                    )
                )
                continue

            acceptor_match = _NEGATIVE_ACCEPTOR_PATTERN.match(line)
            if acceptor_match is not None:
                records.append(
                    RawErrRecord(
                        source_file=path.name,
                        source_line_no=line_no,
                        raw_line=line,
                        record_kind="negative_acceptor",
                        strand=acceptor_match.group(2),
                        donor_seq_100bp=None,
                        acceptor_seq_100bp=_normalize_query_seq(
                            seq=acceptor_match.group(1),
                            source_path=path,
                            line_no=line_no,
                            expected_length=None,
                        ),
                        transcript_id=None,
                        intron_half_length=None,
                    )
                )
                continue

            pair_match = _NEGATIVE_PAIR_PATTERN.match(line)
            if pair_match is not None:
                records.append(
                    RawErrRecord(
                        source_file=path.name,
                        source_line_no=line_no,
                        raw_line=line,
                        record_kind="negative_pair",
                        strand=pair_match.group(3),
                        donor_seq_100bp=_normalize_query_seq(
                            seq=pair_match.group(1),
                            source_path=path,
                            line_no=line_no,
                            expected_length=None,
                        ),
                        acceptor_seq_100bp=_normalize_query_seq(
                            seq=pair_match.group(2),
                            source_path=path,
                            line_no=line_no,
                            expected_length=None,
                        ),
                        transcript_id=None,
                        intron_half_length=int(pair_match.group(4)),
                    )
                )
                continue

            raise ValueError(f"Invalid negative record at {path}:{line_no}: {line}")
    return records


def _normalize_query_seq(
    seq: str,
    source_path: Path,
    line_no: int,
    expected_length: int | None,
) -> str:
    """Normalize and validate one query sequence."""
    normalized = seq.upper()
    if len(normalized) == 0:
        raise ValueError(
            f"Expected non-empty query sequence at {source_path}:{line_no}"
        )
    if expected_length is not None and len(normalized) != expected_length:
        raise ValueError(
            f"Expected {expected_length}bp query sequence at "
            f"{source_path}:{line_no}, got length={len(normalized)}"
        )
    return normalized


def _collect_negative_queries(records: Iterable[RawErrRecord]) -> list[str]:
    """Collect unique donor/acceptor queries from negative records."""
    queries: set[str] = set()
    for record in records:
        if record.donor_seq_100bp is not None:
            queries.add(record.donor_seq_100bp)
        if record.acceptor_seq_100bp is not None:
            queries.add(record.acceptor_seq_100bp)
    return sorted(queries)


def _build_positive_output_row(
    species: str,
    record: RawErrRecord,
    candidates: Sequence[IntronCandidate],
    search_index: InMemoryFastaSearchIndex,
) -> dict[str, str]:
    """Build one positive output row."""
    introns = _sorted_unique_introns(
        [
            IntronHit(
                contig=candidate.chrom,
                start=candidate.intron_start,
                end=candidate.intron_end,
                strand=candidate.strand,
            )
            for candidate in candidates
        ],
        search_index=search_index,
    )

    _ = species
    row = _base_output_row(record=record)
    row.update(
        {
            "hit_count": str(len(introns)),
            "hit_coords": _format_intron_positions(introns),
        }
    )
    return row


def _build_negative_output_row(
    species: str,
    record: RawErrRecord,
    query_hits: dict[str, tuple[WindowHit, ...]],
    search_index: InMemoryFastaSearchIndex,
) -> dict[str, str]:
    """Build one negative output row."""
    _ = species
    row = _base_output_row(record=record)
    if record.record_kind == "negative_donor":
        donor_hits = _filter_hits_by_strand(
            hits=query_hits.get(
                _require_str(record.donor_seq_100bp, "donor_seq_100bp"),
                (),
            ),
            strand=record.strand,
        )
        site_hits = _sorted_unique_sites(
            [
                _site_from_donor_window(
                    hit=hit,
                    window_length=len(
                        _require_str(record.donor_seq_100bp, "donor_seq_100bp")
                    ),
                )
                for hit in donor_hits
            ],
            search_index=search_index,
        )
        row.update(
            {
                "hit_count": str(len(site_hits)),
                "hit_coords": _format_site_positions(site_hits),
            }
        )
        return row

    if record.record_kind == "negative_acceptor":
        acceptor_hits = _filter_hits_by_strand(
            hits=query_hits.get(
                _require_str(record.acceptor_seq_100bp, "acceptor_seq_100bp"),
                (),
            ),
            strand=record.strand,
        )
        site_hits = _sorted_unique_sites(
            [
                _site_from_acceptor_window(
                    hit=hit,
                    window_length=len(
                        _require_str(
                            record.acceptor_seq_100bp,
                            "acceptor_seq_100bp",
                        )
                    ),
                )
                for hit in acceptor_hits
            ],
            search_index=search_index,
        )
        row.update(
            {
                "hit_count": str(len(site_hits)),
                "hit_coords": _format_site_positions(site_hits),
            }
        )
        return row

    donor_hits = _filter_hits_by_strand(
        hits=query_hits.get(
            _require_str(record.donor_seq_100bp, "donor_seq_100bp"),
            (),
        ),
        strand=record.strand,
    )
    acceptor_hits = _filter_hits_by_strand(
        hits=query_hits.get(
            _require_str(record.acceptor_seq_100bp, "acceptor_seq_100bp"),
            (),
        ),
        strand=record.strand,
    )
    donor_sites = _sorted_unique_sites(
        [
            _site_from_donor_window(
                hit=hit,
                window_length=len(
                    _require_str(record.donor_seq_100bp, "donor_seq_100bp")
                ),
            )
            for hit in donor_hits
        ],
        search_index=search_index,
    )
    acceptor_sites = _sorted_unique_sites(
        [
            _site_from_acceptor_window(
                hit=hit,
                window_length=len(
                    _require_str(record.acceptor_seq_100bp, "acceptor_seq_100bp")
                ),
            )
            for hit in acceptor_hits
        ],
        search_index=search_index,
    )
    pair_hits = _pair_negative_hits(
        donor_sites=donor_sites,
        acceptor_sites=acceptor_sites,
        strand=record.strand,
        intron_half_length=_require_int(
            record.intron_half_length,
            "intron_half_length",
        ),
        search_index=search_index,
    )
    row.update(
        {
            "hit_count": str(len(pair_hits)),
            "hit_coords": _format_intron_positions(pair_hits),
        }
    )
    return row


def _pair_negative_hits(
    donor_sites: Sequence[SiteHit],
    acceptor_sites: Sequence[SiteHit],
    strand: str,
    intron_half_length: int,
    search_index: InMemoryFastaSearchIndex,
) -> tuple[IntronHit, ...]:
    """Pair donor and acceptor hits into intron candidates."""
    introns: list[IntronHit] = []
    for donor_site in donor_sites:
        for acceptor_site in acceptor_sites:
            if donor_site.contig != acceptor_site.contig:
                continue
            if donor_site.strand != acceptor_site.strand:
                continue
            if donor_site.strand != strand:
                continue

            if strand == "+":
                if acceptor_site.position <= donor_site.position:
                    continue
                intron = IntronHit(
                    contig=donor_site.contig,
                    start=donor_site.position,
                    end=acceptor_site.position - 1,
                    strand=strand,
                )
            else:
                if donor_site.position <= acceptor_site.position:
                    continue
                intron = IntronHit(
                    contig=donor_site.contig,
                    start=acceptor_site.position + 1,
                    end=donor_site.position,
                    strand=strand,
                )

            intron_length = intron.end - intron.start + 1
            if intron_length <= 0:
                continue
            if intron_length // 2 != intron_half_length:
                continue
            introns.append(intron)
    return _sorted_unique_introns(introns, search_index=search_index)


def _site_from_donor_window(hit: WindowHit, window_length: int) -> SiteHit:
    """Recover donor splice-site coordinate from one donor window hit."""
    if hit.strand == "+":
        position = hit.start + 3
    else:
        position = hit.start + window_length - 4
    return SiteHit(contig=hit.contig, position=position, strand=hit.strand)


def _site_from_acceptor_window(hit: WindowHit, window_length: int) -> SiteHit:
    """Recover acceptor splice-site coordinate from one acceptor window hit."""
    if hit.strand == "+":
        position = hit.start + window_length - 3
    else:
        position = hit.start + 2
    return SiteHit(contig=hit.contig, position=position, strand=hit.strand)


def _filter_hits_by_strand(
    hits: Sequence[WindowHit],
    strand: str,
) -> tuple[WindowHit, ...]:
    """Keep only hits that match the requested strand."""
    return tuple(hit for hit in hits if hit.strand == strand)


def _sorted_unique_sites(
    hits: Iterable[SiteHit],
    search_index: InMemoryFastaSearchIndex,
) -> tuple[SiteHit, ...]:
    """Sort and deduplicate site coordinates."""
    unique = {(hit.contig, hit.position, hit.strand): hit for hit in hits}
    return tuple(
        unique[key]
        for key in sorted(
            unique,
            key=lambda item: (
                search_index.contig_sort_key(item[0]),
                item[1],
                item[2],
            ),
        )
    )


def _sorted_unique_introns(
    hits: Iterable[IntronHit],
    search_index: InMemoryFastaSearchIndex,
) -> tuple[IntronHit, ...]:
    """Sort and deduplicate intron coordinates."""
    unique = {(hit.contig, hit.start, hit.end, hit.strand): hit for hit in hits}
    return tuple(
        unique[key]
        for key in sorted(
            unique,
            key=lambda item: (
                search_index.contig_sort_key(item[0]),
                item[1],
                item[2],
                item[3],
            ),
        )
    )


def _format_site_positions(hits: Sequence[SiteHit]) -> str:
    """Format site positions for compact TSV output."""
    return ";".join(str(hit.position) for hit in hits)


def _format_intron_positions(hits: Sequence[IntronHit]) -> str:
    """Format intron boundary pairs for compact TSV output."""
    return ";".join(f"{hit.start} {hit.end}" for hit in hits)


def _maybe_update_record_progress(
    reporter: ProgressReporter,
    label: str,
    current: int,
    total: int,
) -> None:
    """Update record-level progress at a bounded refresh rate."""
    if not reporter.enabled or total <= 0:
        return
    step = max(1, total // 200)
    if current == 1 or current == total or current % step == 0:
        reporter.update(label=label, current=current, total=total)


def _base_output_row(record: RawErrRecord) -> dict[str, str]:
    """Create the shared compact output row payload."""
    return {
        "donor_seq": record.donor_seq_100bp or "",
        "acceptor_seq": record.acceptor_seq_100bp or "",
        "hit_count": "",
        "hit_coords": "",
    }


def _write_output_rows(path: Path, rows: Sequence[dict[str, str]]) -> None:
    """Write output rows as TSV."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=_DEFAULT_OUTPUT_COLUMNS,
            delimiter="\t",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _select_seed_code(search_seq: bytes, seed_len: int) -> tuple[int, int] | None:
    """Select the first all-ACGT seed and return its code and offset."""
    limit = len(search_seq) - seed_len + 1
    if limit < 1:
        return None
    for offset in range(limit):
        seed_bytes = search_seq[offset : offset + seed_len]
        code = _encode_acgt_seed(seed_bytes)
        if code is not None:
            return code, offset
    return None


def _encode_acgt_seed(seed_bytes: bytes) -> int | None:
    """Encode one seed into a compact 2-bit integer."""
    code = 0
    for base in seed_bytes:
        encoded = _BASE_TO_BITS.get(base)
        if encoded is None:
            return None
        code = (code << 2) | encoded
    return code


def _require_value(value: str | int | None, name: str) -> str | int:
    """Assert that an optional value is present."""
    if value is None:
        raise ValueError(f"Required value is missing: {name}")
    return value


def _require_str(value: str | None, name: str) -> str:
    """Assert that an optional string value is present."""
    result = _require_value(value, name)
    if not isinstance(result, str):
        raise TypeError(f"Expected string value for {name}, got {type(result)!r}")
    return result


def _require_int(value: int | None, name: str) -> int:
    """Assert that an optional integer value is present."""
    result = _require_value(value, name)
    if not isinstance(result, int):
        raise TypeError(f"Expected integer value for {name}, got {type(result)!r}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
