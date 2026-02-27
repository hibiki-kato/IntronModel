"""Shared data processing utilities for site-level model workflows.

This module centralizes:
- species-based data path resolution
- training/test file discovery
- donor/acceptor window validation and reshaping
- parsing of training and test sequence records
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Dict, List, Literal, Optional, Sequence, Tuple

NAME_FIELD_CHOICES: tuple[str, ...] = (
    "bp_avg",
    "bp_window",
    "donor_len",
    "acceptor_len",
    "epochs",
    "batch_size",
    "lr",
    "conv_channels",
    "kernel_size",
    "dropout",
    "fc_hidden",
    "weight_decay",
    "eta_min_ratio",
    "val_frac",
    "grad_clip",
    "loss",
    "pos_weight_cap",
    "lightweight",
    "compile",
    "intron_score_op",
    "transcript_score_agg",
    "softmin_tau",
    "seed",
    "focal_gamma",
    "focal_alpha_pos",
    "asym_gamma_pos",
    "asym_gamma_neg",
    "asym_alpha_pos",
    "tag",
)
NAME_FIELD_LABELS: dict[str, str] = {
    "donor_len": "dlen",
    "acceptor_len": "alen",
    "epochs": "ep",
    "batch_size": "bs",
    "lr": "lr",
    "conv_channels": "ch",
    "kernel_size": "ks",
    "dropout": "do",
    "fc_hidden": "fch",
    "weight_decay": "wd",
    "eta_min_ratio": "emr",
    "val_frac": "vf",
    "grad_clip": "gc",
    "loss": "loss",
    "pos_weight_cap": "pwc",
    "lightweight": "lw",
    "compile": "comp",
    "intron_score_op": "iop",
    "transcript_score_agg": "tagg",
    "softmin_tau": "stau",
    "seed": "seed",
    "focal_gamma": "fg",
    "focal_alpha_pos": "fa",
    "asym_gamma_pos": "agp",
    "asym_gamma_neg": "agn",
    "asym_alpha_pos": "aap",
    "tag": "tag",
}

TrainingRecordType = Literal["donor", "acceptor", "pair"]


@dataclass(frozen=True)
class ParsedTrainingRecord:
    """Structured result from one ``DEBUG`` training-data line.

    Attributes
    ----------
    record_type : {"donor", "acceptor", "pair"}
        Record category parsed from the line.
    donor_seq : str | None
        Donor-side sequence for donor/pair records.
    acceptor_seq : str | None
        Acceptor-side sequence for acceptor/pair records.
    strand : str | None
        Strand sign when present (``+`` or ``-``).
    transcript_id : str | None
        Transcript identifier from new positive pair format.
    intron_half_length : int | None
        Half intron length from new positive/negative pair formats.
    """

    record_type: TrainingRecordType
    donor_seq: str | None
    acceptor_seq: str | None
    strand: str | None
    transcript_id: str | None
    intron_half_length: int | None


@dataclass(frozen=True)
class SiteTrainingExample:
    """One task-specific training example with optional parsed metadata.

    Attributes
    ----------
    sequence : str
        Task-ready sequence after donor/acceptor reshaping.
    label : int
        Binary class label (1: positive, 0: negative).
    transcript_id : str | None
        Transcript identifier when available from source line.
    intron_half_length : int | None
        Half intron length when available from source line.
    source_record_type : {"donor", "acceptor", "pair"}
        Original record category before task filtering.
    strand : str | None
        Strand sign from source line when present.
    """

    sequence: str
    label: int
    transcript_id: str | None
    intron_half_length: int | None
    source_record_type: TrainingRecordType
    strand: str | None


def project_root() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))


def data_root() -> str:
    """Return data root directory, overridable by environment variable."""
    root = os.environ.get("INTRONMODEL_DATA_ROOT")
    if root is None or root.strip() == "":
        return os.path.join(project_root(), "data")
    if os.path.isabs(root):
        return root
    return os.path.normpath(os.path.join(project_root(), root))


def model_root() -> str:
    """Return model root directory, overridable by environment variable."""
    root = os.environ.get("INTRONMODEL_MODEL_ROOT")
    if root is None or root.strip() == "":
        return os.path.join(project_root(), "model")
    if os.path.isabs(root):
        return root
    return os.path.normpath(os.path.join(project_root(), root))


def species_data_dirs(species: str) -> Dict[str, str]:
    """Return canonical data directories for a species.

    Parameters
    ----------
    species : str
        Species folder name under ``data/``.

    Returns
    -------
    dict[str, str]
        Paths for base/raw/train/trans_score/site_score/eval_score directories.
    """
    base = os.path.join(data_root(), species)
    return {
        "base": base,
        "raw": os.path.join(base, "raw"),
        "train": os.path.join(base, "train"),
        "trans_score": os.path.join(base, "trans_score"),
        "site_score": os.path.join(base, "site_score"),
        "eval_score": os.path.join(base, "eval_score"),
    }


def parse_name_fields(name_fields: Optional[str]) -> List[str]:
    """Parse comma-separated output naming fields.

    Parameters
    ----------
    name_fields : str | None
        Comma-separated field names. ``None`` and empty string mean default.
        The special value ``none`` means no extra suffix.

    Returns
    -------
    list[str]
        Parsed field list. Default is an empty list.

    Raises
    ------
    ValueError
        If unknown field names are provided.
    """
    if name_fields in (None, ""):
        return []
    raw_fields = [
        field.strip()
        for field in str(name_fields).split(",")
        if field.strip()
    ]
    if not raw_fields:
        return []
    if len(raw_fields) == 1 and raw_fields[0] == "none":
        return []
    unknown = [field for field in raw_fields if field not in NAME_FIELD_CHOICES]
    if unknown:
        known = ", ".join(NAME_FIELD_CHOICES)
        unknown_text = ", ".join(unknown)
        raise ValueError(
            f"Unknown --name_fields value: {unknown_text}. Supported: {known}, none"
        )
    return raw_fields


def _format_name_value(value: object) -> str:
    """Format and sanitize a name token value."""
    if isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    text = text.replace("+", "plus")
    text = text.replace("*", "x")
    text = text.replace("-", "m")
    text = text.replace(".", "p")
    text = re.sub(r"[^A-Za-z0-9_]", "", text)
    return text


def _average_bp_label(
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    fallback_train_len: Optional[int],
) -> Optional[str]:
    """Return average bp label like ``100bp`` from donor/acceptor lengths."""
    if donor_len is not None and acceptor_len is not None:
        avg = (donor_len + acceptor_len) / 2.0
    elif donor_len is not None:
        avg = float(donor_len)
    elif acceptor_len is not None:
        avg = float(acceptor_len)
    elif fallback_train_len is not None:
        avg = float(fallback_train_len)
    else:
        return None
    avg_int = int(round(avg))
    return f"{avg_int}bp"


def build_output_stem(
    model_name: str,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    fallback_train_len: Optional[int],
    name_fields: Sequence[str],
    name_params: Dict[str, object],
) -> str:
    """Build output filename stem from selected naming fields.

    Parameters
    ----------
    model_name : str
        Model identifier.
    donor_len : int | None
        Effective donor window length.
    acceptor_len : int | None
        Effective acceptor window length.
    fallback_train_len : int | None
        Fallback length inferred from training files.
    name_fields : Sequence[str]
        Selected fields from ``NAME_FIELD_CHOICES``.
    name_params : dict[str, object]
        Runtime parameter mapping (usually ``vars(args)``).

    Returns
    -------
    str
        Output file stem without extension.
    """
    if not name_fields:
        return model_name

    pieces: List[str] = []
    for field in name_fields:
        if field == "bp_avg":
            avg_label = _average_bp_label(
                donor_len=donor_len,
                acceptor_len=acceptor_len,
                fallback_train_len=fallback_train_len,
            )
            if avg_label:
                pieces.append(avg_label)
            continue

        if field == "bp_window":
            w_suffix = window_suffix(
                donor_len=donor_len,
                acceptor_len=acceptor_len,
                fallback_train_len=fallback_train_len,
            )
            if w_suffix != "default":
                pieces.append(w_suffix)
            continue

        value = name_params.get(field)
        if value is None:
            continue
        label = NAME_FIELD_LABELS.get(field, field)
        pieces.append(f"{label}{_format_name_value(value)}")

    if not pieces:
        return model_name

    first = pieces[0]
    if first.endswith("bp"):
        return f"{model_name}{first}" + "".join(f"_{p}" for p in pieces[1:])
    return f"{model_name}_{'_'.join(pieces)}"


def window_suffix(
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    fallback_train_len: Optional[int] = None,
) -> str:
    """Build a compact window-length suffix for output names.

    Parameters
    ----------
    donor_len : int | None
        Donor window length.
    acceptor_len : int | None
        Acceptor window length.
    fallback_train_len : int | None, default=None
        Fallback training length when explicit lengths are omitted.

    Returns
    -------
    str
        Suffix such as ``100bp``, ``d80bp_a120bp``, or ``default``.
    """
    if donor_len is not None and acceptor_len is not None:
        if donor_len == acceptor_len:
            return f"{donor_len}bp"
        return f"d{donor_len}bp_a{acceptor_len}bp"
    if donor_len is not None:
        return f"d{donor_len}bp"
    if acceptor_len is not None:
        return f"a{acceptor_len}bp"
    if fallback_train_len is not None:
        return f"{fallback_train_len}bp"
    return "default"


def default_site_output_path(
    species: str,
    model_name: str,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    fallback_train_len: Optional[int] = None,
    name_fields: Optional[Sequence[str]] = None,
    name_params: Optional[Dict[str, object]] = None,
) -> str:
    """Return default site-score TSV output path."""
    dirs = species_data_dirs(species)
    fields = list(name_fields) if name_fields is not None else []
    params = name_params or {}
    stem = build_output_stem(
        model_name=model_name,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        fallback_train_len=fallback_train_len,
        name_fields=fields,
        name_params=params,
    )
    return os.path.join(dirs["site_score"], f"{stem}.tsv")


def default_transcript_output_path(
    species: str,
    model_name: str,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    fallback_train_len: Optional[int] = None,
    name_fields: Optional[Sequence[str]] = None,
    name_params: Optional[Dict[str, object]] = None,
) -> str:
    """Return default transcript-score TSV output path."""
    dirs = species_data_dirs(species)
    fields = list(name_fields) if name_fields is not None else []
    params = name_params or {}
    stem = build_output_stem(
        model_name=model_name,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        fallback_train_len=fallback_train_len,
        name_fields=fields,
        name_params=params,
    )
    return os.path.join(dirs["trans_score"], f"{stem}.tsv")


def build_run_name(
    model_name: str,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    lr: float,
    batch_size: int,
    epochs: int,
    tag: Optional[str] = None,
) -> str:
    """Build a stable run-name token from key hyperparameters."""
    suffix = window_suffix(donor_len=donor_len, acceptor_len=acceptor_len)
    base = f"{model_name}_{suffix}_lr{lr:g}_bs{batch_size}_ep{epochs}"
    if tag:
        return f"{base}_{tag}"
    return base


def infer_default_model_dir(species: str, task: str, model_name: str) -> str:
    """Return strict default model directory for a species/task/model."""
    return os.path.join(model_root(), species, task, model_name)


def validate_window_args(
    donor_len: Optional[int],
    acceptor_len: Optional[int],
):
    if donor_len is not None:
        if donor_len <= 0:
            raise ValueError("--donor_len must be > 0")

    if acceptor_len is not None:
        if acceptor_len <= 0:
            raise ValueError("--acceptor_len must be > 0")


def detect_raw_reference_files(raw_dir: str) -> Dict[str, Optional[str]]:
    """Detect single reference files in raw dir by extension family."""
    out: Dict[str, Optional[str]] = {"fna": None, "gff": None, "gtf": None}
    if not os.path.isdir(raw_dir):
        return out

    files = [
        os.path.join(raw_dir, name)
        for name in os.listdir(raw_dir)
        if os.path.isfile(os.path.join(raw_dir, name))
    ]

    fna = [p for p in files if p.endswith(".fna")]
    gff = [
        p
        for p in files
        if p.endswith(".gff") or p.endswith(".gff3") or ".gff." in os.path.basename(p)
    ]
    gtf = [p for p in files if p.endswith(".gtf")]

    out["fna"] = fna[0] if len(fna) == 1 else None
    out["gff"] = gff[0] if len(gff) == 1 else None
    out["gtf"] = gtf[0] if len(gtf) == 1 else None
    return out


def list_available_train_lengths(train_dir: str) -> List[int]:
    if not os.path.isdir(train_dir):
        return []

    pos_lengths = set()
    neg_lengths = set()
    for name in os.listdir(train_dir):
        m_pos = re.match(r"^(\d+)bp\.err$", name)
        if m_pos:
            pos_lengths.add(int(m_pos.group(1)))
            continue
        m_neg = re.match(r"^(\d+)bp\.neg\.err$", name)
        if m_neg:
            neg_lengths.add(int(m_neg.group(1)))

    return sorted(pos_lengths & neg_lengths)


def infer_default_train_paths(
    train_dir: str,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
) -> Tuple[str, str, int]:
    available = list_available_train_lengths(train_dir)
    if not available:
        raise ValueError(
            "No paired training files found in "
            f"{train_dir}. Expected <N>bp.err and <N>bp.neg.err."
        )

    requested = [x for x in [donor_len, acceptor_len] if x is not None]
    if requested:
        required_len = max(requested)
        candidates = [x for x in available if x >= required_len]
        if not candidates:
            raise ValueError(
                "No training length >= requested window "
                f"({required_len}) in {train_dir}. "
                f"Available lengths: {available}"
            )
        chosen = min(candidates)
    else:
        chosen = max(available)

    pos = os.path.join(train_dir, f"{chosen}bp.err")
    neg = os.path.join(train_dir, f"{chosen}bp.neg.err")
    return pos, neg, chosen


def resolve_train_paths(
    species: str,
    train_pos_path: Optional[str],
    train_neg_path: Optional[str],
    donor_len: Optional[int],
    acceptor_len: Optional[int],
) -> Tuple[str, str, Optional[int]]:
    dirs = species_data_dirs(species)
    inferred_train_len: Optional[int] = None
    default_pos = None
    default_neg = None

    if train_pos_path is None or train_neg_path is None:
        default_pos, default_neg, inferred_train_len = infer_default_train_paths(
            train_dir=dirs["raw"], donor_len=donor_len, acceptor_len=acceptor_len
        )

    pos_path = train_pos_path or default_pos
    neg_path = train_neg_path or default_neg

    if pos_path is None or neg_path is None:
        raise ValueError(
            "Could not infer default training file path from species. "
            "Specify --train_pos_path and --train_neg_path."
        )

    return pos_path, neg_path, inferred_train_len


def resolve_test_tsv(species: str, test_tsv: Optional[str]) -> str:
    if test_tsv:
        return test_tsv
    dirs = species_data_dirs(species)
    return os.path.join(dirs["raw"], "transcripts.tsv")


def resolve_effective_window_lengths(
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    inferred_train_len: Optional[int],
) -> Tuple[Optional[int], Optional[int]]:
    effective_donor_len = donor_len
    effective_acceptor_len = acceptor_len

    if inferred_train_len is not None:
        if effective_donor_len is None:
            effective_donor_len = inferred_train_len
        if effective_acceptor_len is None:
            effective_acceptor_len = inferred_train_len

    return effective_donor_len, effective_acceptor_len


def reshape_site_sequence(
    seq: str,
    site_type: str,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
) -> Optional[str]:
    """Reshape site sequence by fixed donor/acceptor window lengths."""
    seq = seq.upper()

    if site_type == "donor":
        if donor_len is None:
            return seq
        if donor_len > len(seq):
            return None
        return seq[:donor_len]

    if site_type == "acceptor":
        if acceptor_len is None:
            return seq
        if acceptor_len > len(seq):
            return None
        return seq[-acceptor_len:]

    return None


def _parse_optional_strand(
    tokens: Sequence[str],
    index: int,
) -> tuple[str | None, int]:
    """Parse optional strand token (``+`` or ``-``) and updated index."""
    if index < len(tokens) and tokens[index] in {"+", "-"}:
        return tokens[index], index + 1
    return None, index


def _parse_required_int(token: str) -> int | None:
    """Parse required integer token.

    Parameters
    ----------
    token : str
        Raw token expected to represent an integer.

    Returns
    -------
    int | None
        Parsed integer value, or ``None`` when parsing fails.
    """
    try:
        return int(token)
    except ValueError:
        return None


def parse_debug_training_record(line: str) -> ParsedTrainingRecord | None:
    """Parse one training line from old/new ``DEBUG`` formats.

    Parameters
    ----------
    line : str
        Raw training-data line.

    Returns
    -------
    ParsedTrainingRecord | None
        Parsed structured record, or ``None`` for unsupported/malformed lines.
    """
    tokens = line.strip().split()
    if len(tokens) < 3 or tokens[0] != "DEBUG":
        return None

    record_type = tokens[1]
    if record_type == "pair":
        if len(tokens) < 4:
            return None
        donor_seq = tokens[2].upper()
        acceptor_seq = tokens[3].upper()
        index = 4
        strand, index = _parse_optional_strand(tokens, index)
        intron_half_length: int | None = None
        if index < len(tokens):
            parsed = _parse_required_int(tokens[index])
            if parsed is None:
                return None
            intron_half_length = parsed
            index += 1
        if index != len(tokens):
            return None
        return ParsedTrainingRecord(
            record_type="pair",
            donor_seq=donor_seq,
            acceptor_seq=acceptor_seq,
            strand=strand,
            transcript_id=None,
            intron_half_length=intron_half_length,
        )

    if record_type == "donor":
        donor_seq = tokens[2].upper()
        if len(tokens) >= 5 and tokens[3] == "acceptor":
            acceptor_seq = tokens[4].upper()
            index = 5
            strand, index = _parse_optional_strand(tokens, index)
            transcript_id: str | None = None
            intron_half_length: int | None = None
            if index < len(tokens):
                if index + 1 >= len(tokens):
                    return None
                transcript_id = tokens[index]
                parsed = _parse_required_int(tokens[index + 1])
                if parsed is None:
                    return None
                intron_half_length = parsed
                index += 2
            if index != len(tokens):
                return None
            return ParsedTrainingRecord(
                record_type="pair",
                donor_seq=donor_seq,
                acceptor_seq=acceptor_seq,
                strand=strand,
                transcript_id=transcript_id,
                intron_half_length=intron_half_length,
            )

        strand, index = _parse_optional_strand(tokens, 3)
        if index != len(tokens):
            return None
        return ParsedTrainingRecord(
            record_type="donor",
            donor_seq=donor_seq,
            acceptor_seq=None,
            strand=strand,
            transcript_id=None,
            intron_half_length=None,
        )

    if record_type == "acceptor":
        acceptor_seq = tokens[2].upper()
        strand, index = _parse_optional_strand(tokens, 3)
        if index != len(tokens):
            return None
        return ParsedTrainingRecord(
            record_type="acceptor",
            donor_seq=None,
            acceptor_seq=acceptor_seq,
            strand=strand,
            transcript_id=None,
            intron_half_length=None,
        )

    return None


def parse_pair_sequences(line: str) -> Optional[Tuple[str, str]]:
    """Parse one pair-record line into donor and acceptor sequences.

    Parameters
    ----------
    line : str
        Raw line from training file.

    Returns
    -------
    tuple[str, str] | None
        ``(donor_seq, acceptor_seq)`` when the line is a supported pair record;
        otherwise ``None``.
    """
    parsed = parse_debug_training_record(line)
    if parsed is None or parsed.record_type != "pair":
        return None
    if parsed.donor_seq is None or parsed.acceptor_seq is None:
        return None
    return parsed.donor_seq, parsed.acceptor_seq


def read_examples_single_task_with_metadata(
    pos_path: str,
    neg_path: str,
    task: str,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
) -> List[SiteTrainingExample]:
    """Read one-task training examples with parsed line metadata.

    Parameters
    ----------
    pos_path : str
        Positive training file path.
    neg_path : str
        Negative training file path.
    task : str
        Target task (``"donor"`` or ``"acceptor"``).
    donor_len : int | None
        Donor window length. ``None`` keeps original donor-side length.
    acceptor_len : int | None
        Acceptor window length. ``None`` keeps original acceptor-side length.

    Returns
    -------
    list[SiteTrainingExample]
        Task-specific examples with optional metadata fields populated.

    Raises
    ------
    ValueError
        If ``task`` is neither ``"donor"`` nor ``"acceptor"``.
    """
    if task not in {"donor", "acceptor"}:
        raise ValueError("task must be either 'donor' or 'acceptor'.")

    examples: List[SiteTrainingExample] = []

    def read_one_set(path: str, label: int) -> None:
        with open(path, "r", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("DEBUG"):
                    continue

                parsed = parse_debug_training_record(line)
                if parsed is None:
                    continue

                if parsed.record_type == "pair":
                    raw_seq = (
                        parsed.donor_seq
                        if task == "donor"
                        else parsed.acceptor_seq
                    )
                elif parsed.record_type == task:
                    raw_seq = (
                        parsed.donor_seq
                        if task == "donor"
                        else parsed.acceptor_seq
                    )
                else:
                    continue

                if raw_seq is None:
                    continue

                reshaped = reshape_site_sequence(
                    raw_seq,
                    task,
                    donor_len=donor_len,
                    acceptor_len=acceptor_len,
                )
                if reshaped is None:
                    continue
                examples.append(
                    SiteTrainingExample(
                        sequence=reshaped,
                        label=label,
                        transcript_id=parsed.transcript_id,
                        intron_half_length=parsed.intron_half_length,
                        source_record_type=parsed.record_type,
                        strand=parsed.strand,
                    )
                )

    read_one_set(pos_path, label=1)
    read_one_set(neg_path, label=0)
    return examples


def read_examples_single_task(
    pos_path: str,
    neg_path: str,
    task: str,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
) -> List[Tuple[str, int]]:
    """Read one-task training examples as ``(sequence, label)`` pairs.

    This compatibility wrapper intentionally drops extra metadata fields.
    """
    examples_with_metadata = read_examples_single_task_with_metadata(
        pos_path=pos_path,
        neg_path=neg_path,
        task=task,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
    )
    return [(item.sequence, item.label) for item in examples_with_metadata]


def read_test_site_rows(
    test_tsv: str,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
) -> Tuple[List[Dict[str, object]], int]:
    rows: List[Dict[str, object]] = []
    skipped_short = 0

    with open(test_tsv, "r") as f:
        _ = next(f, None)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 8:
                continue

            site_type = parts[2]
            reshaped = reshape_site_sequence(
                parts[7],
                site_type,
                donor_len=donor_len,
                acceptor_len=acceptor_len,
            )
            if not reshaped:
                skipped_short += 1
                continue

            rows.append(
                {
                    "transcript_id": parts[0],
                    "site_type": site_type,
                    "intron_index": int(parts[3]),
                    "seq": reshaped,
                }
            )

    return rows, skipped_short
