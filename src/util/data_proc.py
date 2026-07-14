"""Shared data processing utilities for site-level model workflows.

This module centralizes:
- species-based data path resolution
- training/test file discovery
- donor/acceptor window validation and reshaping
- parsing of training and test sequence records
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os
import re
from typing import Dict, List, Literal, Optional, Sequence, Tuple

from util.unique_intron import UNIQUE_MAP_TSV_NAME, load_unique_half_lengths

NAME_FIELD_CHOICES: tuple[str, ...] = (
    "bp_avg",
    "bp_window",
    "donor_len",
    "acceptor_len",
    "donor_upstream",
    "donor_downstream",
    "acceptor_upstream",
    "acceptor_downstream",
    "epochs",
    "batch_size",
    "lr",
    "conv_channels",
    "kernel_sizes",
    "kernel_size",
    "conv_stride",
    "head_type",
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
    "donor_upstream": "dup",
    "donor_downstream": "ddn",
    "acceptor_upstream": "aup",
    "acceptor_downstream": "adn",
    "epochs": "ep",
    "batch_size": "bs",
    "lr": "lr",
    "conv_channels": "ch",
    "kernel_sizes": "kss",
    "kernel_size": "ks",
    "conv_stride": "cst",
    "head_type": "head",
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
TrainingFileSignature = Tuple[str, int, int]
_TRAINING_EXAMPLE_CACHE_MAXSIZE: int = 4
_MIXED_PAIR_NEGATIVE_PATTERN = re.compile(r".*mixed_one_side.*\.neg\.err$")


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
    chrom: str | None
    pos: int | None


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


@dataclass(frozen=True)
class PairTrainingExample:
    """One pair-task training example with optional parsed metadata.

    Attributes
    ----------
    donor_sequence : str
        Donor-side sequence after donor-window reshaping.
    acceptor_sequence : str
        Acceptor-side sequence after acceptor-window reshaping.
    label : int
        Binary class label (1: positive, 0: negative).
    transcript_id : str | None
        Transcript identifier when available from source line.
    intron_half_length : int | None
        Half intron length when available from source line.
    source_record_type : {"pair"}
        Original record category before pair-task filtering.
    strand : str | None
        Strand sign from source line when present.
    """

    donor_sequence: str
    acceptor_sequence: str
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
        Paths for base/raw/processed/train/site_score/intron_score/trans_score/
        learning_metric/eval_score directories.
    """
    base = os.path.join(data_root(), species)
    return {
        "base": base,
        "raw": os.path.join(base, "raw"),
        "processed": os.path.join(base, "processed"),
        "train": os.path.join(base, "train"),
        "trans_score": os.path.join(base, "trans_score"),
        "site_score": os.path.join(base, "site_score"),
        "intron_score": os.path.join(base, "intron_score"),
        "learning_metric": os.path.join(base, "learning_metric"),
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
        field.strip() for field in str(name_fields).split(",") if field.strip()
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


def normalize_tag_name_value(model_name: str, tag_value: object) -> str:
    """Normalize a tag token for file/checkpoint naming.

    Parameters
    ----------
    model_name : str
        Model prefix already present in generated names.
    tag_value : object
        Raw tag value provided by the user.

    Returns
    -------
    str
        Sanitized tag token with a duplicated model-name token removed.
    """
    text = str(tag_value).strip()
    if text == model_name:
        return ""
    prefix = f"{model_name}_"
    if text.startswith(prefix):
        text = text[len(prefix) :]
    return _format_name_value(text)


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
        if field == "tag":
            normalized_tag = normalize_tag_name_value(model_name, value)
            if normalized_tag != "":
                pieces.append(normalized_tag)
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


def default_intron_output_path(
    species: str,
    model_name: str,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    fallback_train_len: Optional[int] = None,
    name_fields: Optional[Sequence[str]] = None,
    name_params: Optional[Dict[str, object]] = None,
) -> str:
    """Return default intron-score TSV output path."""
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
    return os.path.join(dirs["intron_score"], f"{stem}.tsv")


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
    del donor_len, acceptor_len
    base = f"{model_name}_lr{lr:g}_bs{batch_size}_ep{epochs}"
    if tag:
        normalized_tag = normalize_tag_name_value(model_name, tag)
        if normalized_tag != "":
            return f"{base}_{normalized_tag}"
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


def validate_window_args_4p(
    donor_upstream: Optional[int],
    donor_downstream: Optional[int],
    acceptor_upstream: Optional[int],
    acceptor_downstream: Optional[int],
) -> None:
    """Validate the four independent splice-site window parameters."""
    for name, val in [
        ("--donor_upstream", donor_upstream),
        ("--donor_downstream", donor_downstream),
        ("--acceptor_upstream", acceptor_upstream),
        ("--acceptor_downstream", acceptor_downstream),
    ]:
        if val is not None and val <= 0:
            raise ValueError(f"{name} must be > 0")
    if (donor_upstream is None) != (donor_downstream is None):
        raise ValueError(
            "--donor_upstream and --donor_downstream must be set together"
        )
    if (acceptor_upstream is None) != (acceptor_downstream is None):
        raise ValueError(
            "--acceptor_upstream and --acceptor_downstream must be set together"
        )


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


def _required_train_window_length(
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    donor_upstream: Optional[int] = None,
    donor_downstream: Optional[int] = None,
    acceptor_upstream: Optional[int] = None,
    acceptor_downstream: Optional[int] = None,
) -> Optional[int]:
    """Return the largest requested site window length, if one was specified."""
    requested = [length for length in (donor_len, acceptor_len) if length is not None]
    if donor_upstream is not None and donor_downstream is not None:
        requested.append(donor_upstream + donor_downstream)
    if acceptor_upstream is not None and acceptor_downstream is not None:
        requested.append(acceptor_upstream + acceptor_downstream)
    return max(requested) if requested else None


def _first_site_sequence_length(path: str) -> Optional[int]:
    """Return the first DEBUG site-sequence length from one processed ERR file."""
    try:
        with open(path, encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                fields = line.split()
                if len(fields) >= 3 and fields[0] == "DEBUG":
                    return len(fields[2])
    except OSError:
        return None
    return None


def _infer_processed_site_flank_paths(
    processed_dir: str,
    required_len: Optional[int],
) -> Optional[Tuple[str, str, int]]:
    """Resolve generated 100-nt-per-side data when a full 200-nt window is requested."""
    if required_len is None:
        return None
    pos_path = os.path.join(processed_dir, "site_flank100.coding.err")
    neg_path = os.path.join(processed_dir, "site_flank100.neg.err")
    if not (os.path.isfile(pos_path) and os.path.isfile(neg_path)):
        return None
    pos_len = _first_site_sequence_length(pos_path)
    neg_len = _first_site_sequence_length(neg_path)
    if pos_len is None or neg_len is None:
        return None
    available_len = min(pos_len, neg_len)
    if available_len < required_len:
        return None
    return pos_path, neg_path, available_len


def infer_default_train_paths(
    train_dir: str,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    donor_upstream: Optional[int] = None,
    donor_downstream: Optional[int] = None,
    acceptor_upstream: Optional[int] = None,
    acceptor_downstream: Optional[int] = None,
) -> Tuple[str, str, int]:
    available = list_available_train_lengths(train_dir)
    if not available:
        raise ValueError(
            "No paired training files found in "
            f"{train_dir}. Expected <N>bp.err and <N>bp.neg.err."
        )

    required_len = _required_train_window_length(
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        donor_upstream=donor_upstream,
        donor_downstream=donor_downstream,
        acceptor_upstream=acceptor_upstream,
        acceptor_downstream=acceptor_downstream,
    )
    if required_len is not None:
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
    donor_upstream: Optional[int] = None,
    donor_downstream: Optional[int] = None,
    acceptor_upstream: Optional[int] = None,
    acceptor_downstream: Optional[int] = None,
) -> Tuple[str, str, Optional[int]]:
    dirs = species_data_dirs(species)
    inferred_train_len: Optional[int] = None
    default_pos = None
    default_neg = None

    if train_pos_path is None or train_neg_path is None:
        try:
            default_pos, default_neg, inferred_train_len = infer_default_train_paths(
                train_dir=dirs["raw"],
                donor_len=donor_len,
                acceptor_len=acceptor_len,
                donor_upstream=donor_upstream,
                donor_downstream=donor_downstream,
                acceptor_upstream=acceptor_upstream,
                acceptor_downstream=acceptor_downstream,
            )
        except ValueError as raw_error:
            required_len = _required_train_window_length(
                donor_len=donor_len,
                acceptor_len=acceptor_len,
                donor_upstream=donor_upstream,
                donor_downstream=donor_downstream,
                acceptor_upstream=acceptor_upstream,
                acceptor_downstream=acceptor_downstream,
            )
            processed_paths = _infer_processed_site_flank_paths(
                dirs["processed"], required_len
            )
            if processed_paths is None:
                raise raw_error
            default_pos, default_neg, inferred_train_len = processed_paths

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
    processed_path = os.path.join(dirs["base"], "processed", "transcripts.unique.tsv")
    if os.path.isfile(processed_path):
        return processed_path
    raise FileNotFoundError(
        "Missing required processed unique transcript TSV. "
        f"species={species} path={processed_path}. "
        "Generate it with run/make_unique_intron_assets.sh "
        "or pass --test_tsv explicitly."
    )


@lru_cache(maxsize=8)
def _load_test_half_length_map(test_tsv: str) -> dict[tuple[str, int], int]:
    """Load half lengths from a sibling unique map file when available."""
    map_path = Path(test_tsv).with_name(UNIQUE_MAP_TSV_NAME)
    if not map_path.is_file():
        return {}
    return load_unique_half_lengths(map_path)


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


# Position (0-indexed) of the splice-site dinucleotide (GT/AG) in the 102bp
# sequences from *.coding.pwm.err and *.neg.pwm.err raw files.
SPLICE_SITE_OFFSET: int = 50
LEGACY_TEST_EXON_BP: int = 5


def _infer_splice_site_offset(seq_len: int) -> int:
    """Infer splice-site offset for supported raw/processed site windows."""
    if seq_len == 102:
        return SPLICE_SITE_OFFSET
    return seq_len // 2


def reshape_site_sequence_4p(
    seq: str,
    site_type: str,
    upstream: Optional[int],
    downstream: Optional[int],
    splice_site_offset: Optional[int] = None,
) -> Optional[str]:
    """Slice a splice-site sequence using independent upstream/downstream windows.

    Parameters
    ----------
    seq:
        Raw nucleotide sequence (any length).
    site_type:
        ``"donor"`` or ``"acceptor"``.
    upstream:
        Bases to include before *splice_site_offset*. ``None`` = from start.
    downstream:
        Bases to include from *splice_site_offset* onwards. ``None`` = to end.
    splice_site_offset:
        0-indexed position of the first dinucleotide base (GT for donor,
        AG for acceptor) within *seq*.

    Returns
    -------
    str | None
        Sliced sequence, or ``None`` when requested window exceeds *seq*.
    """
    if site_type not in {"donor", "acceptor"}:
        return None
    seq = seq.upper()
    if splice_site_offset is None:
        splice_site_offset = _infer_splice_site_offset(len(seq))
    start = splice_site_offset - upstream if upstream is not None else 0
    end = splice_site_offset + downstream if downstream is not None else len(seq)
    if start < 0 or end > len(seq) or start >= end:
        return None
    return seq[start:end]


def _resolve_explicit_test_request_context(
    *,
    site_type: str,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    donor_upstream: Optional[int],
    donor_downstream: Optional[int],
    acceptor_upstream: Optional[int],
    acceptor_downstream: Optional[int],
    source_upstream: int,
    source_downstream: int,
) -> tuple[Optional[int], Optional[int]]:
    """Resolve requested upstream/downstream context for explicit-window TSV rows."""
    requested_len = donor_len if site_type == "donor" else acceptor_len
    if (
        requested_len is not None
        and requested_len == source_upstream + source_downstream
        and (
            (site_type == "donor" and donor_upstream is None and donor_downstream is None)
            or (
                site_type == "acceptor"
                and acceptor_upstream is None
                and acceptor_downstream is None
            )
        )
    ):
        # An explicit TSV already supplies exactly the requested window.
        # Preserve its splice-site alignment instead of applying the legacy
        # 5-bp-exon convention used by shorter historical CNN windows.
        return source_upstream, source_downstream

    if site_type == "donor":
        if donor_upstream is not None or donor_downstream is not None:
            return donor_upstream, donor_downstream
        if donor_len is None:
            return None, None
        upstream = min(LEGACY_TEST_EXON_BP, donor_len)
        return upstream, donor_len - upstream

    if site_type == "acceptor":
        if acceptor_upstream is not None or acceptor_downstream is not None:
            return acceptor_upstream, acceptor_downstream
        if acceptor_len is None:
            return None, None
        downstream = min(LEGACY_TEST_EXON_BP, acceptor_len)
        return acceptor_len - downstream, downstream

    return None, None


def _reshape_explicit_test_site_sequence(
    *,
    seq: str,
    source_upstream: int,
    source_downstream: int,
    requested_upstream: Optional[int],
    requested_downstream: Optional[int],
) -> Optional[str]:
    """Slice or pad one explicit-context site sequence."""
    seq_upper = seq.upper()
    expected_len = source_upstream + source_downstream
    if expected_len < 0:
        return None

    start = source_upstream - requested_upstream if requested_upstream is not None else 0
    end = (
        source_upstream + requested_downstream
        if requested_downstream is not None
        else len(seq_upper)
    )
    if start >= end:
        return None

    left_pad = max(0, -start)
    right_pad = max(0, end - len(seq_upper))
    bounded_start = max(0, start)
    bounded_end = min(len(seq_upper), end)
    middle = seq_upper[bounded_start:bounded_end]
    return ("N" * left_pad) + middle + ("N" * right_pad)


def _reshape_or_pad_test_site_sequence(
    seq: str,
    site_type: str,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    donor_upstream: Optional[int] = None,
    donor_downstream: Optional[int] = None,
    acceptor_upstream: Optional[int] = None,
    acceptor_downstream: Optional[int] = None,
    source_upstream: Optional[int] = None,
    source_downstream: Optional[int] = None,
) -> Optional[str]:
    """Reshape one inference-time site sequence to fixed length.

    Test TSVs generated for mask-mode evaluation may clip intronic context for
    short introns, which yields variable-length sequences. Inference models are
    trained on fixed-width windows, so short donor windows must be padded on the
    right and short acceptor windows must be padded on the left with ``N``.

    Parameters
    ----------
    seq : str
        Raw site sequence from the test TSV.
    site_type : str
        Site type label. Supported values are ``donor`` and ``acceptor``.
    donor_len : int | None
        Requested fixed donor window length.
    acceptor_len : int | None
        Requested fixed acceptor window length.

    Returns
    -------
    str | None
        Fixed-length sequence ready for inference, or ``None`` for unsupported
        site types.
    """
    if source_upstream is not None and source_downstream is not None:
        requested_upstream, requested_downstream = _resolve_explicit_test_request_context(
            site_type=site_type,
            donor_len=donor_len,
            acceptor_len=acceptor_len,
            donor_upstream=donor_upstream,
            donor_downstream=donor_downstream,
            acceptor_upstream=acceptor_upstream,
            acceptor_downstream=acceptor_downstream,
            source_upstream=source_upstream,
            source_downstream=source_downstream,
        )
        return _reshape_explicit_test_site_sequence(
            seq=seq,
            source_upstream=source_upstream,
            source_downstream=source_downstream,
            requested_upstream=requested_upstream,
            requested_downstream=requested_downstream,
        )

    upstream = donor_upstream if site_type == "donor" else acceptor_upstream
    downstream = donor_downstream if site_type == "donor" else acceptor_downstream
    use_4p = upstream is not None or downstream is not None
    if use_4p:
        reshaped = reshape_site_sequence_4p(
            seq=seq,
            site_type=site_type,
            upstream=upstream,
            downstream=downstream,
        )
    else:
        reshaped = reshape_site_sequence(
            seq=seq,
            site_type=site_type,
            donor_len=donor_len,
            acceptor_len=acceptor_len,
        )
    if reshaped is not None:
        return reshaped

    seq_upper = seq.upper()
    if site_type == "donor":
        if use_4p:
            if donor_upstream is None or donor_downstream is None:
                return seq_upper
            donor_len = donor_upstream + donor_downstream
        if donor_len is None:
            return seq_upper
        if len(seq_upper) > donor_len:
            return seq_upper[:donor_len]
        return seq_upper + ("N" * (donor_len - len(seq_upper)))

    if site_type == "acceptor":
        if use_4p:
            if acceptor_upstream is None or acceptor_downstream is None:
                return seq_upper
            acceptor_len = acceptor_upstream + acceptor_downstream
        if acceptor_len is None:
            return seq_upper
        if len(seq_upper) > acceptor_len:
            return seq_upper[-acceptor_len:]
        return ("N" * (acceptor_len - len(seq_upper))) + seq_upper

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


def _parse_required_number(token: str) -> int | None:
    """Parse required numeric token (int or float) as int."""
    try:
        return int(float(token))
    except (ValueError, TypeError):
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
            parsed = _parse_required_number(tokens[index])
            if parsed is None:
                return None
            intron_half_length = parsed
            index += 1
        # Accept optional trailing chrom/pos fields from new format; skip them.
        while index < len(tokens):
            index += 1
        return ParsedTrainingRecord(
            record_type="pair",
            donor_seq=donor_seq,
            acceptor_seq=acceptor_seq,
            strand=strand,
            transcript_id=None,
            intron_half_length=intron_half_length,
            chrom=None,
            pos=None,
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
                parsed = _parse_required_number(tokens[index + 1])
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
                chrom=None,
                pos=None,
            )

        strand, index = _parse_optional_strand(tokens, 3)
        # Accept optional trailing: <intron_half_len> <chrom> <pos>
        intron_half_length = None
        chrom: str | None = None
        pos: int | None = None
        if index < len(tokens):
            parsed = _parse_required_number(tokens[index])
            if parsed is None:
                return None
            intron_half_length = parsed
            index += 1
            if index < len(tokens):
                chrom = tokens[index]
                index += 1
                if index < len(tokens):
                    pos = _parse_required_int(tokens[index])
                    if pos is None:
                        return None
                    index += 1
        if index != len(tokens):
            return None
        return ParsedTrainingRecord(
            record_type="donor",
            donor_seq=donor_seq,
            acceptor_seq=None,
            strand=strand,
            transcript_id=None,
            intron_half_length=intron_half_length,
            chrom=chrom,
            pos=pos,
        )

    if record_type == "acceptor":
        acceptor_seq = tokens[2].upper()
        strand, index = _parse_optional_strand(tokens, 3)
        # Accept optional trailing: <intron_half_len> <chrom> <pos>
        intron_half_length = None
        chrom = None
        pos = None
        if index < len(tokens):
            parsed = _parse_required_number(tokens[index])
            if parsed is None:
                return None
            intron_half_length = parsed
            index += 1
            if index < len(tokens):
                chrom = tokens[index]
                index += 1
                if index < len(tokens):
                    pos = _parse_required_int(tokens[index])
                    if pos is None:
                        return None
                    index += 1
        if index != len(tokens):
            return None
        return ParsedTrainingRecord(
            record_type="acceptor",
            donor_seq=None,
            acceptor_seq=acceptor_seq,
            strand=strand,
            transcript_id=None,
            intron_half_length=intron_half_length,
            chrom=chrom,
            pos=pos,
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


def _resolve_training_file_signature(path: str) -> TrainingFileSignature:
    """Return cache key tuple for one training file.

    Parameters
    ----------
    path : str
        Input file path.

    Returns
    -------
    tuple[str, int, int]
        Real path, mtime in nanoseconds, and file size in bytes.
    """
    resolved_path = os.path.realpath(path)
    stat_result = os.stat(resolved_path)
    return (
        resolved_path,
        int(stat_result.st_mtime_ns),
        int(stat_result.st_size),
    )


def clear_training_example_caches() -> None:
    """Clear in-memory caches for parsed training examples."""
    _read_examples_single_task_with_metadata_cached.cache_clear()
    _read_examples_pair_task_with_metadata_cached.cache_clear()


def _read_examples_single_task_with_metadata_uncached(
    *,
    pos_path: str,
    neg_path: str,
    task: str,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    donor_upstream: Optional[int] = None,
    donor_downstream: Optional[int] = None,
    acceptor_upstream: Optional[int] = None,
    acceptor_downstream: Optional[int] = None,
) -> Tuple[SiteTrainingExample, ...]:
    """Read one-task examples from disk without cache lookup."""
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
                        parsed.donor_seq if task == "donor" else parsed.acceptor_seq
                    )
                elif parsed.record_type == task:
                    raw_seq = (
                        parsed.donor_seq if task == "donor" else parsed.acceptor_seq
                    )
                else:
                    continue

                if raw_seq is None:
                    continue

                up = donor_upstream if task == "donor" else acceptor_upstream
                dn = donor_downstream if task == "donor" else acceptor_downstream
                if up is not None or dn is not None:
                    reshaped = reshape_site_sequence_4p(raw_seq, task, up, dn)
                else:
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
    return tuple(examples)


@lru_cache(maxsize=_TRAINING_EXAMPLE_CACHE_MAXSIZE)
def _read_examples_single_task_with_metadata_cached(
    pos_signature: TrainingFileSignature,
    neg_signature: TrainingFileSignature,
    task: str,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    donor_upstream: Optional[int] = None,
    donor_downstream: Optional[int] = None,
    acceptor_upstream: Optional[int] = None,
    acceptor_downstream: Optional[int] = None,
) -> Tuple[SiteTrainingExample, ...]:
    """Read one-task examples with cache keyed by file signatures."""
    return _read_examples_single_task_with_metadata_uncached(
        pos_path=pos_signature[0],
        neg_path=neg_signature[0],
        task=task,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        donor_upstream=donor_upstream,
        donor_downstream=donor_downstream,
        acceptor_upstream=acceptor_upstream,
        acceptor_downstream=acceptor_downstream,
    )


def read_examples_single_task_with_metadata(
    pos_path: str,
    neg_path: str,
    task: str,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    donor_upstream: Optional[int] = None,
    donor_downstream: Optional[int] = None,
    acceptor_upstream: Optional[int] = None,
    acceptor_downstream: Optional[int] = None,
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
    pos_signature = _resolve_training_file_signature(pos_path)
    neg_signature = _resolve_training_file_signature(neg_path)
    cached_examples = _read_examples_single_task_with_metadata_cached(
        pos_signature,
        neg_signature,
        task,
        donor_len,
        acceptor_len,
        donor_upstream,
        donor_downstream,
        acceptor_upstream,
        acceptor_downstream,
    )
    return list(cached_examples)


def read_examples_single_task(
    pos_path: str,
    neg_path: str,
    task: str,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    donor_upstream: Optional[int] = None,
    donor_downstream: Optional[int] = None,
    acceptor_upstream: Optional[int] = None,
    acceptor_downstream: Optional[int] = None,
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
        donor_upstream=donor_upstream,
        donor_downstream=donor_downstream,
        acceptor_upstream=acceptor_upstream,
        acceptor_downstream=acceptor_downstream,
    )
    return [(item.sequence, item.label) for item in examples_with_metadata]


def read_examples_pair_task_with_metadata(
    pos_path: str,
    neg_path: str,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    donor_upstream: Optional[int] = None,
    donor_downstream: Optional[int] = None,
    acceptor_upstream: Optional[int] = None,
    acceptor_downstream: Optional[int] = None,
    *,
    negative_pair_only: bool = True,
) -> List[PairTrainingExample]:
    """Read pair-task training examples with parsed metadata.

    Parameters
    ----------
    pos_path : str
        Positive training file path.
    neg_path : str
        Negative training file path.
    donor_len : int | None
        Donor window length. ``None`` keeps original donor-side length.
    acceptor_len : int | None
        Acceptor window length. ``None`` keeps original acceptor-side length.
    negative_pair_only : bool, default=True
        If ``True``, negative examples are restricted to ``DEBUG pair`` rows.

    Returns
    -------
    list[PairTrainingExample]
        Pair-task examples with optional metadata fields populated.

    Notes
    -----
    When available, this loader automatically appends additional negative rows
    from ``processed/*mixed_one_side*.neg.err`` under the same species
    directory as ``pos_path``/``neg_path``.
    """
    pos_signature = _resolve_training_file_signature(pos_path)
    neg_signature = _resolve_training_file_signature(neg_path)
    extra_negative_paths = discover_default_pair_extra_negative_paths(
        pos_path=pos_signature[0],
        neg_path=neg_signature[0],
    )
    extra_negative_signatures = tuple(
        _resolve_training_file_signature(path) for path in extra_negative_paths
    )
    cached_examples = _read_examples_pair_task_with_metadata_cached(
        pos_signature,
        neg_signature,
        extra_negative_signatures,
        donor_len,
        acceptor_len,
        donor_upstream,
        donor_downstream,
        acceptor_upstream,
        acceptor_downstream,
        negative_pair_only,
    )
    return list(cached_examples)


def _read_examples_pair_task_with_metadata_uncached(
    *,
    pos_path: str,
    neg_path: str,
    extra_neg_paths: Sequence[str],
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    donor_upstream: Optional[int] = None,
    donor_downstream: Optional[int] = None,
    acceptor_upstream: Optional[int] = None,
    acceptor_downstream: Optional[int] = None,
    negative_pair_only: bool = True,
) -> Tuple[PairTrainingExample, ...]:
    """Read pair-task examples from disk without cache lookup."""
    examples: List[PairTrainingExample] = []

    def read_one_set(path: str, label: int) -> None:
        with open(path, "r", errors="ignore") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or not line.startswith("DEBUG"):
                    continue
                parsed = parse_debug_training_record(line)
                if parsed is None or parsed.record_type != "pair":
                    continue
                if (
                    label == 0
                    and negative_pair_only
                    and not line.startswith("DEBUG pair ")
                ):
                    continue
                if parsed.donor_seq is None or parsed.acceptor_seq is None:
                    continue

                if donor_upstream is not None or donor_downstream is not None:
                    donor_seq = reshape_site_sequence_4p(
                        parsed.donor_seq,
                        "donor",
                        donor_upstream,
                        donor_downstream,
                    )
                else:
                    donor_seq = reshape_site_sequence(
                        parsed.donor_seq,
                        "donor",
                        donor_len=donor_len,
                        acceptor_len=acceptor_len,
                    )
                if acceptor_upstream is not None or acceptor_downstream is not None:
                    acceptor_seq = reshape_site_sequence_4p(
                        parsed.acceptor_seq,
                        "acceptor",
                        acceptor_upstream,
                        acceptor_downstream,
                    )
                else:
                    acceptor_seq = reshape_site_sequence(
                        parsed.acceptor_seq,
                        "acceptor",
                        donor_len=donor_len,
                        acceptor_len=acceptor_len,
                    )
                if donor_seq is None or acceptor_seq is None:
                    continue

                examples.append(
                    PairTrainingExample(
                        donor_sequence=donor_seq,
                        acceptor_sequence=acceptor_seq,
                        label=label,
                        transcript_id=parsed.transcript_id,
                        intron_half_length=parsed.intron_half_length,
                        source_record_type=parsed.record_type,
                        strand=parsed.strand,
                    )
                )

    read_one_set(pos_path, label=1)
    read_one_set(neg_path, label=0)
    for extra_neg_path in extra_neg_paths:
        read_one_set(extra_neg_path, label=0)
    return tuple(examples)


@lru_cache(maxsize=_TRAINING_EXAMPLE_CACHE_MAXSIZE)
def _read_examples_pair_task_with_metadata_cached(
    pos_signature: TrainingFileSignature,
    neg_signature: TrainingFileSignature,
    extra_neg_signatures: tuple[TrainingFileSignature, ...],
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    donor_upstream: Optional[int] = None,
    donor_downstream: Optional[int] = None,
    acceptor_upstream: Optional[int] = None,
    acceptor_downstream: Optional[int] = None,
    negative_pair_only: bool = True,
) -> Tuple[PairTrainingExample, ...]:
    """Read pair-task examples with cache keyed by file signatures."""
    return _read_examples_pair_task_with_metadata_uncached(
        pos_path=pos_signature[0],
        neg_path=neg_signature[0],
        extra_neg_paths=tuple(signature[0] for signature in extra_neg_signatures),
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        donor_upstream=donor_upstream,
        donor_downstream=donor_downstream,
        acceptor_upstream=acceptor_upstream,
        acceptor_downstream=acceptor_downstream,
        negative_pair_only=negative_pair_only,
    )


def _infer_processed_dir_from_training_path(path: str) -> str | None:
    """Infer one species ``processed`` directory from a train data path.

    Parameters
    ----------
    path : str
        Training file path.

    Returns
    -------
    str | None
        Canonical processed directory path, or ``None`` when not inferrable.
    """
    resolved_path = os.path.realpath(path)
    parent_dir = os.path.dirname(resolved_path)
    parent_name = os.path.basename(parent_dir).strip().lower()
    if parent_name not in {"raw", "processed"}:
        return None
    species_dir = os.path.dirname(parent_dir)
    processed_dir = os.path.join(species_dir, "processed")
    if not os.path.isdir(processed_dir):
        return None
    return os.path.realpath(processed_dir)


def _discover_default_pair_extra_negative_paths(
    *,
    pos_path: str,
    neg_path: str,
) -> tuple[str, ...]:
    """Discover extra mixed negative files for one pair-training run.

    Parameters
    ----------
    pos_path : str
        Positive training file path.
    neg_path : str
        Primary negative training file path.

    Returns
    -------
    tuple[str, ...]
        Sorted unique extra negative file paths.
    """
    processed_dirs: set[str] = set()
    for candidate_path in (pos_path, neg_path):
        processed_dir = _infer_processed_dir_from_training_path(candidate_path)
        if processed_dir is not None:
            processed_dirs.add(processed_dir)

    resolved_pos = os.path.realpath(pos_path)
    resolved_neg = os.path.realpath(neg_path)
    discovered_paths: list[str] = []
    seen_paths: set[str] = set()

    for processed_dir in sorted(processed_dirs):
        try:
            entries = sorted(
                entry.name for entry in os.scandir(processed_dir) if entry.is_file()
            )
        except OSError:
            continue
        for entry_name in entries:
            if _MIXED_PAIR_NEGATIVE_PATTERN.fullmatch(entry_name) is None:
                continue
            candidate = os.path.realpath(os.path.join(processed_dir, entry_name))
            if candidate in {resolved_pos, resolved_neg}:
                continue
            if candidate in seen_paths:
                continue
            seen_paths.add(candidate)
            discovered_paths.append(candidate)
    return tuple(discovered_paths)


def discover_default_pair_extra_negative_paths(
    *,
    pos_path: str,
    neg_path: str,
) -> tuple[str, ...]:
    """Discover default mixed negative files for pair-task training.

    Parameters
    ----------
    pos_path : str
        Positive training file path.
    neg_path : str
        Primary negative training file path.

    Returns
    -------
    tuple[str, ...]
        Sorted unique extra negative file paths under the inferred processed
        directory.
    """
    return _discover_default_pair_extra_negative_paths(
        pos_path=pos_path,
        neg_path=neg_path,
    )


def read_examples_pair_task(
    pos_path: str,
    neg_path: str,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    *,
    negative_pair_only: bool = True,
) -> List[Tuple[Tuple[str, str], int]]:
    """Read pair-task examples as ``((donor_seq, acceptor_seq), label)`` tuples."""
    examples_with_metadata = read_examples_pair_task_with_metadata(
        pos_path=pos_path,
        neg_path=neg_path,
        donor_len=donor_len,
        acceptor_len=acceptor_len,
        negative_pair_only=negative_pair_only,
    )
    return [
        ((item.donor_sequence, item.acceptor_sequence), item.label)
        for item in examples_with_metadata
    ]


def _parse_test_header_indices(header_line: str) -> Dict[str, int]:
    """Resolve required and optional column indices for ``transcripts.tsv``."""
    header = header_line.rstrip("\n").split("\t")
    index_by_name = {name: idx for idx, name in enumerate(header)}

    required = ("transcript_id", "site_type", "intron_index", "seq")
    missing = [name for name in required if name not in index_by_name]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            "Missing required columns in test TSV header: "
            f"{missing_text}. Found: {header}"
        )

    out = {
        "transcript_id": index_by_name["transcript_id"],
        "site_type": index_by_name["site_type"],
        "intron_index": index_by_name["intron_index"],
        "seq": index_by_name["seq"],
    }
    if "upstream_bp" in index_by_name:
        out["upstream_bp"] = index_by_name["upstream_bp"]
    if "downstream_bp" in index_by_name:
        out["downstream_bp"] = index_by_name["downstream_bp"]
    if "intron_half_length" in index_by_name:
        out["intron_half_length"] = index_by_name["intron_half_length"]
    return out


def read_test_site_rows(
    test_tsv: str,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    donor_upstream: Optional[int] = None,
    donor_downstream: Optional[int] = None,
    acceptor_upstream: Optional[int] = None,
    acceptor_downstream: Optional[int] = None,
) -> Tuple[List[Dict[str, object]], int]:
    """Read inference-time site rows from one transcript TSV.

    Short mask-mode windows are padded with ``N`` to the requested fixed length
    instead of being dropped. If the TSV omits ``intron_half_length``, this
    function backfills it from a sibling ``transcripts.unique.map.tsv`` when
    available.

    Parameters
    ----------
    test_tsv : str
        Test TSV path.
    donor_len : int | None
        Requested fixed donor window length.
    acceptor_len : int | None
        Requested fixed acceptor window length.

    Returns
    -------
    tuple[list[dict[str, object]], int]
        Parsed rows and the count of rows skipped due to invalid data.
    """
    rows: List[Dict[str, object]] = []
    skipped_short = 0

    with open(test_tsv, "r") as f:
        header_line = next(f, None)
        if header_line is None:
            return rows, skipped_short
        idx = _parse_test_header_indices(header_line)
        required_max_index = max(idx.values())
        half_length_map = _load_test_half_length_map(test_tsv)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= required_max_index:
                continue

            site_type = parts[idx["site_type"]]
            source_upstream: Optional[int] = None
            source_downstream: Optional[int] = None
            source_upstream_idx = idx.get("upstream_bp")
            source_downstream_idx = idx.get("downstream_bp")
            if source_upstream_idx is not None and source_downstream_idx is not None:
                raw_upstream = parts[source_upstream_idx].strip()
                raw_downstream = parts[source_downstream_idx].strip()
                if raw_upstream != "" and raw_downstream != "":
                    source_upstream = _parse_required_int(raw_upstream)
                    source_downstream = _parse_required_int(raw_downstream)
                    if source_upstream is None or source_downstream is None:
                        raise ValueError(
                            "Invalid upstream_bp/downstream_bp in test TSV: "
                            f"{raw_upstream}, {raw_downstream}"
                        )
            reshaped = _reshape_or_pad_test_site_sequence(
                seq=parts[idx["seq"]],
                site_type=site_type,
                donor_len=donor_len,
                acceptor_len=acceptor_len,
                donor_upstream=donor_upstream,
                donor_downstream=donor_downstream,
                acceptor_upstream=acceptor_upstream,
                acceptor_downstream=acceptor_downstream,
                source_upstream=source_upstream,
                source_downstream=source_downstream,
            )
            if not reshaped:
                skipped_short += 1
                continue

            intron_half_length: int | None = None
            intron_half_length_idx = idx.get("intron_half_length")
            if intron_half_length_idx is not None:
                raw_value = parts[intron_half_length_idx].strip()
                if raw_value != "":
                    intron_half_length = _parse_required_int(raw_value)
                    if intron_half_length is None:
                        raise ValueError(
                            f"Invalid intron_half_length value in test TSV: {raw_value}"
                        )
            if intron_half_length is None:
                intron_half_length = half_length_map.get(
                    (parts[idx["transcript_id"]], int(parts[idx["intron_index"]]))
                )

            rows.append(
                {
                    "transcript_id": parts[idx["transcript_id"]],
                    "site_type": site_type,
                    "intron_index": int(parts[idx["intron_index"]]),
                    "seq": reshaped,
                    "intron_half_length": intron_half_length,
                }
            )

    return rows, skipped_short


def read_test_pair_rows(
    test_tsv: str,
    donor_len: Optional[int],
    acceptor_len: Optional[int],
    donor_upstream: Optional[int] = None,
    donor_downstream: Optional[int] = None,
    acceptor_upstream: Optional[int] = None,
    acceptor_downstream: Optional[int] = None,
) -> Tuple[List[Dict[str, object]], int, int]:
    """Read and pair donor/acceptor test rows into pair-task records.

    If the TSV omits ``intron_half_length``, this function backfills it from a
    sibling ``transcripts.unique.map.tsv`` when available.

    Parameters
    ----------
    test_tsv : str
        Test TSV path generated by test-data utilities.
    donor_len : int | None
        Donor window length.
    acceptor_len : int | None
        Acceptor window length.

    Returns
    -------
    tuple[list[dict[str, object]], int, int]
        ``(rows, skipped_short, skipped_unpaired)`` where each row includes
        ``transcript_id``, ``intron_index``, ``donor_seq``, ``acceptor_seq``,
        and optional ``intron_half_length``.
    """
    grouped: dict[tuple[str, int], dict[str, object]] = defaultdict(dict)
    skipped_short = 0

    with open(test_tsv, "r") as handle:
        header_line = next(handle, None)
        if header_line is None:
            return [], 0, 0
        idx = _parse_test_header_indices(header_line)
        required_max_index = max(idx.values())
        half_length_map = _load_test_half_length_map(test_tsv)

        for raw_line in handle:
            parts = raw_line.rstrip("\n").split("\t")
            if len(parts) <= required_max_index:
                continue

            site_type = parts[idx["site_type"]]
            if site_type not in {"donor", "acceptor"}:
                continue
            source_upstream: Optional[int] = None
            source_downstream: Optional[int] = None
            source_upstream_idx = idx.get("upstream_bp")
            source_downstream_idx = idx.get("downstream_bp")
            if source_upstream_idx is not None and source_downstream_idx is not None:
                raw_upstream = parts[source_upstream_idx].strip()
                raw_downstream = parts[source_downstream_idx].strip()
                if raw_upstream != "" and raw_downstream != "":
                    source_upstream = _parse_required_int(raw_upstream)
                    source_downstream = _parse_required_int(raw_downstream)
                    if source_upstream is None or source_downstream is None:
                        raise ValueError(
                            "Invalid upstream_bp/downstream_bp in test TSV: "
                            f"{raw_upstream}, {raw_downstream}"
                        )
            reshaped = _reshape_or_pad_test_site_sequence(
                seq=parts[idx["seq"]],
                site_type=site_type,
                donor_len=donor_len,
                acceptor_len=acceptor_len,
                donor_upstream=donor_upstream,
                donor_downstream=donor_downstream,
                acceptor_upstream=acceptor_upstream,
                acceptor_downstream=acceptor_downstream,
                source_upstream=source_upstream,
                source_downstream=source_downstream,
            )
            if reshaped is None:
                skipped_short += 1
                continue

            transcript_id = parts[idx["transcript_id"]]
            intron_index = int(parts[idx["intron_index"]])
            key = (transcript_id, intron_index)
            bucket = grouped[key]
            bucket["transcript_id"] = transcript_id
            bucket["intron_index"] = intron_index
            bucket[f"{site_type}_seq"] = reshaped

            intron_half_length_idx = idx.get("intron_half_length")
            if intron_half_length_idx is not None:
                raw_value = parts[intron_half_length_idx].strip()
                if raw_value != "":
                    parsed_half = _parse_required_int(raw_value)
                    if parsed_half is None:
                        raise ValueError(
                            f"Invalid intron_half_length value in test TSV: {raw_value}"
                        )
                    existing = bucket.get("intron_half_length")
                    if (
                        existing is not None
                        and isinstance(existing, int)
                        and existing != parsed_half
                    ):
                        raise ValueError(
                            "Mismatched intron_half_length within one pair: "
                            f"{existing} != {parsed_half}"
                        )
                    bucket["intron_half_length"] = parsed_half
            if "intron_half_length" not in bucket:
                bucket["intron_half_length"] = half_length_map.get(
                    (transcript_id, intron_index)
                )

    rows: List[Dict[str, object]] = []
    skipped_unpaired = 0
    for key in sorted(grouped.keys(), key=lambda item: (item[0], item[1])):
        bucket = grouped[key]
        donor_seq = bucket.get("donor_seq")
        acceptor_seq = bucket.get("acceptor_seq")
        if not isinstance(donor_seq, str) or not isinstance(acceptor_seq, str):
            skipped_unpaired += 1
            continue
        rows.append(
            {
                "transcript_id": str(bucket["transcript_id"]),
                "intron_index": int(bucket["intron_index"]),
                "donor_seq": donor_seq,
                "acceptor_seq": acceptor_seq,
                "intron_half_length": bucket.get("intron_half_length"),
            }
        )

    return rows, skipped_short, skipped_unpaired
