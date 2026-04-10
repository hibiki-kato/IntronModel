"""Scan GT and AG splice-site candidates and score them with best checkpoints.

The helper reads one DNA sequence, extracts canonical donor and acceptor
candidates on the forward strand, skips candidates that cannot supply the full
trained window, scores the retained windows, and writes one score file per site
type.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
from torch import nn

from models import dnabert as dnabert_model
from models import cnn_v3 as cnn_v3_model
from models.cnn import load_task_model as load_cnn_v2_task_model
from models.cnn import score_sequences as score_cnn_v2_sequences
from util.data_proc import model_root, project_root
from util.checkpoint_io import (
    extract_task_checkpoint_path,
    read_json_object,
    resolve_existing_checkpoint_path,
)
from util.model_runtime import pick_device
from util.score_format import format_score_text
from util.score_test_suite_pair_filter import (
    apply_pair_score_adjustments,
    build_pair_candidates,
    write_sparse_scores,
)
from util.transcript_eval import coerce_score_to_probability
from util.versioned_artifacts import (
    is_active_public_model,
    resolve_latest_published_name,
)

CandidateKind = Literal["gt", "ag"]


@dataclass(frozen=True)
class ScoredCandidate:
    """One splice-site candidate and its extracted scoring window."""

    kind: CandidateKind
    coordinate: int
    window: str


@dataclass(frozen=True)
class ResolvedBestModelPaths:
    """Resolved model and checkpoint metadata for one best site-scoring run."""

    best_config_path: Path
    donor_checkpoint_path: Path
    acceptor_checkpoint_path: Path
    donor_window_len: int
    acceptor_window_len: int
    resolved_model_name: str | None = None


@dataclass(frozen=True)
class ScoreTestSuiteCase:
    """One discovered score-test-suite case FASTA."""

    case_name: str
    fasta_path: Path


@dataclass(frozen=True)
class ScoreTestSuiteSummary:
    """Summary for one generated pair of score-test-suite outputs."""

    case_name: str
    donor_candidate_count: int
    acceptor_candidate_count: int
    donor_output_path: Path
    acceptor_output_path: Path


@dataclass(frozen=True)
class LoadedDnabertPairModel:
    """Loaded DNABERT pair model with scoring metadata."""

    model: nn.Module
    tokenizer: object
    max_tokens: int
    input_kmer: int | None
    donor_window_len: int
    acceptor_window_len: int


@dataclass(frozen=True)
class PairScoringConfig:
    """Optional pair-scoring pass applied after individual site scoring."""

    loaded_pair_model: LoadedDnabertPairModel
    inactive_score: float = -1000.0
    pair_score_center: float = -2.0
    pair_score_scale: float = 50.0
    pair_delta_min: float = -150.0
    pair_delta_max: float = 100.0
    no_pair_penalty: float = -150.0
    min_intron_length: int = 30
    pair_batch_size: int = 256


def _is_cnn_v3_model_name(model_name: str) -> bool:
    """Return whether one model name should use the cnn_v3 loader."""
    normalized = model_name.strip().lower()
    return normalized.startswith("cnn_v3") or normalized.startswith("cnn_pair_v3")


def _is_dnabert_model_name(model_name: str) -> bool:
    """Return whether one model name should use the DNABERT loader."""
    normalized = model_name.strip().lower()
    return normalized.startswith("dnabert")


def _optional_positive_int(raw_value: object) -> int | None:
    """Normalize one optional positive integer from checkpoint metadata."""
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        return None
    if isinstance(raw_value, int):
        return raw_value if raw_value > 0 else None
    if isinstance(raw_value, float) and raw_value.is_integer():
        parsed_value = int(raw_value)
        return parsed_value if parsed_value > 0 else None
    if isinstance(raw_value, str):
        text = raw_value.strip()
        if text == "":
            return None
        try:
            parsed_value = int(text)
        except ValueError:
            return None
        return parsed_value if parsed_value > 0 else None
    return None


def load_site_model(
    checkpoint_path: str,
    device: str,
    model_name: str,
) -> tuple[nn.Module, dict[str, object]]:
    """Load one site model checkpoint with the matching architecture."""
    if _is_cnn_v3_model_name(model_name):
        return cnn_v3_model.load_task_model(checkpoint_path, device)
    if _is_dnabert_model_name(model_name):
        model, model_config, tokenizer = dnabert_model.load_task_model(
            checkpoint_path,
            device,
        )
        metadata = dict(model_config)
        metadata["tokenizer"] = tokenizer
        return model, metadata
    return load_cnn_v2_task_model(checkpoint_path, device)


def score_site_sequences(
    model: nn.Module,
    sequences: Sequence[str],
    window_len: int,
    device: str,
    batch_size: int,
    model_name: str,
    model_metadata: dict[str, object] | None = None,
) -> np.ndarray:
    """Score site sequences with the matching CNN implementation."""
    if _is_dnabert_model_name(model_name):
        if model_metadata is None:
            raise ValueError("DNABERT scoring requires model metadata.")
        tokenizer = model_metadata.get("tokenizer")
        if tokenizer is None:
            raise ValueError("DNABERT scoring requires a tokenizer.")
        max_tokens = _optional_positive_int(model_metadata.get("max_tokens"))
        if max_tokens is None:
            max_tokens = window_len
        input_kmer = _optional_positive_int(model_metadata.get("input_kmer"))
        scores = dnabert_model.score_sequences(
            model=model,
            sequences=sequences,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            device=device,
            batch_size=batch_size,
            task_name="score_test_suite",
            input_kmer=input_kmer,
            use_amp=False,
            amp_dtype=None,
        )
        return np.asarray(scores, dtype=np.float64)
    if _is_cnn_v3_model_name(model_name):
        return cnn_v3_model.score_sequences(
            model=model,
            sequences=sequences,
            window_len=window_len,
            device=device,
            batch_size=batch_size,
            use_amp=False,
            amp_dtype=None,
        )
    return score_cnn_v2_sequences(
        model=model,
        sequences=sequences,
        window_len=window_len,
        device=device,
        batch_size=batch_size,
        use_amp=False,
        amp_dtype=None,
    )


def normalize_sequence_text(raw_sequence: str) -> str:
    """Normalize a raw FASTA-like sequence string.

    Parameters
    ----------
    raw_sequence : str
        Raw sequence text. FASTA headers and whitespace are ignored.

    Returns
    -------
    str
        Upper-cased contiguous DNA sequence.

    Raises
    ------
    ValueError
        If the normalized sequence is empty.
    """
    parts: list[str] = []
    for raw_line in raw_sequence.splitlines():
        line = raw_line.strip()
        if line == "" or line.startswith(">"):
            continue
        parts.append(re.sub(r"\s+", "", line))

    sequence = "".join(parts).upper()
    if sequence == "":
        raise ValueError("Sequence is empty after normalization.")
    return sequence


def scan_motif_coordinates(sequence: str, motif: str) -> list[int]:
    """Return all 0-based motif start coordinates in one sequence.

    Parameters
    ----------
    sequence : str
        Upper-cased DNA sequence.
    motif : str
        Two-base motif to search for.

    Returns
    -------
    list[int]
        Sorted 0-based motif start coordinates.

    Raises
    ------
    ValueError
        If ``motif`` is not exactly two characters long.
    """
    motif_upper = motif.upper()
    if len(motif_upper) != 2:
        raise ValueError("motif must contain exactly two bases.")

    coordinates: list[int] = []
    last_index = len(sequence) - 1
    for index in range(last_index):
        if sequence[index : index + 2] == motif_upper:
            coordinates.append(index)
    return coordinates


def extract_candidate_window(
    sequence: str,
    *,
    coordinate: int,
    window_len: int,
    left_offset: int,
) -> str | None:
    """Extract one fixed-length window around one candidate coordinate.

    Parameters
    ----------
    sequence : str
        Upper-cased DNA sequence.
    coordinate : int
        0-based coordinate of the motif's first base.
    window_len : int
        Required fixed window length.
    left_offset : int
        Number of bases to keep to the left of the candidate coordinate.

    Returns
    -------
    str | None
        Extracted window when the full context exists; otherwise ``None``.

    Raises
    ------
    ValueError
        If ``window_len`` or ``left_offset`` are invalid.
    """
    if window_len <= 0:
        raise ValueError("window_len must be positive.")
    if left_offset < 0:
        raise ValueError("left_offset must be non-negative.")
    if left_offset >= window_len:
        raise ValueError("left_offset must be smaller than window_len.")

    start = coordinate - left_offset
    end = start + window_len
    if start < 0 or end > len(sequence):
        return None
    return sequence[start:end]


def build_candidate_windows(
    sequence: str,
    *,
    donor_window_len: int,
    acceptor_window_len: int,
) -> tuple[list[ScoredCandidate], list[ScoredCandidate]]:
    """Build score-ready donor and acceptor candidate windows.

    Parameters
    ----------
    sequence : str
        Upper-cased DNA sequence.
    donor_window_len : int
        Fixed donor-model window length.
    acceptor_window_len : int
        Fixed acceptor-model window length.

    Returns
    -------
    tuple[list[ScoredCandidate], list[ScoredCandidate]]
        Donor ``GT`` candidates and acceptor ``AG`` candidates.
    """
    if donor_window_len < 5:
        raise ValueError("donor_window_len must be at least 5.")
    if acceptor_window_len < 5:
        raise ValueError("acceptor_window_len must be at least 5.")

    donor_candidates: list[ScoredCandidate] = []
    for coordinate in scan_motif_coordinates(sequence, "GT"):
        window = extract_candidate_window(
            sequence,
            coordinate=coordinate,
            window_len=donor_window_len,
            left_offset=3,
        )
        if window is None:
            continue
        donor_candidates.append(
            ScoredCandidate(kind="gt", coordinate=coordinate, window=window)
        )

    # Training-time acceptor windows are anchored on the first exonic base of the
    # downstream exon, with the last two intronic bases forming the terminal AG.
    # The stored sparse score coordinate remains the 0-based A position, so the
    # extracted window must start two bases to the right of the old AG-anchored
    # convention.
    acceptor_left_offset = acceptor_window_len - 5
    acceptor_candidates: list[ScoredCandidate] = []
    for coordinate in scan_motif_coordinates(sequence, "AG"):
        window = extract_candidate_window(
            sequence,
            coordinate=coordinate,
            window_len=acceptor_window_len,
            left_offset=acceptor_left_offset,
        )
        if window is None:
            continue
        acceptor_candidates.append(
            ScoredCandidate(kind="ag", coordinate=coordinate, window=window)
        )

    return donor_candidates, acceptor_candidates


def resolve_task_best_config_path(
    data_root: Path,
    species: str,
    model_name: str,
    task: str,
) -> Path:
    """Resolve the canonical best-config path for one site-scoring task.

    Parameters
    ----------
    data_root : Path
        Repository data root.
    species : str
        Species identifier.
    model_name : str
        Model name, for example ``cnn_v2``.
    task : str
        Task name, expected to be ``donor`` or ``acceptor``.

    Returns
    -------
    Path
        Best-config JSON path.

    Raises
    ------
    FileNotFoundError
        If no suitable best-config file exists.
    ValueError
        If ``task`` is unsupported.
    """
    if task not in {"donor", "acceptor"}:
        raise ValueError(f"Unsupported task: {task}")
    model_dir = data_root / species / "tuning" / model_name
    task_dir = model_dir / task
    candidate = task_dir / "best_config.json"
    if candidate.is_file():
        return candidate.resolve()
    fallback = model_dir / "checkpoint_prune_top3.json"
    if fallback.is_file():
        return fallback.resolve()
    fallback = task_dir / "checkpoint_prune_top3.json"
    if fallback.is_file():
        return fallback.resolve()
    raise FileNotFoundError(
        "No canonical task-specific best_config.json or model-level "
        "checkpoint_prune_top3.json "
        f"species={species} model={model_name} task={task} under {data_root}."
    )


def _resolve_json_path(raw_path: str, base_dir: Path) -> Path:
    """Resolve one JSON path string relative to a base directory."""
    path = Path(raw_path.strip())
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def load_task_checkpoint_path(best_config_path: Path, task: str) -> Path:
    """Load one task checkpoint path from a best-config payload.

    Parameters
    ----------
    best_config_path : Path
        Canonical best-config path.
    task : str
        Task name, expected to be ``donor`` or ``acceptor``.

    Returns
    -------
    Path
        Resolved checkpoint path.

    Raises
    ------
    FileNotFoundError
        If the payload or referenced checkpoint path cannot be resolved.
    ValueError
        If the best-config payload is malformed.
    """
    if task not in {"donor", "acceptor"}:
        raise ValueError(f"Unsupported task: {task}")
    payload = read_json_object(best_config_path)
    if payload is None:
        raise FileNotFoundError(f"best_config not found: {best_config_path}")
    status = str(payload.get("status", "")).strip().lower()
    if status != "ok":
        raise ValueError(
            f"Expected best_config status='ok', got: {status or '<missing>'}"
        )

    checkpoint = extract_task_checkpoint_path(
        payload,
        task=task,
        base_dir=best_config_path.parent,
    )
    if checkpoint is None:
        source_best_config = payload.get(f"source_{task}_best_config")
        if not isinstance(source_best_config, str):
            raise FileNotFoundError(
                f"best_config does not contain a {task} checkpoint path or "
                f"source best-config reference: {best_config_path}"
            )
        source_path = _resolve_json_path(source_best_config, best_config_path.parent)
        source_payload = read_json_object(source_path)
        if source_payload is None:
            raise FileNotFoundError(
                f"Unable to read source best-config payload referenced by "
                f"{best_config_path}: {source_path}"
            )
        checkpoint = extract_task_checkpoint_path(
            source_payload,
            task=task,
            base_dir=source_path.parent,
        )
        if checkpoint is None:
            raise FileNotFoundError(
                f"Unable to resolve {task} checkpoint path from {best_config_path}."
            )
    root_dir = Path(model_root()).resolve()
    return resolve_existing_checkpoint_path(
        checkpoint,
        model_root_dir=root_dir,
    )


def _load_window_len_from_checkpoint(
    checkpoint_path: Path,
    *,
    device: str,
) -> int:
    """Read one model's fixed input window length from its checkpoint."""
    _ = device
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid checkpoint payload: {checkpoint_path}")
    window_len = int(payload.get("window_len", 50))
    if window_len <= 0:
        raise ValueError(f"window_len must be positive: {checkpoint_path}")
    return window_len


def load_resolved_best_model_paths(
    data_root: Path,
    species: str,
    model_name: str,
    *,
    device: str,
) -> ResolvedBestModelPaths:
    """Resolve best-config and best checkpoint metadata for one run."""
    published_name: str | None = None
    normalized_model_name = model_name.strip()

    if is_active_public_model(normalized_model_name):
        published_name = resolve_latest_published_name(
            data_root,
            species,
            normalized_model_name,
        )

    donor_best_config_path = resolve_task_best_config_path(
        data_root,
        species,
        model_name,
        "donor",
    )
    acceptor_best_config_path = resolve_task_best_config_path(
        data_root,
        species,
        model_name,
        "acceptor",
    )
    model_candidates = [
        candidate
        for candidate in (published_name, normalized_model_name)
        if candidate is not None and candidate.strip() != ""
    ]
    donor_checkpoint_path = _load_task_checkpoint_path_with_local_fallback(
        best_config_path=donor_best_config_path,
        task="donor",
        species=species,
        model_names=model_candidates,
    )
    acceptor_checkpoint_path = _load_task_checkpoint_path_with_local_fallback(
        best_config_path=acceptor_best_config_path,
        task="acceptor",
        species=species,
        model_names=model_candidates,
    )

    donor_window_len = _load_window_len_from_checkpoint(
        donor_checkpoint_path,
        device=device,
    )
    acceptor_window_len = _load_window_len_from_checkpoint(
        acceptor_checkpoint_path,
        device=device,
    )
    return ResolvedBestModelPaths(
        best_config_path=donor_best_config_path,
        donor_checkpoint_path=donor_checkpoint_path,
        acceptor_checkpoint_path=acceptor_checkpoint_path,
        donor_window_len=donor_window_len,
        acceptor_window_len=acceptor_window_len,
        resolved_model_name=published_name or normalized_model_name,
    )


def resolve_latest_local_task_checkpoint(
    *,
    species: str,
    task: str,
    model_names: Sequence[str],
) -> Path | None:
    """Return the best local checkpoint for one species/task/model list.

    Parameters
    ----------
    species : str
        Species identifier such as ``Dmel``.
    task : str
        Task name such as ``donor`` or ``acceptor``.
    model_names : Sequence[str]
        Ordered model-name candidates. Earlier names take precedence, so a
        published version such as ``cnn_v2.01`` can win over a raw public model
        stem such as ``cnn_v2``.

    Returns
    -------
    Path | None
        Newest matching local checkpoint, or ``None`` when no candidate exists.
    """
    task_dir = Path(model_root()).resolve() / species / task
    if not task_dir.is_dir():
        return None

    candidates = [
        candidate
        for candidate in task_dir.iterdir()
        if candidate.is_file() and candidate.suffix == ".pt"
    ]
    normalized_model_names = [
        str(model_name).strip()
        for model_name in model_names
        if str(model_name).strip() != ""
    ]
    for model_name in normalized_model_names:
        exact_matches = [
            candidate
            for candidate in candidates
            if candidate.name == f"{model_name}.pt"
        ]
        if exact_matches:
            exact_matches.sort(
                key=lambda path: (path.stat().st_mtime_ns, path.name),
                reverse=True,
            )
            return exact_matches[0].resolve()

        prefix_matches = [
            candidate
            for candidate in candidates
            if candidate.name.startswith(f"{model_name}_")
            or candidate.name.startswith(f"{model_name}.")
        ]
        if prefix_matches:
            prefix_matches.sort(
                key=lambda path: (path.stat().st_mtime_ns, path.name),
                reverse=True,
            )
            return prefix_matches[0].resolve()
    return None


def _load_task_checkpoint_path_with_local_fallback(
    *,
    best_config_path: Path,
    task: str,
    species: str,
    model_names: Sequence[str],
) -> Path:
    """Load one task checkpoint path with local latest-checkpoint fallback.

    Parameters
    ----------
    best_config_path : Path
        Canonical best-config JSON path.
    task : str
        Task name, expected to be ``donor`` or ``acceptor``.
    species : str
        Species identifier.
    model_names : Sequence[str]
        Ordered model-name candidates used for local fallback discovery.

    Returns
    -------
    Path
        Resolved checkpoint path.

    Raises
    ------
    FileNotFoundError
        If neither the payload reference nor the local fallback can be
        resolved.
    """
    fallback_only_payload = best_config_path.name == "checkpoint_prune_top3.json"
    try:
        return load_task_checkpoint_path(best_config_path, task)
    except (FileNotFoundError, ValueError) as exc:
        if not fallback_only_payload:
            raise
        fallback = resolve_latest_local_task_checkpoint(
            species=species,
            task=task,
            model_names=model_names,
        )
        if fallback is not None:
            label = ", ".join(str(model_name) for model_name in model_names)
            print(
                "[scan_splice_candidate_sites] "
                f"missing {task} checkpoint from {best_config_path}; "
                f"using latest local checkpoint from [{label}]: {fallback}"
            )
            return fallback
        raise FileNotFoundError(
            f"{exc}. No local fallback checkpoint found for "
            f"species={species} task={task} "
            f"models={[str(model_name) for model_name in model_names]}."
        ) from exc


def score_candidate_windows(
    *,
    candidates: Sequence[ScoredCandidate],
    checkpoint_path: Path,
    window_len: int,
    device: str,
    batch_size: int,
    model_name: str,
) -> list[float]:
    """Score one sequence candidate list with one loaded checkpoint.

    Parameters
    ----------
    candidates : Sequence[ScoredCandidate]
        Score-ready candidate windows.
    checkpoint_path : Path
        Checkpoint path for the corresponding task.
    window_len : int
        Fixed model input length.
    device : str
        PyTorch device string.
    batch_size : int
        Inference batch size.

    Returns
    -------
    list[float]
        Model scores in candidate order.
    """
    if not candidates:
        return []

    model, model_metadata = load_site_model(str(checkpoint_path), device, model_name)
    windows = [candidate.window for candidate in candidates]
    scores = score_site_sequences(
        model=model,
        sequences=windows,
        window_len=window_len,
        device=device,
        batch_size=batch_size,
        model_name=model_name,
        model_metadata=model_metadata,
    )
    return [float(score) for score in np.asarray(scores, dtype=np.float64)]


def write_scores(
    output_path: Path,
    candidates: Sequence[ScoredCandidate],
    scores: Sequence[float],
) -> None:
    """Write one coordinate-score table to disk."""
    if len(candidates) != len(scores):
        raise ValueError("candidates and scores must have the same length.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        for candidate, score in zip(candidates, scores):
            probability_score = coerce_score_to_probability(float(score))
            handle.write(
                f"{candidate.coordinate}\t{format_score_text(probability_score)}\n"
            )


def resolve_dnabert_pair_checkpoint_path(
    data_root: Path,
    species: str,
    pair_model_name: str,
    *,
    explicit_checkpoint_path: Path | None,
) -> Path:
    """Resolve the best pair checkpoint for one dnabert pair model.

    Parameters
    ----------
    data_root : Path
        Repository data root.
    species : str
        Species identifier.
    pair_model_name : str
        Pair model name, e.g. ``dnabert2_pair``.
    explicit_checkpoint_path : Path | None
        Optional explicit checkpoint override.

    Returns
    -------
    Path
        Resolved checkpoint path.

    Raises
    ------
    FileNotFoundError
        If no checkpoint can be resolved.
    """
    if explicit_checkpoint_path is not None:
        return explicit_checkpoint_path.resolve()

    best_config_path = (
        data_root / species / "tuning" / pair_model_name / "pair" / "best_config.json"
    )
    if best_config_path.is_file():
        payload = read_json_object(best_config_path)
        if payload is not None and str(payload.get("status", "")).strip().lower() == "ok":
            checkpoint = extract_task_checkpoint_path(
                payload,
                task="pair",
                base_dir=best_config_path.parent,
            )
            if checkpoint is not None:
                root_dir = Path(model_root()).resolve()
                return resolve_existing_checkpoint_path(
                    checkpoint, model_root_dir=root_dir
                )

    local_pair_dir = Path(model_root()).resolve() / species / "pair"
    if local_pair_dir.is_dir():
        for path in sorted(
            local_pair_dir.iterdir(),
            key=lambda p: (p.stat().st_mtime_ns, p.name),
            reverse=True,
        ):
            if path.is_file() and path.suffix == ".pt" and (
                path.name == f"{pair_model_name}.pt"
                or path.name.startswith(f"{pair_model_name}_")
                or path.name.startswith(f"{pair_model_name}.")
            ):
                return path.resolve()

    raise FileNotFoundError(
        f"No pair checkpoint found for species={species} model={pair_model_name}."
    )


def load_dnabert_pair_model(
    checkpoint_path: Path,
    device: str,
) -> LoadedDnabertPairModel:
    """Load one DNABERT pair model checkpoint.

    Parameters
    ----------
    checkpoint_path : Path
        Path to the ``*.pt`` pair checkpoint.
    device : str
        PyTorch device string.

    Returns
    -------
    LoadedDnabertPairModel
        Loaded model with tokenizer and window metadata.
    """
    raw_ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(raw_ckpt, dict):
        raise ValueError(f"Invalid pair checkpoint payload: {checkpoint_path}")
    donor_window_len = int(raw_ckpt.get("donor_window_len", 100))
    acceptor_window_len = int(raw_ckpt.get("acceptor_window_len", 100))

    model, model_config, tokenizer = dnabert_model.load_task_model(
        str(checkpoint_path), device
    )
    max_tokens = _optional_positive_int(model_config.get("max_tokens")) or donor_window_len
    input_kmer = _optional_positive_int(model_config.get("input_kmer"))

    return LoadedDnabertPairModel(
        model=model,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        input_kmer=input_kmer,
        donor_window_len=donor_window_len,
        acceptor_window_len=acceptor_window_len,
    )


def score_dnabert_pair_candidates(
    loaded_pair: LoadedDnabertPairModel,
    pair_candidates: Sequence,
    device: str,
    batch_size: int,
) -> list[float]:
    """Score donor/acceptor pair candidates with a DNABERT pair model.

    Parameters
    ----------
    loaded_pair : LoadedDnabertPairModel
        Loaded pair model.
    pair_candidates : Sequence
        ``PairCandidate`` objects with ``donor_window`` and ``acceptor_window``.
    device : str
        PyTorch device string.
    batch_size : int
        Inference batch size.

    Returns
    -------
    list[float]
        Pair scores in candidate order.
    """
    if not pair_candidates:
        return []
    donor_seqs = [c.donor_window for c in pair_candidates]
    acceptor_seqs = [c.acceptor_window for c in pair_candidates]
    scores = dnabert_model.score_sequence_pairs(
        model=loaded_pair.model,
        donor_sequences=donor_seqs,
        acceptor_sequences=acceptor_seqs,
        tokenizer=loaded_pair.tokenizer,
        max_tokens=loaded_pair.max_tokens,
        device=device,
        batch_size=batch_size,
        task_name="pair_scan",
        input_kmer=loaded_pair.input_kmer,
        use_amp=False,
        amp_dtype=None,
    )
    return [float(s) for s in np.asarray(scores, dtype=np.float64)]


def discover_score_test_suite_cases(suite_root: Path) -> list[ScoreTestSuiteCase]:
    """Discover score-test-suite FASTA cases under one suite root.

    Parameters
    ----------
    suite_root : Path
        Directory that contains case subdirectories such as ``cds-*`` or
        ``rna-*``.

    Returns
    -------
    list[ScoreTestSuiteCase]
        Sorted discovered cases.

    Raises
    ------
    FileNotFoundError
        If the suite root does not exist or one discovered case is missing its
        expected FASTA file.
    ValueError
        If no score-test-suite cases are found.
    """
    if not suite_root.is_dir():
        raise FileNotFoundError(f"score_test_suite directory not found: {suite_root}")

    cases: list[ScoreTestSuiteCase] = []
    for child in sorted(suite_root.iterdir(), key=lambda path: path.name):
        if not child.is_dir():
            continue
        if not (child.name.startswith("cds-") or child.name.startswith("rna-")):
            continue
        fasta_path = child / f"{child.name}.fa"
        if not fasta_path.is_file():
            raise FileNotFoundError(
                "Expected one matching FASTA file for score-test-suite case: "
                f"{fasta_path}"
            )
        cases.append(
            ScoreTestSuiteCase(
                case_name=child.name,
                fasta_path=fasta_path.resolve(),
            )
        )

    if not cases:
        raise ValueError(f"No score-test-suite cases found under {suite_root}")
    return cases


def build_score_test_suite_output_paths(
    *,
    students_dir: Path,
    case_name: str,
    tag: str,
) -> tuple[Path, Path]:
    """Build output paths for one score-test-suite case.

    Parameters
    ----------
    students_dir : Path
        Output directory for generated student-style score tables.
    case_name : str
        Case directory stem such as ``cds-NP_477286.2``.
    tag : str
        Variant suffix such as ``h``.

    Returns
    -------
    tuple[Path, Path]
        Donor and acceptor output paths.

    Raises
    ------
    ValueError
        If ``case_name`` or ``tag`` are empty.
    """
    if case_name.strip() == "":
        raise ValueError("case_name must be non-empty.")
    if tag.strip() == "":
        raise ValueError("tag must be non-empty.")

    donor_output_path = students_dir / f"out.gt.{case_name}.{tag}.txt"
    acceptor_output_path = students_dir / f"out.ag.{case_name}.{tag}.txt"
    return donor_output_path, acceptor_output_path


def score_one_sequence(
    *,
    sequence: str,
    resolved: ResolvedBestModelPaths,
    device: str,
    batch_size: int,
) -> tuple[
    list[ScoredCandidate],
    list[float],
    list[ScoredCandidate],
    list[float],
]:
    """Build candidate windows and score one normalized DNA sequence.

    Parameters
    ----------
    sequence : str
        Upper-cased contiguous DNA sequence.
    resolved : ResolvedBestModelPaths
        Resolved checkpoint metadata for donor and acceptor models.
    device : str
        Torch device string.
    batch_size : int
        Inference batch size.

    Returns
    -------
    tuple[list[ScoredCandidate], list[float], list[ScoredCandidate], list[float]]
        Donor candidates, donor scores, acceptor candidates, and acceptor
        scores in candidate order.
    """
    donor_candidates, acceptor_candidates = build_candidate_windows(
        sequence,
        donor_window_len=resolved.donor_window_len,
        acceptor_window_len=resolved.acceptor_window_len,
    )
    donor_scores = score_candidate_windows(
        candidates=donor_candidates,
        checkpoint_path=resolved.donor_checkpoint_path,
        window_len=resolved.donor_window_len,
        device=device,
        batch_size=batch_size,
        model_name=resolved.resolved_model_name or "",
    )
    acceptor_scores = score_candidate_windows(
        candidates=acceptor_candidates,
        checkpoint_path=resolved.acceptor_checkpoint_path,
        window_len=resolved.acceptor_window_len,
        device=device,
        batch_size=batch_size,
        model_name=resolved.resolved_model_name or "",
    )
    return donor_candidates, donor_scores, acceptor_candidates, acceptor_scores


def score_test_suite_cases(
    *,
    suite_root: Path,
    students_dir: Path,
    tag: str,
    resolved: ResolvedBestModelPaths,
    device: str,
    batch_size: int,
    pair_config: PairScoringConfig | None = None,
) -> list[ScoreTestSuiteSummary]:
    """Score every discovered score-test-suite FASTA and write student outputs.

    Parameters
    ----------
    suite_root : Path
        Root directory that contains score-test-suite case subdirectories.
    students_dir : Path
        Output directory for generated student-style ``out.gt`` and
        ``out.ag`` tables.
    tag : str
        Variant suffix such as ``h``.
    resolved : ResolvedBestModelPaths
        Resolved checkpoint metadata for donor and acceptor models.
    device : str
        Torch device string.
    batch_size : int
        Inference batch size.
    pair_config : PairScoringConfig | None
        Optional DNABERT pair model config. When provided, site scores are
        adjusted using the best pair score touching each candidate site.

    Returns
    -------
    list[ScoreTestSuiteSummary]
        One summary per scored case.
    """
    cases = discover_score_test_suite_cases(suite_root)
    summaries: list[ScoreTestSuiteSummary] = []

    for case in cases:
        sequence = normalize_sequence_text(case.fasta_path.read_text(encoding="utf-8"))
        donor_candidates, donor_scores, acceptor_candidates, acceptor_scores = (
            score_one_sequence(
                sequence=sequence,
                resolved=resolved,
                device=device,
                batch_size=batch_size,
            )
        )
        donor_output_path, acceptor_output_path = build_score_test_suite_output_paths(
            students_dir=students_dir,
            case_name=case.case_name,
            tag=tag,
        )
        write_scores(donor_output_path, donor_candidates, donor_scores)
        write_scores(acceptor_output_path, acceptor_candidates, acceptor_scores)

        if pair_config is not None:
            _apply_pair_scoring_to_case(
                sequence=sequence,
                donor_candidates=donor_candidates,
                donor_scores=donor_scores,
                acceptor_candidates=acceptor_candidates,
                acceptor_scores=acceptor_scores,
                donor_output_path=donor_output_path,
                acceptor_output_path=acceptor_output_path,
                pair_config=pair_config,
                device=device,
                case_name=case.case_name,
            )

        summaries.append(
            ScoreTestSuiteSummary(
                case_name=case.case_name,
                donor_candidate_count=len(donor_candidates),
                acceptor_candidate_count=len(acceptor_candidates),
                donor_output_path=donor_output_path,
                acceptor_output_path=acceptor_output_path,
            )
        )

    return summaries


def _apply_pair_scoring_to_case(
    *,
    sequence: str,
    donor_candidates: Sequence[ScoredCandidate],
    donor_scores: Sequence[float],
    acceptor_candidates: Sequence[ScoredCandidate],
    acceptor_scores: Sequence[float],
    donor_output_path: Path,
    acceptor_output_path: Path,
    pair_config: PairScoringConfig,
    device: str,
    case_name: str,
) -> None:
    """Score GT/AG pair combinations and adjust site scores in-place on disk."""
    donor_score_map = {
        c.coordinate: float(s)
        for c, s in zip(donor_candidates, donor_scores)
    }
    acceptor_score_map = {
        c.coordinate: float(s)
        for c, s in zip(acceptor_candidates, acceptor_scores)
    }

    lpm = pair_config.loaded_pair_model
    pair_candidates = build_pair_candidates(
        sequence=sequence,
        donor_scores=donor_score_map,
        acceptor_scores=acceptor_score_map,
        donor_window_len=lpm.donor_window_len,
        acceptor_window_len=lpm.acceptor_window_len,
        inactive_score=pair_config.inactive_score,
        min_intron_length=pair_config.min_intron_length,
    )

    pair_scores_list = score_dnabert_pair_candidates(
        lpm,
        pair_candidates,
        device,
        pair_config.pair_batch_size,
    )

    donor_adjusted, acceptor_adjusted, summary = apply_pair_score_adjustments(
        donor_scores=donor_score_map,
        acceptor_scores=acceptor_score_map,
        pair_candidates=pair_candidates,
        pair_scores=pair_scores_list,
        inactive_score=pair_config.inactive_score,
        pair_score_center=pair_config.pair_score_center,
        pair_score_scale=pair_config.pair_score_scale,
        pair_delta_min=pair_config.pair_delta_min,
        pair_delta_max=pair_config.pair_delta_max,
        no_pair_penalty=pair_config.no_pair_penalty,
    )

    write_sparse_scores(donor_adjusted, donor_output_path)
    write_sparse_scores(acceptor_adjusted, acceptor_output_path)

    print(
        f"[scan_splice_candidate_sites] pair_scoring case={case_name} "
        f"pair_candidates={len(pair_candidates)} "
        f"donor(+{summary.donor_bonus_count}/-{summary.donor_penalty_count}/"
        f"nopair={summary.donor_no_pair_count}) "
        f"acceptor(+{summary.acceptor_bonus_count}/-{summary.acceptor_penalty_count}/"
        f"nopair={summary.acceptor_no_pair_count})"
    )


def resolve_model_paths_from_args(
    args: argparse.Namespace,
    *,
    device: str,
) -> ResolvedBestModelPaths:
    """Resolve donor and acceptor checkpoint metadata from CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    device : str
        Torch device string.

    Returns
    -------
    ResolvedBestModelPaths
        Resolved best-config and checkpoint metadata for both site tasks.

    Raises
    ------
    ValueError
        If ``--best-config-path`` does not point at a donor or acceptor
        best-config file.
    """
    if args.best_config_path is None:
        return load_resolved_best_model_paths(
            data_root=args.data_root.resolve(),
            species=str(args.species),
            model_name=str(args.model),
            device=device,
        )

    best_config_path = args.best_config_path.resolve()
    task_dir = best_config_path.parent.name
    if task_dir not in {"donor", "acceptor"}:
        raise ValueError(
            "--best-config-path must point to a donor or acceptor "
            "best_config.json file."
        )
    donor_best_config_path = (
        best_config_path
        if task_dir == "donor"
        else best_config_path.parent.parent / "donor" / "best_config.json"
    )
    acceptor_best_config_path = (
        best_config_path
        if task_dir == "acceptor"
        else best_config_path.parent.parent / "acceptor" / "best_config.json"
    )
    donor_checkpoint_path = load_task_checkpoint_path(
        donor_best_config_path,
        "donor",
    )
    acceptor_checkpoint_path = load_task_checkpoint_path(
        acceptor_best_config_path,
        "acceptor",
    )
    donor_window_len = _load_window_len_from_checkpoint(
        donor_checkpoint_path,
        device=device,
    )
    acceptor_window_len = _load_window_len_from_checkpoint(
        acceptor_checkpoint_path,
        device=device,
    )
    return ResolvedBestModelPaths(
        best_config_path=best_config_path,
        donor_checkpoint_path=donor_checkpoint_path,
        acceptor_checkpoint_path=acceptor_checkpoint_path,
        donor_window_len=donor_window_len,
        acceptor_window_len=acceptor_window_len,
        resolved_model_name=str(args.model).strip(),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Scan GT and AG splice-site candidates on the forward strand, "
            "score them with best checkpoints, and write coordinate-score TSVs."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--species", type=str, default="Dmel")
    parser.add_argument("--model", type=str, default="cnn_v2")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--sequence-file", type=Path, default=None)
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--suite-root", type=Path, default=None)
    parser.add_argument("--students-dir", type=Path, default=None)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--best-config-path", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    # Optional DNABERT pair model for score adjustment after site scoring.
    parser.add_argument("--pair-model-name", type=str, default=None)
    parser.add_argument("--pair-checkpoint-path", type=Path, default=None)
    parser.add_argument("--pair-batch-size", type=int, default=None)
    parser.add_argument("--pair-inactive-score", type=float, default=-1000.0)
    parser.add_argument("--pair-score-center", type=float, default=-2.0)
    parser.add_argument("--pair-score-scale", type=float, default=50.0)
    parser.add_argument("--pair-delta-min", type=float, default=-150.0)
    parser.add_argument("--pair-delta-max", type=float, default=100.0)
    parser.add_argument("--pair-no-pair-penalty", type=float, default=-150.0)
    parser.add_argument("--pair-min-intron-length", type=int, default=30)
    return parser


def _read_sequence(args: argparse.Namespace) -> str:
    """Read and normalize one sequence from CLI arguments."""
    if args.sequence_file is not None:
        raw_sequence = args.sequence_file.read_text(encoding="utf-8")
    elif args.sequence is not None:
        raw_sequence = str(args.sequence)
    else:
        raise ValueError("Either --sequence-file or --sequence must be set.")
    return normalize_sequence_text(raw_sequence)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the splice-candidate scoring command-line utility."""
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    device = pick_device(str(args.device))
    resolved = resolve_model_paths_from_args(args, device=device)

    if args.suite_root is not None:
        if args.sequence_file is not None or args.sequence is not None:
            raise ValueError(
                "--suite-root cannot be combined with --sequence-file or --sequence."
            )
        tag = "" if args.tag is None else str(args.tag).strip()
        if tag == "":
            raise ValueError("--tag is required when --suite-root is set.")

        suite_root = args.suite_root.resolve()
        students_dir = (
            args.students_dir.resolve()
            if args.students_dir is not None
            else (suite_root / "Students").resolve()
        )

        pair_config: PairScoringConfig | None = None
        pair_model_name = (
            str(args.pair_model_name).strip()
            if args.pair_model_name is not None
            else ""
        )
        if pair_model_name != "":
            pair_checkpoint_path = resolve_dnabert_pair_checkpoint_path(
                data_root=args.data_root.resolve(),
                species=str(args.species),
                pair_model_name=pair_model_name,
                explicit_checkpoint_path=args.pair_checkpoint_path,
            )
            loaded_pair = load_dnabert_pair_model(pair_checkpoint_path, device)
            pair_batch_size = (
                int(args.pair_batch_size)
                if args.pair_batch_size is not None
                else int(args.batch_size)
            )
            pair_config = PairScoringConfig(
                loaded_pair_model=loaded_pair,
                inactive_score=float(args.pair_inactive_score),
                pair_score_center=float(args.pair_score_center),
                pair_score_scale=float(args.pair_score_scale),
                pair_delta_min=float(args.pair_delta_min),
                pair_delta_max=float(args.pair_delta_max),
                no_pair_penalty=float(args.pair_no_pair_penalty),
                min_intron_length=int(args.pair_min_intron_length),
                pair_batch_size=pair_batch_size,
            )
            print(
                "[scan_splice_candidate_sites] "
                f"pair_model={pair_model_name} "
                f"pair_checkpoint={pair_checkpoint_path}"
            )

        summaries = score_test_suite_cases(
            suite_root=suite_root,
            students_dir=students_dir,
            tag=tag,
            resolved=resolved,
            device=device,
            batch_size=int(args.batch_size),
            pair_config=pair_config,
        )
        print(
            "[scan_splice_candidate_sites] "
            f"model={resolved.resolved_model_name or str(args.model)} "
            f"best_config={resolved.best_config_path} "
            f"cases={len(summaries)} students_dir={students_dir}"
        )
        return 0

    if args.name is None or str(args.name).strip() == "":
        raise ValueError("--name is required when scoring one sequence.")

    sequence = _read_sequence(args)
    donor_candidates, donor_scores, acceptor_candidates, acceptor_scores = (
        score_one_sequence(
            sequence=sequence,
            resolved=resolved,
            device=device,
            batch_size=int(args.batch_size),
        )
    )

    output_dir = args.output_dir.resolve()
    donor_output_path = output_dir / f"{args.name}.gt.txt"
    acceptor_output_path = output_dir / f"{args.name}.ag.txt"
    write_scores(donor_output_path, donor_candidates, donor_scores)
    write_scores(acceptor_output_path, acceptor_candidates, acceptor_scores)

    print(
        "[scan_splice_candidate_sites] "
        f"model={resolved.resolved_model_name or str(args.model)} "
        f"best_config={resolved.best_config_path} "
        f"donor={len(donor_candidates)} acceptor={len(acceptor_candidates)} "
        f"output_dir={output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
