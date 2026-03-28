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

from models.cnn import load_task_model, score_sequences
from util.data_proc import model_root
from util.checkpoint_io import extract_task_checkpoint_path, read_json_object
from util.model_runtime import pick_device

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

    acceptor_left_offset = acceptor_window_len - 3
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
    candidate = data_root / species / "tuning" / model_name / task / "best_config.json"
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(
        "No canonical task-specific best_config.json found for "
        f"species={species} model={model_name} task={task} under {data_root}."
    )


def _resolve_json_path(raw_path: str, base_dir: Path) -> Path:
    """Resolve one JSON path string relative to a base directory."""
    path = Path(raw_path.strip())
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _resolve_existing_checkpoint_path(
    checkpoint_path: Path,
    *,
    model_root_dir: Path,
) -> Path:
    """Resolve one checkpoint path, falling back to the local model root."""
    if checkpoint_path.is_file():
        return checkpoint_path.resolve()

    path_parts = checkpoint_path.parts
    if "model" in path_parts:
        model_index = path_parts.index("model")
        relative_parts = path_parts[model_index + 1 :]
        if relative_parts:
            candidate = model_root_dir.joinpath(*relative_parts)
            if candidate.is_file():
                return candidate.resolve()

    basename = checkpoint_path.name
    if basename != "":
        candidates = sorted(
            model_root_dir.rglob(basename),
            key=lambda path: (len(path.parts), str(path)),
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()

    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


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
    return _resolve_existing_checkpoint_path(
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
        best_config_path=donor_best_config_path,
        donor_checkpoint_path=donor_checkpoint_path,
        acceptor_checkpoint_path=acceptor_checkpoint_path,
        donor_window_len=donor_window_len,
        acceptor_window_len=acceptor_window_len,
    )


def score_candidate_windows(
    *,
    candidates: Sequence[ScoredCandidate],
    checkpoint_path: Path,
    window_len: int,
    device: str,
    batch_size: int,
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

    model, _ = load_task_model(str(checkpoint_path), device)
    windows = [candidate.window for candidate in candidates]
    scores = score_sequences(
        model=model,
        sequences=windows,
        window_len=window_len,
        device=device,
        batch_size=batch_size,
        use_amp=False,
        amp_dtype=None,
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
            handle.write(f"{candidate.coordinate}\t{score:.6f}\n")


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
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--sequence-file", type=Path, default=None)
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--best-config-path", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
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
    sequence = _read_sequence(args)
    device = pick_device(str(args.device))

    if args.best_config_path is None:
        resolved = load_resolved_best_model_paths(
            data_root=args.data_root.resolve(),
            species=str(args.species),
            model_name=str(args.model),
            device=device,
        )
    else:
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
        resolved = ResolvedBestModelPaths(
            best_config_path=best_config_path,
            donor_checkpoint_path=donor_checkpoint_path,
            acceptor_checkpoint_path=acceptor_checkpoint_path,
            donor_window_len=donor_window_len,
            acceptor_window_len=acceptor_window_len,
        )

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
        batch_size=int(args.batch_size),
    )
    acceptor_scores = score_candidate_windows(
        candidates=acceptor_candidates,
        checkpoint_path=resolved.acceptor_checkpoint_path,
        window_len=resolved.acceptor_window_len,
        device=device,
        batch_size=int(args.batch_size),
    )

    output_dir = args.output_dir.resolve()
    donor_output_path = output_dir / f"{args.name}.gt.txt"
    acceptor_output_path = output_dir / f"{args.name}.ag.txt"
    write_scores(donor_output_path, donor_candidates, donor_scores)
    write_scores(acceptor_output_path, acceptor_candidates, acceptor_scores)

    print(
        "[scan_splice_candidate_sites] "
        f"best_config={resolved.best_config_path} "
        f"donor={len(donor_candidates)} acceptor={len(acceptor_candidates)} "
        f"output_dir={output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
