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


def resolve_best_config_path(data_root: Path, species: str, model_name: str) -> Path:
    """Resolve the canonical best-config path for one site-scoring model.

    Parameters
    ----------
    data_root : Path
        Repository data root.
    species : str
        Species identifier.
    model_name : str
        Model name, for example ``cnn_v2``.

    Returns
    -------
    Path
        Best-config JSON path.

    Raises
    ------
    FileNotFoundError
        If no suitable best-config file exists.
    """
    candidates = [
        data_root / species / "tuning" / model_name / "both" / "best_config.json",
        data_root / species / "tuning" / model_name / "best_config.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "No canonical best_config.json found for "
        f"species={species} model={model_name} under {data_root}."
    )


def _resolve_json_path(raw_path: str, base_dir: Path) -> Path:
    """Resolve one JSON path string relative to a base directory."""
    path = Path(raw_path.strip())
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def load_best_checkpoint_paths(best_config_path: Path) -> tuple[Path, Path]:
    """Load donor and acceptor checkpoint paths from one best-config payload.

    Parameters
    ----------
    best_config_path : Path
        Canonical best-config path.

    Returns
    -------
    tuple[Path, Path]
        Donor and acceptor checkpoint paths.

    Raises
    ------
    FileNotFoundError
        If the payload or referenced checkpoint paths cannot be resolved.
    ValueError
        If the best-config payload is malformed.
    """
    payload = read_json_object(best_config_path)
    if payload is None:
        raise FileNotFoundError(f"best_config not found: {best_config_path}")
    status = str(payload.get("status", "")).strip().lower()
    if status != "ok":
        raise ValueError(
            f"Expected best_config status='ok', got: {status or '<missing>'}"
        )

    donor_checkpoint = extract_task_checkpoint_path(
        payload,
        task="donor",
        base_dir=best_config_path.parent,
    )
    acceptor_checkpoint = extract_task_checkpoint_path(
        payload,
        task="acceptor",
        base_dir=best_config_path.parent,
    )
    if donor_checkpoint is not None and acceptor_checkpoint is not None:
        return donor_checkpoint, acceptor_checkpoint

    source_donor_best_config = payload.get("source_donor_best_config")
    source_acceptor_best_config = payload.get("source_acceptor_best_config")
    if not isinstance(source_donor_best_config, str) or not isinstance(
        source_acceptor_best_config, str
    ):
        raise FileNotFoundError(
            "best_config does not contain donor/acceptor checkpoints or "
            "source best-config references."
        )

    donor_source_path = _resolve_json_path(
        source_donor_best_config,
        best_config_path.parent,
    )
    acceptor_source_path = _resolve_json_path(
        source_acceptor_best_config,
        best_config_path.parent,
    )
    donor_payload = read_json_object(donor_source_path)
    acceptor_payload = read_json_object(acceptor_source_path)
    if donor_payload is None or acceptor_payload is None:
        raise FileNotFoundError(
            "Unable to read source best-config payloads referenced by "
            f"{best_config_path}."
        )

    donor_checkpoint = extract_task_checkpoint_path(
        donor_payload,
        task="donor",
        base_dir=donor_source_path.parent,
    )
    acceptor_checkpoint = extract_task_checkpoint_path(
        acceptor_payload,
        task="acceptor",
        base_dir=acceptor_source_path.parent,
    )
    if donor_checkpoint is None or acceptor_checkpoint is None:
        raise FileNotFoundError(
            "Unable to resolve donor and acceptor checkpoint paths from "
            f"{best_config_path}."
        )
    return donor_checkpoint, acceptor_checkpoint


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
    best_config_path = resolve_best_config_path(data_root, species, model_name)
    donor_checkpoint_path, acceptor_checkpoint_path = load_best_checkpoint_paths(
        best_config_path
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
        donor_checkpoint_path, acceptor_checkpoint_path = load_best_checkpoint_paths(
            best_config_path
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
