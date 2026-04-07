"""Prune score-test-suite site candidates with a pair model.

This helper reads existing sparse donor/acceptor score tables, enumerates
valid donor/acceptor combinations, scores those combinations with a pair model,
and drops site candidates that never participate in a good pair.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Sequence

from models import cnn_v2
from util.checkpoint_io import (
    extract_task_checkpoint_path,
    read_json_object,
    resolve_existing_checkpoint_path,
)
from util.data_proc import data_root, model_root
from util.model_runtime import pick_device
from util.path_format import resolve_path_string
from util.score_test_suite_pair_filter import (
    PairCandidate,
    apply_pair_score_filter,
    build_pair_candidates,
    read_fasta_sequence,
    read_sparse_scores,
    write_sparse_scores,
)

DEFAULT_PAIR_MODEL_NAME = "cnn_pair_v2"
LEGACY_MODEL_ALIASES: dict[str, tuple[str, ...]] = {
    "cnn_pair_v2": ("cnn_v2",),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Optional CLI argument sequence. ``None`` uses ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed runtime configuration.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Use a pair model to prune donor/acceptor candidates before the "
            "score-test-suite HMM step."
        )
    )
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--donor-input", type=Path, required=True)
    parser.add_argument("--acceptor-input", type=Path, required=True)
    parser.add_argument("--donor-output", type=Path, required=True)
    parser.add_argument("--acceptor-output", type=Path, required=True)
    parser.add_argument("--species", type=str, default="Dmel")
    parser.add_argument("--model-name", type=str, default=DEFAULT_PAIR_MODEL_NAME)
    parser.add_argument("--data-root", type=Path, default=Path(data_root()))
    parser.add_argument("--best-config-path", type=Path, default=None)
    parser.add_argument("--pair-checkpoint-path", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--inactive-score", type=float, default=-1000.0)
    parser.add_argument("--pair-min-score", type=float, default=-2.0)
    parser.add_argument("--min-intron-length", type=int, default=30)
    parser.add_argument(
        "--missing-pair-model-mode",
        choices=("error", "skip"),
        default="skip",
    )
    return parser.parse_args(argv)


def default_best_config_path(
    *,
    data_root_dir: Path,
    species: str,
    model_name: str,
) -> Path:
    """Return the default pair best-config location.

    Parameters
    ----------
    data_root_dir : Path
        Project data root.
    species : str
        Species identifier.
    model_name : str
        Pair model name.

    Returns
    -------
    Path
        Default ``best_config.json`` path for the pair model.
    """
    return data_root_dir / species / "tuning" / model_name / "pair" / "best_config.json"


def ordered_model_names(model_name: str, published_name: str | None) -> list[str]:
    """Build ordered checkpoint-name candidates for local fallback lookup.

    Parameters
    ----------
    model_name : str
        Requested public model name.
    published_name : str | None
        Published checkpoint stem from best-config metadata, when available.

    Returns
    -------
    list[str]
        Ordered unique model-name stems.
    """
    ordered: list[str] = []
    for candidate in (
        published_name,
        model_name.strip(),
        *LEGACY_MODEL_ALIASES.get(model_name.strip(), ()),
    ):
        if candidate is None:
            continue
        normalized = candidate.strip()
        if normalized == "" or normalized in ordered:
            continue
        ordered.append(normalized)
    return ordered


def load_checkpoint_candidates_from_best_config(
    best_config_path: Path,
) -> tuple[list[Path], str | None]:
    """Extract checkpoint candidates from one pair best-config payload.

    Parameters
    ----------
    best_config_path : Path
        Best-config JSON path.

    Returns
    -------
    tuple[list[Path], str | None]
        Candidate checkpoint paths plus optional published name.
    """
    payload = read_json_object(best_config_path)
    if payload is None:
        return [], None

    candidates: list[Path] = []
    published_name_obj = payload.get("published_name")
    published_name = (
        str(published_name_obj).strip()
        if isinstance(published_name_obj, str) and str(published_name_obj).strip() != ""
        else None
    )

    checkpoint = extract_task_checkpoint_path(
        payload,
        task="pair",
        base_dir=best_config_path.parent,
    )
    if checkpoint is not None:
        candidates.append(checkpoint)

    metrics_json_obj = payload.get("metrics_json")
    if isinstance(metrics_json_obj, str) and metrics_json_obj.strip() != "":
        metrics_path = resolve_path_string(
            metrics_json_obj,
            base_dir=best_config_path.parent,
        )
        metrics_payload = read_json_object(metrics_path)
        if metrics_payload is not None:
            metrics_checkpoint = extract_task_checkpoint_path(
                metrics_payload,
                task="pair",
                base_dir=metrics_path.parent,
            )
            if metrics_checkpoint is not None:
                candidates.append(metrics_checkpoint)

    return _deduplicate_paths(candidates), published_name


def resolve_local_pair_checkpoint_candidates(
    *,
    species: str,
    model_names: Sequence[str],
) -> list[Path]:
    """Discover locally available pair checkpoints by model-name prefix.

    Parameters
    ----------
    species : str
        Species identifier.
    model_names : Sequence[str]
        Ordered public or legacy model-name stems.

    Returns
    -------
    list[Path]
        Existing local checkpoint candidates sorted by discovery priority and
        modification time.
    """
    pair_dir = Path(model_root()).resolve() / species / "pair"
    if not pair_dir.is_dir():
        return []

    all_candidates = sorted(
        (
            path
            for path in pair_dir.iterdir()
            if path.is_file() and path.suffix == ".pt"
        ),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )

    ordered: list[Path] = []
    for model_name in model_names:
        exact_name = f"{model_name}.pt"
        for path in all_candidates:
            if path.name == exact_name:
                ordered.append(path.resolve())
        for path in all_candidates:
            if path.name.startswith(f"{model_name}_") or path.name.startswith(
                f"{model_name}."
            ):
                ordered.append(path.resolve())
    return _deduplicate_paths(ordered)


def load_pair_model_with_fallback(
    *,
    species: str,
    model_name: str,
    device: str,
    best_config_path: Path | None,
    explicit_checkpoint_path: Path | None,
) -> tuple[object, dict[str, object], Path]:
    """Load one usable pair model, trying best-config and local fallbacks.

    Parameters
    ----------
    species : str
        Species identifier.
    model_name : str
        Requested pair model name.
    device : str
        PyTorch device string.
    best_config_path : Path | None
        Best-config JSON path, when available.
    explicit_checkpoint_path : Path | None
        Explicit checkpoint override.

    Returns
    -------
    tuple[object, dict[str, object], Path]
        Loaded model, checkpoint payload, and resolved checkpoint path.

    Raises
    ------
    RuntimeError
        If no candidate checkpoint can be loaded successfully.
    """
    model_root_dir = Path(model_root()).resolve()
    published_name: str | None = None
    raw_candidates: list[Path] = []

    if explicit_checkpoint_path is not None:
        raw_candidates.append(explicit_checkpoint_path.resolve())

    if best_config_path is not None:
        best_candidates, published_name = load_checkpoint_candidates_from_best_config(
            best_config_path
        )
        raw_candidates.extend(best_candidates)

    model_names = ordered_model_names(model_name, published_name)
    raw_candidates.extend(
        resolve_local_pair_checkpoint_candidates(
            species=species,
            model_names=model_names,
        )
    )

    candidate_paths = _deduplicate_paths(raw_candidates)
    errors: list[str] = []
    for candidate_path in candidate_paths:
        resolved_path = candidate_path
        try:
            resolved_path = resolve_existing_checkpoint_path(
                candidate_path,
                model_root_dir=model_root_dir,
            )
        except FileNotFoundError:
            if not candidate_path.is_file():
                errors.append(f"{candidate_path.name}: file not found")
                continue
            resolved_path = candidate_path.resolve()

        try:
            model, checkpoint_payload = cnn_v2.load_pair_model(
                str(resolved_path),
                device,
            )
            return model, checkpoint_payload, resolved_path
        except (
            FileNotFoundError,
            OSError,
            RuntimeError,
            ValueError,
            pickle.UnpicklingError,
        ) as exc:
            errors.append(_summarize_load_error(resolved_path, exc))

    error_lines = _format_load_error_lines(errors)
    raise RuntimeError(
        "Unable to load a usable pair checkpoint for "
        f"species={species} model={model_name}.\n{error_lines}"
    )


def pair_candidates_to_sequences(
    pair_candidates: Sequence[PairCandidate],
) -> list[tuple[str, str]]:
    """Convert structured pair candidates into model input tuples.

    Parameters
    ----------
    pair_candidates : Sequence[PairCandidate]
        Structured candidate list.

    Returns
    -------
    list[tuple[str, str]]
        ``(donor_window, acceptor_window)`` tuples in candidate order.
    """
    return [
        (candidate.donor_window, candidate.acceptor_window)
        for candidate in pair_candidates
    ]


def run(argv: Sequence[str] | None = None) -> int:
    """Run the CLI entry point.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Optional CLI token sequence. ``None`` uses ``sys.argv``.

    Returns
    -------
    int
        Process exit status.
    """
    args = parse_args(argv)
    best_config_path = args.best_config_path
    if best_config_path is None:
        best_config_path = default_best_config_path(
            data_root_dir=args.data_root.resolve(),
            species=args.species,
            model_name=args.model_name,
        )

    donor_scores = read_sparse_scores(args.donor_input)
    acceptor_scores = read_sparse_scores(args.acceptor_input)
    sequence = read_fasta_sequence(args.fasta)
    device_name = pick_device(args.device)

    try:
        pair_model, checkpoint_payload, checkpoint_path = load_pair_model_with_fallback(
            species=args.species,
            model_name=args.model_name,
            device=device_name,
            best_config_path=best_config_path,
            explicit_checkpoint_path=args.pair_checkpoint_path,
        )
    except RuntimeError as exc:
        if args.missing_pair_model_mode == "skip":
            write_sparse_scores(donor_scores, args.donor_output)
            write_sparse_scores(acceptor_scores, args.acceptor_output)
            print(
                "[filter_score_test_suite_pairs] "
                f"pair filtering skipped: {exc}",
                file=sys.stderr,
            )
            return 0
        raise

    donor_window_len = int(checkpoint_payload.get("donor_window_len", 50))
    acceptor_window_len = int(checkpoint_payload.get("acceptor_window_len", 50))
    pair_candidates = build_pair_candidates(
        sequence=sequence,
        donor_scores=donor_scores,
        acceptor_scores=acceptor_scores,
        donor_window_len=donor_window_len,
        acceptor_window_len=acceptor_window_len,
        inactive_score=float(args.inactive_score),
        min_intron_length=int(args.min_intron_length),
    )

    model_config_obj = checkpoint_payload.get("model_config", {})
    model_config = (
        model_config_obj if isinstance(model_config_obj, dict) else {}
    )
    pair_scores: list[float] = []
    if pair_candidates:
        scores_array = cnn_v2.score_pair_sequences(
            model=pair_model,
            pairs=pair_candidates_to_sequences(pair_candidates),
            donor_window_len=donor_window_len,
            acceptor_window_len=acceptor_window_len,
            device=device_name,
            input_mode=str(model_config.get("input_mode", "onehot")),
            bpe_pretrained_model_name=str(
                model_config.get(
                    "bpe_pretrained_model_name",
                    cnn_v2.BPE_DEFAULT_MODEL_NAME,
                )
            ),
            bpe_pretrained_revision=(
                str(model_config["bpe_pretrained_revision"])
                if model_config.get("bpe_pretrained_revision") is not None
                else None
            ),
            bpe_trust_remote_code=bool(
                model_config.get("bpe_trust_remote_code", False)
            ),
            batch_size=int(args.batch_size),
            use_amp=False,
            amp_dtype=None,
        )
        pair_scores = [float(score) for score in scores_array.tolist()]

    donor_output, acceptor_output, summary = apply_pair_score_filter(
        donor_scores=donor_scores,
        acceptor_scores=acceptor_scores,
        pair_candidates=pair_candidates,
        pair_scores=pair_scores,
        inactive_score=float(args.inactive_score),
        pair_keep_threshold=float(args.pair_min_score),
    )
    write_sparse_scores(donor_output, args.donor_output)
    write_sparse_scores(acceptor_output, args.acceptor_output)

    donor_remaining = sum(
        1 for score in donor_output.values() if score > float(args.inactive_score)
    )
    acceptor_remaining = sum(
        1 for score in acceptor_output.values() if score > float(args.inactive_score)
    )
    print(
        "[filter_score_test_suite_pairs] "
        f"checkpoint={checkpoint_path} "
        f"pair_candidates={summary.pair_candidate_count} "
        f"donor_active={summary.donor_input_active_count}->{donor_remaining} "
        f"acceptor_active={summary.acceptor_input_active_count}->{acceptor_remaining} "
        f"pair_min_score={args.pair_min_score}",
        file=sys.stderr,
    )
    return 0


def _deduplicate_paths(paths: Sequence[Path]) -> list[Path]:
    """Return deterministic unique paths while preserving first occurrence."""
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in paths:
        resolved = path.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(resolved)
    return ordered


def _summarize_load_error(path: Path, exc: BaseException) -> str:
    """Collapse one checkpoint-load failure into a short log line."""
    first_line = str(exc).splitlines()[0].strip()
    if first_line == "":
        first_line = exc.__class__.__name__
    return f"{path.name}: {exc.__class__.__name__}: {first_line}"


def _format_load_error_lines(errors: Sequence[str], *, limit: int = 6) -> str:
    """Format a bounded list of checkpoint-load errors for user-facing logs."""
    if not errors:
        return "no candidates discovered"
    displayed = list(errors[:limit])
    if len(errors) > limit:
        displayed.append(f"... {len(errors) - limit} more candidates omitted")
    return "\n".join(displayed)


if __name__ == "__main__":
    raise SystemExit(run())
