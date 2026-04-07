"""Reweight score-test-suite site candidates with a pair model.

This helper reads existing sparse donor/acceptor score tables, enumerates
valid donor/acceptor combinations, scores those combinations with a pair model,
and then either prunes or additively reweights site candidates before the HMM
step.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import pickle
import sys
from pathlib import Path
from typing import Callable, Sequence

from models import cnn_pair, cnn_pair_v3, cnn_v2
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
    PairScoreAdjustmentSummary,
    apply_pair_score_adjustments,
    apply_pair_score_filter,
    build_pair_candidates,
    read_fasta_sequence,
    read_sparse_scores,
    write_sparse_scores,
)

DEFAULT_PAIR_MODEL_NAME = "cnn_pair_v2"
LEGACY_MODEL_ALIASES: dict[str, tuple[str, ...]] = {
    "cnn_pair_v2": ("cnn_pair", "cnn_v2"),
    "cnn_v2_pair": ("cnn_pair_v2", "cnn_pair", "cnn_v2"),
    "cnn_pair_v3": ("cnn_pair_v3",),
    "cnn_v3_pair": ("cnn_pair_v3",),
}

PairLoadFn = Callable[[str, str], tuple[object, dict[str, object]]]
PairScoreFn = Callable[
    [object, dict[str, object], Sequence[tuple[str, str]], str, int],
    list[float],
]


@dataclass(frozen=True)
class PairBackendSpec:
    """One pair-model backend implementation."""

    name: str
    load_model: PairLoadFn
    score_pairs: PairScoreFn


@dataclass(frozen=True)
class LoadedPairModel:
    """One loaded pair model plus backend metadata."""

    backend_name: str
    model: object
    checkpoint_payload: dict[str, object]
    checkpoint_path: Path


def _score_pairs_with_cnn_pair(
    model: object,
    checkpoint_payload: dict[str, object],
    pairs: Sequence[tuple[str, str]],
    device: str,
    batch_size: int,
) -> list[float]:
    """Score pairs with the legacy ``cnn_pair`` backend."""
    scores = cnn_pair.score_pair_sequences(
        model=model,
        pairs=pairs,
        donor_window_len=int(checkpoint_payload.get("donor_window_len", 50)),
        acceptor_window_len=int(checkpoint_payload.get("acceptor_window_len", 50)),
        device=device,
        batch_size=batch_size,
        use_amp=False,
        amp_dtype=None,
    )
    return [float(score) for score in scores.tolist()]


def _score_pairs_with_cnn_pair_v3(
    model: object,
    checkpoint_payload: dict[str, object],
    pairs: Sequence[tuple[str, str]],
    device: str,
    batch_size: int,
) -> list[float]:
    """Score pairs with the ``cnn_pair_v3`` backend."""
    model_config_obj = checkpoint_payload.get("model_config", {})
    model_config = model_config_obj if isinstance(model_config_obj, dict) else {}
    scores = cnn_pair_v3.score_pair_sequences(
        model=model,
        pairs=pairs,
        donor_window_len=int(checkpoint_payload.get("donor_window_len", 50)),
        acceptor_window_len=int(checkpoint_payload.get("acceptor_window_len", 50)),
        device=device,
        input_mode=str(model_config.get("input_mode", "onehot")),
        bpe_pretrained_model_name=str(
            model_config.get(
                "bpe_pretrained_model_name",
                cnn_pair_v3.BPE_DEFAULT_MODEL_NAME,
            )
        ),
        bpe_pretrained_revision=(
            str(model_config["bpe_pretrained_revision"])
            if model_config.get("bpe_pretrained_revision") is not None
            else None
        ),
        bpe_trust_remote_code=bool(model_config.get("bpe_trust_remote_code", False)),
        batch_size=batch_size,
        use_amp=False,
        amp_dtype=None,
    )
    return [float(score) for score in scores.tolist()]


def _score_pairs_with_cnn_v2(
    model: object,
    checkpoint_payload: dict[str, object],
    pairs: Sequence[tuple[str, str]],
    device: str,
    batch_size: int,
) -> list[float]:
    """Score pairs with the ``cnn_v2`` pair backend."""
    model_config_obj = checkpoint_payload.get("model_config", {})
    model_config = model_config_obj if isinstance(model_config_obj, dict) else {}
    scores = cnn_v2.score_pair_sequences(
        model=model,
        pairs=pairs,
        donor_window_len=int(checkpoint_payload.get("donor_window_len", 50)),
        acceptor_window_len=int(checkpoint_payload.get("acceptor_window_len", 50)),
        device=device,
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
        bpe_trust_remote_code=bool(model_config.get("bpe_trust_remote_code", False)),
        batch_size=batch_size,
        use_amp=False,
        amp_dtype=None,
    )
    return [float(score) for score in scores.tolist()]


PAIR_BACKEND_SPECS: dict[str, PairBackendSpec] = {
    "cnn_pair": PairBackendSpec(
        name="cnn_pair",
        load_model=cnn_pair.load_pair_model,
        score_pairs=_score_pairs_with_cnn_pair,
    ),
    "cnn_pair_v3": PairBackendSpec(
        name="cnn_pair_v3",
        load_model=cnn_pair_v3.load_pair_model,
        score_pairs=_score_pairs_with_cnn_pair_v3,
    ),
    "cnn_v2": PairBackendSpec(
        name="cnn_v2",
        load_model=cnn_v2.load_pair_model,
        score_pairs=_score_pairs_with_cnn_v2,
    ),
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
            "Use a pair model to update donor/acceptor candidates before the "
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
    parser.add_argument(
        "--site-score-mode",
        choices=("additive", "hard_reject"),
        default="additive",
    )
    parser.add_argument("--pair-min-score", type=float, default=-2.0)
    parser.add_argument("--pair-score-center", type=float, default=-2.0)
    parser.add_argument("--pair-score-scale", type=float, default=50.0)
    parser.add_argument("--pair-delta-min", type=float, default=-150.0)
    parser.add_argument("--pair-delta-max", type=float, default=100.0)
    parser.add_argument("--no-pair-penalty", type=float, default=-150.0)
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


def backend_names_for_model_name(model_name: str) -> tuple[str, ...]:
    """Resolve backend preference order for one model/checkpoint name."""
    normalized = model_name.strip()
    if normalized in LEGACY_MODEL_ALIASES:
        return LEGACY_MODEL_ALIASES[normalized]
    if normalized.startswith("cnn_pair_v3"):
        return ("cnn_pair_v3",)
    if normalized.startswith("cnn_pair_v2"):
        return ("cnn_pair", "cnn_v2")
    if normalized.startswith("cnn_v3_pair"):
        return ("cnn_pair_v3",)
    if normalized.startswith("cnn_v2_pair"):
        return ("cnn_pair", "cnn_v2")
    if normalized.startswith("cnn_pair"):
        return ("cnn_pair",)
    if normalized.startswith("cnn_v2"):
        return ("cnn_v2", "cnn_pair")
    return ("cnn_pair", "cnn_pair_v3", "cnn_v2")


def ordered_backend_specs(model_names: Sequence[str]) -> list[PairBackendSpec]:
    """Return backend specs in deterministic preference order."""
    ordered_names: list[str] = []
    for model_name in model_names:
        for backend_name in backend_names_for_model_name(model_name):
            if backend_name not in ordered_names:
                ordered_names.append(backend_name)
    return [PAIR_BACKEND_SPECS[backend_name] for backend_name in ordered_names]


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
) -> LoadedPairModel:
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
    LoadedPairModel
        Loaded model, backend name, checkpoint payload, and checkpoint path.

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

        backend_specs = ordered_backend_specs(
            [candidate_path.stem, resolved_path.stem, *model_names]
        )
        for backend_spec in backend_specs:
            try:
                model, checkpoint_payload = backend_spec.load_model(
                    str(resolved_path),
                    device,
                )
                return LoadedPairModel(
                    backend_name=backend_spec.name,
                    model=model,
                    checkpoint_payload=checkpoint_payload,
                    checkpoint_path=resolved_path,
                )
            except (
                FileNotFoundError,
                OSError,
                RuntimeError,
                ValueError,
                pickle.UnpicklingError,
            ) as exc:
                errors.append(
                    _summarize_load_error(
                        resolved_path,
                        exc,
                        backend_name=backend_spec.name,
                    )
                )

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


def score_pairs_with_backend(
    loaded_model: LoadedPairModel,
    *,
    pairs: Sequence[tuple[str, str]],
    device: str,
    batch_size: int,
) -> list[float]:
    """Score pair windows with the selected backend."""
    backend_spec = PAIR_BACKEND_SPECS[loaded_model.backend_name]
    return backend_spec.score_pairs(
        loaded_model.model,
        loaded_model.checkpoint_payload,
        pairs,
        device,
        batch_size,
    )


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
        loaded_model = load_pair_model_with_fallback(
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

    donor_window_len = int(loaded_model.checkpoint_payload.get("donor_window_len", 50))
    acceptor_window_len = int(
        loaded_model.checkpoint_payload.get("acceptor_window_len", 50)
    )
    pair_candidates = build_pair_candidates(
        sequence=sequence,
        donor_scores=donor_scores,
        acceptor_scores=acceptor_scores,
        donor_window_len=donor_window_len,
        acceptor_window_len=acceptor_window_len,
        inactive_score=float(args.inactive_score),
        min_intron_length=int(args.min_intron_length),
    )

    pair_scores: list[float] = []
    if pair_candidates:
        pair_scores = score_pairs_with_backend(
            loaded_model,
            pairs=pair_candidates_to_sequences(pair_candidates),
            device=device_name,
            batch_size=int(args.batch_size),
        )

    donor_input_active_count = sum(
        1 for score in donor_scores.values() if score > float(args.inactive_score)
    )
    acceptor_input_active_count = sum(
        1 for score in acceptor_scores.values() if score > float(args.inactive_score)
    )
    score_summary_threshold = (
        float(args.pair_min_score)
        if args.site_score_mode == "hard_reject"
        else float(args.pair_score_center)
    )

    if args.site_score_mode == "hard_reject":
        donor_output, acceptor_output, filter_summary = apply_pair_score_filter(
            donor_scores=donor_scores,
            acceptor_scores=acceptor_scores,
            pair_candidates=pair_candidates,
            pair_scores=pair_scores,
            inactive_score=float(args.inactive_score),
            pair_keep_threshold=float(args.pair_min_score),
        )
        update_summary_text = (
            "pair_updates="
            f"donor_pruned={filter_summary.donor_pruned_count} "
            f"acceptor_pruned={filter_summary.acceptor_pruned_count}"
        )
        donor_input_active_count = filter_summary.donor_input_active_count
        acceptor_input_active_count = filter_summary.acceptor_input_active_count
    else:
        donor_output, acceptor_output, adjustment_summary = (
            apply_pair_score_adjustments(
                donor_scores=donor_scores,
                acceptor_scores=acceptor_scores,
                pair_candidates=pair_candidates,
                pair_scores=pair_scores,
                inactive_score=float(args.inactive_score),
                pair_score_center=float(args.pair_score_center),
                pair_score_scale=float(args.pair_score_scale),
                pair_delta_min=float(args.pair_delta_min),
                pair_delta_max=float(args.pair_delta_max),
                no_pair_penalty=float(args.no_pair_penalty),
            )
        )
        update_summary_text = format_adjustment_summary(adjustment_summary)
        donor_input_active_count = adjustment_summary.donor_input_active_count
        acceptor_input_active_count = (
            adjustment_summary.acceptor_input_active_count
        )
    write_sparse_scores(donor_output, args.donor_output)
    write_sparse_scores(acceptor_output, args.acceptor_output)

    donor_remaining = sum(
        1 for score in donor_output.values() if score > float(args.inactive_score)
    )
    acceptor_remaining = sum(
        1 for score in acceptor_output.values() if score > float(args.inactive_score)
    )
    score_summary = format_pair_score_summary(
        pair_scores,
        threshold=score_summary_threshold,
    )
    print(
        "[filter_score_test_suite_pairs] "
        f"site_score_mode={args.site_score_mode} "
        f"backend={loaded_model.backend_name} "
        f"checkpoint={loaded_model.checkpoint_path} "
        f"pair_candidates={len(pair_candidates)} "
        f"{score_summary} "
        f"{update_summary_text} "
        f"donor_active={donor_input_active_count}->{donor_remaining} "
        f"acceptor_active={acceptor_input_active_count}->{acceptor_remaining} "
        f"pair_min_score={args.pair_min_score} "
        f"pair_score_center={args.pair_score_center}",
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


def format_pair_score_summary(scores: Sequence[float], *, threshold: float) -> str:
    """Build one concise summary string for pair-score distributions."""
    if not scores:
        return "pair_scores=none"
    sorted_scores = sorted(float(score) for score in scores)
    below_count = sum(score <= threshold for score in sorted_scores)
    fraction = below_count / len(sorted_scores)
    median = sorted_scores[len(sorted_scores) // 2]
    p90_index = int(round((len(sorted_scores) - 1) * 0.9))
    p90 = sorted_scores[p90_index]
    minimum = sorted_scores[0]
    maximum = sorted_scores[-1]
    return (
        "pair_scores="
        f"{below_count}/{len(sorted_scores)}<=thresh({fraction:.3f}) "
        f"min={minimum:.3f} median={median:.3f} p90={p90:.3f} max={maximum:.3f}"
    )


def format_adjustment_summary(summary: PairScoreAdjustmentSummary) -> str:
    """Build one concise summary string for additive pair-score updates."""
    return (
        "pair_adjustments="
        f"donor(+{summary.donor_bonus_count}/-{summary.donor_penalty_count}/"
        f"nopair{summary.donor_no_pair_count}) "
        f"acceptor(+{summary.acceptor_bonus_count}/"
        f"-{summary.acceptor_penalty_count}/"
        f"nopair{summary.acceptor_no_pair_count})"
    )


def _summarize_load_error(
    path: Path,
    exc: BaseException,
    *,
    backend_name: str,
) -> str:
    """Collapse one checkpoint-load failure into a short log line."""
    first_line = str(exc).splitlines()[0].strip()
    if first_line == "":
        first_line = exc.__class__.__name__
    return (
        f"{path.name} via {backend_name}: "
        f"{exc.__class__.__name__}: {first_line}"
    )


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
