"""Migrate legacy CNN-pair best settings into ``cnn_pair_v2`` runs.

This tool automates:

1. Re-running legacy pair best configs with ``cnn_pair_v2`` in one-hot 100bp mode.
2. Promoting candidate outputs to canonical ``cnn_pair_v2`` outputs.
3. Archiving legacy non-tuning artifacts into an archive directory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class VariantSpec:
    """Configuration for one legacy pair variant to replay."""

    name: str
    tuned_model_name: str
    target_model_name: str
    target_artifact_stem: str
    pair_mode: str
    train_target: str
    mask_mode: bool


@dataclass(frozen=True)
class CandidateResult:
    """Execution result for one species/variant candidate run."""

    species: str
    variant: str
    return_code: int
    baseline_variant_best_f1: float | None
    baseline_target_best_f1: float | None
    candidate_best_f1: float | None
    promoted: bool
    run_dir: Path


LEGACY_VARIANTS: dict[str, VariantSpec] = {
    "cnn_pair": VariantSpec(
        name="cnn_pair",
        tuned_model_name="cnn_pair",
        target_model_name="cnn_pair_v2",
        target_artifact_stem="cnn_pair_v2",
        pair_mode="pair",
        train_target="pair",
        mask_mode=False,
    ),
    "cnn_pair_mask": VariantSpec(
        name="cnn_pair_mask",
        tuned_model_name="cnn_pair_mask",
        target_model_name="cnn_pair_v2",
        target_artifact_stem="cnn_pair_v2",
        pair_mode="pair",
        train_target="pair",
        mask_mode=True,
    ),
}

_PAIR_ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        "batch_size",
        "lr",
        "loss",
        "conv_channels",
        "donor_conv_channels",
        "acceptor_conv_channels",
        "kernel_size",
        "kernel_sizes",
        "donor_kernel_size",
        "acceptor_kernel_size",
        "donor_kernel_sizes",
        "acceptor_kernel_sizes",
        "max_pool_size",
        "conv_stride",
        "head_type",
        "fusion_mode",
        "dropout",
        "fc_hidden",
        "weight_decay",
        "eta_min_ratio",
        "val_frac",
        "grad_clip",
        "pos_weight_cap",
        "focal_gamma",
        "focal_alpha_pos",
        "f1_lambda",
        "asym_gamma_pos",
        "asym_gamma_neg",
        "asym_alpha_pos",
    }
) 
_INDEPENDENT_PREFIX_KEYS: frozenset[str] = frozenset(
    {
        "batch_size",
        "lr",
        "loss",
        "conv_channels",
        "kernel_size",
        "kernel_sizes",
        "max_pool_size",
        "conv_stride",
        "head_type",
        "dropout",
        "fc_hidden",
        "weight_decay",
        "eta_min_ratio",
        "val_frac",
        "grad_clip",
        "pos_weight_cap",
        "focal_gamma",
        "focal_alpha_pos",
        "f1_lambda",
        "asym_gamma_pos",
        "asym_gamma_neg",
        "asym_alpha_pos",
    }
)
_INDEPENDENT_SHARED_KEYS: frozenset[str] = frozenset()


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Replay legacy cnn_pair best configs with cnn_pair_v2 and promote."
        )
    )
    parser.add_argument(
        "--species",
        default="Athal,Dmel,Hsap,Mmus",
        help="Comma-separated species list.",
    )
    parser.add_argument(
        "--variants",
        default="cnn_pair,cnn_pair_mask",
        help="Comma-separated legacy variants to replay.",
    )
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Project root path.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used to run src/run_model.py.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--promote-epsilon",
        type=float,
        default=0.0,
        help="Minimum strict improvement required to promote candidate outputs.",
    )
    parser.add_argument(
        "--promote-if-better",
        action="store_true",
        help=(
            "Promote candidate outputs to cnn_v2 or cnn_pair_v2 when better than "
            "current best."
        ),
    )
    parser.add_argument(
        "--archive-legacy",
        action="store_true",
        help="Move legacy cnn/cnn_pair non-tuning artifacts into archive.",
    )
    parser.add_argument(
        "--archive-root",
        default="archive",
        help="Archive root directory (relative to project root when not absolute).",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Optional cap for candidate runs (0 means no cap).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without running training or moving files.",
    )
    return parser.parse_args(argv)


def _parse_csv(raw_value: str) -> list[str]:
    """Parse one comma-separated list with stable order and deduplication."""
    values: list[str] = []
    seen: set[str] = set()
    for token in raw_value.split(","):
        item = token.strip()
        if item == "" or item in seen:
            continue
        seen.add(item)
        values.append(item)
    if not values:
        raise ValueError("Expected at least one value.")
    return values


def _read_json_object(path: Path) -> dict[str, object]:
    """Read one JSON object from disk."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _parse_best_f1(eval_path: Path) -> float | None:
    """Parse best F1 score from one eval score text file."""
    if not eval_path.is_file():
        return None
    best_f1: float | None = None
    with eval_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line == "":
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                f1_value = float(parts[-1])
            except ValueError:
                continue
            if best_f1 is None or f1_value > best_f1:
                best_f1 = f1_value
    return best_f1


def _format_cli_value(value: object) -> str:
    """Convert one JSON-like value into CLI text."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(value, ".15g")
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return ",".join(_format_cli_value(item) for item in value)
    return str(value)


def _resolve_mask_train_paths(data_root: Path, species: str) -> tuple[Path, Path]:
    """Resolve default mask-mode training paths."""
    processed_dir = data_root / species / "processed"
    return (
        processed_dir / "100bp_trimmed_npad.err",
        processed_dir / "100bp_trimmed_npad.neg.err",
    )


def _resolve_default_train_paths(data_root: Path, species: str) -> tuple[Path, Path]:
    """Resolve default non-mask training paths."""
    raw_dir = data_root / species / "raw"
    return (raw_dir / "100bp.err", raw_dir / "100bp.neg.err")


def _resolve_default_test_tsv(data_root: Path, species: str) -> Path | None:
    """Resolve default test TSV path used by wrapper pipeline."""
    unique_path = data_root / species / "processed" / "transcripts.unique.tsv"
    if unique_path.is_file():
        return unique_path
    raw_dir = data_root / species / "raw"
    for candidate_name in (
        "transcripts_mask.tsv",
        "transcripts_masked.tsv",
        "transcripts_with_intron_half.tsv",
    ):
        candidate = raw_dir / candidate_name
        if candidate.is_file():
            return candidate
    return None


def _load_sampled_params(path: Path) -> dict[str, object]:
    """Load ``sampled_params`` from one ``best_config.json``."""
    payload = _read_json_object(path)
    status = str(payload.get("status", "")).strip().lower()
    if status != "ok":
        raise ValueError(f"best_config status must be 'ok': {path}")
    sampled_params = payload.get("sampled_params")
    if not isinstance(sampled_params, dict):
        raise ValueError(f"sampled_params is missing: {path}")
    return sampled_params


def _build_pair_overrides(sampled_params: Mapping[str, object]) -> dict[str, str]:
    """Build CLI overrides for ``cnn_pair_v2``."""
    overrides: dict[str, str] = {}
    for key, value in sampled_params.items():
        if key in {"donor_len", "acceptor_len"}:
            continue
        if key not in _PAIR_ALLOWED_KEYS:
            continue
        if value is None:
            continue
        overrides[key] = _format_cli_value(value)
    return overrides


def _build_independent_overrides(
    donor_params: Mapping[str, object],
    acceptor_params: Mapping[str, object],
) -> dict[str, str]:
    """Build CLI overrides for ``cnn_v2 --pair_mode independent``."""
    overrides: dict[str, str] = {}
    for key in sorted(_INDEPENDENT_PREFIX_KEYS):
        donor_value = donor_params.get(key)
        if donor_value is not None:
            overrides[f"donor_{key}"] = _format_cli_value(donor_value)
        acceptor_value = acceptor_params.get(key)
        if acceptor_value is not None:
            overrides[f"acceptor_{key}"] = _format_cli_value(acceptor_value)

    # Keep room for future shared-only parameters that do not expose
    # donor_/acceptor_ prefixed overrides.
    for key in sorted(_INDEPENDENT_SHARED_KEYS):
        donor_value = donor_params.get(key)
        acceptor_value = acceptor_params.get(key)
        selected = donor_value if donor_value is not None else acceptor_value
        if selected is not None:
            overrides[key] = _format_cli_value(selected)
    return overrides


def _build_run_args(
    *,
    species: str,
    variant_spec: VariantSpec,
    data_root: Path,
    run_dir: Path,
    device: str,
    seed: int,
    sampled_pair: Mapping[str, object] | None,
    sampled_donor: Mapping[str, object] | None,
    sampled_acceptor: Mapping[str, object] | None,
) -> list[str]:
    """Build one ``src/run_model.py`` argument list for candidate execution."""
    train_pos_path: Path
    train_neg_path: Path
    if variant_spec.mask_mode:
        train_pos_path, train_neg_path = _resolve_mask_train_paths(
            data_root,
            species,
        )
    else:
        train_pos_path, train_neg_path = _resolve_default_train_paths(
            data_root,
            species,
        )

    if not train_pos_path.is_file() or not train_neg_path.is_file():
        missing = [
            str(path)
            for path in (train_pos_path, train_neg_path)
            if not path.is_file()
        ]
        raise FileNotFoundError(
            "Missing training data for "
            f"{species}/{variant_spec.name}: {', '.join(missing)}"
        )

    test_tsv = _resolve_default_test_tsv(data_root, species)
    class_file = data_root / species / "raw" / "transcript_class.txt"

    args: list[str] = [
        "--model",
        variant_spec.target_model_name,
        "--species",
        species,
        "--donor_len",
        "100",
        "--acceptor_len",
        "100",
        "--device",
        device,
        "--seed",
        str(seed),
        "--name_fields",
        "none",
        "--train_target",
        variant_spec.train_target,
        "--sequence_transform",
        "none",
        "--input_mode",
        "onehot",
        "--pair_mode",
        variant_spec.pair_mode,
        "--train_pos_path",
        str(train_pos_path),
        "--train_neg_path",
        str(train_neg_path),
        "--visualize",
        "none",
        "--site_output_tsv",
        str(run_dir / "site_score.tsv"),
        "--intron_output_tsv",
        str(run_dir / "intron_score.tsv"),
        "--transcript_output_tsv",
        str(run_dir / "trans_score.tsv"),
        "--eval_output_txt",
        str(run_dir / "eval_score.txt"),
        "--metrics_json",
        str(run_dir / "train_metrics.json"),
    ]
    if test_tsv is not None:
        args.extend(["--test_tsv", str(test_tsv)])
    if class_file.is_file():
        args.extend(["--class_file", str(class_file)])

    assert sampled_pair is not None
    overrides = _build_pair_overrides(sampled_pair)
    for key in sorted(overrides):
        args.extend([f"--{key}", overrides[key]])
    return args


def _copy_file(src: Path, dst: Path, *, dry_run: bool) -> None:
    """Copy one file with parent-directory creation."""
    if not src.is_file():
        raise FileNotFoundError(f"Missing source file: {src}")
    if dry_run:
        print(f"[dry-run] copy: {src} -> {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _promote_candidate_outputs(
    *,
    project_root: Path,
    species: str,
    run_dir: Path,
    variant: str,
    target_stem: str,
    dry_run: bool,
) -> None:
    """Promote candidate outputs into canonical target-model artifact paths."""
    species_root = project_root / "data" / species
    mapping = {
        run_dir / "site_score.tsv": species_root / "site_score" / f"{target_stem}.tsv",
        run_dir / "intron_score.tsv": (
            species_root / "intron_score" / f"{target_stem}.tsv"
        ),
        run_dir / "trans_score.tsv": species_root / "trans_score" / f"{target_stem}.tsv",
        run_dir / "eval_score.txt": species_root / "eval_score" / f"{target_stem}.txt",
        run_dir / "train_metrics.json": (
            species_root / "learning_metric" / f"{target_stem}.train.json"
        ),
    }
    for src, dst in mapping.items():
        _copy_file(src, dst, dry_run=dry_run)
    promotion_meta = (
        species_root / "learning_metric" / f"{target_stem}.promoted_from.json"
    )
    payload = {
        "promoted_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "species": species,
        "variant": variant,
        "run_dir": str(run_dir),
    }
    if dry_run:
        print(f"[dry-run] write: {promotion_meta}")
        return
    promotion_meta.parent.mkdir(parents=True, exist_ok=True)
    promotion_meta.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _is_legacy_cnn_stem(stem: str) -> bool:
    """Return whether one stem belongs to legacy cnn/cnn_pair outputs."""
    normalized = stem.strip().lower()
    if normalized == "":
        return False
    if normalized.startswith("cnn_v2"):
        return False
    if normalized.startswith("cnn_v3"):
        return False
    if normalized.startswith("cnn_resdil"):
        return False
    if normalized.startswith("cnn_pair"):
        return True
    if normalized == "cnn":
        return True
    return normalized.startswith("cnn_")


def _archive_file(
    *,
    project_root: Path,
    source_path: Path,
    archive_root: Path,
    dry_run: bool,
) -> None:
    """Move one file into archive while preserving project-relative structure."""
    relative = source_path.resolve().relative_to(project_root.resolve())
    destination = archive_root / relative
    if dry_run:
        print(f"[dry-run] move: {source_path} -> {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        destination = destination.with_name(f"{destination.name}.{timestamp}.bak")
    shutil.move(str(source_path), str(destination))


def _candidate_output_dirs(species_root: Path) -> Iterable[Path]:
    """Yield legacy output directories to archive."""
    for child in (
        "site_score",
        "intron_score",
        "trans_score",
        "eval_score",
        "learning_metric",
    ):
        yield species_root / child


def _archive_legacy_outputs_for_species(
    *,
    project_root: Path,
    species: str,
    archive_root: Path,
    dry_run: bool,
) -> None:
    """Archive legacy cnn/cnn_pair non-tuning outputs for one species."""
    species_root = project_root / "data" / species
    for directory in _candidate_output_dirs(species_root):
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            stem = path.stem
            if path.name.endswith(".train.json"):
                stem = path.name[: -len(".train.json")]
            elif path.name.endswith("_learning_curve.png"):
                stem = path.name[: -len("_learning_curve.png")]
            if not _is_legacy_cnn_stem(stem):
                continue
            _archive_file(
                project_root=project_root,
                source_path=path,
                archive_root=archive_root,
                dry_run=dry_run,
            )

    model_species_root = project_root / "model" / species
    for task_dir_name in ("donor", "acceptor", "pair"):
        task_dir = model_species_root / task_dir_name
        if not task_dir.is_dir():
            continue
        for path in sorted(task_dir.iterdir()):
            if not path.is_file() or path.suffix != ".pt":
                continue
            if not _is_legacy_cnn_stem(path.stem):
                continue
            _archive_file(
                project_root=project_root,
                source_path=path,
                archive_root=archive_root,
                dry_run=dry_run,
            )


def _run_candidate(
    *,
    project_root: Path,
    python_bin: str,
    species: str,
    variant_spec: VariantSpec,
    run_root: Path,
    device: str,
    seed: int,
    promote_epsilon: float,
    promote_if_better: bool,
    dry_run: bool,
) -> CandidateResult:
    """Execute one candidate replay and optionally promote target artifacts."""
    data_root = project_root / "data"
    tuned_root = data_root / species / "tuning" / variant_spec.tuned_model_name
    sampled_pair: dict[str, object] | None = None
    sampled_donor: dict[str, object] | None = None
    sampled_acceptor: dict[str, object] | None = None
    if variant_spec.pair_mode == "pair":
        pair_best = tuned_root / "pair" / "best_config.json"
        sampled_pair = _load_sampled_params(pair_best)
    else:
        donor_best = tuned_root / "donor" / "best_config.json"
        acceptor_best = tuned_root / "acceptor" / "best_config.json"
        sampled_donor = _load_sampled_params(donor_best)
        sampled_acceptor = _load_sampled_params(acceptor_best)

    run_dir = run_root / species / variant_spec.name
    run_dir.mkdir(parents=True, exist_ok=True)
    args = _build_run_args(
        species=species,
        variant_spec=variant_spec,
        data_root=data_root,
        run_dir=run_dir,
        device=device,
        seed=seed,
        sampled_pair=sampled_pair,
        sampled_donor=sampled_donor,
        sampled_acceptor=sampled_acceptor,
    )

    baseline_variant_eval = (
        data_root / species / "eval_score" / f"{variant_spec.name}.txt"
    )
    baseline_variant_best_f1 = _parse_best_f1(baseline_variant_eval)
    baseline_target_eval = (
        data_root
        / species
        / "eval_score"
        / f"{variant_spec.target_artifact_stem}.txt"
    )
    baseline_target_best_f1 = _parse_best_f1(baseline_target_eval)

    if dry_run:
        print(
            f"[dry-run] would run species={species} variant={variant_spec.name} "
            f"target_model={variant_spec.target_model_name} "
            f"pair_mode={variant_spec.pair_mode}"
        )
        return CandidateResult(
            species=species,
            variant=variant_spec.name,
            return_code=0,
            baseline_variant_best_f1=baseline_variant_best_f1,
            baseline_target_best_f1=baseline_target_best_f1,
            candidate_best_f1=None,
            promoted=False,
            run_dir=run_dir,
        )

    cmd = [python_bin, "-u", str(project_root / "src" / "run_model.py"), *args]
    print(
        f"[run] species={species} variant={variant_spec.name} "
        f"target_model={variant_spec.target_model_name}"
    )
    completed = subprocess.run(cmd, check=False)
    candidate_best_f1 = _parse_best_f1(run_dir / "eval_score.txt")
    promoted = False
    if (
        promote_if_better
        and
        completed.returncode == 0
        and candidate_best_f1 is not None
        and (
            baseline_target_best_f1 is None
            or candidate_best_f1 > (baseline_target_best_f1 + promote_epsilon)
        )
    ):
        _promote_candidate_outputs(
            project_root=project_root,
            species=species,
            run_dir=run_dir,
            variant=variant_spec.name,
            target_stem=variant_spec.target_artifact_stem,
            dry_run=False,
        )
        promoted = True
    return CandidateResult(
        species=species,
        variant=variant_spec.name,
        return_code=completed.returncode,
        baseline_variant_best_f1=baseline_variant_best_f1,
        baseline_target_best_f1=baseline_target_best_f1,
        candidate_best_f1=candidate_best_f1,
        promoted=promoted,
        run_dir=run_dir,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    project_root = Path(args.project_root).resolve()
    archive_root = Path(args.archive_root)
    if not archive_root.is_absolute():
        archive_root = (project_root / archive_root).resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_root = project_root / "data" / "migration_runs" / f"cnn_merge_{stamp}"

    species_list = _parse_csv(str(args.species))
    variant_names = _parse_csv(str(args.variants))
    variant_specs: list[VariantSpec] = []
    for name in variant_names:
        spec = LEGACY_VARIANTS.get(name)
        if spec is None:
            known = ", ".join(sorted(LEGACY_VARIANTS))
            raise ValueError(f"Unknown variant '{name}'. Supported: {known}")
        variant_specs.append(spec)

    all_results: list[CandidateResult] = []
    run_counter = 0
    for species in species_list:
        for variant_spec in variant_specs:
            if args.max_runs > 0 and run_counter >= int(args.max_runs):
                break
            try:
                result = _run_candidate(
                    project_root=project_root,
                    python_bin=str(args.python_bin),
                    species=species,
                    variant_spec=variant_spec,
                    run_root=run_root,
                    device=str(args.device),
                    seed=int(args.seed),
                    promote_epsilon=float(args.promote_epsilon),
                    promote_if_better=bool(args.promote_if_better),
                    dry_run=bool(args.dry_run),
                )
            except Exception as exc:  # pragma: no cover
                print(
                    "[error] "
                    f"species={species} variant={variant_spec.name} failed: {exc}"
                )
                result = CandidateResult(
                    species=species,
                    variant=variant_spec.name,
                    return_code=1,
                    baseline_variant_best_f1=None,
                    baseline_target_best_f1=None,
                    candidate_best_f1=None,
                    promoted=False,
                    run_dir=run_root / species / variant_spec.name,
                )
            all_results.append(result)
            run_counter += 1
        if args.max_runs > 0 and run_counter >= int(args.max_runs):
            break

    if args.archive_legacy:
        archive_stamp_root = archive_root / f"cnn_legacy_{stamp}"
        for species in species_list:
            _archive_legacy_outputs_for_species(
                project_root=project_root,
                species=species,
                archive_root=archive_stamp_root,
                dry_run=bool(args.dry_run),
            )

    summary_path = run_root / "summary.json"
    summary_payload = {
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "project_root": str(project_root),
        "archive_root": str(archive_root),
        "dry_run": bool(args.dry_run),
        "archive_legacy": bool(args.archive_legacy),
        "promote_if_better": bool(args.promote_if_better),
        "results": [
            {
                "species": item.species,
                "variant": item.variant,
                "return_code": item.return_code,
                "baseline_variant_best_f1": item.baseline_variant_best_f1,
                "baseline_target_best_f1": item.baseline_target_best_f1,
                "candidate_best_f1": item.candidate_best_f1,
                "promoted": item.promoted,
                "run_dir": str(item.run_dir),
            }
            for item in all_results
        ],
    }
    if args.dry_run:
        print(f"[dry-run] write summary: {summary_path}")
    else:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        print(f"[done] summary={summary_path}")

    failures = [item for item in all_results if item.return_code != 0]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
