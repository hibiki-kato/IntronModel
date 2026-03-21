"""Split ``cnn_v2/both`` best configs into donor/acceptor best configs.

This utility reads:

- ``data/<species>/tuning/cnn_v2/both/best_config.json``

and writes:

- ``data/<species>/tuning/cnn_v2/donor/best_config.json``
- ``data/<species>/tuning/cnn_v2/acceptor/best_config.json``

The source payload is reused, but each target config updates:

- ``objective_metric`` to ``donor_pr_auc`` / ``acceptor_pr_auc``
- ``objective_score`` and ``selection_score`` to that target metric
- ``sampled_params.train_target`` to the target
- branch-specific keys (for example ``donor_conv_channels``) into shared
  ``conv_channels`` / ``kernel_sizes`` when present.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence


TARGETS: tuple[str, str] = ("donor", "acceptor")


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse CLI args for split operation.

    Parameters
    ----------
    argv : Sequence[str]
        Raw CLI argument tokens (without executable name).

    Returns
    -------
    argparse.Namespace
        Parsed argument namespace.

    Raises
    ------
    SystemExit
        Raised by ``argparse`` when arguments are invalid.

    Complexity
    ----------
    O(n) time and O(n) memory for ``n`` CLI tokens.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Split cnn_v2 both best_config.json files into donor/acceptor "
            "best configs."
        )
    )
    parser.add_argument(
        "--species",
        default="Athal,Dmel,Hsap,Mmus",
        help="Comma-separated species list.",
    )
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Project root path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without writing files.",
    )
    return parser.parse_args(argv)


def _parse_csv(raw_value: str) -> list[str]:
    """Parse a CSV string into ordered unique tokens.

    Parameters
    ----------
    raw_value : str
        Comma-separated string.

    Returns
    -------
    list[str]
        Deduplicated ordered tokens.

    Raises
    ------
    ValueError
        If no non-empty token is found.

    Complexity
    ----------
    O(k) time and O(k) memory for ``k`` comma-separated tokens.
    """

    values: list[str] = []
    seen: set[str] = set()
    for token in raw_value.split(","):
        value = token.strip()
        if value == "" or value in seen:
            continue
        seen.add(value)
        values.append(value)
    if not values:
        raise ValueError("Expected at least one species value.")
    return values


def _read_json_object(path: Path) -> dict[str, object]:
    """Read one JSON object from disk.

    Parameters
    ----------
    path : Path
        JSON file path.

    Returns
    -------
    dict[str, object]
        Parsed JSON object.

    Raises
    ------
    ValueError
        If the payload is not a JSON object.
    json.JSONDecodeError
        If JSON parsing fails.

    Complexity
    ----------
    O(n) time and O(n) memory for file size ``n``.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _as_finite_float(value: object) -> float | None:
    """Convert a scalar-like value to finite float.

    Parameters
    ----------
    value : object
        Scalar-like value.

    Returns
    -------
    float | None
        Converted finite float when possible, else ``None``.

    Complexity
    ----------
    O(1) time and O(1) memory.
    """

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        try:
            numeric = float(stripped)
        except ValueError:
            return None
        if math.isfinite(numeric):
            return numeric
    return None


def _build_target_sampled_params(
    *,
    sampled_params: Mapping[str, object],
    target: str,
) -> dict[str, object]:
    """Build target-specific sampled params from one ``both`` payload.

    Parameters
    ----------
    sampled_params : Mapping[str, object]
        Source ``sampled_params`` from ``both`` best config.
    target : str
        Site target, either ``"donor"`` or ``"acceptor"``.

    Returns
    -------
    dict[str, object]
        Target-specific params dictionary.

    Raises
    ------
    ValueError
        If ``target`` is invalid.

    Complexity
    ----------
    O(m) time and O(m) memory for ``m`` sampled params.
    """

    if target not in TARGETS:
        raise ValueError(f"Unsupported target: {target}")

    result: dict[str, object] = {}
    for key, value in sampled_params.items():
        if key == "train_target":
            continue
        if key.startswith("donor_") or key.startswith("acceptor_"):
            continue
        result[key] = value

    result["train_target"] = target
    result["pair_mode"] = "independent"

    conv_key = f"{target}_conv_channels"
    kernel_key = f"{target}_kernel_sizes"
    if conv_key in sampled_params and sampled_params[conv_key] is not None:
        result["conv_channels"] = sampled_params[conv_key]
    if kernel_key in sampled_params and sampled_params[kernel_key] is not None:
        result["kernel_sizes"] = sampled_params[kernel_key]
    return result


def _build_target_payload(
    *,
    source_payload: Mapping[str, object],
    source_path: Path,
    target: str,
) -> dict[str, object]:
    """Build one donor/acceptor best payload from a ``both`` payload.

    Parameters
    ----------
    source_payload : Mapping[str, object]
        Source ``both`` best payload.
    source_path : Path
        Source path for provenance.
    target : str
        Site target, either ``"donor"`` or ``"acceptor"``.

    Returns
    -------
    dict[str, object]
        Transformed payload for target best config.

    Raises
    ------
    ValueError
        If required fields are missing or invalid.

    Complexity
    ----------
    O(m) time and O(m) memory for ``m`` sampled params.
    """

    objective_metric = f"{target}_pr_auc"
    objective_value = _as_finite_float(source_payload.get(objective_metric))
    if objective_value is None:
        raise ValueError(
            f"Missing finite {objective_metric} in source payload: {source_path}"
        )

    sampled_params_raw = source_payload.get("sampled_params")
    if not isinstance(sampled_params_raw, Mapping):
        raise ValueError(f"Missing sampled_params object: {source_path}")

    payload: dict[str, object] = dict(source_payload)
    payload["objective_metric"] = objective_metric
    payload["objective_score"] = objective_value
    payload["selection_score"] = objective_value
    if target == "donor":
        payload["donor_pr_auc"] = objective_value
        payload["acceptor_pr_auc"] = None
    else:
        payload["donor_pr_auc"] = None
        payload["acceptor_pr_auc"] = objective_value
    payload["mean_pr_auc"] = None
    payload["sampled_params"] = _build_target_sampled_params(
        sampled_params=sampled_params_raw,
        target=target,
    )

    hparam_context_raw = source_payload.get("hparam_context")
    if isinstance(hparam_context_raw, Mapping):
        hparam_context: dict[str, object] = dict(hparam_context_raw)
        hparam_context["objective_metric"] = objective_metric
        fixed_run_args_raw = hparam_context.get("fixed_run_args")
        if isinstance(fixed_run_args_raw, Mapping):
            fixed_run_args: dict[str, object] = dict(fixed_run_args_raw)
            fixed_run_args["train_target"] = target
            hparam_context["fixed_run_args"] = fixed_run_args
        payload["hparam_context"] = hparam_context

    payload["source_best_config"] = str(source_path)
    payload["generated_by"] = "split_cnn_v2_both_to_site_best.py"
    payload["generated_at_utc"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return payload


def _write_json(path: Path, payload: Mapping[str, object], *, dry_run: bool) -> None:
    """Write one JSON object payload.

    Parameters
    ----------
    path : Path
        Output JSON path.
    payload : Mapping[str, object]
        JSON-serializable object payload.
    dry_run : bool
        If ``True``, only print planned write path.

    Returns
    -------
    None

    Complexity
    ----------
    O(n) time and O(n) memory for serialized size ``n``.
    """

    if dry_run:
        print(f"[dry-run] write: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _split_species(*, data_root: Path, species: str, dry_run: bool) -> None:
    """Split one species ``both`` best into donor/acceptor best configs.

    Parameters
    ----------
    data_root : Path
        Root data directory.
    species : str
        Species name.
    dry_run : bool
        If ``True``, preview writes only.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If source ``both`` best config is missing.
    ValueError
        If source payload is invalid.

    Complexity
    ----------
    O(m) time and O(m) memory for ``m`` sampled params per species.
    """

    source_path = data_root / species / "tuning" / "cnn_v2" / "both" / "best_config.json"
    if not source_path.is_file():
        raise FileNotFoundError(f"Source best_config not found: {source_path}")

    source_payload = _read_json_object(source_path)
    status = str(source_payload.get("status", "")).strip().lower()
    if status != "ok":
        raise ValueError(f"Expected status=ok in source payload: {source_path}")

    for target in TARGETS:
        output_path = (
            data_root
            / species
            / "tuning"
            / "cnn_v2"
            / target
            / "best_config.json"
        )
        payload = _build_target_payload(
            source_payload=source_payload,
            source_path=source_path,
            target=target,
        )
        _write_json(output_path, payload, dry_run=dry_run)
        print(
            "[split] "
            f"species={species} target={target} "
            f"objective={payload['objective_metric']} "
            f"score={payload['objective_score']}",
            flush=True,
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run CLI entrypoint.

    Parameters
    ----------
    argv : Sequence[str] | None, default=None
        Optional argument list for testing.

    Returns
    -------
    int
        Process exit code.

    Complexity
    ----------
    O(s * m) time for ``s`` species and ``m`` sampled params per species.
    """

    args = _parse_args(sys.argv[1:] if argv is None else argv)
    species_list = _parse_csv(str(args.species))
    project_root = Path(args.project_root).resolve()
    data_root = project_root / "data"
    dry_run = bool(args.dry_run)

    for species in species_list:
        _split_species(data_root=data_root, species=species, dry_run=dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
