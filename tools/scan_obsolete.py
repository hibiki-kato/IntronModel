#!/usr/bin/env python3
"""Scan repository consistency and identify legacy/unconnected modules.

This tool inspects the model registry, model files, README references, and
wrapper scripts. It produces a Markdown or JSON report used for repository
cleanup before introducing additional models.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DefaultMismatch:
    """Mismatch between script usage text and script variable defaults.

    Attributes
    ----------
    script : str
        Script path relative to repository root.
    option : str
        CLI option name without leading dashes.
    usage_default : str
        Default value documented in the usage block.
    variable_name : str
        Backing shell variable used for this option.
    variable_default : str
        Actual assigned default value in the shell script.
    """

    script: str
    option: str
    usage_default: str
    variable_name: str
    variable_default: str


@dataclass(frozen=True, slots=True)
class ScanResult:
    """Structured output for repository scan results."""

    generated_at_utc: str
    registry_models: dict[str, str]
    connected_model_files: list[str]
    discovered_model_files: list[str]
    unconnected_model_files: list[str]
    readme_run_scripts: list[str]
    run_scripts_on_disk: list[str]
    run_scripts_missing_from_readme: list[str]
    run_scripts_mentioned_but_missing: list[str]
    script_default_mismatches: list[DefaultMismatch]
    readme_option_mismatches: list[str]
    notes: list[str]


def _repo_root() -> Path:
    """Return repository root path based on this script location."""

    return Path(__file__).resolve().parents[1]


def _load_registry_models(registry_path: Path) -> dict[str, str]:
    """Parse `_MODEL_TO_MODULE` dictionary from registry source.

    Parameters
    ----------
    registry_path : pathlib.Path
        Path to `src/models/registry.py`.

    Returns
    -------
    dict[str, str]
        Mapping of model name to module import path.

    Raises
    ------
    ValueError
        If `_MODEL_TO_MODULE` is missing or not a literal dictionary.
    """

    source = registry_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    def parse_mapping_node(value: ast.AST) -> dict[str, str]:
        if not isinstance(value, ast.Dict):
            raise ValueError("_MODEL_TO_MODULE is not a literal dictionary.")
        mapping: dict[str, str] = {}
        for key_node, value_node in zip(value.keys, value.values):
            if not isinstance(key_node, ast.Constant):
                raise ValueError("Registry model key is not a string literal.")
            if not isinstance(value_node, ast.Constant):
                raise ValueError("Registry model value is not a string literal.")
            key_value = key_node.value
            model_path = value_node.value
            if not isinstance(key_value, str) or not isinstance(model_path, str):
                raise ValueError("Registry entries must be string literals.")
            mapping[key_value] = model_path
        return mapping

    for node in tree.body:
        if isinstance(node, ast.Assign):
            target_names = [
                target.id for target in node.targets if isinstance(target, ast.Name)
            ]
            if "_MODEL_TO_MODULE" in target_names:
                return parse_mapping_node(node.value)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                if node.target.id == "_MODEL_TO_MODULE" and node.value is not None:
                    return parse_mapping_node(node.value)

    raise ValueError("Could not find _MODEL_TO_MODULE in registry file.")


def _connected_model_files(registry_models: dict[str, str]) -> set[str]:
    """Derive connected model filenames from registry module paths."""

    connected: set[str] = set()
    for module_path in registry_models.values():
        if not module_path.startswith("models."):
            continue
        connected.add(module_path.split(".")[-1])
    return connected


def _discover_model_files(models_dir: Path) -> set[str]:
    """Return model implementation file stems under `src/models`."""

    stems: set[str] = set()
    for path in sorted(models_dir.glob("*.py")):
        if path.name in {"__init__.py", "registry.py"}:
            continue
        stems.add(path.stem)
    return stems


def _discover_run_scripts(run_dir: Path) -> set[str]:
    """Return `run/*.sh` paths relative to repository root."""

    root = run_dir.parent
    return {
        str(path.relative_to(root).as_posix())
        for path in sorted(run_dir.glob("*.sh"))
    }


def _readme_run_script_refs(readme_text: str) -> set[str]:
    """Extract referenced `run/*.sh` paths from README text."""

    refs = re.findall(r"run/[A-Za-z0-9_\-]+\.sh", readme_text)
    return {ref for ref in refs}


def _parse_usage_defaults(script_text: str) -> dict[str, str]:
    """Extract defaults from usage text lines.

    Returns
    -------
    dict[str, str]
        Mapping of option name (without `--`) to default text.
    """

    defaults: dict[str, str] = {}
    pattern = re.compile(
        r"^\s*--([a-z0-9\-]+)[^\n]*\(default:\s*([^\)]+)\)",
        re.MULTILINE,
    )
    for match in pattern.finditer(script_text):
        option = match.group(1)
        value = match.group(2).strip()
        if ";" in value:
            value = value.split(";", maxsplit=1)[0].strip()
        defaults[option] = value
    return defaults


def _parse_assignments(script_text: str) -> dict[str, str]:
    """Extract `VAR="value"` assignments from shell script text."""

    assignments: dict[str, str] = {}
    pattern = re.compile(r"^([A-Z_]+)=\"([^\"]*)\"", re.MULTILINE)
    for match in pattern.finditer(script_text):
        assignments[match.group(1)] = match.group(2)
    return assignments


def _option_variable_mapping(option_name: str) -> str | None:
    """Map option names to conventional shell variable names."""

    mapping: dict[str, str] = {
        "species": "SPECIES",
        "donor-len": "DONOR_LEN",
        "acceptor-len": "ACCEPTOR_LEN",
        "epochs": "EPOCHS",
        "batch-size": "BATCH_SIZE",
        "lr": "LR",
        "loss": "LOSS",
        "seed": "SEED",
        "device": "DEVICE",
        "visualize": "VISUALIZE",
        "softmin-tau": "SOFTMIN_TAU",
        "x-min": "X_MIN",
        "x-max": "X_MAX",
        "y-min": "Y_MIN",
        "y-max": "Y_MAX",
        "conda-env": "CONDA_ENV",
        "feature": "FEATURE",
        "limit": "LIMIT",
    }
    return mapping.get(option_name)


def _collect_script_default_mismatches(root: Path) -> list[DefaultMismatch]:
    """Detect usage/default mismatches for all `run/*.sh` wrappers."""

    mismatches: list[DefaultMismatch] = []
    for script_path in sorted((root / "run").glob("*.sh")):
        text = script_path.read_text(encoding="utf-8")
        usage_defaults = _parse_usage_defaults(text)
        assignments = _parse_assignments(text)

        for option, usage_default in usage_defaults.items():
            variable_name = _option_variable_mapping(option)
            if variable_name is None:
                continue
            variable_default = assignments.get(variable_name)
            if variable_default is None:
                continue
            if usage_default != variable_default:
                mismatches.append(
                    DefaultMismatch(
                        script=str(script_path.relative_to(root).as_posix()),
                        option=option,
                        usage_default=usage_default,
                        variable_name=variable_name,
                        variable_default=variable_default,
                    )
                )
    return mismatches


def _readme_option_mismatches(root: Path, readme_text: str) -> list[str]:
    """Detect option references in README that wrappers do not support."""

    mismatches: list[str] = []
    cnn_wrapper = root / "run" / "cnn.sh"
    if not cnn_wrapper.exists():
        return ["run/cnn.sh is missing."]

    cnn_text = cnn_wrapper.read_text(encoding="utf-8")
    if "--site-score-tsv" in readme_text and "--site-score-tsv)" not in cnn_text:
        mismatches.append(
            "README references --site-score-tsv but run/cnn.sh does not parse it."
        )
    return mismatches


def _build_scan_result(root: Path) -> ScanResult:
    """Assemble scan results from repository files."""

    registry_path = root / "src" / "models" / "registry.py"
    models_dir = root / "src" / "models"
    run_dir = root / "run"
    readme_path = root / "README.md"

    registry_models = _load_registry_models(registry_path)
    connected = _connected_model_files(registry_models)
    discovered = _discover_model_files(models_dir)
    unconnected = sorted(discovered - connected)

    run_scripts_on_disk = _discover_run_scripts(run_dir)
    readme_text = readme_path.read_text(encoding="utf-8")
    readme_run_scripts = _readme_run_script_refs(readme_text)

    missing_from_readme = sorted(run_scripts_on_disk - readme_run_scripts)
    mentioned_but_missing = sorted(readme_run_scripts - run_scripts_on_disk)

    default_mismatches = _collect_script_default_mismatches(root)
    option_mismatches = _readme_option_mismatches(root, readme_text)

    notes: list[str] = []
    if unconnected:
        notes.append(
            "Unconnected model files are treated as legacy/experimental and are "
            "kept intentionally."
        )
    if missing_from_readme or mentioned_but_missing:
        notes.append("README and run script references are not fully aligned.")
    if default_mismatches:
        notes.append("Wrapper usage defaults do not match shell variable defaults.")

    return ScanResult(
        generated_at_utc=datetime.now(tz=UTC).isoformat(),
        registry_models=dict(sorted(registry_models.items())),
        connected_model_files=sorted(connected),
        discovered_model_files=sorted(discovered),
        unconnected_model_files=unconnected,
        readme_run_scripts=sorted(readme_run_scripts),
        run_scripts_on_disk=sorted(run_scripts_on_disk),
        run_scripts_missing_from_readme=missing_from_readme,
        run_scripts_mentioned_but_missing=mentioned_but_missing,
        script_default_mismatches=default_mismatches,
        readme_option_mismatches=option_mismatches,
        notes=notes,
    )


def _render_markdown(result: ScanResult) -> str:
    """Render scan result in Markdown format."""

    lines: list[str] = []
    lines.append("# Repository Scan Report")
    lines.append("")
    lines.append(f"- Generated at (UTC): `{result.generated_at_utc}`")
    lines.append("")

    lines.append("## Registry vs Model Files")
    lines.append("")
    lines.append("### Registry Models")
    lines.append("")
    for model_name, module_path in result.registry_models.items():
        lines.append(f"- `{model_name}` -> `{module_path}`")
    lines.append("")

    lines.append("### Discovered Model Files")
    lines.append("")
    for stem in result.discovered_model_files:
        lines.append(f"- `src/models/{stem}.py`")
    lines.append("")

    lines.append("### Unconnected (Legacy Candidates)")
    lines.append("")
    if result.unconnected_model_files:
        for stem in result.unconnected_model_files:
            lines.append(f"- `src/models/{stem}.py` (legacy; keep, not delete)")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## README vs run/*.sh")
    lines.append("")
    lines.append("### Scripts Missing from README")
    lines.append("")
    if result.run_scripts_missing_from_readme:
        for path in result.run_scripts_missing_from_readme:
            lines.append(f"- `{path}`")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("### Scripts Mentioned in README but Missing on Disk")
    lines.append("")
    if result.run_scripts_mentioned_but_missing:
        for path in result.run_scripts_mentioned_but_missing:
            lines.append(f"- `{path}`")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("### Wrapper Default Mismatches")
    lines.append("")
    if result.script_default_mismatches:
        for mismatch in result.script_default_mismatches:
            lines.append(
                "- "
                f"`{mismatch.script}` `--{mismatch.option}` usage default "
                f"`{mismatch.usage_default}` != {mismatch.variable_name} "
                f"`{mismatch.variable_default}`"
            )
    else:
        lines.append("- None")
    lines.append("")

    lines.append("### README Option Mismatches")
    lines.append("")
    if result.readme_option_mismatches:
        for item in result.readme_option_mismatches:
            lines.append(f"- {item}")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    if result.notes:
        for note in result.notes:
            lines.append(f"- {note}")
    else:
        lines.append("- No notable issues detected.")
    lines.append("")

    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Scan model registry, wrapper scripts, and README references for "
            "repository consistency."
        )
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output report path.",
    )
    parser.add_argument(
        "--format",
        choices=["md", "json"],
        default="md",
        help="Output format.",
    )
    return parser


def main() -> None:
    """CLI entrypoint."""

    args = _build_parser().parse_args()
    root = _repo_root()
    result = _build_scan_result(root)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "json":
        payload = {
            **asdict(result),
            "script_default_mismatches": [
                asdict(mismatch) for mismatch in result.script_default_mismatches
            ],
        }
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        output_path.write_text(_render_markdown(result), encoding="utf-8")


if __name__ == "__main__":
    main()
