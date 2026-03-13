from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_bash(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-lc", command],
        capture_output=True,
        text=True,
        check=False,
        cwd=_project_root(),
    )


def test_prepare_species_data_accepts_hsap(tmp_path: Path) -> None:
    prepare_script = _project_root() / "src" / "scripts" / "prepare_species_data.sh"
    missing_source_root = tmp_path / "missing_source"
    command = " ".join(
        [
            "bash",
            shlex.quote(str(prepare_script)),
            "--species",
            "Hsap",
            "--source-root",
            shlex.quote(str(missing_source_root)),
            "--target-root",
            shlex.quote(str(tmp_path / "target")),
        ]
    )

    run = _run_bash(command)

    assert run.returncode != 0
    assert "Invalid --species value" not in run.stderr
    assert "Source root not found" in run.stderr


def test_fetch_reference_data_all_includes_hsap(tmp_path: Path) -> None:
    fetch_script = _project_root() / "src" / "scripts" / "fetch_reference_data.sh"
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    for species in ("Athal", "Dmel", "Hsap", "Mmus"):
        raw_dir = source_root / species / "raw"
        raw_dir.mkdir(parents=True)
        (raw_dir / f"{species}.fna").write_text(">chr1\nACGT\n", encoding="utf-8")

    command = " ".join(
        [
            "bash",
            shlex.quote(str(fetch_script)),
            "--species",
            "all",
            "--source-root",
            shlex.quote(str(source_root)),
            "--target-root",
            shlex.quote(str(target_root)),
        ]
    )

    run = _run_bash(command)

    assert run.returncode == 0, run.stderr
    assert (target_root / "Hsap" / "raw" / "Hsap.fna").is_file()
    assert "species=Hsap" in run.stdout
