from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import scan_splice_candidate_sites as scan


def _write_text(path: Path, text: str) -> None:
    """Write UTF-8 text to one path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write one JSON object to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_normalize_sequence_text_strips_headers_and_whitespace() -> None:
    """Normalize one FASTA-like string to one upper-cased sequence."""
    raw = ">chr1\nacg t\nA C\n"

    normalized = scan.normalize_sequence_text(raw)

    assert normalized == "ACGTAC"


def test_scan_motif_coordinates_finds_overlapping_candidates() -> None:
    """Return all 0-based motif coordinates on the forward strand."""
    sequence = "GTGTAG"

    assert scan.scan_motif_coordinates(sequence, "GT") == [0, 2]
    assert scan.scan_motif_coordinates(sequence, "AG") == [4]


def test_build_candidate_windows_skips_edge_candidates() -> None:
    """Skip candidates that cannot supply the full scoring window."""
    sequence = "AGTAAAGTAAAG"

    donor_candidates, acceptor_candidates = scan.build_candidate_windows(
        sequence,
        donor_window_len=6,
        acceptor_window_len=6,
    )

    assert [candidate.coordinate for candidate in donor_candidates] == [6]
    assert [candidate.coordinate for candidate in acceptor_candidates] == [5]
    assert donor_candidates[0].window == "AAAGTA"
    assert acceptor_candidates[0].window == "TAAAGT"


def test_load_best_checkpoint_paths_follows_source_configs(
    tmp_path: Path,
) -> None:
    """Resolve donor and acceptor checkpoints through a summary best-config."""
    donor_best = tmp_path / "donor" / "best_config.json"
    acceptor_best = tmp_path / "acceptor" / "best_config.json"
    donor_ckpt = tmp_path / "model" / "donor.pt"
    acceptor_ckpt = tmp_path / "model" / "acceptor.pt"

    _write_json(
        donor_best,
        {
            "status": "ok",
            "donor_checkpoint_path": str(donor_ckpt),
        },
    )
    _write_json(
        acceptor_best,
        {
            "status": "ok",
            "acceptor_checkpoint_path": str(acceptor_ckpt),
        },
    )

    summary_best = tmp_path / "cnn_v2" / "best_config.json"
    _write_json(
        summary_best,
        {
            "status": "ok",
            "source_donor_best_config": str(donor_best),
            "source_acceptor_best_config": str(acceptor_best),
        },
    )

    resolved = scan.load_best_checkpoint_paths(summary_best)

    assert resolved == (donor_ckpt.resolve(), acceptor_ckpt.resolve())


def test_main_writes_output_files_and_skips_edges(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Run the command-line path with fake models and deterministic scores."""
    sequence_file = tmp_path / "sequence.txt"
    output_dir = tmp_path / "out"
    _write_text(sequence_file, ">chr1\nAGTAAAGTAAAG\n")

    resolved = scan.ResolvedBestModelPaths(
        best_config_path=tmp_path / "best_config.json",
        donor_checkpoint_path=tmp_path / "donor.pt",
        acceptor_checkpoint_path=tmp_path / "acceptor.pt",
        donor_window_len=6,
        acceptor_window_len=6,
    )

    monkeypatch.setattr(scan, "pick_device", lambda device: "cpu")
    monkeypatch.setattr(
        scan,
        "load_resolved_best_model_paths",
        lambda **kwargs: resolved,
    )
    monkeypatch.setattr(
        scan,
        "load_task_model",
        lambda checkpoint_path, device: (f"model:{checkpoint_path}", {"window_len": 6}),
    )
    monkeypatch.setattr(
        scan,
        "score_sequences",
        lambda **kwargs: [float(len(sequence)) for sequence in kwargs["sequences"]],
    )

    exit_code = scan.main(
        [
            "--data-root",
            str(tmp_path / "data"),
            "--species",
            "Dmel",
            "--model",
            "cnn_v2",
            "--name",
            "NC_004354.4",
            "--sequence-file",
            str(sequence_file),
            "--output-dir",
            str(output_dir),
            "--batch-size",
            "8",
        ]
    )

    donor_output = output_dir / "NC_004354.4.gt.txt"
    acceptor_output = output_dir / "NC_004354.4.ag.txt"

    assert exit_code == 0
    assert donor_output.read_text(encoding="utf-8").splitlines() == ["6\t6.000000"]
    assert acceptor_output.read_text(encoding="utf-8").splitlines() == ["5\t6.000000"]
