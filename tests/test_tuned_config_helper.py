from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _project_root() -> Path:
    """Return repository root inferred from this test file."""

    return Path(__file__).resolve().parents[1]


def _run_tuned_config_helper(config_path: Path) -> list[str]:
    """Execute tuned-config helper and return stripped output lines."""

    helper_path = _project_root() / "run" / "lib" / "tuned_config.sh"
    command = (
        f'source "{helper_path}" && '
        f'intronmodel_load_tuned_overrides "{config_path}"'
    )
    run = subprocess.run(
        ["bash", "-lc", command],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    return [line for line in run.stdout.splitlines() if line.strip() != ""]


def test_load_tuned_overrides_merges_fixed_and_sampled_params(
    tmp_path: Path,
) -> None:
    """Best-config loader should merge fixed fields before sampled fields."""

    config_path = tmp_path / "best_config.json"
    config_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "hparam_context": {
                    "fixed_run_args": {
                        "model": "cnn_v2",
                        "train_target": "donor",
                        "input_mode": "onehot",
                    }
                },
                "sampled_params": {
                    "mask": "on",
                    "batch_size": 256,
                    "lr": 0.001,
                },
            }
        ),
        encoding="utf-8",
    )

    assert _run_tuned_config_helper(config_path) == [
        "input_mode\tonehot",
        "model\tcnn_v2",
        "train_target\tdonor",
        "sequence_transform\tmask_outside_intron_n",
        "batch_size\t256",
        "lr\t0.001",
    ]


def test_load_tuned_overrides_ignores_script_name_metadata(
    tmp_path: Path,
) -> None:
    """Best-config loader should not forward scheduler-only metadata keys."""

    config_path = tmp_path / "best_config.json"
    config_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "hparam_context": {
                    "fixed_run_args": {
                        "model": "cnn_pair_v3",
                        "script_name": "tune_cnn_pair_v3_time.sh",
                        "train_target": "pair",
                    }
                },
                "sampled_params": {
                    "batch_size": 512,
                    "lr": 0.0005,
                },
            }
        ),
        encoding="utf-8",
    )

    assert _run_tuned_config_helper(config_path) == [
        "model\tcnn_pair_v3",
        "train_target\tpair",
        "batch_size\t512",
        "lr\t0.0005",
    ]


def test_load_tuned_overrides_disables_mask_for_independent_cnn_v2(
    tmp_path: Path,
) -> None:
    """Independent cnn_v2 configs should always emit sequence_transform=none."""

    config_path = tmp_path / "best_config.json"
    config_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "hparam_context": {
                    "fixed_run_args": {
                        "model": "cnn_v2",
                        "pair_mode": "independent",
                        "train_target": "acceptor",
                    }
                },
                "sampled_params": {
                    "mask": "on",
                    "batch_size": 256,
                },
            }
        ),
        encoding="utf-8",
    )

    assert _run_tuned_config_helper(config_path) == [
        "model\tcnn_v2",
        "pair_mode\tindependent",
        "train_target\tacceptor",
        "sequence_transform\tnone",
        "batch_size\t256",
    ]


def test_load_tuned_overrides_drops_irrelevant_site_length_for_donor_task(
    tmp_path: Path,
) -> None:
    """Independent donor configs should normalize legacy lengths to flanks."""

    config_path = tmp_path / "best_config.json"
    config_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "hparam_context": {
                    "fixed_run_args": {
                        "model": "cnn_v2",
                        "pair_mode": "independent",
                        "train_target": "donor",
                    }
                },
                "sampled_params": {
                    "donor_len": 50,
                    "acceptor_len": 70,
                    "batch_size": 256,
                },
            }
        ),
        encoding="utf-8",
    )

    assert _run_tuned_config_helper(config_path) == [
        "model\tcnn_v2",
        "pair_mode\tindependent",
        "train_target\tdonor",
        "sequence_transform\tnone",
        "batch_size\t256",
        "donor_downstream\t45",
        "donor_upstream\t5",
    ]


def test_load_tuned_overrides_prefers_sampled_active_flanks_over_fixed_defaults(
    tmp_path: Path,
) -> None:
    """Independent donor configs should emit the effective donor flank values."""

    config_path = tmp_path / "best_config.json"
    config_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "hparam_context": {
                    "fixed_run_args": {
                        "model": "cnn_v2",
                        "pair_mode": "independent",
                        "train_target": "donor",
                        "donor_upstream": 5,
                        "donor_downstream": 95,
                        "acceptor_upstream": 95,
                        "acceptor_downstream": 5,
                    }
                },
                "sampled_params": {
                    "donor_upstream": 20,
                    "donor_downstream": 100,
                    "acceptor_upstream": 80,
                    "acceptor_downstream": 40,
                },
            }
        ),
        encoding="utf-8",
    )

    assert _run_tuned_config_helper(config_path) == [
        "model\tcnn_v2",
        "pair_mode\tindependent",
        "train_target\tdonor",
        "sequence_transform\tnone",
        "donor_downstream\t100",
        "donor_upstream\t20",
    ]


def test_load_tuned_overrides_keeps_legacy_lengths_for_non_cnn_v2_models(
    tmp_path: Path,
) -> None:
    """Length-based wrappers should still receive legacy donor/acceptor lengths."""

    config_path = tmp_path / "best_config.json"
    config_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "hparam_context": {
                    "fixed_run_args": {
                        "model": "cnn_v3",
                        "train_target": "pair",
                    }
                },
                "sampled_params": {
                    "donor_len": 90,
                    "acceptor_len": 70,
                },
            }
        ),
        encoding="utf-8",
    )

    assert _run_tuned_config_helper(config_path) == [
        "model\tcnn_v3",
        "train_target\tpair",
        "acceptor_len\t70",
        "donor_len\t90",
    ]


def test_resolve_tuned_config_path_ignores_legacy_root_for_cnn_v2(
    tmp_path: Path,
) -> None:
    """cnn_v2 should not fall back to one shared root-level best config."""

    helper_path = _project_root() / "run" / "lib" / "tuned_config.sh"
    species = "Dmel"
    legacy_path = tmp_path / species / "tuning" / "cnn_v2" / "best_config.json"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text("{}", encoding="utf-8")
    command = (
        f'source "{helper_path}" && '
        f'intronmodel_resolve_tuned_config_path '
        f'"{tmp_path}" "{species}" "cnn_v2" "donor" "" "{legacy_path}"'
    )
    run = subprocess.run(
        ["bash", "-lc", command],
        capture_output=True,
        text=True,
        check=False,
    )

    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == ""


def test_resolve_tuned_config_path_ignores_legacy_root_for_cnn_v3(
    tmp_path: Path,
) -> None:
    """cnn_v3 should resolve only task-scoped tuned configs."""

    helper_path = _project_root() / "run" / "lib" / "tuned_config.sh"
    species = "Dmel"
    legacy_path = tmp_path / species / "tuning" / "cnn_v3" / "best_config.json"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text("{}", encoding="utf-8")
    command = (
        f'source "{helper_path}" && '
        f'intronmodel_resolve_tuned_config_path '
        f'"{tmp_path}" "{species}" "cnn_v3" "pair" "" "{legacy_path}"'
    )
    run = subprocess.run(
        ["bash", "-lc", command],
        capture_output=True,
        text=True,
        check=False,
    )

    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == ""


def test_resolve_tuned_config_path_ignores_legacy_root_for_generic_public_pair_model(
    tmp_path: Path,
) -> None:
    """Public pair models should resolve only task-scoped tuned configs."""

    helper_path = _project_root() / "run" / "lib" / "tuned_config.sh"
    common_path = _project_root() / "run" / "lib" / "common.sh"
    species = "Dmel"
    legacy_path = tmp_path / species / "tuning" / "bilstm_pair" / "best_config.json"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text("{}", encoding="utf-8")
    command = (
        f'source "{common_path}" && '
        f'source "{helper_path}" && '
        f'intronmodel_resolve_tuned_config_path '
        f'"{tmp_path}" "{species}" "bilstm_pair" "pair" "" "{legacy_path}"'
    )
    run = subprocess.run(
        ["bash", "-lc", command],
        capture_output=True,
        text=True,
        check=False,
    )

    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == ""


def test_load_tuned_overrides_accepts_legacy_sequence_transform(
    tmp_path: Path,
) -> None:
    """Legacy sequence_transform-only configs should still load cleanly."""

    config_path = tmp_path / "best_config.json"
    config_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "sampled_params": {
                    "sequence_transform": "truncate_outside_intron",
                    "lr": 0.001,
                },
            }
        ),
        encoding="utf-8",
    )

    assert _run_tuned_config_helper(config_path) == [
        "sequence_transform\ttruncate_outside_intron",
        "lr\t0.001",
    ]
