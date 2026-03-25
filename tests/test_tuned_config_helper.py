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
                        "train_target": "both",
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
        "train_target\tboth",
        "sequence_transform\tmask_outside_intron_n",
        "batch_size\t256",
        "lr\t0.001",
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
                        "train_target": "both",
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
        "train_target\tboth",
        "sequence_transform\tnone",
        "batch_size\t256",
    ]


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
