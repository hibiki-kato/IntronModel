from __future__ import annotations

import json
from pathlib import Path

from tools import tune_time_scheduler


def test_load_config_keeps_bare_python_command_name(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    data_root = project_root / "data"
    model_root = project_root / "model"
    config_root = project_root / "scheduler"
    output_parent_dir = project_root / "runs"
    template_config_path = project_root / "templates" / "job.json"
    jobs_file = config_root / "jobs.jsonl"
    config_path = config_root / "scheduler_config.json"

    data_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    output_parent_dir.mkdir(parents=True, exist_ok=True)
    template_config_path.parent.mkdir(parents=True, exist_ok=True)
    config_root.mkdir(parents=True, exist_ok=True)
    template_config_path.write_text("{}", encoding="utf-8")
    jobs_file.write_text(
        json.dumps(
            {
                "species": "Hsap",
                "target_name": "donor",
                "seed": 1337,
                "tuning_model_name": "cnn_v3",
                "template_config_path": "../templates/job.json",
                "output_parent_dir": "../runs",
                "plot_target_name": "donor",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config_path.write_text(
        json.dumps(
            {
                "script_name": "tune_cnn_v3_time.sh",
                "project_root": str(project_root),
                "data_root": "data",
                "model_root": "model",
                "python_bin": "python3",
                "time_budget_minutes": 10,
                "timeout_grace_seconds": 30,
                "selected_gpu_ids": ["0"],
                "parallel_slot_count": 1,
                "start_epoch": "2026-03-29T00:00:00Z",
                "jobs_file": "scheduler/jobs.jsonl",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    config = tune_time_scheduler._load_config(config_path)

    assert config.project_root == project_root.resolve()
    assert config.python_bin == "python3"
    assert config.jobs[0].template_config_path == template_config_path.resolve()
