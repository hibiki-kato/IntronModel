from __future__ import annotations

from pathlib import Path
import resource

import tools.run_wrapper_pipeline as run_wrapper_pipeline
from tools.run_wrapper_pipeline import WrapperSpec
from tools.run_wrapper_pipeline import _apply_species_parallel_env_overrides
from tools.run_wrapper_pipeline import _resolve_parallel_auto_num_workers
from tools.run_wrapper_pipeline import _resolve_species_gpu_slots
from tools.run_wrapper_pipeline import _run_species_batch


def _build_spec() -> WrapperSpec:
    """Build a minimal wrapper spec for scheduler tests."""

    return WrapperSpec(
        script_name="unit.sh",
        model_env_name="cnn",
        supports_tuned_hparams=False,
        tuned_key_map={},
        stem_param_builder="cnn",
        required_arg_keys=(),
        per_task_override_keys=(),
    )


def test_resolve_species_gpu_slots_prefers_visible_env(
    monkeypatch: object,
) -> None:
    """Use CUDA_VISIBLE_DEVICES ordering when available."""

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,2")
    assert _resolve_species_gpu_slots("auto", "auto") == ["5", "2"]


def test_run_species_batch_parallel_assigns_gpu_per_species(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    """Assign one visible GPU to each species subprocess."""

    assignments: list[tuple[str, str | None, str | None]] = []

    def _fake_run_single_species(
        spec: WrapperSpec,
        *,
        project_root: Path,
        data_root: Path,
        env: dict[str, str],
        species: str,
        process_env: dict[str, str],
    ) -> int:
        del spec, project_root, data_root
        assignments.append(
            (
                species,
                process_env.get("CUDA_VISIBLE_DEVICES"),
                env.get("NUM_WORKERS"),
            )
        )
        return 0

    monkeypatch.setattr(
        run_wrapper_pipeline,
        "_run_single_species",
        _fake_run_single_species,
    )
    monkeypatch.setattr(
        run_wrapper_pipeline,
        "_resolve_species_gpu_slots",
        lambda _device, _gpu_ids_setting: ["0", "1"],
    )
    monkeypatch.setattr(
        run_wrapper_pipeline,
        "_resolve_parallel_auto_num_workers",
        lambda parallel_species: parallel_species + 1,
    )

    result = _run_species_batch(
        _build_spec(),
        project_root=tmp_path,
        data_root=tmp_path,
        base_env={"DEVICE": "auto", "NUM_WORKERS": "auto"},
        base_process_env={},
        species_list=["Athal", "Dmel", "Mmus"],
    )

    assert result == 0
    assert len(assignments) == 3
    assert {
        species for species, _gpu, _num_workers in assignments
    } == {"Athal", "Dmel", "Mmus"}
    assert {gpu for _species, gpu, _num_workers in assignments} == {"0", "1"}
    assert {
        num_workers for _species, _gpu, num_workers in assignments
    } == {"3"}


def test_apply_species_parallel_env_overrides_resolves_auto_num_workers(
    monkeypatch: object,
) -> None:
    """Lower auto worker count for parallel species runs."""

    monkeypatch.setattr(
        run_wrapper_pipeline,
        "_resolve_parallel_auto_num_workers",
        lambda parallel_species: 3 + parallel_species,
    )

    resolved = _apply_species_parallel_env_overrides(
        env={"MODEL": "cnn", "NUM_WORKERS": "auto", "DEVICE": "auto"},
        parallel_species=2,
        script_name="unit.sh",
    )

    assert resolved["NUM_WORKERS"] == "5"
    assert resolved["REPORT_TRAIN_METRICS"] == "0"


def test_apply_species_parallel_env_overrides_keeps_explicit_num_workers() -> None:
    """Leave explicit worker settings unchanged."""

    resolved = _apply_species_parallel_env_overrides(
        env={"MODEL": "cnn", "NUM_WORKERS": "6", "DEVICE": "auto"},
        parallel_species=3,
        script_name="unit.sh",
    )

    assert resolved["NUM_WORKERS"] == "6"
    assert resolved["REPORT_TRAIN_METRICS"] == "0"


def test_apply_species_parallel_env_overrides_leaves_serial_reporting_alone() -> None:
    """Do not touch train reporting when only one GPU slot is active."""

    resolved = _apply_species_parallel_env_overrides(
        env={"MODEL": "cnn", "NUM_WORKERS": "auto", "DEVICE": "auto"},
        parallel_species=1,
        script_name="unit.sh",
    )

    assert "REPORT_TRAIN_METRICS" not in resolved


def test_resolve_parallel_auto_num_workers_uses_cpu_and_gpu_process_count(
    monkeypatch: object,
) -> None:
    """Scale workers from CPU count and concurrent GPU process count."""

    monkeypatch.setattr(run_wrapper_pipeline.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(
        run_wrapper_pipeline.resource,
        "getrlimit",
        lambda _kind: (4096, resource.RLIM_INFINITY),
    )

    resolved = _resolve_parallel_auto_num_workers(4)

    assert resolved == 2


def test_run_species_batch_uses_serial_fallback_for_single_gpu(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    """Keep species order and preserve inherited visibility for single-GPU runs."""

    call_order: list[tuple[str, str | None]] = []

    def _fake_run_single_species(
        spec: WrapperSpec,
        *,
        project_root: Path,
        data_root: Path,
        env: dict[str, str],
        species: str,
        process_env: dict[str, str],
    ) -> int:
        del spec, project_root, data_root, env
        call_order.append((species, process_env.get("CUDA_VISIBLE_DEVICES")))
        return 0

    monkeypatch.setattr(
        run_wrapper_pipeline,
        "_run_single_species",
        _fake_run_single_species,
    )
    monkeypatch.setattr(
        run_wrapper_pipeline,
        "_resolve_species_gpu_slots",
        lambda _device, _gpu_ids_setting: ["0"],
    )

    result = _run_species_batch(
        _build_spec(),
        project_root=tmp_path,
        data_root=tmp_path,
        base_env={"DEVICE": "auto"},
        base_process_env={"CUDA_VISIBLE_DEVICES": "0"},
        species_list=["Athal", "Dmel"],
    )

    assert result == 0
    assert call_order == [("Athal", "0"), ("Dmel", "0")]
