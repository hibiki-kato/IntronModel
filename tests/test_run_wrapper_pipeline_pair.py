from __future__ import annotations

from pathlib import Path

import tools.run_wrapper_pipeline as run_wrapper_pipeline
from tools.run_wrapper_pipeline import WrapperSpec
from tools.run_wrapper_pipeline import (
    _apply_mask_mode_defaults,
    _build_run_args,
    _resolve_species_path_template,
    _resolve_expected_checkpoint_paths_for_run,
    _resolve_tasks_for_target,
)


def test_resolve_tasks_for_target_accepts_single_task_model() -> None:
    tasks = _resolve_tasks_for_target(
        train_target="pair",
        model_tasks=("pair",),
        train_only=False,
    )
    assert tasks == ("pair",)


def test_resolve_expected_checkpoint_paths_for_pair_model() -> None:
    paths, required_tasks = _resolve_expected_checkpoint_paths_for_run(
        [
            "--model",
            "cnn_pair",
            "--species",
            "Dmel",
            "--train_target",
            "pair",
        ]
    )

    assert required_tasks == ("pair",)
    assert set(paths.keys()) == {"pair"}


def test_build_run_args_accepts_train_data_path_overrides() -> None:
    spec = WrapperSpec(
        script_name="unit.sh",
        model_env_name="cnn_pair",
        supports_tuned_hparams=False,
        tuned_key_map={},
        stem_param_builder="cnn",
        required_arg_keys=(),
        per_task_override_keys=("KERNEL_SIZES",),
    )
    env = {
        "MODEL": "cnn_pair",
        "TRAIN_POS_PATH": "/tmp/train.pos.err",
        "TRAIN_NEG_PATH": "/tmp/train.neg.err",
        "KERNEL_SIZES": "11,7,5",
        "DONOR_KERNEL_SIZES": "13,9,5",
        "ACCEPTOR_KERNEL_SIZES": "9,7,5",
    }
    args = _build_run_args(spec, env)
    assert "--train_pos_path" in args
    assert "/tmp/train.pos.err" in args
    assert "--train_neg_path" in args
    assert "/tmp/train.neg.err" in args
    assert "--kernel_sizes" in args
    assert "11,7,5" in args
    assert "--donor_kernel_sizes" in args
    assert "13,9,5" in args
    assert "--acceptor_kernel_sizes" in args
    assert "9,7,5" in args


def test_build_run_args_forwards_pair_fusion_mode() -> None:
    spec = WrapperSpec(
        script_name="unit.sh",
        model_env_name="cnn_pair",
        supports_tuned_hparams=False,
        tuned_key_map={},
        stem_param_builder="cnn",
        required_arg_keys=("FUSION_MODE",),
        per_task_override_keys=(),
    )
    env = {
        "MODEL": "cnn_pair",
        "FUSION_MODE": "early",
    }

    args = _build_run_args(spec, env)

    assert "--fusion_mode" in args
    assert "early" in args


def test_resolve_species_path_template_replaces_all_tokens() -> None:
    resolved = _resolve_species_path_template(
        "data/{species}/raw/${SPECIES}/{SPECIES}.tsv",
        "Dmel",
    )
    assert resolved == "data/Dmel/raw/Dmel/Dmel.tsv"


def test_apply_mask_mode_defaults_sets_tag_and_paths(tmp_path: Path) -> None:
    raw_dir = tmp_path / "Dmel" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "transcripts_mask.tsv").write_text(
        "transcript_id\tsite_type\tintron_index\tseq\n",
        encoding="utf-8",
    )
    env: dict[str, str] = {
        "MASK_MODE": "on",
        "DONOR_LEN": "80",
        "ACCEPTOR_LEN": "100",
        "NAME_FIELDS": "none",
        "TAG": "",
        "TRAIN_POS_PATH": "",
        "TRAIN_NEG_PATH": "",
        "TEST_TSV_PATH": "",
    }

    _apply_mask_mode_defaults(
        env=env,
        data_root=tmp_path,
        species="Dmel",
        process_env={},
    )

    assert env["TAG"] == "mask"
    assert env["NAME_FIELDS"] == "tag"
    assert env["TRAIN_POS_PATH"] == str(raw_dir / "100bp_trimmed_npad.err")
    assert env["TRAIN_NEG_PATH"] == str(raw_dir / "100bp_trimmed_npad.neg.err")
    assert env["TEST_TSV_PATH"] == str(raw_dir / "transcripts_mask.tsv")


def test_apply_mask_mode_defaults_builds_mask_test_tsv_when_missing(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    raw_dir = tmp_path / "Dmel" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    generated = raw_dir / "transcripts_with_intron_half.tsv"
    generated.write_text(
        "transcript_id\tsite_type\tintron_index\tseq\n",
        encoding="utf-8",
    )

    env: dict[str, str] = {
        "MASK_MODE": "on",
        "DONOR_LEN": "100",
        "ACCEPTOR_LEN": "100",
        "NAME_FIELDS": "none",
        "TAG": "",
        "TRAIN_POS_PATH": "",
        "TRAIN_NEG_PATH": "",
        "TEST_TSV_PATH": "",
    }

    def _fake_build_mask_test_tsv(**_: object) -> Path:
        return generated

    monkeypatch.setattr(
        run_wrapper_pipeline,
        "_build_mask_test_tsv",
        _fake_build_mask_test_tsv,
    )

    _apply_mask_mode_defaults(
        env=env,
        data_root=tmp_path,
        species="Dmel",
        process_env={},
    )

    assert env["TEST_TSV_PATH"] == str(generated)
