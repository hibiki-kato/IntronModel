from __future__ import annotations

from pathlib import Path

import pytest

import tools.run_wrapper_pipeline as run_wrapper_pipeline
from tools.run_wrapper_pipeline import SPECS, WrapperSpec
from tools.run_wrapper_pipeline import (
    _apply_wrapper_defaults,
    _apply_mask_mode_defaults,
    _build_run_args,
    _resolve_species_path_template,
    _resolve_expected_checkpoint_paths_for_run,
    _resolve_tasks_for_target,
    _stem_params,
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


def test_build_run_args_forwards_pair_max_pool_flag() -> None:
    spec = WrapperSpec(
        script_name="unit.sh",
        model_env_name="cnn_pair",
        supports_tuned_hparams=False,
        tuned_key_map={},
        stem_param_builder="cnn",
        required_arg_keys=("MAX_POOL_SIZE", "CONV_STRIDE", "HEAD_TYPE"),
        per_task_override_keys=(),
    )
    env = {
        "MODEL": "cnn_pair",
        "MAX_POOL_SIZE": "1",
        "CONV_STRIDE": "2",
        "HEAD_TYPE": "center",
    }

    args = _build_run_args(spec, env)

    assert "--max_pool_size" in args
    assert args[args.index("--max_pool_size") + 1] == "1"
    assert "--conv_stride" in args
    assert args[args.index("--conv_stride") + 1] == "2"
    assert "--head_type" in args
    assert args[args.index("--head_type") + 1] == "center"


def test_build_run_args_forwards_dnabert_infer_runtime_overrides() -> None:
    spec = WrapperSpec(
        script_name="unit.sh",
        model_env_name="dnabert_pair",
        supports_tuned_hparams=False,
        tuned_key_map={},
        stem_param_builder="dnabert_pair",
        required_arg_keys=(),
        per_task_override_keys=(),
    )
    args = _build_run_args(
        spec,
        {
            "MODEL": "dnabert_pair",
            "INFER_BATCH_SIZE": "384",
            "INFER_USE_AMP": "1",
            "INFER_AMP_DTYPE": "fp16",
            "INFER_COMPILE": "1",
            "INFER_COMPILE_MODE": "on",
        },
    )

    assert "--infer_batch_size" in args
    assert args[args.index("--infer_batch_size") + 1] == "384"
    assert "--infer_use_amp" in args
    assert args[args.index("--infer_use_amp") + 1] == "1"
    assert "--infer_amp_dtype" in args
    assert args[args.index("--infer_amp_dtype") + 1] == "fp16"
    assert "--infer_compile" in args
    assert args[args.index("--infer_compile") + 1] == "1"
    assert "--infer_compile_mode" in args
    assert args[args.index("--infer_compile_mode") + 1] == "on"


def test_stem_params_include_pair_max_pool_flag() -> None:
    params = _stem_params(
        "cnn_pair",
        {
            "DONOR_LEN": "100",
            "ACCEPTOR_LEN": "100",
            "EPOCHS": "10",
            "BATCH_SIZE": "256",
            "LR": "0.001",
            "LOSS": "focal",
            "WEIGHT_DECAY": "0.01",
            "ETA_MIN_RATIO": "0.01",
            "GRAD_CLIP": "5.0",
            "VAL_FRAC": "0.1",
            "TRANSCRIPT_SCORE_AGG": "min",
            "SOFTMIN_TAU": "1.0",
            "SEED": "1337",
            "TRAIN_TARGET": "pair",
            "CONV_CHANNELS": "64,128,256",
            "KERNEL_SIZES": "7,7,7",
            "MAX_POOL_SIZE": "1",
            "CONV_STRIDE": "2",
            "HEAD_TYPE": "center",
            "DROPOUT": "0.3",
            "FC_HIDDEN": "128",
        },
    )

    assert params["max_pool_size"] == 1
    assert params["conv_stride"] == 2
    assert params["head_type"] == "center"


def test_apply_wrapper_defaults_fills_single_task_pair_values() -> None:
    spec = WrapperSpec(
        script_name="unit.sh",
        model_env_name="cnn_pair",
        supports_tuned_hparams=False,
        tuned_key_map={},
        stem_param_builder="cnn_pair",
        required_arg_keys=(),
        per_task_override_keys=(),
    )
    env: dict[str, str] = {}

    _apply_wrapper_defaults(spec, env)

    assert env["MODEL"] == "cnn_pair"
    assert env["TRAIN_TARGET"] == "pair"


def test_cnn_pair_spec_does_not_require_redundant_pair_envs() -> None:
    spec = SPECS["cnn_pair.sh"]

    assert spec.stem_param_builder == "cnn_pair"
    assert "TRAIN_TARGET" not in spec.required_arg_keys
    assert "INTRON_SCORE_OP" not in spec.required_arg_keys


def test_markov_xgboost_spec_single_task_pair() -> None:
    spec = SPECS["markov_xgboost.sh"]

    assert spec.model_env_name == "markov_xgboost"
    assert spec.stem_param_builder == "markov_xgboost"
    assert "TRAIN_TARGET" not in spec.required_arg_keys
    assert "MARKOV_ORDER" in spec.required_arg_keys
    assert "MARKOV_FEATURE_MODE" in spec.required_arg_keys
    assert "MARKOV_CACHE_MODE" in spec.required_arg_keys
    assert "XGB_N_ESTIMATORS" in spec.required_arg_keys


def test_dnabert_pair_spec_single_task_pair() -> None:
    spec = SPECS["dnabert_pair.sh"]

    assert spec.model_env_name == "dnabert_pair"
    assert spec.stem_param_builder == "dnabert_pair"
    assert "TRAIN_TARGET" in spec.required_arg_keys
    assert "INTRON_SCORE_OP" not in spec.required_arg_keys


def test_bilstm_pair_spec_supports_tuned_hparams() -> None:
    spec = SPECS["bilstm_pair.sh"]

    assert spec.model_env_name == "bilstm_pair"
    assert spec.supports_tuned_hparams is True
    assert spec.tuned_key_map["hidden_size"] == "HIDDEN_SIZE"
    assert spec.tuned_key_map["input_mode"] == "INPUT_MODE"
    assert (
        spec.tuned_key_map["bpe_pretrained_model_name"]
        == "BPE_PRETRAINED_MODEL_NAME"
    )


def test_bilstm_pair_spec_excludes_intron_score_op() -> None:
    spec = SPECS["bilstm_pair.sh"]

    assert "TRAIN_TARGET" not in spec.required_arg_keys
    assert "INTRON_SCORE_OP" not in spec.required_arg_keys


def test_stem_params_for_markov_xgboost_builder() -> None:
    params = _stem_params(
        "markov_xgboost",
        {
            "DONOR_LEN": "100",
            "ACCEPTOR_LEN": "100",
            "MARKOV_ORDER": "2",
            "MARKOV_ALPHA": "0.5",
            "MARKOV_FEATURE_MODE": "per_base",
            "VAL_FRAC": "0.1",
            "TRANSCRIPT_SCORE_AGG": "min",
            "SOFTMIN_TAU": "1.0",
            "SEED": "1337",
            "TRAIN_TARGET": "pair",
            "TAG": "mxgb",
        },
    )

    assert params["markov_order"] == 2
    assert params["markov_alpha"] == 0.5
    assert params["markov_feature_mode"] == "per_base"
    assert params["train_target"] == "pair"
    assert params["tag"] == "mxgb"


def test_stem_params_for_dnabert_pair_excludes_intron_score_op() -> None:
    params = _stem_params(
        "dnabert_pair",
        {
            "DONOR_LEN": "100",
            "ACCEPTOR_LEN": "100",
            "EPOCHS": "6",
            "BATCH_SIZE": "32",
            "LR": "2e-5",
            "LOSS": "weighted_bce",
            "WEIGHT_DECAY": "0.01",
            "ETA_MIN_RATIO": "0.01",
            "GRAD_CLIP": "1.0",
            "VAL_FRAC": "0.1",
            "TRANSCRIPT_SCORE_AGG": "min",
            "SOFTMIN_TAU": "1.0",
            "SEED": "1337",
            "TRAIN_TARGET": "pair",
            "MAX_TOKENS": "auto",
            "DROPOUT": "0.1",
            "HEAD_LAYER_NORM": "1",
            "READOUT_TYPE": "cnn",
            "READOUT_CNN_KERNEL_SIZE": "3",
            "READOUT_MLP_HIDDEN_DIM": "256",
            "READOUT_MLP_LAYERS": "1",
        },
    )

    assert params["train_target"] == "pair"
    assert "intron_score_op" not in params
    assert params["readout_type"] == "cnn"
    assert params["readout_cnn_kernel_size"] == 3


def test_resolve_species_path_template_replaces_all_tokens() -> None:
    resolved = _resolve_species_path_template(
        "data/{species}/raw/${SPECIES}/{SPECIES}.tsv",
        "Dmel",
    )
    assert resolved == "data/Dmel/raw/Dmel/Dmel.tsv"


def test_apply_mask_mode_defaults_sets_tag_and_paths(tmp_path: Path) -> None:
    raw_dir = tmp_path / "Dmel" / "raw"
    processed_dir = tmp_path / "Dmel" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
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
    assert env["TRAIN_POS_PATH"] == str(processed_dir / "100bp_trimmed_npad.err")
    assert env["TRAIN_NEG_PATH"] == str(processed_dir / "100bp_trimmed_npad.neg.err")
    assert env["TEST_TSV_PATH"] == str(raw_dir / "transcripts_mask.tsv")


def test_apply_mask_mode_defaults_prefers_unique_tsv_over_detected(
    tmp_path: Path,
) -> None:
    """transcripts.unique.tsv must be used before falling back to detection."""
    processed_dir = tmp_path / "Dmel" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    # Place both unique TSV and a legacy mask TSV; unique must win.
    unique_tsv = processed_dir / "transcripts.unique.tsv"
    unique_tsv.write_text(
        "transcript_id\tsite_type\tintron_index\tseq\n",
        encoding="utf-8",
    )
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

    assert env["TEST_TSV_PATH"] == str(unique_tsv)


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


def test_run_single_species_requires_processed_transcripts_tsv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    species = "Dmel"
    spec = SPECS["cnn_pair.sh"]
    env: dict[str, str] = {}
    _apply_wrapper_defaults(spec, env)
    env["MPS_MAX_BATCH_SIZE"] = "2048"
    env["MODEL"] = "cnn_pair"
    env["DONOR_LEN"] = "100"
    env["ACCEPTOR_LEN"] = "100"

    species_raw_dir = tmp_path / species / "raw"
    species_raw_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(run_wrapper_pipeline, "_stem_params", lambda *_args: {})
    monkeypatch.setattr(
        run_wrapper_pipeline,
        "build_output_stem",
        lambda **_kwargs: "unit_stem",
    )

    with pytest.raises(
        FileNotFoundError,
        match="Missing required processed unique transcript TSV",
    ):
        run_wrapper_pipeline._run_single_species(
            spec=spec,
            project_root=tmp_path,
            data_root=tmp_path,
            env=env,
            species=species,
            process_env={},
        )
