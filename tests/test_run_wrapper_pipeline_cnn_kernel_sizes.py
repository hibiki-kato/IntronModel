from __future__ import annotations

from tools.run_wrapper_pipeline import WrapperSpec
from tools.run_wrapper_pipeline import _build_run_args
from tools.run_wrapper_pipeline import _stem_params


def test_build_run_args_accepts_cnn_kernel_size_overrides() -> None:
    spec = WrapperSpec(
        script_name="unit.sh",
        model_env_name="cnn",
        supports_tuned_hparams=False,
        tuned_key_map={},
        stem_param_builder="cnn",
        required_arg_keys=(),
        per_task_override_keys=("KERNEL_SIZES",),
    )
    env = {
        "MODEL": "cnn",
        "KERNEL_SIZES": "11,9,7",
        "DONOR_KERNEL_SIZES": "13,11",
        "ACCEPTOR_KERNEL_SIZES": "9,7,5",
    }
    args = _build_run_args(spec, env, tuned_config_paths={})
    assert "--kernel_sizes" in args
    assert "11,9,7" in args
    assert "--donor_kernel_sizes" in args
    assert "13,11" in args
    assert "--acceptor_kernel_sizes" in args
    assert "9,7,5" in args


def test_stem_params_include_cnn_arch_flags() -> None:
    params = _stem_params(
        "cnn",
        {
            "DONOR_LEN": "100",
            "ACCEPTOR_LEN": "100",
            "EPOCHS": "15",
            "BATCH_SIZE": "256",
            "LR": "0.001",
            "LOSS": "focal",
            "WEIGHT_DECAY": "0.01",
            "ETA_MIN_RATIO": "0.01",
            "GRAD_CLIP": "5.0",
            "VAL_FRAC": "0.1",
            "INTRON_SCORE_OP": "*",
            "TRANSCRIPT_SCORE_AGG": "min",
            "SOFTMIN_TAU": "1.0",
            "SEED": "1337",
            "TRAIN_TARGET": "both",
            "CONV_CHANNELS": "64,128,256",
            "KERNEL_SIZES": "7,7,7",
            "CONV_STRIDE": "2",
            "HEAD_TYPE": "center",
            "DROPOUT": "0.3",
            "FC_HIDDEN": "128",
        },
    )

    assert params["conv_stride"] == 2
    assert params["head_type"] == "center"


def test_build_run_args_includes_cnn_arch_flags_when_required() -> None:
    spec = WrapperSpec(
        script_name="unit.sh",
        model_env_name="cnn",
        supports_tuned_hparams=False,
        tuned_key_map={},
        stem_param_builder="cnn",
        required_arg_keys=("CONV_STRIDE", "HEAD_TYPE"),
        per_task_override_keys=(),
    )
    args = _build_run_args(
        spec,
        {
            "MODEL": "cnn",
            "CONV_STRIDE": "2",
            "HEAD_TYPE": "center",
        },
        tuned_config_paths={},
    )

    assert "--conv_stride" in args
    assert args[args.index("--conv_stride") + 1] == "2"
    assert "--head_type" in args
    assert args[args.index("--head_type") + 1] == "center"


def test_build_run_args_forwards_infer_runtime_overrides() -> None:
    spec = WrapperSpec(
        script_name="unit.sh",
        model_env_name="cnn",
        supports_tuned_hparams=False,
        tuned_key_map={},
        stem_param_builder="cnn",
        required_arg_keys=(),
        per_task_override_keys=(),
    )
    args = _build_run_args(
        spec,
        {
            "MODEL": "cnn",
            "INFER_BATCH_SIZE": "1536",
            "INFER_USE_AMP": "1",
            "INFER_AMP_DTYPE": "bf16",
            "INFER_COMPILE": "0",
            "INFER_COMPILE_MODE": "auto",
        },
        tuned_config_paths={},
    )

    assert "--infer_batch_size" in args
    assert args[args.index("--infer_batch_size") + 1] == "1536"
    assert "--infer_use_amp" in args
    assert args[args.index("--infer_use_amp") + 1] == "1"
    assert "--infer_amp_dtype" in args
    assert args[args.index("--infer_amp_dtype") + 1] == "bf16"
    assert "--infer_compile" in args
    assert args[args.index("--infer_compile") + 1] == "0"
    assert "--infer_compile_mode" in args
    assert args[args.index("--infer_compile_mode") + 1] == "auto"
