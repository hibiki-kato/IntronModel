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
        "MAX_POOL_SIZE": "1",
        "KERNEL_SIZES": "11,9,7",
        "DONOR_KERNEL_SIZES": "13,11",
        "ACCEPTOR_KERNEL_SIZES": "9,7,5",
    }
    args = _build_run_args(spec, env)
    assert "--max_pool_size" not in args
    assert "--kernel_sizes" in args
    assert "11,9,7" in args
    assert "--donor_kernel_sizes" in args
    assert "13,11" in args
    assert "--acceptor_kernel_sizes" in args
    assert "9,7,5" in args


def test_stem_params_include_cnn_max_pool_flag() -> None:
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
            "MAX_POOL_SIZE": "1",
            "DROPOUT": "0.3",
            "FC_HIDDEN": "128",
        },
    )

    assert params["max_pool_size"] == 1


def test_build_run_args_includes_cnn_max_pool_flag_when_required() -> None:
    spec = WrapperSpec(
        script_name="unit.sh",
        model_env_name="cnn",
        supports_tuned_hparams=False,
        tuned_key_map={},
        stem_param_builder="cnn",
        required_arg_keys=("MAX_POOL_SIZE",),
        per_task_override_keys=(),
    )
    args = _build_run_args(
        spec,
        {
            "MODEL": "cnn",
            "MAX_POOL_SIZE": "1",
        },
    )

    assert "--max_pool_size" in args
    assert args[args.index("--max_pool_size") + 1] == "1"
