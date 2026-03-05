from __future__ import annotations

from tools.run_wrapper_pipeline import WrapperSpec
from tools.run_wrapper_pipeline import (
    _build_run_args,
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
