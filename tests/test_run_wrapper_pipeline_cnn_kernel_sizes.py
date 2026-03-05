from __future__ import annotations

from tools.run_wrapper_pipeline import WrapperSpec
from tools.run_wrapper_pipeline import _build_run_args


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
    args = _build_run_args(spec, env)
    assert "--kernel_sizes" in args
    assert "11,9,7" in args
    assert "--donor_kernel_sizes" in args
    assert "13,11" in args
    assert "--acceptor_kernel_sizes" in args
    assert "9,7,5" in args
