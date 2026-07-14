from __future__ import annotations

import argparse

import pytest
import torch
import torch.nn.functional as F

from dev.IntronModel.src.models import cnn_v3, cnn_v4
from dev.IntronModel.src.models.cnn_pair_v3 import OrganicBranchLayout
from dev.IntronModel.src.models.registry import available_models, load_model_module
from dev.IntronModel.src.run_model import _build_checkpoint_paths


def _arch() -> cnn_v3.TaskOrganicArchParams:
    return cnn_v3.TaskOrganicArchParams(
        layout=OrganicBranchLayout(
            channels=[8, 8],
            kernel_sizes=[3, 3],
            dilations=[1, 1],
            residual_channels=[4, 4],
        ),
        head_type="gap",
        fc_hidden=8,
    )


def test_deformable_conv_zero_offsets_matches_grouped_conv1d() -> None:
    layer = cnn_v4.DeformableConv1d(4, 4, 3, groups=2)
    x = torch.randn(2, 4, 19)

    expected = F.conv1d(x, layer.weight, layer.bias, padding=1, groups=2)

    assert torch.allclose(layer(x), expected, atol=1e-5, rtol=1e-5)


def test_grouped_deformable_cnn_forward_shape() -> None:
    model = cnn_v4.OrganicSiteCNN(
        arch_params=_arch(),
        dropout=0.1,
        deformable_params=cnn_v4.GroupedDeformableParams(groups=2, kernel_size=3),
    )

    assert model(torch.randn(3, 4, 31)).shape == (3,)


@pytest.mark.parametrize("groups", [0, 3])
def test_grouped_deformable_stem_rejects_invalid_groups(groups: int) -> None:
    with pytest.raises(ValueError, match="groups|divide"):
        _ = cnn_v4.GroupedDeformableStem(groups=groups, kernel_size=3)


def test_resolve_task_deformable_params_prefers_task_override() -> None:
    args = argparse.Namespace(
        deformable_groups=2,
        donor_deformable_groups=4,
        acceptor_deformable_groups=None,
        deformable_kernel_size=3,
        donor_deformable_kernel_size=5,
        acceptor_deformable_kernel_size=None,
    )

    donor = cnn_v4._resolve_task_deformable_params("donor", args)
    acceptor = cnn_v4._resolve_task_deformable_params("acceptor", args)

    assert donor == cnn_v4.GroupedDeformableParams(groups=4, kernel_size=5)
    assert acceptor == cnn_v4.GroupedDeformableParams(groups=2, kernel_size=3)


def test_cnn_v4_is_registered() -> None:
    assert "cnn_v4" in available_models()


def test_cnn_v4_registry_loads_common_model_contract() -> None:
    module = load_model_module("cnn_v4")
    assert callable(module.train)
    assert callable(module.infer_site)


def test_checkpoint_paths_are_species_local() -> None:
    hsap = _build_checkpoint_paths("Hsap", "cnn_v4_unit")
    dmel = _build_checkpoint_paths("Dmel", "cnn_v4_unit")

    assert hsap["donor"] != dmel["donor"]
    assert "/Hsap/donor/" in hsap["donor"]
    assert "/Dmel/donor/" in dmel["donor"]


def test_cnn_v4_checkpoint_round_trip(tmp_path: Path) -> None:
    params = cnn_v4.GroupedDeformableParams(groups=2, kernel_size=3)
    model = cnn_v4.OrganicSiteCNN(arch_params=_arch(), dropout=0.1, deformable_params=params)
    model.eval()
    checkpoint_path = tmp_path / "Hsap_donor.pt"
    torch.save(
        {
            "task": "donor",
            "window_len": 31,
            "model_config": {
                "site_arch": "organic_resdil_grouped_deformable",
                "conv_channels": [8, 8],
                "kernel_sizes": [3, 3],
                "block_dilations": [1, 1],
                "residual_channels": [4, 4],
                "head_type": "gap",
                "dropout": 0.1,
                "fc_hidden": 8,
                "deformable_groups": 2,
                "deformable_kernel_size": 3,
            },
            "model_state": model.state_dict(),
        },
        checkpoint_path,
    )

    loaded, payload = cnn_v4.load_task_model(str(checkpoint_path), "cpu")
    x = torch.randn(2, 4, 31)

    assert payload["window_len"] == 31
    assert torch.allclose(model(x), loaded(x))
