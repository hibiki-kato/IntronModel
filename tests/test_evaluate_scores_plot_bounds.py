from __future__ import annotations

import pytest

from evaluate_scores import resolve_plot_bounds, resolve_plot_output


def test_resolve_plot_bounds_for_known_species() -> None:
    bounds = resolve_plot_bounds(
        species="Mmus",
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
    )
    assert bounds == (10.0, 18.0, 40.0, 46.0)


def test_resolve_plot_bounds_for_hsap_defaults() -> None:
    bounds = resolve_plot_bounds(
        species="Hsap",
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
    )
    assert bounds == (10.0, 19.0, 26.0, 35.0)


def test_resolve_plot_bounds_unknown_species_requires_explicit_bounds() -> None:
    with pytest.raises(ValueError, match="Unknown species"):
        resolve_plot_bounds(
            species="Mmel",
            x_min=None,
            x_max=None,
            y_min=None,
            y_max=None,
        )


def test_resolve_plot_bounds_unknown_species_with_explicit_bounds() -> None:
    bounds = resolve_plot_bounds(
        species="Mmel",
        x_min=1.0,
        x_max=2.0,
        y_min=3.0,
        y_max=4.0,
    )
    assert bounds == (1.0, 2.0, 3.0, 4.0)


def test_resolve_plot_output_defaults_to_species_snpr_name() -> None:
    output_path = resolve_plot_output(species="Mmus", output_png=None)
    assert output_path.endswith("/data/Mmus/Mmus_snpr.png")
