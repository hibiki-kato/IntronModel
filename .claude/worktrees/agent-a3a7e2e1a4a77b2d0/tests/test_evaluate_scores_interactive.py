from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from evaluate_scores import (
    LEGEND_HIDDEN_ALPHA,
    LEGEND_VISIBLE_ALPHA,
    _connect_interactive_legend_toggle,
    _validate_interactive_backend,
)


def test_interactive_legend_toggle_updates_scatter_visibility() -> None:
    fig, ax = plt.subplots()
    scatter_artist = ax.scatter([1.0, 2.0], [3.0, 4.0], label="model_a")
    max_f1_artist = ax.scatter([2.0], [4.0], color="black")
    legend = ax.legend()

    try:
        callback = _connect_interactive_legend_toggle(
            fig=fig,
            legend=legend,
            labeled_artists={"model_a": (scatter_artist, max_f1_artist)},
        )

        legend_handle = legend.legend_handles[0]
        legend_text = legend.get_texts()[0]

        assert scatter_artist.get_visible() is True
        assert max_f1_artist.get_visible() is True
        assert legend_handle.get_alpha() == pytest.approx(LEGEND_VISIBLE_ALPHA)
        assert legend_text.get_alpha() == pytest.approx(LEGEND_VISIBLE_ALPHA)

        callback(SimpleNamespace(artist=legend_handle))
        assert scatter_artist.get_visible() is False
        assert max_f1_artist.get_visible() is False
        assert legend_handle.get_alpha() == pytest.approx(LEGEND_HIDDEN_ALPHA)
        assert legend_text.get_alpha() == pytest.approx(LEGEND_HIDDEN_ALPHA)

        callback(SimpleNamespace(artist=legend_text))
        assert scatter_artist.get_visible() is True
        assert max_f1_artist.get_visible() is True
        assert legend_handle.get_alpha() == pytest.approx(LEGEND_VISIBLE_ALPHA)
        assert legend_text.get_alpha() == pytest.approx(LEGEND_VISIBLE_ALPHA)
    finally:
        plt.close(fig)


def test_interactive_legend_toggle_updates_outside_legend_visibility() -> None:
    fig, ax = plt.subplots()
    scatter_artist = ax.scatter([1.0, 2.0], [3.0, 4.0], label="model_a")
    max_f1_artist = ax.scatter([2.0], [4.0], color="black")
    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    try:
        callback = _connect_interactive_legend_toggle(
            fig=fig,
            legend=legend,
            labeled_artists={"model_a": (scatter_artist, max_f1_artist)},
        )

        legend_handle = legend.legend_handles[0]
        legend_text = legend.get_texts()[0]

        callback(SimpleNamespace(artist=legend_handle))
        assert scatter_artist.get_visible() is False
        assert max_f1_artist.get_visible() is False
        assert legend_handle.get_alpha() == pytest.approx(LEGEND_HIDDEN_ALPHA)
        assert legend_text.get_alpha() == pytest.approx(LEGEND_HIDDEN_ALPHA)

        callback(SimpleNamespace(artist=legend_text))
        assert scatter_artist.get_visible() is True
        assert max_f1_artist.get_visible() is True
        assert legend_handle.get_alpha() == pytest.approx(LEGEND_VISIBLE_ALPHA)
        assert legend_text.get_alpha() == pytest.approx(LEGEND_VISIBLE_ALPHA)
    finally:
        plt.close(fig)


def test_interactive_legend_toggle_requires_matching_label() -> None:
    fig, ax = plt.subplots()
    _ = ax.scatter([1.0], [2.0], label="model_a")
    legend = ax.legend()

    try:
        with pytest.raises(ValueError, match="does not match a plotted artist"):
            _connect_interactive_legend_toggle(
                fig=fig,
                legend=legend,
                labeled_artists={},
            )
    finally:
        plt.close(fig)


def test_validate_interactive_backend_rejects_agg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("evaluate_scores.matplotlib.get_backend", lambda: "agg")

    with pytest.raises(RuntimeError, match="GUI Matplotlib backend"):
        _validate_interactive_backend()


def test_validate_interactive_backend_accepts_gui_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("evaluate_scores.matplotlib.get_backend", lambda: "TkAgg")

    _validate_interactive_backend()
