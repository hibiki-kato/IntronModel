"""Evaluate transcript-level scores and optionally plot SN/PR scatter data."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.backend_bases import PickEvent
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.text import Text
from util.versioned_artifacts import finalize_ready_published_outputs_for_species

PLOT_BOUNDS_BY_SPECIES: dict[str, tuple[float, float, float, float]] = {
    "Athal": (10.0, 52.0, 48.0, 75.0),
    "Dmel": (40.0, 52.0, 40.0, 55.0),
    "Mmus": (10.0, 18.0, 40.0, 46.0),
    "Hsap": (10.0, 19.0, 27.0, 38.0),
}
FALLBACK_PLOT_BOUNDS: tuple[float, float, float, float] = (40.0, 50.0, 40.0, 50.0)

LEGEND_FONT_SIZE = 14
AXIS_TICK_FONT_SIZE = 16
AXIS_LABEL_FONT_SIZE = 18
TITLE_FONT_SIZE = 20
LEGEND_HIDDEN_ALPHA = 0.2
LEGEND_VISIBLE_ALPHA = 1.0
LEGEND_LOCATION = "upper left"
LEGEND_BBOX_ANCHOR: tuple[float, float] = (1.02, 1.0)
NON_GUI_BACKENDS: set[str] = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}


def resolve_project_root() -> Path:
    """Resolve the repository root for maintenance helpers."""

    return Path(__file__).resolve().parents[1]


def load_class_dict(class_file: str | Path) -> dict[str, str]:
    """Load transcript -> class mapping from ``transcript_class.txt``.

    Parameters
    ----------
    class_file : str | pathlib.Path
        Path to a whitespace-delimited file with two columns:
        transcript_id and class_code.

    Returns
    -------
    dict[str, str]
        Mapping from transcript id to class code.

    Raises
    ------
    ValueError
        If a non-empty line does not contain exactly two columns.
    """

    mapping: dict[str, str] = {}
    with Path(class_file).open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(
                    f"Invalid class line at {class_file}:{line_no}: {raw_line.rstrip()}"
                )
            transcript_id, class_code = fields
            mapping[transcript_id] = class_code
    return mapping


def count_reference_transcripts(ref_gff: str | Path) -> int:
    """Count reference transcripts from a GFF feature stream.

    This follows the legacy shell pipeline logic exactly:
    count consecutive ``exon`` runs whose run-length is greater than 1.

    Parameters
    ----------
    ref_gff : str | pathlib.Path
        Path to the reference GFF file.

    Returns
    -------
    int
        Number of counted reference transcript-like exon runs.

    Raises
    ------
    ValueError
        If no eligible exon run is found.
    """

    run_feature: str | None = None
    run_length = 0
    reference_count = 0

    with Path(ref_gff).open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 3:
                continue
            feature = fields[2]

            if feature == run_feature:
                run_length += 1
                continue

            if run_feature == "exon" and run_length > 1:
                reference_count += 1
            run_feature = feature
            run_length = 1

    if run_feature == "exon" and run_length > 1:
        reference_count += 1

    if reference_count <= 0:
        raise ValueError(
            f"Failed to derive positive reference count from ref_gff: {ref_gff}"
        )
    return reference_count


def _truncate_percent(numerator: int, denominator: int) -> float:
    """Compute truncated percentage with two decimals."""

    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return int((numerator / denominator) * 10000.0) / 100.0


def evaluate_score_file(
    class_file: str | Path,
    score_file: str | Path,
    ref_gff: str | Path,
) -> list[str]:
    """Evaluate transcript scores into SN/PR/F1 rows.

    Parameters
    ----------
    class_file : str | pathlib.Path
        Transcript class file (``transcript_id class_code``).
    score_file : str | pathlib.Path
        Transcript score table. The 1st column is transcript id and the 5th
        column is the score used for sorting.
    ref_gff : str | pathlib.Path
        Reference GFF path used to compute the sensitivity denominator.

    Returns
    -------
    list[str]
        Lines formatted as:
        ``transcript_id score class_code sensitivity precision f1``.
    """

    class_dict = load_class_dict(class_file)
    reference_count = count_reference_transcripts(ref_gff)

    filtered_data: list[tuple[str, float, str]] = []
    running_total = 0
    running_good = 0

    with Path(score_file).open("r", encoding="utf-8") as f:
        next(f, None)
        for raw_line in f:
            fields = raw_line.strip().split()
            if len(fields) < 5:
                continue

            transcript_id = fields[0]
            class_code = class_dict.get(transcript_id)
            if class_code is None or class_code == "c":
                continue

            try:
                score = float(fields[4])
                last_field = float(fields[-1])
            except ValueError:
                continue
            if last_field == 10000:
                continue

            filtered_data.append((transcript_id, score, class_code))
            running_total += 1
            if class_code == "=":
                running_good += 1

    filtered_data.sort(key=lambda row: row[1])

    output_lines: list[str] = []
    for transcript_id, score, class_code in filtered_data:
        if class_code == "=":
            running_good -= 1
        running_total -= 1
        if running_total <= 0:
            continue

        sensitivity = _truncate_percent(running_good, reference_count)
        precision = _truncate_percent(running_good, running_total)
        f1 = 0.0
        if sensitivity + precision > 0.0:
            f1 = 2.0 * (sensitivity * precision) / (sensitivity + precision)
        output_lines.append(
            f"{transcript_id} {score} {class_code} {sensitivity} {precision} {f1}"
        )

    return output_lines


def infer_species_from_path(path: str, default_species: str = "Dmel") -> str:
    """Infer species name from a path that includes ``data/<species>/...``."""

    parts = os.path.normpath(path).split(os.sep)
    if "data" in parts:
        idx = parts.index("data")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return default_species


def resolve_eval_output_file(
    score_file: str,
    output_file: str | None,
    species: str | None = None,
) -> str:
    """Resolve output file path for evaluation text."""

    if output_file not in (None, "", "None"):
        return output_file

    inferred_species = species or infer_species_from_path(score_file)
    base = os.path.splitext(os.path.basename(score_file))[0]
    out_dir = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            inferred_species,
            "eval_score",
        )
    )
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{base}.txt")


def resolve_eval_dir(species: str) -> str:
    """Resolve ``data/<species>/eval_score`` directory path."""

    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "data", species, "eval_score")
    )


def resolve_plot_output(species: str, output_png: str | None) -> str:
    """Resolve output PNG path for summary scatter plot."""

    if output_png:
        return output_png
    return os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            species,
            f"{species}_snpr.png",
        )
    )


def resolve_plot_bounds(
    species: str,
    x_min: float | None,
    x_max: float | None,
    y_min: float | None,
    y_max: float | None,
) -> tuple[float, float, float, float]:
    """Resolve plotting bounds with species defaults."""

    default_bounds = PLOT_BOUNDS_BY_SPECIES.get(species)
    if default_bounds is None:
        missing_all_bounds = (
            x_min is None and x_max is None and y_min is None and y_max is None
        )
        if missing_all_bounds:
            supported = ", ".join(sorted(PLOT_BOUNDS_BY_SPECIES))
            raise ValueError(
                f"Unknown species '{species}' for default plot bounds. "
                f"Supported species: {supported}. "
                "Specify --x_min/--x_max/--y_min/--y_max explicitly."
            )
        default_bounds = FALLBACK_PLOT_BOUNDS
    resolved_x_min = default_bounds[0] if x_min is None else x_min
    resolved_x_max = default_bounds[1] if x_max is None else x_max
    resolved_y_min = default_bounds[2] if y_min is None else y_min
    resolved_y_max = default_bounds[3] if y_max is None else y_max
    return resolved_x_min, resolved_x_max, resolved_y_min, resolved_y_max


def _sync_legend_entry_visibility(
    legend_handle: Artist,
    legend_text: Text,
    target_artists: tuple[Artist, ...],
) -> None:
    """Match legend entry opacity to target artist visibility."""

    alpha = LEGEND_HIDDEN_ALPHA
    if any(artist.get_visible() for artist in target_artists):
        alpha = LEGEND_VISIBLE_ALPHA
    legend_handle.set_alpha(alpha)
    legend_text.set_alpha(alpha)


def _connect_interactive_legend_toggle(
    fig: Figure,
    legend: Legend,
    labeled_artists: dict[str, tuple[Artist, ...]],
) -> Callable[[PickEvent], None]:
    """Attach legend click handlers that toggle plotted artist visibility.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure that owns the legend and receives pick events.
    legend : matplotlib.legend.Legend
        Legend whose handles and texts should be clickable.
    labeled_artists : dict[str, tuple[matplotlib.artist.Artist, ...]]
        Mapping from legend label text to plotted artists that should be
        shown or hidden together.

    Returns
    -------
    collections.abc.Callable[[matplotlib.backend_bases.PickEvent], None]
        The registered pick-event callback.

    Raises
    ------
    ValueError
        If a legend label does not have a matching plotted artist.
    """

    legend_targets: dict[Artist, tuple[Artist, Text, tuple[Artist, ...]]] = {}
    legend_handles = list(legend.legend_handles)
    legend_texts = list(legend.get_texts())

    for legend_handle, legend_text in zip(legend_handles, legend_texts, strict=True):
        label = legend_text.get_text()
        target_artists = labeled_artists.get(label)
        if target_artists is None:
            raise ValueError(f"Legend label '{label}' does not match a plotted artist.")
        legend_handle.set_picker(True)
        legend_text.set_picker(True)
        _sync_legend_entry_visibility(
            legend_handle=legend_handle,
            legend_text=legend_text,
            target_artists=target_artists,
        )
        legend_targets[legend_handle] = (legend_handle, legend_text, target_artists)
        legend_targets[legend_text] = (legend_handle, legend_text, target_artists)

    def on_pick(event: PickEvent) -> None:
        """Toggle the artist associated with a picked legend entry."""

        target = legend_targets.get(event.artist)
        if target is None:
            return
        legend_handle, legend_text, target_artists = target
        currently_visible = any(artist.get_visible() for artist in target_artists)
        new_visible = not currently_visible
        for artist in target_artists:
            artist.set_visible(new_visible)
        _sync_legend_entry_visibility(
            legend_handle=legend_handle,
            legend_text=legend_text,
            target_artists=target_artists,
        )
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", on_pick)
    return on_pick


def _validate_interactive_backend() -> None:
    """Fail clearly when interactive plotting uses a non-GUI backend."""

    backend = matplotlib.get_backend().lower()
    if backend in NON_GUI_BACKENDS:
        raise RuntimeError(
            "Interactive plotting requires a GUI Matplotlib backend, "
            f"but the current backend is '{matplotlib.get_backend()}'. "
            "Ensure DISPLAY/XAUTHORITY are available or disable auto tmux."
        )


def plot_eval_scores(
    species: str,
    output_png: str | None = None,
    interactive: bool = False,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    """Plot sensitivity/precision points from evaluation score text files.

    Parameters
    ----------
    species : str
        Species name used to resolve the evaluation directory and default
        output path.
    output_png : str | None, optional
        Output PNG path. If omitted, a species-specific default path is used.
    interactive : bool, optional
        If ``True``, validate that the active Matplotlib backend supports GUI
        interaction and enable clickable legend toggles.
    x_min : float | None, optional
        Minimum x-axis value in sensitivity units.
    x_max : float | None, optional
        Maximum x-axis value in sensitivity units.
    y_min : float | None, optional
        Minimum y-axis value in precision units.
    y_max : float | None, optional
        Maximum y-axis value in precision units.

    Returns
    -------
    None
        This function writes the plot to disk and optionally shows it.

    Raises
    ------
    FileNotFoundError
        If the evaluation directory does not exist or contains no score files.
    ValueError
        If the interactive backend is invalid, plot bounds are invalid, or
        duplicate legend labels are detected.
    """

    if interactive:
        _validate_interactive_backend()

    eval_dir = resolve_eval_dir(species)
    if not os.path.isdir(eval_dir):
        raise FileNotFoundError(f"eval_score directory not found: {eval_dir}")

    files = [f for f in os.listdir(eval_dir) if f.endswith(".txt")]
    files.sort()
    if not files:
        raise FileNotFoundError(f"No .txt files found under: {eval_dir}")

    fig, ax = plt.subplots(figsize=(16, 12), dpi=100)
    markers = ["o", "s", "^", "D", "v", "p", "*", "h", "x", "+"]
    labeled_artists: dict[str, tuple[Artist, ...]] = {}

    best_model_label: str = ""
    best_model_f1: float = -1.0

    for index, filename in enumerate(files):
        data = np.loadtxt(os.path.join(eval_dir, filename), usecols=(3, 4, 5))
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)
        sensitivity = data[:, 0]
        precision = data[:, 1]
        f1_scores = data[:, 2]
        label = filename[:-4]
        scatter_artist = ax.scatter(
            sensitivity,
            precision,
            s=2,
            marker=markers[index % len(markers)],
            label=label,
        )
        if label in labeled_artists:
            raise ValueError(f"Duplicate legend label detected: {label}")
        labeled_artists[label] = scatter_artist

        max_f1_idx = int(np.argmax(f1_scores))
        model_max_f1 = float(f1_scores[max_f1_idx])
        max_f1_artist = ax.scatter(
            sensitivity[max_f1_idx],
            precision[max_f1_idx],
            s=20,
            color="black",
            zorder=5,
        )
        labeled_artists[label] = (scatter_artist, max_f1_artist)
        if model_max_f1 > best_model_f1:
            best_model_f1 = model_max_f1
            best_model_label = label

    (
        x_min_resolved,
        x_max_resolved,
        y_min_resolved,
        y_max_resolved,
    ) = resolve_plot_bounds(
        species=species,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )
    print(
        "[plot_eval] bounds: "
        f"species={species} "
        f"x=({x_min_resolved}, {x_max_resolved}) "
        f"y=({y_min_resolved}, {y_max_resolved})"
    )

    ax.set_xlabel("Sensitivity", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel("Precision", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_xlim(x_min_resolved, x_max_resolved)
    ax.set_ylim(y_min_resolved, y_max_resolved)
    ax.set_title(
        species,
        fontsize=TITLE_FONT_SIZE,
        loc="left",
    )
    ax.text(
        1.0,
        1.0,
        f"best: {best_model_label}",
        transform=ax.transAxes,
        fontsize=TITLE_FONT_SIZE,
        ha="right",
        va="bottom",
    )
    ax.tick_params(axis="both", labelsize=AXIS_TICK_FONT_SIZE)
    ax.set_aspect("equal")

    # Draw F1-score iso-curves behind scatter points.
    _x = np.linspace(x_min_resolved, x_max_resolved, 400)
    _y = np.linspace(y_min_resolved, y_max_resolved, 400)
    _sn, _pr = np.meshgrid(_x, _y)
    with np.errstate(invalid="ignore", divide="ignore"):
        _f1 = np.where(
            (_sn + _pr) > 0,
            2.0 * _sn * _pr / (_sn + _pr),
            0.0,
        )
    _cs = ax.contour(
        _sn,
        _pr,
        _f1,
        levels=10,
        colors="lightgray",
        linewidths=0.5,
        alpha=0.8,
        zorder=0,
    )
    ax.clabel(_cs, fmt="F1=%.1f", fontsize=9, inline=True)
    fig.subplots_adjust(right=0.78)
    legend = ax.legend(
        markerscale=7,
        fontsize=LEGEND_FONT_SIZE,
        loc=LEGEND_LOCATION,
        bbox_to_anchor=LEGEND_BBOX_ANCHOR,
        borderaxespad=0.0,
    )
    if interactive:
        _connect_interactive_legend_toggle(
            fig=fig,
            legend=legend,
            labeled_artists=labeled_artists,
        )

    final_output = resolve_plot_output(species, output_png)
    out_dir = os.path.dirname(final_output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(final_output, bbox_inches="tight")
    print(f"Saved plot to {final_output}")
    if interactive:
        plt.show(block=True)


def run_eval_command(args: argparse.Namespace) -> None:
    """Run `eval` subcommand."""

    species_for_output = args.species or infer_species_from_path(args.score_file)
    output_lines = evaluate_score_file(
        class_file=args.class_file,
        score_file=args.score_file,
        ref_gff=args.ref_gff,
    )
    output_file = resolve_eval_output_file(
        score_file=args.score_file,
        output_file=args.output_file,
        species=species_for_output,
    )
    with Path(output_file).open("w", encoding="utf-8") as f:
        if output_lines:
            f.write("\n".join(output_lines))
            f.write("\n")
    print(f"Evaluation scores saved to {output_file}")

    if args.visualize != "none":
        plot_eval_scores(
            species=species_for_output,
            output_png=args.output_png,
            interactive=(args.visualize == "interactive"),
            x_min=args.x_min,
            x_max=args.x_max,
            y_min=args.y_min,
            y_max=args.y_max,
        )


def run_plot_command(args: argparse.Namespace) -> None:
    """Run `plot` subcommand."""

    finalized_entries = finalize_ready_published_outputs_for_species(
        project_root=resolve_project_root(),
        species=args.species,
    )
    if finalized_entries:
        kept = ", ".join(
            sorted(entry.published_name for entry in finalized_entries)
        )
        print(f"[plot_eval] finalized published outputs: {kept}")

    plot_eval_scores(
        species=args.species,
        output_png=args.output_png,
        interactive=args.interactive,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build top-level CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate transcript scores and/or visualize sensitivity-precision curves"
        )
    )
    subparsers = parser.add_subparsers(dest="command")

    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluate one score TSV into eval_score txt",
    )
    eval_parser.add_argument("class_file", help="Path to class file")
    eval_parser.add_argument("score_file", help="Path to score file")
    eval_parser.add_argument("ref_gff", help="Path to reference GFF")
    eval_parser.add_argument("--output_file", default=None, help="Path to output txt")
    eval_parser.add_argument(
        "--species",
        default=None,
        help="Species override for default output path",
    )
    eval_parser.add_argument(
        "--visualize",
        choices=["none", "true", "interactive"],
        default="none",
        help=(
            "Plot after evaluation: none (default), true (save png), "
            "interactive (save + show)"
        ),
    )
    eval_parser.add_argument(
        "--output_png",
        default=None,
        help="Output PNG path when --visualize is true/interactive",
    )
    eval_parser.add_argument("--x_min", type=float, default=None)
    eval_parser.add_argument("--x_max", type=float, default=None)
    eval_parser.add_argument("--y_min", type=float, default=None)
    eval_parser.add_argument("--y_max", type=float, default=None)
    eval_parser.set_defaults(func=run_eval_command)

    plot_parser = subparsers.add_parser(
        "plot",
        help="Plot all eval_score txt files for a species",
    )
    plot_parser.add_argument(
        "species",
        nargs="?",
        default="Dmel",
        help="Folder name under data/ (default: Dmel)",
    )
    plot_parser.add_argument(
        "--output_png",
        default=None,
        help="Output PNG path (default: data/{species}/{species}_snpr.png)",
    )
    plot_parser.add_argument(
        "--interactive",
        action="store_true",
        help="Show the plot interactively instead of saving only",
    )
    plot_parser.add_argument("--x_min", type=float, default=None)
    plot_parser.add_argument("--x_max", type=float, default=None)
    plot_parser.add_argument("--y_min", type=float, default=None)
    plot_parser.add_argument("--y_max", type=float, default=None)
    plot_parser.set_defaults(func=run_plot_command)

    return parser


def build_legacy_eval_parser() -> argparse.ArgumentParser:
    """Build parser for backward-compatible no-subcommand mode."""

    parser = argparse.ArgumentParser()
    parser.add_argument("class_file", help="Path to class file")
    parser.add_argument("score_file", help="Path to score file")
    parser.add_argument("ref_gff", help="Path to reference GFF")
    parser.add_argument("--output_file", help="Path to output file", default=None)
    parser.add_argument(
        "--species",
        default=None,
        help="Species override for default output path",
    )
    parser.add_argument(
        "--visualize",
        choices=["none", "true", "interactive"],
        default="none",
        help=(
            "Plot after evaluation: none (default), true (save png), "
            "interactive (save + show)"
        ),
    )
    parser.add_argument(
        "--output_png",
        default=None,
        help="Output PNG path when --visualize is true/interactive",
    )
    parser.add_argument("--x_min", type=float, default=None)
    parser.add_argument("--x_max", type=float, default=None)
    parser.add_argument("--y_min", type=float, default=None)
    parser.add_argument("--y_max", type=float, default=None)
    return parser


def main() -> None:
    """CLI entrypoint."""

    argv = sys.argv[1:]
    if argv and argv[0] in {"eval", "plot"}:
        parser = build_parser()
        args = parser.parse_args(argv)
        args.func(args)
        return

    args = build_legacy_eval_parser().parse_args(argv)
    run_eval_command(args)


if __name__ == "__main__":
    main()
