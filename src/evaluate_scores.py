import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

PLOT_BOUNDS_BY_SPECIES: dict[str, tuple[float, float, float, float]] = {
    "Athal": (10.0, 39.0, 48.0, 80.0),
    "Dmel": (40.0, 48.0, 40.0, 50.0),
    "Mmus": (5.0, 16.0, 35.0, 45.0),
}
FALLBACK_PLOT_BOUNDS: tuple[float, float, float, float] = (40.0, 50.0, 40.0, 50.0)

LEGEND_FONT_SIZE = 14
AXIS_TICK_FONT_SIZE = 16
AXIS_LABEL_FONT_SIZE = 18
TITLE_FONT_SIZE = 20


def load_class_dict(class_file: str) -> Dict[str, str]:
    class_dict: Dict[str, str] = {}
    with open(class_file, "r") as f:
        for line in f:
            tr, cl = line.strip().split()
            class_dict[tr] = cl
    return class_dict


def evaluate_score_file(
    class_file: str,
    score_file: str,
    good: int,
    total: int,
    ref: int,
) -> List[str]:
    class_dict = load_class_dict(class_file)

    filtered_data: List[Tuple[str, float, str]] = []
    with open(score_file, "r") as f:
        next(f, None)
        for line in f:
            fields = line.strip().split()
            if len(fields) < 5:
                continue
            tr = fields[0]
            score = float(fields[4])
            last_field = float(fields[-1])
            if tr in class_dict and class_dict[tr] != "c" and last_field != 10000:
                filtered_data.append((tr, score, class_dict[tr]))

    filtered_data.sort(key=lambda x: x[1])

    output_lines: List[str] = []
    current_good = good
    current_total = total
    for tr, score, cl in filtered_data:
        if cl == "=":
            current_good -= 1
        current_total -= 1
        if current_total == 0:
            current_total = 1
        sn = int(current_good / ref * 10000) / 100
        pr = int(current_good / current_total * 10000) / 100
        f1 = 0.0 if (sn + pr) == 0.0 else 2.0 * (sn * pr) / (sn + pr)
        output_lines.append(f"{tr} {score} {cl} {sn} {pr} {f1}")

    return output_lines


def infer_species_from_path(path: str, default_species: str = "Dmel") -> str:
    parts = os.path.normpath(path).split(os.sep)
    if "data" in parts:
        idx = parts.index("data")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return default_species


def resolve_eval_output_file(
    score_file: str, output_file: Optional[str], species: Optional[str] = None
) -> str:
    if output_file not in (None, "", "None"):
        return output_file

    inferred_species = species or infer_species_from_path(score_file)
    base = os.path.splitext(os.path.basename(score_file))[0]
    out_dir = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__), "..", "data", inferred_species, "eval_score"
        )
    )
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{base}.txt")


def resolve_eval_dir(species: str) -> str:
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "data", species, "eval_score")
    )


def resolve_plot_output(species: str, output_png: Optional[str]) -> str:
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
    x_min: Optional[float],
    x_max: Optional[float],
    y_min: Optional[float],
    y_max: Optional[float],
) -> tuple[float, float, float, float]:
    default_bounds = PLOT_BOUNDS_BY_SPECIES.get(species)
    if default_bounds is None:
        missing_all_bounds = (
            x_min is None
            and x_max is None
            and y_min is None
            and y_max is None
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


def plot_eval_scores(
    species: str,
    output_png: Optional[str] = None,
    interactive: bool = False,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
):
    eval_dir = resolve_eval_dir(species)
    if not os.path.isdir(eval_dir):
        raise FileNotFoundError(f"eval_score directory not found: {eval_dir}")

    files = [f for f in os.listdir(eval_dir) if f.endswith(".txt")]
    files.sort()
    if not files:
        raise FileNotFoundError(f"No .txt files found under: {eval_dir}")

    fig, ax = plt.subplots(figsize=(16, 12), dpi=100)
    markers = ["o", "s", "^", "D", "v", "p", "*", "h", "x", "+"]

    for i, file in enumerate(files):
        data = np.loadtxt(os.path.join(eval_dir, file), usecols=(3, 4))
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)
        sensitivity = data[:, 0]
        precision = data[:, 1]
        ax.scatter(
            sensitivity,
            precision,
            s=2,
            marker=markers[i % len(markers)],
            label=file[:-4],
        )

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
    ax.set_title(species, fontsize=TITLE_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=AXIS_TICK_FONT_SIZE)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(markerscale=7, fontsize=LEGEND_FONT_SIZE, loc="lower left")

    final_output = resolve_plot_output(species, output_png)
    out_dir = os.path.dirname(final_output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(final_output)
    print(f"Saved plot to {final_output}")
    if interactive:
        plt.show()


def run_eval_command(args):
    species_for_output = args.species or infer_species_from_path(args.score_file)
    output_lines = evaluate_score_file(
        class_file=args.class_file,
        score_file=args.score_file,
        good=args.good,
        total=args.total,
        ref=args.ref,
    )
    output_file = resolve_eval_output_file(
        score_file=args.score_file,
        output_file=args.output_file,
        species=species_for_output,
    )
    with open(output_file, "w") as f:
        f.write("\n".join(output_lines) + "\n")
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


def run_plot_command(args):
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
    parser = argparse.ArgumentParser(
        description="Evaluate transcript scores and/or visualize sensitivity-precision curves"
    )
    subparsers = parser.add_subparsers(dest="command")

    eval_parser = subparsers.add_parser(
        "eval", help="Evaluate one score TSV into eval_score txt"
    )
    eval_parser.add_argument("class_file", help="Path to class file")
    eval_parser.add_argument("score_file", help="Path to score file")
    eval_parser.add_argument("--output_file", default=None, help="Path to output txt")
    eval_parser.add_argument(
        "--species", default=None, help="Species override for default output path"
    )
    eval_parser.add_argument("--good", type=int, default=15169)
    eval_parser.add_argument("--total", type=int, default=38235)
    eval_parser.add_argument("--ref", type=int, default=32288)
    eval_parser.add_argument(
        "--visualize",
        choices=["none", "true", "interactive"],
        default="none",
        help="Plot after evaluation: none (default), true (save png), interactive (save + show)",
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
        "plot", help="Plot all eval_score txt files for a species"
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
    parser = argparse.ArgumentParser()
    parser.add_argument("class_file", help="Path to class file")
    parser.add_argument("score_file", help="Path to score file")
    parser.add_argument("--output_file", help="Path to output file", default=None)
    parser.add_argument(
        "--species", default=None, help="Species override for default output path"
    )
    parser.add_argument("--good", type=int, default=15169)
    parser.add_argument("--total", type=int, default=38235)
    parser.add_argument("--ref", type=int, default=32288)
    parser.add_argument(
        "--visualize",
        choices=["none", "true", "interactive"],
        default="none",
        help="Plot after evaluation: none (default), true (save png), interactive (save + show)",
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


def main():
    argv = sys.argv[1:]
    if argv and argv[0] in {"eval", "plot"}:
        parser = build_parser()
        args = parser.parse_args(argv)
        args.func(args)
        return

    # Backward-compatible mode: treat as `eval`
    args = build_legacy_eval_parser().parse_args(argv)
    run_eval_command(args)


if __name__ == "__main__":
    main()
