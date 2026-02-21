"""Generate SVG architecture figures for project documentation.

This script intentionally uses only the Python standard library so figure
creation works in minimal development environments.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
import math
from pathlib import Path
from typing import Sequence

CANVAS_WIDTH: int = 1280
CANVAS_HEIGHT: int = 360
MARGIN_X: int = 36
MARGIN_TOP: int = 72
MARGIN_BOTTOM: int = 24


@dataclass(frozen=True)
class BoxSpec:
    """Rounded-rectangle box in normalized coordinates."""

    x: float
    y: float
    width: float
    height: float
    label: str
    fill: str


@dataclass(frozen=True)
class ArrowSpec:
    """Arrow in normalized coordinates with an optional label."""

    start: tuple[float, float]
    end: tuple[float, float]
    label: str | None = None


def _validate_unit_interval(value: float, name: str) -> None:
    """Validate that a coordinate-like value is in [0, 1]."""
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got: {value}")


def _plot_width() -> float:
    return float(CANVAS_WIDTH - 2 * MARGIN_X)


def _plot_height() -> float:
    return float(CANVAS_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM)


def _x(value: float) -> float:
    _validate_unit_interval(value, "x")
    return float(MARGIN_X) + (value * _plot_width())


def _y(value: float) -> float:
    _validate_unit_interval(value, "y")
    return float(MARGIN_TOP) + (value * _plot_height())


def _svg_header(title: str) -> str:
    safe_title = escape(title)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{CANVAS_WIDTH}" height="{CANVAS_HEIGHT}" '
        f'viewBox="0 0 {CANVAS_WIDTH} {CANVAS_HEIGHT}">'
        "<defs>"
        '<marker id="arrowhead" markerWidth="10" markerHeight="7" '
        'refX="9" refY="3.5" orient="auto">'
        '<polygon points="0 0, 10 3.5, 0 7" fill="#253746" />'
        "</marker>"
        "</defs>"
        '<rect width="100%" height="100%" fill="#ffffff" />'
        f'<text x="{CANVAS_WIDTH / 2:.1f}" y="38" '
        'font-family="Arial, sans-serif" font-size="24" '
        'font-weight="700" text-anchor="middle" fill="#1f2a33">'
        f"{safe_title}</text>"
    )


def _svg_footer() -> str:
    return "</svg>"


def _box_to_svg(spec: BoxSpec) -> str:
    """Convert a box spec into SVG rectangle and text elements."""
    for value_name, value in (
        ("x", spec.x),
        ("y", spec.y),
        ("width", spec.width),
        ("height", spec.height),
    ):
        _validate_unit_interval(value, f"box.{value_name}")

    left = _x(spec.x)
    top = _y(spec.y)
    width_px = spec.width * _plot_width()
    height_px = spec.height * _plot_height()
    center_x = left + (width_px / 2.0)
    center_y = top + (height_px / 2.0)

    rect = (
        f'<rect x="{left:.1f}" y="{top:.1f}" '
        f'width="{width_px:.1f}" height="{height_px:.1f}" '
        'rx="16" ry="16" '
        f'fill="{escape(spec.fill)}" stroke="#253746" stroke-width="2" />'
    )

    lines = [escape(part) for part in spec.label.split("\n")]
    line_height = 16.0
    top_line_y = center_y - ((len(lines) - 1) * line_height / 2.0)
    text_parts: list[str] = [
        (
            f'<text x="{center_x:.1f}" y="{top_line_y:.1f}" '
            'font-family="Arial, sans-serif" font-size="13" '
            'text-anchor="middle" fill="#111111">'
        )
    ]
    for index, line in enumerate(lines):
        dy = "0" if index == 0 else f"{line_height:.1f}"
        text_parts.append(f'<tspan x="{center_x:.1f}" dy="{dy}">{line}</tspan>')
    text_parts.append("</text>")
    return rect + "".join(text_parts)


def _arrow_to_svg(spec: ArrowSpec) -> str:
    """Convert an arrow spec to SVG line and optional label elements."""
    x1 = _x(spec.start[0])
    y1 = _y(spec.start[1])
    x2 = _x(spec.end[0])
    y2 = _y(spec.end[1])

    base = (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        'stroke="#253746" stroke-width="2" marker-end="url(#arrowhead)" />'
    )
    if spec.label is None or spec.label.strip() == "":
        return base

    label_x = (x1 + x2) / 2.0
    label_y = (y1 + y2) / 2.0 - 8.0
    safe_label = escape(spec.label)
    label = (
        f'<text x="{label_x:.1f}" y="{label_y:.1f}" '
        'font-family="Arial, sans-serif" font-size="12" '
        'text-anchor="middle" fill="#253746">'
        f"{safe_label}</text>"
    )
    return base + label


def _write_diagram(
    output_path: Path,
    title: str,
    boxes: Sequence[BoxSpec],
    arrows: Sequence[ArrowSpec],
) -> None:
    """Write one complete SVG diagram."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = [_svg_header(title=title)]
    parts.extend(_box_to_svg(spec) for spec in boxes)
    parts.extend(_arrow_to_svg(spec) for spec in arrows)
    parts.append(_svg_footer())
    output_path.write_text("".join(parts), encoding="utf-8")


def _polyline(points: Sequence[tuple[float, float]], stroke: str) -> str:
    """Build a polyline SVG element from absolute canvas coordinates."""
    encoded_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (
        f'<polyline points="{encoded_points}" fill="none" '
        f'stroke="{stroke}" stroke-width="2.4" />'
    )


def _line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str = "#253746",
    width: float = 1.8,
    dashed: bool = False,
) -> str:
    """Build a line SVG element."""
    dash_part = ' stroke-dasharray="6 4"' if dashed else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width:.1f}"{dash_part} />'
    )


def _text(
    x: float,
    y: float,
    label: str,
    *,
    size: int = 12,
    anchor: str = "start",
    color: str = "#1f2a33",
) -> str:
    """Build a text SVG element."""
    safe_label = escape(label)
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}" fill="{color}">'
        f"{safe_label}</text>"
    )


def _render_pipeline_overview(output_dir: Path) -> None:
    boxes = [
        BoxSpec(0.02, 0.28, 0.15, 0.46, "Train\n(donor/acceptor)", "#cfe8ff"),
        BoxSpec(0.21, 0.28, 0.15, 0.46, "Site Infer\n(logit -> score)", "#d8f7d2"),
        BoxSpec(0.40, 0.28, 0.16, 0.46, "Intron Merge\n(donor+acceptor)", "#ffe8b5"),
        BoxSpec(0.60, 0.28, 0.16, 0.46, "Transcript Agg\n(min/softmin/...)", "#ffd8e8"),
        BoxSpec(0.80, 0.28, 0.18, 0.46, "SN / PR / F1\nEvaluation", "#ede2ff"),
    ]
    arrows = [
        ArrowSpec((0.17, 0.51), (0.21, 0.51), "checkpoint"),
        ArrowSpec((0.36, 0.51), (0.40, 0.51), "site_score.tsv"),
        ArrowSpec((0.56, 0.51), (0.60, 0.51), "intron score"),
        ArrowSpec((0.76, 0.51), (0.80, 0.51), "trans_score.tsv"),
    ]
    _write_diagram(
        output_path=output_dir / "pipeline_overview.svg",
        title="IntronModel Unified Pipeline",
        boxes=boxes,
        arrows=arrows,
    )


def _render_cnn_layers(output_dir: Path) -> None:
    boxes = [
        BoxSpec(0.02, 0.28, 0.12, 0.46, "Input\nB x 4 x L", "#d6ecff"),
        BoxSpec(0.17, 0.28, 0.15, 0.46, "Conv-BN-ReLU\nPool-2 + Drop", "#d7f9e1"),
        BoxSpec(0.35, 0.28, 0.15, 0.46, "Conv-BN-ReLU\nPool-2 + Drop", "#d7f9e1"),
        BoxSpec(0.53, 0.28, 0.15, 0.46, "Conv-BN-ReLU\nPool-2 + Drop", "#d7f9e1"),
        BoxSpec(0.71, 0.28, 0.10, 0.46, "GAP\nB x C", "#fff3c7"),
        BoxSpec(0.84, 0.28, 0.14, 0.46, "MLP\nC -> h -> 1", "#ffe0cc"),
    ]
    arrows = [
        ArrowSpec((0.14, 0.51), (0.17, 0.51), "x3 blocks"),
        ArrowSpec((0.32, 0.51), (0.35, 0.51)),
        ArrowSpec((0.50, 0.51), (0.53, 0.51)),
        ArrowSpec((0.68, 0.51), (0.71, 0.51)),
        ArrowSpec((0.81, 0.51), (0.84, 0.51), "logit"),
    ]
    _write_diagram(
        output_path=output_dir / "cnn_layers.svg",
        title="Baseline CNN (src/models/cnn.py)",
        boxes=boxes,
        arrows=arrows,
    )


def _render_cnn_resdil_layers(output_dir: Path) -> None:
    boxes = [
        BoxSpec(0.02, 0.28, 0.13, 0.46, "Input\nB x 4 x L", "#d6ecff"),
        BoxSpec(0.19, 0.28, 0.14, 0.46, "Stem\nConv+BN+ReLU", "#d7f9e1"),
        BoxSpec(0.37, 0.28, 0.22, 0.46, "ResDil Block\nd in {1,2,4,8}", "#fff3c7"),
        BoxSpec(0.63, 0.28, 0.10, 0.46, "Pool-2", "#ffe0cc"),
        BoxSpec(0.77, 0.28, 0.21, 0.46, "GAP + MLP\nC -> h -> 1", "#ede2ff"),
    ]
    arrows = [
        ArrowSpec((0.15, 0.51), (0.19, 0.51)),
        ArrowSpec((0.33, 0.51), (0.37, 0.51), "repeat over channels"),
        ArrowSpec((0.59, 0.51), (0.63, 0.51)),
        ArrowSpec((0.73, 0.51), (0.77, 0.51), "logit"),
        ArrowSpec((0.39, 0.37), (0.57, 0.37), "conv1d(d) x2"),
        ArrowSpec((0.39, 0.66), (0.57, 0.66), "skip + add"),
    ]
    _write_diagram(
        output_path=output_dir / "cnn_resdil_layers.svg",
        title="Residual Dilated CNN (src/models/cnn_resdil.py)",
        boxes=boxes,
        arrows=arrows,
    )


def _render_tcn_layers(output_dir: Path) -> None:
    boxes = [
        BoxSpec(0.02, 0.28, 0.13, 0.46, "Input\nB x 4 x L", "#d6ecff"),
        BoxSpec(0.19, 0.28, 0.14, 0.46, "Stem\nConv+BN+ReLU", "#d7f9e1"),
        BoxSpec(
            0.37,
            0.28,
            0.22,
            0.46,
            "TCN Block Stack\ndilation: 1,2,4,...",
            "#fff3c7",
        ),
        BoxSpec(0.63, 0.28, 0.10, 0.46, "Residual\nadd", "#ffe0cc"),
        BoxSpec(0.77, 0.28, 0.21, 0.46, "GAP + MLP\nC -> h -> 1", "#ede2ff"),
    ]
    arrows = [
        ArrowSpec((0.15, 0.51), (0.19, 0.51)),
        ArrowSpec((0.33, 0.51), (0.37, 0.51), "repeat by channels"),
        ArrowSpec((0.59, 0.51), (0.63, 0.51)),
        ArrowSpec((0.73, 0.51), (0.77, 0.51), "logit"),
        ArrowSpec((0.39, 0.36), (0.57, 0.36), "optional causal chomp"),
        ArrowSpec((0.39, 0.66), (0.57, 0.66), "conv1d(d) x2 + skip"),
    ]
    _write_diagram(
        output_path=output_dir / "tcn_layers.svg",
        title="Temporal CNN (src/models/tcn.py)",
        boxes=boxes,
        arrows=arrows,
    )


def _render_reservoir_layers(output_dir: Path) -> None:
    boxes = [
        BoxSpec(0.02, 0.28, 0.16, 0.46, "Tokens\n(onehot or k-mer)", "#d6ecff"),
        BoxSpec(
            0.22,
            0.28,
            0.20,
            0.46,
            "Fixed projection\n(token_proj, W_in)",
            "#d7f9e1",
        ),
        BoxSpec(
            0.46,
            0.28,
            0.20,
            0.46,
            "Fixed reservoir\nh_t = (1-leak)h_{t-1}\n+ leak*tanh(...)",
            "#fff3c7",
        ),
        BoxSpec(
            0.70,
            0.28,
            0.13,
            0.46,
            "Pooling\nmean/max/attn",
            "#ffe0cc",
        ),
        BoxSpec(
            0.86,
            0.28,
            0.12,
            0.46,
            "Readout\nMLP or sum",
            "#ede2ff",
        ),
    ]
    arrows = [
        ArrowSpec((0.18, 0.51), (0.22, 0.51)),
        ArrowSpec((0.42, 0.51), (0.46, 0.51)),
        ArrowSpec((0.66, 0.51), (0.70, 0.51), "washout / preroll"),
        ArrowSpec((0.83, 0.51), (0.86, 0.51), "logit"),
        ArrowSpec((0.48, 0.36), (0.64, 0.36), "spectral_radius scaling"),
    ]
    _write_diagram(
        output_path=output_dir / "reservoir_layers.svg",
        title="Reservoir Readout (src/models/reservoir.py)",
        boxes=boxes,
        arrows=arrows,
    )


def _render_bert_layers(output_dir: Path) -> None:
    boxes = [
        BoxSpec(0.02, 0.28, 0.16, 0.46, "k-mer tokens\n+ attention mask", "#d6ecff"),
        BoxSpec(0.22, 0.28, 0.16, 0.46, "Token + Pos\nEmbedding", "#d7f9e1"),
        BoxSpec(
            0.42,
            0.28,
            0.22,
            0.46,
            "Transformer Encoder\n(n_layers, n_heads)",
            "#fff3c7",
        ),
        BoxSpec(0.68, 0.28, 0.10, 0.46, "[CLS]\nstate", "#ffe0cc"),
        BoxSpec(0.82, 0.28, 0.16, 0.46, "Dropout + Linear\n-> logit", "#ede2ff"),
    ]
    arrows = [
        ArrowSpec((0.18, 0.51), (0.22, 0.51)),
        ArrowSpec((0.38, 0.51), (0.42, 0.51)),
        ArrowSpec((0.64, 0.51), (0.68, 0.51)),
        ArrowSpec((0.78, 0.51), (0.82, 0.51), "binary"),
    ]
    _write_diagram(
        output_path=output_dir / "bert_layers.svg",
        title="Small BERT (src/models/bert.py)",
        boxes=boxes,
        arrows=arrows,
    )


def _render_dnabert_layers(output_dir: Path) -> None:
    boxes = [
        BoxSpec(0.03, 0.28, 0.18, 0.46, "Tokenizer output\ninput_ids, mask", "#d6ecff"),
        BoxSpec(
            0.25,
            0.28,
            0.33,
            0.46,
            "Pretrained DNABERT\nAutoModel backbone",
            "#d7f9e1",
        ),
        BoxSpec(0.62, 0.28, 0.11, 0.46, "[CLS]\nhidden", "#fff3c7"),
        BoxSpec(0.77, 0.28, 0.20, 0.46, "Dropout + Linear\n-> logit", "#ede2ff"),
    ]
    arrows = [
        ArrowSpec((0.21, 0.51), (0.25, 0.51)),
        ArrowSpec((0.58, 0.51), (0.62, 0.51), "last_hidden_state"),
        ArrowSpec((0.73, 0.51), (0.77, 0.51), "binary"),
    ]
    _write_diagram(
        output_path=output_dir / "dnabert_layers.svg",
        title="DNABERT Classifier (src/models/dnabert.py)",
        boxes=boxes,
        arrows=arrows,
    )


def _render_training_optimization(output_dir: Path) -> None:
    """Render one-epoch optimization flow used by all trainable models."""
    boxes = [
        BoxSpec(0.02, 0.28, 0.14, 0.46, "Batch\nx, y", "#d6ecff"),
        BoxSpec(0.20, 0.28, 0.16, 0.46, "Forward\nlogits = f_theta(x)", "#d7f9e1"),
        BoxSpec(0.40, 0.28, 0.15, 0.46, "Loss\nL(logits, y)", "#fff3c7"),
        BoxSpec(0.59, 0.28, 0.14, 0.46, "Backward\nnabla_theta L", "#ffe0cc"),
        BoxSpec(0.77, 0.28, 0.21, 0.46, "AdamW step +\nCosine LR update", "#ede2ff"),
    ]
    arrows = [
        ArrowSpec((0.16, 0.51), (0.20, 0.51)),
        ArrowSpec((0.36, 0.51), (0.40, 0.51), "autocast on CUDA"),
        ArrowSpec((0.55, 0.51), (0.59, 0.51)),
        ArrowSpec((0.73, 0.51), (0.77, 0.51), "clip grad, scaler(fp16)"),
    ]
    _write_diagram(
        output_path=output_dir / "training_optimization.svg",
        title="Training Optimization Flow (per task)",
        boxes=boxes,
        arrows=arrows,
    )


def _render_eval_sweep(output_dir: Path) -> None:
    """Render evaluation-threshold sweep logic."""
    boxes = [
        BoxSpec(
            0.02,
            0.28,
            0.16,
            0.46,
            "Load\nclass_file + score_file",
            "#d6ecff",
        ),
        BoxSpec(
            0.22,
            0.28,
            0.16,
            0.46,
            "Filter\nclass != c and\nlast_field != 10000",
            "#d7f9e1",
        ),
        BoxSpec(0.42, 0.28, 0.14, 0.46, "Sort by\nscore asc", "#fff3c7"),
        BoxSpec(
            0.60,
            0.28,
            0.16,
            0.46,
            "Sweep\nremove current row",
            "#ffe0cc",
        ),
        BoxSpec(
            0.80,
            0.28,
            0.18,
            0.46,
            "Compute\nSN, PR, F1\n(truncated %)",
            "#ede2ff",
        ),
    ]
    arrows = [
        ArrowSpec((0.18, 0.51), (0.22, 0.51)),
        ArrowSpec((0.38, 0.51), (0.42, 0.51)),
        ArrowSpec((0.56, 0.51), (0.60, 0.51)),
        ArrowSpec((0.76, 0.51), (0.80, 0.51)),
    ]
    _write_diagram(
        output_path=output_dir / "eval_sweep.svg",
        title="Evaluation Sweep (src/evaluate_scores.py)",
        boxes=boxes,
        arrows=arrows,
    )


def _render_loss_curves(output_dir: Path) -> None:
    """Render conceptual loss-function curves used in binary classification."""
    title = "Loss Curves (conceptual)"
    parts: list[str] = [_svg_header(title=title)]

    left_x0 = _x(0.06)
    left_x1 = _x(0.46)
    right_x0 = _x(0.54)
    right_x1 = _x(0.94)
    top = _y(0.18)
    bottom = _y(0.84)

    def panel_point(
        x0: float,
        x1: float,
        x_ratio: float,
        y_ratio: float,
    ) -> tuple[float, float]:
        px = x0 + (x1 - x0) * x_ratio
        py = bottom - (bottom - top) * y_ratio
        return px, py

    # Axes and titles.
    for x0, x1 in ((left_x0, left_x1), (right_x0, right_x1)):
        parts.append(
            f'<rect x="{x0:.1f}" y="{top:.1f}" '
            f'width="{(x1 - x0):.1f}" height="{(bottom - top):.1f}" '
            'fill="#fafbfd" stroke="#d0d7de" stroke-width="1.5" />'
        )
        parts.append(_line(x0, bottom, x1, bottom, color="#495057"))
        parts.append(_line(x0, top, x0, bottom, color="#495057"))
        parts.append(_line(x0, (top + bottom) / 2.0, x1, (top + bottom) / 2.0))

    parts.append(_text(left_x0, top - 12.0, "Positive sample: y=1", size=12))
    parts.append(_text(right_x0, top - 12.0, "Negative sample: y=0", size=12))
    parts.append(_text(left_x1, bottom + 18.0, "p = sigmoid(logit)", anchor="end"))
    parts.append(
        _text(
            right_x1,
            bottom + 18.0,
            "p = sigmoid(logit)",
            anchor="end",
        )
    )

    parts.append(_text(left_x0 - 8.0, top + 10.0, "loss", anchor="end"))
    parts.append(_text(right_x0 - 8.0, top + 10.0, "loss", anchor="end"))

    # Curves are clipped to ymax for readability.
    ymax = 5.0
    bce_pos_points: list[tuple[float, float]] = []
    focal_pos_points: list[tuple[float, float]] = []
    bce_neg_points: list[tuple[float, float]] = []
    asym_neg_points: list[tuple[float, float]] = []

    for idx in range(240):
        x_ratio = idx / 239.0
        p = 1e-3 + (1.0 - 2e-3) * x_ratio

        bce_pos = -math.log(p)
        focal_pos = ((1.0 - p) ** 2.0) * bce_pos
        bce_neg = -math.log(1.0 - p)
        asym_neg = (p**4.0) * bce_neg

        bce_pos_norm = min(1.0, bce_pos / ymax)
        focal_pos_norm = min(1.0, focal_pos / ymax)
        bce_neg_norm = min(1.0, bce_neg / ymax)
        asym_neg_norm = min(1.0, asym_neg / ymax)

        bce_pos_points.append(
            panel_point(left_x0, left_x1, x_ratio=x_ratio, y_ratio=bce_pos_norm)
        )
        focal_pos_points.append(
            panel_point(left_x0, left_x1, x_ratio=x_ratio, y_ratio=focal_pos_norm)
        )
        bce_neg_points.append(
            panel_point(right_x0, right_x1, x_ratio=x_ratio, y_ratio=bce_neg_norm)
        )
        asym_neg_points.append(
            panel_point(right_x0, right_x1, x_ratio=x_ratio, y_ratio=asym_neg_norm)
        )

    parts.append(_polyline(bce_pos_points, stroke="#1f77b4"))
    parts.append(_polyline(focal_pos_points, stroke="#ff7f0e"))
    parts.append(_polyline(bce_neg_points, stroke="#1f77b4"))
    parts.append(_polyline(asym_neg_points, stroke="#2ca02c"))

    # Legends.
    legend_y = _y(0.90)
    parts.append(_line(_x(0.08), legend_y, _x(0.11), legend_y, color="#1f77b4"))
    parts.append(_text(_x(0.115), legend_y + 4.0, "BCE"))
    parts.append(_line(_x(0.18), legend_y, _x(0.21), legend_y, color="#ff7f0e"))
    parts.append(_text(_x(0.215), legend_y + 4.0, "Focal (gamma=2)"))
    parts.append(_line(_x(0.36), legend_y, _x(0.39), legend_y, color="#2ca02c"))
    parts.append(_text(_x(0.395), legend_y + 4.0, "Asymmetric neg (gamma_neg=4)"))

    parts.append(_svg_footer())
    output_path = output_dir / "loss_curves.svg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(parts), encoding="utf-8")


def generate_figures(output_dir: Path) -> None:
    """Generate all SVG figures required by model architecture docs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _render_pipeline_overview(output_dir)
    _render_cnn_layers(output_dir)
    _render_cnn_resdil_layers(output_dir)
    _render_tcn_layers(output_dir)
    _render_reservoir_layers(output_dir)
    _render_bert_layers(output_dir)
    _render_dnabert_layers(output_dir)
    _render_training_optimization(output_dir)
    _render_eval_sweep(output_dir)
    _render_loss_curves(output_dir)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate SVG figures used by docs/model-architecture.md.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Target directory. Default: <repo>/docs/_static/figures",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entrypoint."""
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else repo_root / "docs" / "_static" / "figures"
    )
    generate_figures(output_dir=output_dir)
    print(f"Generated SVG figures under: {output_dir}")


if __name__ == "__main__":
    main()
