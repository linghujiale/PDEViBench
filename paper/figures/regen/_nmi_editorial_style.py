"""Shared figure style and export helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

MM_PER_INCH = 25.4
FINAL_WIDTH_MM = 180.0
FINAL_WIDTH_IN = FINAL_WIDTH_MM / MM_PER_INCH

PALETTE = {
    "route_valid": "#496A8D",
    "claim_invalid": "#C8755B",
    "reference_hidden": "#4F9188",
    "neutral": "#C7CDD3",
    "text": "#2F3B46",
    "secondary": "#69747E",
    "grid": "#E7EAED",
    "spine": "#5F6B75",
    "white": "#FFFFFF",
    "codex": "#496A8D",
    "claude": "#D39A58",
}


def apply_editorial_style(*, extended_data: bool = False) -> None:
    minimum = 6.5 if extended_data else 7.0
    mpl.rcParams.update(
        {
            "figure.facecolor": PALETTE["white"],
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.facecolor": PALETTE["white"],
            "savefig.edgecolor": "none",
            "axes.facecolor": PALETTE["white"],
            "axes.edgecolor": PALETTE["spine"],
            "axes.linewidth": 0.5,
            "axes.labelcolor": PALETTE["text"],
            "axes.labelsize": max(7.0, minimum),
            "axes.titlesize": 7.5,
            "axes.titleweight": "normal",
            "axes.titlelocation": "left",
            "axes.titlepad": 5.0,
            "xtick.color": PALETTE["text"],
            "ytick.color": PALETTE["text"],
            "xtick.labelsize": minimum,
            "ytick.labelsize": minimum,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": minimum,
            "text.color": PALETTE["text"],
            "legend.fontsize": minimum,
            "legend.frameon": False,
            "lines.linewidth": 0.9,
            "lines.markersize": 4.0,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.4,
            "grid.alpha": 1.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def figure_size(height_mm: float) -> tuple[float, float]:
    """Return a 180-mm-wide figure size in inches."""
    return FINAL_WIDTH_IN, height_mm / MM_PER_INCH


def clean_axis(ax: plt.Axes, *, grid_axis: str | None = None) -> None:
    """Remove dashboard chrome while retaining useful quantitative guides."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, zorder=0)
        ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str, *, x: float = -0.09, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        fontweight="bold",
        color=PALETTE["text"],
        clip_on=False,
    )


def finding_title(fig: plt.Figure, text: str) -> None:
    fig.suptitle(
        text,
        x=0.0,
        y=0.985,
        ha="left",
        va="top",
        fontsize=10.0,
        fontweight="semibold",
        color=PALETTE["text"],
    )


def save_figure(
    fig: plt.Figure,
    output_stem: Path,
    *,
    png_dpi: int = 300,
    tiff: bool = False,
    close: bool = True,
) -> list[Path]:
    """Export an exact 180-mm canvas as editable PDF/SVG plus PNG QA output."""
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [output_stem.with_suffix(suffix) for suffix in (".pdf", ".svg", ".png")]
    fig.savefig(outputs[0])
    fig.savefig(outputs[1])
    outputs[1].write_text(
        "\n".join(line.rstrip() for line in outputs[1].read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(outputs[2], dpi=png_dpi)
    if tiff:
        tiff_path = output_stem.with_suffix(".tiff")
        fig.savefig(
            tiff_path,
            dpi=300,
            pil_kwargs={"compression": "tiff_lzw"},
        )
        outputs.append(tiff_path)
    if close:
        plt.close(fig)
    return outputs


def power_text(ax, x, y, base: str, exponent: str, *, transform=None, ha: str = "center", va: str = "bottom", fontsize: float = 7.0, color=None, fontweight: str = "normal", zorder: float = 5):
    """Draw ``base`` with a raised ``exponent`` using full-size body-font glyphs.

    Mathtext superscripts shrink to 70% of the body size and fall back to
    DejaVu; this helper keeps every glyph in the figure font at >= fontsize.
    """
    fig = ax.figure
    transform = transform or ax.transData
    color = color or PALETTE["text"]
    renderer = fig.canvas.get_renderer()
    probe_base = ax.text(0, 0, base, fontsize=fontsize, fontweight=fontweight, transform=transform, alpha=0)
    probe_exp = ax.text(0, 0, exponent, fontsize=fontsize, fontweight=fontweight, transform=transform, alpha=0)
    width_base = probe_base.get_window_extent(renderer).width * 72 / fig.dpi
    width_exp = probe_exp.get_window_extent(renderer).width * 72 / fig.dpi
    probe_base.remove()
    probe_exp.remove()
    total = width_base + width_exp - 0.2
    if ha == "center":
        start = -total / 2
    elif ha == "right":
        start = -total
    else:
        start = 0.0
    raise_pts = 0.40 * fontsize
    base_text = ax.annotate(base, xy=(x, y), xycoords=transform, xytext=(start, 0), textcoords="offset points", ha="left", va=va, fontsize=fontsize, color=color, fontweight=fontweight, zorder=zorder, annotation_clip=False)
    exp_text = ax.annotate(exponent, xy=(x, y), xycoords=transform, xytext=(start + width_base - 0.2, raise_pts), textcoords="offset points", ha="left", va=va, fontsize=fontsize, color=color, fontweight=fontweight, zorder=zorder, annotation_clip=False)
    return base_text, exp_text


def log_tick_labels(ax, axis: str, exponents, *, fontsize: float = 7.0, pad_pts: float = 3.0, color=None):
    """Replace mathtext log tick labels with full-size '10^k' labels."""
    import matplotlib.transforms as mtransforms

    color = color or PALETTE["text"]
    if axis == "x":
        ax.set_xticks([10.0**k for k in exponents])
        ax.set_xticklabels([""] * len(exponents))
        transform = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        tick_len = ax.xaxis.get_major_ticks()[0].get_tick_padding() if ax.xaxis.get_major_ticks() else 3.0
        for k in exponents:
            base_text, exp_text = power_text(ax, 10.0**k, 0.0, "10", f"{k}".replace("-", "\u2212"), transform=transform, ha="center", va="top", fontsize=fontsize, color=color)
            for t in (base_text, exp_text):
                offset = t.get_position()
                t.set_position((offset[0], offset[1] - tick_len - pad_pts))
    else:
        ax.set_yticks([10.0**k for k in exponents])
        ax.set_yticklabels([""] * len(exponents))
        transform = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        tick_len = ax.yaxis.get_major_ticks()[0].get_tick_padding() if ax.yaxis.get_major_ticks() else 3.0
        for k in exponents:
            base_text, exp_text = power_text(ax, 0.0, 10.0**k, "10", f"{k}".replace("-", "\u2212"), transform=transform, ha="right", va="center", fontsize=fontsize, color=color)
            for t in (base_text, exp_text):
                offset = t.get_position()
                t.set_position((offset[0] - tick_len - pad_pts, offset[1]))
