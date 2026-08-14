"""Shared publication style for all SN-RCPS figures."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt

# A colorblind-safe, print-friendly palette. The same method receives the same
# color and marker in every experiment.
INK = "#202124"
MUTED = "#70757A"
GRID = "#E5E7EB"
PANEL = "#FAFAFB"
TARGET = "#EAF5EF"
TARGET_LINE = "#3E6B56"
REGIME = ["#F7F8FA", "#F2F6FA", "#F8F4F0", "#F2F7F4"]

METHOD_STYLE: dict[str, dict[str, object]] = {
    "Split conformal": {"color": "#7A7D81", "marker": "o", "linestyle": "-"},
    "i.i.d. RCPS": {"color": "#4C78A8", "marker": "o", "linestyle": "-"},
    "SN-RCPS": {"color": "#009E73", "marker": "s", "linestyle": "-"},
    "SN-RCPS-H": {"color": "#2A7AB0", "marker": "^", "linestyle": "-"},
    "SN-RCPS-M": {"color": "#009E73", "marker": "D", "linestyle": "-"},
    "Episodic SN-RCPS": {"color": "#009E73", "marker": "D", "linestyle": "-"},
    "Bonferroni SN-RCPS": {"color": "#8B6AA8", "marker": "P", "linestyle": "-"},
    "Rolling conformal": {"color": "#E69F00", "marker": "X", "linestyle": "-"},
    "ACI": {"color": "#CC79A7", "marker": "v", "linestyle": "-"},
    "Oracle library member": {"color": "#202124", "marker": "*", "linestyle": "--"},
    "Deployment oracle": {"color": "#202124", "marker": "*", "linestyle": "--"},
    "Exact blocked RCPS": {"color": "#D55E00", "marker": "^", "linestyle": "-"},
    "Static SN-RCPS": {"color": "#6F7378", "marker": "o", "linestyle": ":"},
    "Scheduled SN-RCPS": {"color": "#2A7AB0", "marker": "s", "linestyle": "-"},
    "Triggered SN-RCPS": {"color": "#009E73", "marker": "D", "linestyle": "-"},
    "Change-point oracle": {"color": "#D55E00", "marker": "P", "linestyle": "--"},
}

DISPLAY_NAME = {
    "Bonferroni SN-RCPS": "Bonferroni SN-RCPS",
    "Oracle library member": "Oracle library member",
    "Deployment oracle": "Deployment oracle",
    "Exact blocked RCPS": "Exact blocked RCPS",
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "backend": "Agg",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "stixsans",
            "font.size": 9.0,
            "axes.labelsize": 9.2,
            "axes.titlesize": 10.0,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "legend.fontsize": 7.4,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.facecolor": PANEL,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#B9BEC5",
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.65,
            "grid.alpha": 0.95,
            "xtick.color": "#4F5358",
            "ytick.color": "#4F5358",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.5,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "legend.frameon": False,
            "legend.handlelength": 2.2,
            "legend.columnspacing": 1.2,
            "legend.handletextpad": 0.5,
            "figure.constrained_layout.use": True,
        }
    )


def style_for(method: str) -> dict[str, object]:
    return METHOD_STYLE.get(method, {"color": MUTED, "marker": "o", "linestyle": "-"})


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.105,
        1.065,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11.0,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )


def target_band(
    ax: plt.Axes,
    target: float,
    *,
    direction: str,
    label: str | None = None,
    zorder: float = 0.0,
) -> None:
    """Draw the valid side of a risk or coverage target."""

    if direction not in {"below", "above"}:
        raise ValueError("direction must be 'below' or 'above'")
    low, high = ax.get_ylim()
    if direction == "below":
        ax.axhspan(low, target, color=TARGET, alpha=0.92, zorder=zorder)
    else:
        ax.axhspan(target, high, color=TARGET, alpha=0.92, zorder=zorder)
    ax.axhline(target, color=TARGET_LINE, linestyle=(0, (4, 3)), linewidth=1.05, zorder=1.5)
    if label:
        ax.text(
            0.985,
            target,
            label,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom" if direction == "below" else "top",
            fontsize=7.2,
            color=TARGET_LINE,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.0},
        )


def add_regime_bands(ax: plt.Axes, boundaries: Iterable[float], end: float) -> None:
    starts = [0.0, *[float(x) for x in boundaries]]
    stops = [*[float(x) for x in boundaries], float(end)]
    for index, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        ax.axvspan(start, stop, color=REGIME[index % len(REGIME)], zorder=-5)
    for boundary in boundaries:
        ax.axvline(boundary, color="#9AA0A6", linewidth=0.85, linestyle=(0, (3, 3)), zorder=0)


def tidy_axis(ax: plt.Axes, *, xgrid: bool = False) -> None:
    ax.grid(True, axis="both" if xgrid else "y")
    ax.margins(x=0.02)


def legend_above(
    ax: plt.Axes,
    *,
    ncol: int = 3,
    y: float = 1.02,
    fontsize: float | None = None,
) -> None:
    kwargs: dict[str, object] = {}
    if fontsize is not None:
        kwargs["fontsize"] = fontsize
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, y),
        ncol=ncol,
        borderaxespad=0.0,
        **kwargs,
    )


def save_figure(fig: plt.Figure, figure_dir: Path, name: str, *, dpi: int = 360) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Creator": "SN-RCPS reproducibility package",
        "Title": name.replace("_", " ").title(),
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(
        figure_dir / f"{name}.pdf",
        bbox_inches="tight",
        pad_inches=0.035,
        metadata=metadata,
    )
    fig.savefig(
        figure_dir / f"{name}.png",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.035,
    )
    plt.close(fig)
