"""Shared figure style for paper figures (static PNG + PDF, light background).

Colour roles (validated categorical palette, first three slots are safe for colour-blind readers):
    label 0 (CN / sMCI)   blue
    label 1 (AD / pMCI)   orange
    third series          aqua
Text and axes use neutral greys, never the series colour.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

LABEL_NAMES = {"CN_AD": ("CN", "AD"), "sMCI_pMCI": ("sMCI", "pMCI")}


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "legend.frameon": False, "legend.fontsize": 8, "font.size": 9,
        "lines.linewidth": 2, "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
