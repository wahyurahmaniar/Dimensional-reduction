"""Shared plotting style for the manuscript figures.

Conventions enforced here:
  * Okabe-Ito palette (OI) and the bright palette (BR) used by the figures.
  * STIXGeneral text with the matching STIX math font.
  * No plot titles; panel labels placed bottom-centre, non-bold.
  * 400 dpi PNG output.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito palette
OI = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    # Okabe-Ito reddish purple, darkened from #CC79A7: the original has
    # contrast 3.06 against white, the weakest of the six, which left the
    # cross-sector label faint. Hue is unchanged (326 deg); contrast 5.49,
    # matching blue at 5.19.
    "purple": "#A34A7C",
    "grey": "#4A4A4A",
}

DPI = 400

# Bright palette for the figures: pure blue, red and green for up to three
# series, then magenta, orange, purple and cyan when more are needed.
BR = {
    "blue": "#0000FF",
    "red": "#FF0000",
    "green": "#008C00",
    "magenta": "#D400D4",
    "orange": "#FF7F00",
    "purple": "#7B00E0",
    "cyan": "#00AEC0",
    "black": "#000000",
    "grey": "#4A4A4A",
}

# Every figure is drawn on a canvas W_CANVAS inches wide, saved without a tight
# crop, and placed at \linewidth = 468.33 pt in cas-sc. The scale is therefore
# the same for every figure, and one point size gives one on-page size.
W_CANVAS = 7.0
TEXTWIDTH_IN = 468.3324 / 72.27
PAGE_SCALE = TEXTWIDTH_IN / W_CANVAS
PANEL_ON_PAGE_PT = 9.0
PANEL_PT = PANEL_ON_PAGE_PT / PAGE_SCALE


def tint(base, white=0.62):
    """Mix a palette color with white. Lower `white` gives a stronger fill.

    The fills were originally hand-picked hex values sitting 78 to 93 per cent
    of the way to white, which read as washed out in print. Routing them through
    one function keeps every panel consistent and adjustable.
    """
    b = [int(base[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02X%02X%02X" % tuple(int(round(c + white * (255 - c))) for c in b)


def apply_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 9.0,
            "axes.labelsize": 9.5,
            "axes.titlesize": 9.5,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "lines.linewidth": 1.4,
            "legend.frameon": False,
            "legend.handlelength": 2.4,
            "savefig.dpi": DPI,
            "figure.dpi": DPI,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.02,
        }
    )


def panel_label(ax, text, y=-0.175):
    """Place a non-bold panel label at the bottom centre of an axes."""
    ax.text(
        0.5,
        y,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=12.5,
        fontweight="normal",
    )


def panel_label_fig(fig, x, y, text):
    """Panel label for schematic panels drawn without a data axes."""
    fig.text(x, y, text, ha="center", va="bottom", fontsize=12.5,
             fontweight="normal")


def panel_label_ax(fig, ax, text, y=0.030, size=10.0, at=None):
    """Bottom-centre panel label.

    Centred on the axes box by default. When a panel's drawing does not fill
    its axes, pass `at` as a data x-coordinate inside that axes and the label
    is centred on that instead, so it tracks the drawing rather than the box.
    """
    pos = ax.get_position()
    if at is None:
        x = 0.5 * (pos.x0 + pos.x1)
    else:
        x = fig.transFigure.inverted().transform(
            ax.transData.transform((at, 0)))[0]
    fig.text(x, y, text, ha="center", va="bottom",
             fontsize=size, fontweight="normal")


def save(fig, name):
    fig.savefig(name, dpi=DPI, format="png")
    plt.close(fig)
    print("wrote", name)
