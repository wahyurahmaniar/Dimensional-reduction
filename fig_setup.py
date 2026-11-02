"""Fig. 1: suppression of position along the fiber, with every species and mass kept.

(a) the five-dimensional field on M_4 x S^1_R, with an excitation at a definite
    place y_0 on the fiber, and the tower of four-dimensional particle species
    m_n^2 = m_5^2 + n^2/R^2 that the same field presents to a four-dimensional
    observer, drawn to scale with m_5 R = 0.45;
(b) the dynamics: a phenomenological global coupling (exact model; a local
    coupling reproduces it at leading order) to an environment through
    K_y, whose reduced evolution exp(t L) spreads the excitation round the fiber
    (heat-kernel snapshots, computed);
(c) the one-particle sector, where Q = n: the coherences of the density matrix
    fade with |n - n'| while the sector weights p_Q stay;
(d) a separate two-particle example: under a momentum-conserving interaction a
    pair in the levels (-1, +1) can scatter into (-2, +2) at fixed total Q = 0,
    which energy conservation allows only when the pair's center-of-mass
    energy exceeds 2 m_2.

No equations are drawn inside the panels; they sit in the caption.
"""

import numpy as np
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Ellipse
from matplotlib.lines import Line2D

from figstyle import BR, W_CANVAS, PANEL_PT, apply_style, save, panel_label_ax, tint
import matplotlib.pyplot as plt

apply_style()

FS = 8.0                      # annotation and legend size on the canvas
H_IN = 3.05
fig = plt.figure(figsize=(W_CANVAS, H_IN))
gs = fig.add_gridspec(1, 4, width_ratios=[1.06, 0.92, 0.62, 0.56], wspace=0.05,
                      left=0.008, right=0.992, bottom=0.262, top=0.985)
YLO, YMAX = 0.2, 7.0
fig.canvas.draw()


def aspect(ax):
    """Data x-units per data y-unit that draw as equal lengths on the page."""
    p = ax.get_position()
    x_in = (p.x1 - p.x0) * W_CANVAS / (ax.get_xlim()[1] - ax.get_xlim()[0])
    y_in = (p.y1 - p.y0) * H_IN / (ax.get_ylim()[1] - ax.get_ylim()[0])
    return y_in / x_in


LEG = dict(frameon=False, fontsize=FS, borderaxespad=0.1, labelspacing=0.30,
           loc="upper center", bbox_to_anchor=(0.5, 0.0))

# ================================================== (a) field and its species
axa = fig.add_subplot(gs[0, 0]); axa.set_xlim(0, 10); axa.set_ylim(YLO, YMAX)
axa.axis("off")
ka = aspect(axa)

sheet = np.array([[0.15, 1.30], [4.15, 1.30], [5.00, 2.30], [1.00, 2.30]])
axa.add_patch(mpatches.Polygon(sheet, closed=True, facecolor=tint(BR["blue"], 0.80),
                               edgecolor=BR["blue"], linewidth=1.1, zorder=1))
EW = 0.44                                   # fiber ellipse half-width
for cx, cy, main in [(0.95, 1.62, False), (2.50, 1.80, True), (4.05, 1.98, False)]:
    axa.plot([cx, cx], [cy, cy + 1.30], color=BR["grey"], linewidth=0.55,
             alpha=1.0 if main else 0.5, zorder=2)
    axa.add_patch(Ellipse((cx, cy + 2.62), 2 * EW, 2.00, facecolor="none",
                          edgecolor=BR["black"] if main else BR["grey"],
                          linewidth=1.5 if main else 0.7,
                          alpha=1.0 if main else 0.55, zorder=3))
ex = 2.50 + EW * np.cos(0.60); ey = 4.42 + 1.00 * np.sin(0.60)
axa.plot([ex], [ey], marker="o", markersize=6.0, color=BR["red"], zorder=5)
axa.text(ex + 0.20, ey + 0.10, r"$y_{0}$", ha="left", va="center", fontsize=9.5,
         color=BR["red"])

# the same field read by a four-dimensional observer: a tower of species,
# level heights proportional to m_n = (m_5^2 + n^2/R^2)^(1/2) with m_5 R = 0.45
m5, base, scale = 0.45, 1.30, 1.47
XL, X0, X1 = 6.10, 6.40, 8.20
tops = [base + np.sqrt(m5 ** 2 + n ** 2) * scale for n in range(4)]
axa.add_patch(FancyArrowPatch((XL, base - 0.10), (XL, tops[-1] + 0.55),
                              arrowstyle="-|>", mutation_scale=8, linewidth=0.9,
                              color=BR["black"], zorder=4))
for n, yy in enumerate(tops):
    axa.plot([X0, X1], [yy, yy], color=BR["green"], linewidth=2.6,
             solid_capstyle="butt", zorder=4)
    axa.text(X1 + 0.18, yy, r"$n=0$" if n == 0 else rf"$\pm{n}$", ha="left",
             va="center", fontsize=9.0, color=BR["green"])
axa.text(XL + 0.22, tops[-1] + 0.55, r"$m_{n}$", ha="left", va="center", fontsize=10.0)
axa.add_patch(FancyArrowPatch((4.55, 4.95), (XL - 0.15, tops[2] + 0.35),
                              arrowstyle="-|>", mutation_scale=9, linewidth=0.9,
                              color=BR["grey"], connectionstyle="arc3,rad=-0.25",
                              zorder=6))
axa.text(3.75, 6.42, "as four-dimensional" + chr(10) + "species", ha="center",
         va="center", fontsize=FS, color=BR["grey"], linespacing=1.2)
axa.legend(handles=[
    Line2D([], [], color=BR["black"], linewidth=1.5, label=r"fiber $S^{1}_{R}$"),
    Line2D([], [], color=BR["red"], marker="o", markersize=5, linestyle="none",
           label=r"excitation at $y_{0}$"),
    Line2D([], [], color=BR["blue"], linewidth=1.5, label=r"base $\mathcal{M}_{4}$"),
    Line2D([], [], color=BR["green"], linewidth=2.6, label=r"tower $m_{n}$")],
    ncol=2, handlelength=1.5, columnspacing=1.0, **LEG)

# ================================================== (b) the dynamics
axb = fig.add_subplot(gs[0, 1]); axb.set_xlim(0, 10); axb.set_ylim(YLO, YMAX)
axb.axis("off")
kb = aspect(axb)

axb.text(5.0, 6.80, r"global coupling through $K_{y}$", ha="center", va="center",
         fontsize=FS, color=BR["magenta"])
axb.text(5.0, 6.42, "local coupling: leading order", ha="center", va="center",
         fontsize=FS, color=BR["magenta"])
FY = 5.55
axb.add_patch(Ellipse((1.75, FY), 2 * 0.33 * kb / ka, 1.06,
                      facecolor="none", edgecolor=BR["black"], linewidth=1.4,
                      zorder=3))
axb.text(1.75, FY - 0.78, "field", ha="center", va="center", fontsize=FS,
         color=BR["grey"])
axb.add_patch(mpatches.FancyBboxPatch(
    (5.65, FY - 0.43), 4.05, 0.86, boxstyle="round,pad=0.04,rounding_size=0.12",
    linewidth=1.2, edgecolor=BR["grey"], facecolor="white", zorder=3))
axb.text(7.675, FY, "environment", ha="center", va="center", fontsize=FS,
         color=BR["grey"], zorder=4)
wx = np.linspace(2.30, 5.55, 200)
axb.plot(wx, FY + 0.17 * np.sin((wx - 2.30) * 5.6), color=BR["magenta"],
         linewidth=1.2, zorder=2)
axb.text(5.0, 4.38, r"$\mathrm{e}^{t\mathcal{L}}$", ha="center", va="center", fontsize=11)
axb.add_patch(FancyArrowPatch((0.5, 3.92), (9.5, 3.92), arrowstyle="-|>",
                              mutation_scale=14, linewidth=1.6, color=BR["black"]))
axb.text(5.0, 3.50, "any environment: sectors conserved", ha="center",
         va="center", fontsize=FS, color=BR["grey"])
axb.text(5.0, 3.12, "Gaussian, Markovian: coherence decays", ha="center",
         va="center", fontsize=FS, color=BR["grey"])
ys = np.linspace(0, 2 * np.pi, 240)
SNAP = [(0.0, BR["blue"], "-"), (0.25, BR["red"], (0, (5, 2))),
        (1.2, BR["green"], (0, (1.5, 1.8)))]
mfreq = np.fft.fftfreq(len(ys), 1 / len(ys))
P0b = np.exp(9.0 * np.cos(ys - 2.4))
for gt, col, ls in SNAP:
    P = np.real(np.fft.ifft(np.fft.fft(P0b) * np.exp(-gt * mfreq ** 2)))
    P = P / P0b.max()
    axb.plot(0.8 + 8.4 * ys / (2 * np.pi), 1.35 + 1.45 * P, color=col,
             linestyle=ls, linewidth=1.6,
             label=r"$\gamma t=0$" if gt == 0 else rf"$\gamma t={gt:g}$")
axb.text(5.0, 0.95, "spreads round the fiber", ha="center", va="center",
         fontsize=FS, color=BR["grey"])
axb.legend(ncol=2, handlelength=1.6, columnspacing=1.0, **LEG)

# ================================================== (c) one-particle coherences
axc = fig.add_subplot(gs[0, 2]); axc.set_xlim(0, 10); axc.set_ylim(YLO, YMAX)
axc.axis("off")
kc = aspect(axc)

NG, cy0, celly = 5, 2.95, 0.60
cellx = celly * kc
gx0 = 5.0 - (NG * cellx - 0.12 * cellx) / 2
for i in range(NG):
    for j in range(NG):
        d = abs(i - j)
        col = BR["green"] if d == 0 else BR["red"]
        al = 1.0 if d == 0 else float(np.exp(-0.55 * d ** 2))
        axc.add_patch(mpatches.Rectangle(
            (gx0 + j * cellx, cy0 + (NG - 1 - i) * celly), cellx * 0.88, celly * 0.88,
            facecolor=col, edgecolor="none", alpha=al, zorder=2))
axc.text(5.0, cy0 - 0.45, "coherences fade", ha="center", va="center",
         fontsize=FS, color=BR["red"])
axc.add_patch(FancyArrowPatch((gx0 + 1.65 * cellx, cy0 + NG * celly + 0.58),
                              (gx0 + 0.55 * cellx, cy0 + (NG - 0.2) * celly),
                              arrowstyle="-|>", mutation_scale=8, linewidth=0.8,
                              color=BR["green"], zorder=6))
axc.text(gx0 + 1.80 * cellx, cy0 + NG * celly + 0.62, r"$p_{Q}$ kept", ha="left",
         va="center", fontsize=8.6, color=BR["green"])
axc.legend(handles=[
    Line2D([], [], color=BR["green"], marker="s", markersize=6, linestyle="none",
           label=r"sector weights $p_{Q}$"),
    Line2D([], [], color=BR["red"], marker="s", markersize=6, alpha=0.58,
           linestyle="none", label="coherence, $|n-n$′$|=1$"),
    Line2D([], [], color=BR["red"], marker="s", markersize=6, alpha=0.22,
           markeredgecolor=BR["red"], markeredgewidth=0.5,
           linestyle="none", label="$|n-n$′$|=2$, fainter")],
    ncol=1, handlelength=1.0, handletextpad=0.5, **LEG)

# ================================================== (d) two-particle example
axd = fig.add_subplot(gs[0, 3]); axd.set_xlim(0, 10); axd.set_ylim(YLO, YMAX)
axd.axis("off")

LEV = [-2, -1, 0, 1, 2]
ly0, dly = 2.95, 0.66
LX1, LX2, LW = 1.75, 7.25, 1.35


def occ_ladder(x0, occupied):
    for n in LEV:
        yy = ly0 + (n + 2) * dly
        axd.plot([x0, x0 + LW], [yy, yy], color=BR["grey"], linewidth=0.9, zorder=3)
        if n in occupied:
            axd.plot([x0 + LW / 2], [yy], marker="o", markersize=5.0,
                     color=BR["blue"], zorder=5)


occ_ladder(LX1, {-1, 1})
occ_ladder(LX2, {-2, 2})
for n in LEV:
    axd.text(LX1 - 0.25, ly0 + (n + 2) * dly, rf"${n:+d}$" if n else r"$0$",
             ha="right", va="center", fontsize=FS, color=BR["grey"])
axd.add_patch(FancyArrowPatch((LX1 + LW + 0.30, ly0 + 2 * dly), (LX2 - 0.30, ly0 + 2 * dly),
                              arrowstyle="-|>", mutation_scale=9, linewidth=1.0,
                              color=BR["black"], zorder=6))
axd.text(0.5 * (LX1 + LW + LX2), ly0 + 2 * dly + 0.12, "interaction", ha="center",
         va="bottom", fontsize=FS, color=BR["black"])
axd.text(5.0, ly0 - 0.45, "occupations move", ha="center", va="center",
         fontsize=FS, color=BR["blue"])
axd.text(5.0, ly0 - 0.85, r"total $Q=0$ kept", ha="center", va="center",
         fontsize=FS, color=BR["green"])
axd.text(5.0, ly0 - 1.25, "when the energy allows", ha="center", va="center",
         fontsize=FS, color=BR["grey"])
axd.legend(handles=[
    Line2D([], [], color=BR["blue"], marker="o", markersize=5.0, linestyle="none",
           label="occupied level")],
    ncol=1, handlelength=1.0, handletextpad=0.5, **LEG)

AT = {"a": 4.809, "b": 5.103, "c": 4.981, "d": 4.915}   # centred on the panel ink
panel_label_ax(fig, axa, "(a)", y=0.022, size=PANEL_PT, at=AT["a"])
panel_label_ax(fig, axb, "(b)", y=0.022, size=PANEL_PT, at=AT["b"])
panel_label_ax(fig, axc, "(c)", y=0.022, size=PANEL_PT, at=AT["c"])
panel_label_ax(fig, axd, "(d)", y=0.022, size=PANEL_PT, at=AT["d"])

save(fig, "fig_setup.png")
