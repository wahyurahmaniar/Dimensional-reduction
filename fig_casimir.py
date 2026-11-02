"""Fig. 4: the geometric spectrum of a state and the decay spectrum of its density.

(a) The Laplace-Beltrami ladder of the unit sphere S^2 = SU(2)/U(1),
    lambda_l = l(l+1), with a zonal harmonic for each level, joined to it by a
    leader line so the discs can be drawn large enough to show their nodal
    latitudes. The one-particle state (Y_10 + Y_20)/sqrt(2) occupies the levels
    l = 1 and l = 2, drawn heavy; the other levels are drawn light. Each level
    carries the color of the harmonic L = l in (b), which decays at the rate
    gamma l(l+1). These eigenvalues set the Kaluza-Klein masses of the modes.
(b) The fiber density of that same state under heat-kernel dephasing. Its
    harmonic L decays as exp(-gamma L(L+1) t). The coherence between l = 1 and
    l = 2 feeds only the odd harmonics L = 1 and 3, because Y_10 Y_20 has odd
    parity, and the diagonal terms feed L = 0, 2 and 4, with L = 4 from the
    level l = 2 alone, so the decay spectrum {2, 6, 12, 20} is set by the
    adjoint action on observables and reaches eigenvalues of levels the state
    does not occupy.

The decay curves are the closed form; numcheck.py measures the same rates from
the twirled density matrix of the state, L = 1..4, to 1e-2.
"""

import numpy as np
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from numpy.polynomial import legendre as _leg

from figstyle import BR, W_CANVAS, PANEL_PT, apply_style, save, panel_label_ax, tint
import matplotlib.pyplot as plt

apply_style()

LS = [0, 1, 2, 3, 4]
LAM = [l * (l + 1) for l in LS]                       # 0, 2, 6, 12, 20
OCC = {1, 2}                                          # levels of the state
COLS = [BR["blue"], BR["red"], BR["green"], BR["magenta"], BR["orange"]]
DASH = ["-", (0, (5, 2)), (0, (6, 2, 1.4, 2)), (0, (1.5, 1.8)), (0, (8, 2, 1.4, 2, 1.4, 2))]

H_IN = 2.75
fig = plt.figure(figsize=(W_CANVAS, H_IN))
gs = fig.add_gridspec(1, 2, width_ratios=[0.86, 1.14], wspace=0.26,
                      left=0.030, right=0.975, bottom=0.255, top=0.965)

# ------------------------------------------------- (a) the geometric spectrum
axa = fig.add_subplot(gs[0, 0])
YLO, YHI = -6.2, 23.4
axa.set_xlim(0, 10)
axa.set_ylim(YLO, YHI)
axa.axis("off")

axa.plot([2.55, 2.55], [-0.9, 20.9], color=BR["black"], linewidth=0.8)
axa.text(1.05, 10.0, r"$\lambda_{l}$", ha="center", va="center", fontsize=9.5)

fig.canvas.draw()
_pos = axa.get_position()
_ar = ((_pos.y1 - _pos.y0) * H_IN) / ((_pos.x1 - _pos.x0) * W_CANVAS)
RX = 0.62
RY = RX * ((YHI - YLO) / 10.0) / _ar                 # discs read as circles


def mode_disc(cx, cy, l, col, strong):
    zeros = np.sort(np.real(_leg.legroots([0] * l + [1]))) if l else np.array([])
    edges = np.concatenate(([-1.0], zeros, [1.0]))
    for k in range(len(edges) - 1):
        u = np.linspace(edges[k], edges[k + 1], 60)
        w = RX * np.sqrt(np.clip(1 - u ** 2, 0, None))
        poly = np.concatenate([np.c_[cx - w, cy + RY * u],
                               np.c_[cx + w, cy + RY * u][::-1]])
        dark, light = (0.80, 0.14) if strong else (0.32, 0.07)
        axa.add_patch(mpatches.Polygon(poly, closed=True, facecolor=col,
                                       edgecolor="none",
                                       alpha=dark if k % 2 else light, zorder=3))
    for z in zeros:
        w = RX * np.sqrt(max(1 - z ** 2, 0.0))
        axa.plot([cx - w, cx + w], [cy + RY * z, cy + RY * z],
                 color=col, linewidth=0.8 if strong else 0.6, zorder=4)
    axa.add_patch(Ellipse((cx, cy), 2 * RX, 2 * RY, facecolor="none",
                          edgecolor=col, linewidth=1.3 if strong else 0.8, zorder=5))


YD = [-0.4, 4.85, 10.1, 15.35, 20.6]                 # disc centres, evenly spaced
for l, lam in zip(LS, LAM):
    strong = l in OCC
    col = COLS[l]                                     # color of the harmonic L = l in (b)
    axa.plot([2.55, 4.35], [lam, lam], color=col,
             linewidth=2.6 if strong else 1.2, solid_capstyle="butt", zorder=4)
    axa.text(2.35, lam, rf"${lam}$", ha="right", va="center", fontsize=8.2)
    axa.text(4.55, lam, rf"$l={l}$", ha="left", va="center", fontsize=8.0)
    yd = YD[l]
    axa.plot([5.62, 8.15 - RX - 0.12], [lam, yd], color=col,
             linewidth=0.8 if strong else 0.6, zorder=2)
    mode_disc(8.15, yd, l, col, strong)
axa.text(3.4, -4.0, "eigenvalue", ha="center", va="center", fontsize=7.4)
axa.text(8.15, -4.0, "zonal harmonic", ha="center", va="center", fontsize=7.4)
_occ = tuple(Line2D([], [], color=COLS[l], linewidth=2.6, solid_capstyle="butt") for l in LS if l in OCC)
_emp = tuple(Line2D([], [], color=COLS[l], linewidth=1.2, solid_capstyle="butt") for l in LS if l not in OCC)
axa.legend([_occ, _emp], ["levels of the state", "unoccupied levels"],
           handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
           loc="lower center", frameon=False, fontsize=7.4, handlelength=2.4, ncol=2,
           columnspacing=1.2, borderaxespad=0.0, bbox_to_anchor=(0.5, -0.11))

# ------------------------------------------------- (b) the decay spectrum
axb = fig.add_subplot(gs[0, 1])
gt = np.linspace(0, 1.6, 500)
for L, lam, col, ls in zip(LS, LAM, COLS, DASH):
    axb.plot(gt, np.exp(-lam * gt), color=col, linestyle=ls, linewidth=1.6,
             label=rf"$L={L}$")
axb.axhline(np.exp(-1.0), color=BR["grey"], linewidth=0.6,
            linestyle=(0, (2.4, 2.0)))
axb.text(1.57, np.exp(-1.0) + 0.03, r"$1/\mathrm{e}$", ha="right", va="bottom",
         fontsize=7.6, color=BR["grey"])
for lam, col in zip(LAM, COLS):
    if lam:
        axb.plot([1.0 / lam], [np.exp(-1.0)], marker="o", markersize=3.8,
                 color=col, zorder=5)
axb.set_xlabel(r"$\gamma t$")
axb.set_ylabel(r"$|P_{L}(t)|\,/\,|P_{L}(0)|$")
axb.set_xlim(0, 1.6)
axb.set_ylim(0, 1.08)
axb.legend(loc="center right", frameon=False, fontsize=7.8, handlelength=2.4,
           borderaxespad=0.6, labelspacing=0.32, bbox_to_anchor=(1.0, 0.62))

panel_label_ax(fig, axa, "(a)", y=0.022, size=PANEL_PT, at=5.104)
panel_label_ax(fig, axb, "(b)", y=0.022, size=PANEL_PT)

save(fig, "fig_casimir.png")
print("eigenvalues l(l+1):", LAM)
print("marked 1/e crossings at gamma t = 1/lambda:", [round(1 / c, 4) for c in LAM if c])
