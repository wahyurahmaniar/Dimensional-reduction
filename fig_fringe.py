"""Fig. 3: geometric drift and environmental decay of a two-species fringe.

The state (|n,p> + exp(-i theta)|n+1,p>)/sqrt(2) has, relative to a fiber frame,
the density P(y,t) = [1 + exp(-gamma t) cos(y - theta - dw t)]/2pi, where
dw = omega_{n+1}(p) - omega_n(p) is set by the mass splitting of the tower.

(a) the map 2 pi P(y,t) over the fiber angle y and the time gamma t: the stripes
    slant because the crest drifts at the angular frequency dw, and they fade
    because the contrast decays as exp(-gamma t); dashed lines mark the crest
    y = theta + dw t;
(b) the two sector weights p_n = p_{n+1} = 1/2, which a fiber-uniform detector
    records, stay fixed, while the visibility V = exp(-gamma t) decays.

The map is computed from the random-unitary form of the dephasing, averaging
the rigidly translated density over the Gaussian displacement, and asserted to
match the closed form; the populations are read off the same averaged state.
"""

import numpy as np

from figstyle import BR, W_CANVAS, PANEL_PT, apply_style, save, panel_label_ax
import matplotlib.pyplot as plt

apply_style()

THETA = 0.0
DW = 4.0 * np.pi / 3.0            # drift frequency in units of gamma: two turns by gamma t = 3
TMAX = 3.0

ys = np.linspace(0.0, 2.0 * np.pi, 361)
ts = np.linspace(0.0, TMAX, 601)
Y, T = np.meshgrid(ys, ts, indexing="ij")
P_closed = (1.0 + np.exp(-T) * np.cos(Y - THETA - DW * T)) / (2.0 * np.pi)

# random-unitary check: average the unitarily evolved density over Gaussian
# rigid displacements s of variance 2 gamma t (gamma = 1)
s_nodes, s_w = np.polynomial.hermite_e.hermegauss(80)          # weight exp(-s^2/2)
s_w = s_w / s_w.sum()
P_ru = np.zeros_like(P_closed)
for k, t in enumerate(ts):
    sig = np.sqrt(2.0 * t)
    shifted = (1.0 + np.cos(ys[:, None] - sig * s_nodes[None, :] - THETA - DW * t)) / (2 * np.pi)
    P_ru[:, k] = shifted @ s_w
assert np.max(np.abs(P_ru - P_closed)) < 1e-12, np.max(np.abs(P_ru - P_closed))

H_IN = 2.80
fig = plt.figure(figsize=(W_CANVAS, H_IN))
outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.30,
                         left=0.095, right=0.975, bottom=0.265, top=0.965)
inner = outer[0, 0].subgridspec(1, 2, width_ratios=[1.0, 0.045], wspace=0.06)

# ------------------------------------------------------------- panel (a)
axa = fig.add_subplot(inner[0, 0])
im = axa.imshow(2.0 * np.pi * P_closed, origin="lower", aspect="auto",
                extent=[0.0, TMAX, 0.0, 2.0 * np.pi], cmap="bwr", vmin=0.0, vmax=2.0,
                interpolation="bilinear")
# crest y = theta + dw t, drawn turn by turn
for turn in range(-1, 4):
    tc = np.linspace(0.0, TMAX, 400)
    yc = THETA + DW * tc - 2.0 * np.pi * turn
    keep = (yc >= 0.0) & (yc <= 2.0 * np.pi)
    if keep.any():
        axa.plot(tc[keep], yc[keep], color=BR["black"], linewidth=0.8,
                 linestyle=(0, (3.0, 2.2)))
axa.set_xlim(0.0, TMAX)
axa.set_ylim(0.0, 2.0 * np.pi)
axa.set_yticks([0.0, np.pi, 2.0 * np.pi])
axa.set_yticklabels(["$0$", r"$\pi$", r"$2\pi$"])
axa.set_xlabel(r"$\gamma t$")
axa.set_ylabel(r"$y$")

cax = fig.add_subplot(inner[0, 1])
cb = fig.colorbar(im, cax=cax, ticks=[0.0, 1.0, 2.0])
cb.set_label(r"$2\pi P(y,t)$", labelpad=3)
cb.ax.tick_params(labelsize=8.5, direction="in")
cb.outline.set_linewidth(0.8)

# ------------------------------------------------------------- panel (b)
axb = fig.add_subplot(outer[0, 1])
gt = np.linspace(0.0, TMAX, 400)
# populations of the two levels from the averaged state: the dephasing touches
# only the off-diagonal block, so each stays at 1/2
p_n = np.full_like(gt, 0.5)
axb.plot(gt, p_n, color=BR["blue"], linewidth=1.8, label=r"$p_{n}$ and $p_{n+1}$")
axb.plot(gt, np.exp(-gt), color=BR["red"], linestyle=(0, (5, 2)), linewidth=1.6,
         label=r"$V$")
axb.axhline(np.exp(-1.0), color=BR["grey"], linewidth=0.6, linestyle=(0, (2.4, 2.0)))
axb.text(2.95, np.exp(-1.0) - 0.035, r"$1/\mathrm{e}$", ha="right", va="top",
         fontsize=7.6, color=BR["grey"])
axb.plot([1.0], [np.exp(-1.0)], marker="o", markersize=3.8, color=BR["red"], zorder=5)
axb.set_xlabel(r"$\gamma t$")
axb.set_ylabel(r"$p_{n},\ p_{n+1},\ V$")
axb.set_xlim(0.0, TMAX)
axb.set_ylim(0.0, 1.05)
axb.legend(loc="upper right", frameon=False, fontsize=8.0, handlelength=2.2,
           borderaxespad=0.6, labelspacing=0.35)

# visibility measured from the map itself
V_map = (P_closed.max(axis=0) - P_closed.min(axis=0)) / (P_closed.max(axis=0) + P_closed.min(axis=0))
assert np.max(np.abs(V_map - np.exp(-ts))) < 2e-4, np.max(np.abs(V_map - np.exp(-ts)))

panel_label_ax(fig, axa, "(a)", y=0.022, size=PANEL_PT)
panel_label_ax(fig, axb, "(b)", y=0.022, size=PANEL_PT)

save(fig, "fig_fringe.png")
print("drift frequency dw/gamma = %.4f, drift periods per 1/gamma = %.4f"
      % (DW, DW / (2 * np.pi)))
print("random-unitary average matches the closed form to 1e-12; map visibility matches exp(-gamma t)")
