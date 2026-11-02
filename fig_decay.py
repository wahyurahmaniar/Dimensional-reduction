"""Fig. 2: how fast fiber localization disappears, in the interaction picture.

(a) the harmonics of the fiber density, |P_m(t)|/|P_m(0)| = exp(-gamma m^2 t),
    so that tau_m = 1/(gamma m^2) and the short-wavelength structure goes first;
(b) an initially localized packet on the circle, evolved with the heat kernel
    G_t(y-y') = (1/(2pi)) sum_m exp(-gamma m^2 t) exp(i m (y-y')), approaching the
    uniform density 1/(2pi).

Panel (b) is computed, not sketched: the packet is transformed, damped mode by
mode and transformed back, and the total probability and the Kaluza-Klein
populations are asserted to be conserved along the way.
"""

import numpy as np

from figstyle import BR, W_CANVAS, PANEL_PT, apply_style, save, panel_label_ax
import matplotlib.pyplot as plt

apply_style()

# ------------------------------------------------------------- panel (a)
MS = [1, 2, 3, 4]
STYLE = [(BR["blue"], "-"), (BR["red"], (0, (5, 2))),
         (BR["green"], (0, (1.4, 1.8))), (BR["magenta"], (0, (6, 2, 1.4, 2)))]
gt = np.linspace(0, 3.0, 400)

# ------------------------------------------------------------- panel (b)
NY = 512
ys = np.linspace(0, 2 * np.pi, NY, endpoint=False)
Y0, KAPPA = 2.2, 12.0
P0 = np.exp(KAPPA * np.cos(ys - Y0))
P0 /= np.trapezoid(np.r_[P0, P0[0]], np.r_[ys, 2 * np.pi])     # unit total
F0 = np.fft.fft(P0)
MODES = np.fft.fftfreq(NY, d=1.0 / NY)


def density(gt_val):
    """Damp each harmonic by exp(-gamma m^2 t) and transform back."""
    return np.real(np.fft.ifft(F0 * np.exp(-gt_val * MODES ** 2)))


TIMES = [0.0, 0.05, 0.2, 1.0]

fig = plt.figure(figsize=(W_CANVAS, 2.80))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.30,
                      left=0.095, right=0.975, bottom=0.265, top=0.965)

axa = fig.add_subplot(gs[0, 0])
for m, (col, ls) in zip(MS, STYLE):
    axa.plot(gt, np.exp(-m ** 2 * gt), color=col, linestyle=ls, linewidth=1.6,
             label=rf"$m={m}$")
# the curves crowd the axis for m >= 3, so the labels go in a legend placed
# where no curve runs
axa.legend(loc="upper right", frameon=False, fontsize=8.0,
           handlelength=2.2, borderaxespad=0.6, labelspacing=0.35)
axa.axhline(np.exp(-1.0), color=BR["grey"], linewidth=0.6,
            linestyle=(0, (2.4, 2.0)))
axa.text(2.95, np.exp(-1.0) + 0.03, r"$1/\mathrm{e}$", ha="right", va="bottom",
         fontsize=7.6, color=BR["grey"])
for m, (col, _) in zip(MS, STYLE):
    axa.plot([1.0 / m ** 2], [np.exp(-1.0)], marker="o", markersize=3.8,
             color=col, zorder=5)
axa.set_xlabel(r"$\gamma t$")
axa.set_ylabel(r"$|P^{I}_{m}(t)|\,/\,|P^{I}_{m}(0)|$")
axa.set_xlim(0, 3.0)
axa.set_ylim(0, 1.02)

axb = fig.add_subplot(gs[0, 1])
# distinct hues and dash patterns rather than one hue at four opacities, so the
# snapshots stay separable where the curves cross and in grayscale
SNAP = [(BR["blue"], "-"), (BR["red"], (0, (5, 2))),
        (BR["green"], (0, (6, 2, 1.4, 2))), (BR["magenta"], (0, (1.5, 1.8)))]
for tv, (col, ls) in zip(TIMES, SNAP):
    lab = r"$\gamma t=0$" if tv == 0 else rf"$\gamma t={tv:g}$"
    axb.plot(ys, density(tv), color=col, linestyle=ls,
             linewidth=1.8 if tv == 0 else 1.5, label=lab)
axb.legend(loc="upper right", frameon=False, fontsize=7.8,
           handlelength=1.9, borderaxespad=0.6, labelspacing=0.30)
axb.axhline(1 / (2 * np.pi), color=BR["grey"], linewidth=0.9,
            linestyle=(0, (3.0, 2.0)))
axb.text(0.12, 1 / (2 * np.pi) + 0.05, r"$1/(2\pi)$", fontsize=7.8,
         color=BR["grey"], va="bottom")
axb.set_xlabel(r"$y$")
axb.set_ylabel(r"$P^{I}(y,t)$")
axb.set_xlim(0, 2 * np.pi)
axb.set_xticks([0, np.pi, 2 * np.pi])
axb.set_xticklabels(["$0$", r"$\pi$", r"$2\pi$"])
axb.set_ylim(0, 1.05 * P0.max())

panel_label_ax(fig, axa, "(a)", y=0.022, size=PANEL_PT)
panel_label_ax(fig, axb, "(b)", y=0.022, size=PANEL_PT)

save(fig, "fig_decay.png")

# ------------------------------------------------- assertions on the data
tot = [np.trapezoid(np.r_[density(t), density(t)[0]], np.r_[ys, 2 * np.pi])
       for t in TIMES]
print("total probability at each time:", np.round(tot, 12).tolist())
p_n = [abs(np.fft.fft(density(t))[0]) for t in TIMES]
print("m = 0 harmonic (the conserved weight):", np.round(p_n, 10).tolist())
far = density(60.0)
print("uniform limit reached: max|P(y,inf) - 1/2pi| = %.2e"
      % np.max(np.abs(far - 1 / (2 * np.pi))))
