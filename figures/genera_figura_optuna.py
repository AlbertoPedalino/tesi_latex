"""Figure of Section 6.1: the trials of the hyperparameter search on the plane of their first two principal components,
with the best region marked. Same plane as scripts/plot_era5_optuna_pca.py of the code repository: four coordinates per
trial (log10 of the learning rates of G and D and of the reconstruction weight, log2 of the D:G updates), standardised.

The trials are in figures/dati/era5_optuna_prove.csv, copied from the `report` of
scripts/optuna_era5_stgan.py on the study era5-gan-extended-2 (W&B sweep and first Optuna study imported, then the
trials of the second study). Run from tesi_latex/figures:

    python genera_figura_optuna.py

which writes figures/era5_optuna_pca.pdf and prints the share of the variance of the two components.
"""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

from genera_figure_g_d import MUTED, TEXT, TICKS, size

HERE = pathlib.Path(__file__).parent
TRIALS = HERE / "dati" / "era5_optuna_prove.csv"
# One hue, dark for the lowest MMD (the best), light for the highest.
RAMP = LinearSegmentedColormap.from_list("mmd", ("#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"))
BEST_REGION = ("serene-sweep-14", "optuna-26", "optuna-27")  # lr G 6-8e-4, lr D 1.5-3e-3, weight 60-170, 1:1
LABELS = ("lr G", "lr D", r"peso $\lambda$", "agg. D:G")
HALO = [effects.withStroke(linewidth=2.5, foreground="white")]
REGION = "#e8a33d"
UPDATES = {"1:2": -1.0, "1:1": 0.0, "2:1": 1.0}


def plane(trials):
    points = np.column_stack([np.log10(trials.lr_g), np.log10(trials.lr_d), np.log10(trials.reconstruction_weight),
                              trials.updates_d_g.map(UPDATES)])
    spread = points.std(axis=0)
    standard = (points - points.mean(axis=0)) / np.where(spread > 0, spread, 1)
    _, singular, axes = np.linalg.svd(standard, full_matrices=False)
    variance = singular ** 2 / (singular ** 2).sum()
    coords, loadings = standard @ axes[:2].T, axes[:2].T
    best = trials.mmd.idxmin()
    if coords[best, 0] < 0:  # the best trial on the right, for reading
        coords[:, 0], loadings[:, 0] = -coords[:, 0], -loadings[:, 0]
    return coords, loadings, variance


def main():
    trials = pd.read_csv(TRIALS)
    coords, loadings, variance = plane(trials)
    print(f"share of the variance: first {variance[0]:.2f}, second {variance[1]:.2f}, together {variance[:2].sum():.2f}")
    figure, axis = plt.subplots(figsize=size(0.72), constrained_layout=True)

    # The best region is marked on its trials, not as an area: on two components of four, trials far apart in the
    # four parameters can fall close together.
    region = trials.name.isin(BEST_REGION).to_numpy()
    axis.scatter(coords[region, 0], coords[region, 1], marker="o", s=190, facecolors="none", edgecolors=REGION,
                 linewidths=1.6, zorder=2)

    norm = Normalize(vmin=0.0025, vmax=0.007)
    finished = (trials.state == "finished").to_numpy()
    for phase, marker in (("W&B", "o"), ("Optuna", "s")):
        chosen = finished & (trials.phase == phase).to_numpy()
        mesh = axis.scatter(coords[chosen, 0], coords[chosen, 1], c=trials.mmd[chosen], cmap=RAMP, norm=norm,
                            marker=marker, s=46, edgecolors="white", linewidths=0.8, zorder=3)
    failed = (trials.state == "non-finite").to_numpy()
    axis.scatter(coords[failed, 0], coords[failed, 1], marker="x", s=36, color=MUTED, linewidths=1.2, zorder=3)
    running = (trials.state == "running").to_numpy()
    axis.scatter(coords[running, 0], coords[running, 1], marker="s", s=46, facecolors="none", edgecolors=TEXT,
                 linewidths=1.0, zorder=3)

    best = int(trials.mmd.idxmin())
    axis.annotate(f"migliore\nMMD {trials.mmd[best]:.5f}", coords[best], xytext=(10, -4), textcoords="offset points",
                  fontsize=7.5, color=TEXT, va="top", path_effects=HALO, zorder=4)

    # How each parameter enters the two components, in a corner: towards an arrow, that parameter grows.
    key = axis.inset_axes([0.0, 0.0, 0.3, 0.34])
    for (dx, dy), label in zip(loadings, LABELS):
        key.annotate("", (dx, dy), (0, 0), arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=0.8, shrinkA=0,
                                                           shrinkB=0))
        side = "right" if dx < -0.3 else "left" if dx > 0.3 else "center"  # the label beyond the tip, not on it
        key.annotate(label, (dx, dy), xytext=(4 * np.sign(dx) * (side != "center"), 4 * np.sign(dy) * (side == "center")),
                     textcoords="offset points", ha=side, va="center" if side != "center" else
                     ("bottom" if dy > 0 else "top"), fontsize=7, color=MUTED)
    key.set_xlim(-1.75, 1.75)
    key.set_ylim(-1.5, 1.5)
    key.set_aspect("equal")
    key.set_xticks([])
    key.set_yticks([])
    key.set_title("verso dove cresce\nogni parametro", fontsize=6.5, color=MUTED, pad=2)
    for side in key.spines.values():
        side.set_color("#d9d8d4")

    bar = figure.colorbar(mesh, ax=axis, shrink=0.85, pad=0.02, extend="max")
    bar.set_label("MMD sulla validazione (più scuro = migliore)", fontsize=TICKS)
    bar.ax.tick_params(labelsize=TICKS)
    keys = [Line2D([], [], marker="o", linestyle="", color="#3987e5", markeredgecolor="white", markersize=6.5,
                   label="sweep di W&B"),
            Line2D([], [], marker="s", linestyle="", color="#3987e5", markeredgecolor="white", markersize=6.5,
                   label="Optuna"),
            Line2D([], [], marker="x", linestyle="", color=MUTED, markersize=6, label="perdita non finita"),
            Line2D([], [], marker="s", linestyle="", markerfacecolor="none", markeredgecolor=TEXT, markersize=6.5,
                   label="in corso"),
            Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=REGION,
                   markeredgewidth=1.6, markersize=10, label="regione migliore")]
    axis.legend(handles=keys, frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3,
                handletextpad=0.3, columnspacing=1.0)
    axis.set_xlabel(f"prima componente principale ({100 * variance[0]:.0f}\\%)".replace("\\%", "%"), fontsize=TICKS)
    axis.set_ylabel(f"seconda componente ({100 * variance[1]:.0f}%)", fontsize=TICKS)
    axis.tick_params(labelsize=TICKS, length=2)
    axis.set_aspect("equal", adjustable="datalim")
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    figure.savefig(HERE / "era5_optuna_pca.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()
