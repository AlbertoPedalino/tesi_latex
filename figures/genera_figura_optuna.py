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
from matplotlib.colors import BoundaryNorm, ListedColormap

from genera_figure_g_d import MUTED, TEXT, TICKS, size

HERE = pathlib.Path(__file__).parent
TRIALS = HERE / "dati" / "era5_optuna_prove.csv"
# One hue, dark for the lowest MMD (the best), light for the highest.
BOUNDS = (0.0025, 0.0035, 0.004, 0.0045, 0.0055, 0.007)  # five classes of the MMD distance, then "over"
RAMP = ListedColormap(("#0d366b", "#1f5fae", "#3987e5", "#7fb2ef", "#c4dcf8"), name="mmd")
RAMP.set_over("#eef4fc")
BEST_REGION = ("serene-sweep-14", "optuna-26", "optuna-27")  # lr G 6-8e-4, lr D 1.5-3e-3, weight 60-170, 1:1
LABELS = ("tasso di appr. G", "tasso di appr. D", r"peso $\lambda_G$", "aggiornamenti D:G")
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
    # The key of the arrows in its own panel under the plane, so that it covers no trial.
    figure = plt.figure(figsize=size(0.9), layout="constrained")
    grid = figure.add_gridspec(2, 1, height_ratios=(1, 0.24))
    axis, key = figure.add_subplot(grid[0]), figure.add_subplot(grid[1])

    # The best region is marked on its trials, not as an area: on two components of four, trials far apart in the
    # four parameters can fall close together.
    region = trials.name.isin(BEST_REGION).to_numpy()
    axis.scatter(coords[region, 0], coords[region, 1], marker="o", s=190, facecolors="none", edgecolors=REGION,
                 linewidths=1.6, zorder=2)

    # Five classes instead of a continuous ramp: nearby distances get clearly different blues.
    norm = BoundaryNorm(BOUNDS, RAMP.N)
    finished = (trials.state == "finished").to_numpy()
    mesh = axis.scatter(coords[finished, 0], coords[finished, 1], c=trials.mmd[finished], cmap=RAMP, norm=norm,
                        marker="o", s=46, edgecolors="#8f8d87", linewidths=0.4, zorder=3)
    failed = (trials.state == "non-finite").to_numpy()
    axis.scatter(coords[failed, 0], coords[failed, 1], marker="x", s=36, color=MUTED, linewidths=1.2, zorder=3)
    running = (trials.state == "running").to_numpy()
    axis.scatter(coords[running, 0], coords[running, 1], marker="o", s=46, facecolors="none", edgecolors=TEXT,
                 linewidths=1.0, zorder=3)

    best = int(trials.mmd.idxmin())
    axis.annotate(f"migliore\nMMD {trials.mmd[best]:.5f}", coords[best], xytext=(10, -4), textcoords="offset points",
                  fontsize=7.5, color=TEXT, va="top", path_effects=HALO, zorder=4)

    # How each parameter enters the two components: towards an arrow, that parameter grows.
    for (dx, dy), label in zip(loadings, LABELS):
        key.annotate("", (dx, dy), (0, 0), arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=0.8, shrinkA=0,
                                                           shrinkB=0))
        side = "right" if dx < -0.3 else "left" if dx > 0.3 else "center"  # the label beyond the tip, not on it
        key.annotate(label, (dx, dy), xytext=(4 * np.sign(dx) * (side != "center"), 4 * np.sign(dy) * (side == "center")),
                     textcoords="offset points", ha=side, va="center" if side != "center" else
                     ("bottom" if dy > 0 else "top"), fontsize=7, color=MUTED)
    key.set_xlim(-4.8, 2.4)
    key.set_ylim(-0.7, 1.3)
    key.set_aspect("equal", adjustable="box")
    key.set_xticks([])
    key.set_yticks([])
    key.text(0.02, 0.5, "verso dove cresce\nogni iperparametro", transform=key.transAxes, ha="left", va="center",
             fontsize=6.5, color=MUTED)
    for side in key.spines.values():
        side.set_color("#d9d8d4")

    bar = figure.colorbar(mesh, ax=axis, shrink=0.85, pad=0.02, extend="max")
    bar.set_label("MMD sugli anni di validazione (più scuro = migliore)", fontsize=TICKS)
    bar.ax.tick_params(labelsize=TICKS)
    # The names on the figure, not in a legend: the trials stopped by a non-finite loss, next to the leftmost one;
    # the best region, under its lowest trial.
    if failed.any():
        first = np.flatnonzero(failed)[np.argmin(coords[failed, 0])]
        axis.annotate("loss non finita", coords[first], xytext=(-8, 0), textcoords="offset points", fontsize=7.5,
                      color=MUTED, ha="right", va="center", path_effects=HALO, zorder=4)
    lowest = np.flatnonzero(region)[np.argmin(coords[region, 1])]
    axis.annotate("gruppo migliore", coords[lowest], xytext=(0, -11), textcoords="offset points", fontsize=7.5,
                  color="#a86a10", ha="center", va="top", path_effects=HALO, zorder=4)
    if running.any():
        axis.annotate("in corso", coords[np.flatnonzero(running)[0]], xytext=(8, 0), textcoords="offset points",
                      fontsize=7.5, color=TEXT, va="center")
    axis.set_xlabel(f"prima componente principale ({100 * variance[0]:.0f}\\%)".replace("\\%", "%"), fontsize=TICKS)
    axis.set_ylabel(f"seconda componente principale ({100 * variance[1]:.0f}%)", fontsize=TICKS)
    axis.tick_params(labelsize=TICKS, length=2)
    axis.set_aspect("equal", adjustable="datalim")
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    figure.savefig(HERE / "era5_optuna_pca.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()
