"""Figures of Section 6.7 and 6.8 that compare the nine methods around the concordant clusters.

Run from the thesis folder, after figures/prepara_dati_concordate.py:
    python figures/genera_figure_concordate_confronto.py
which writes
- figures/era5_concordate_sovrapposizione.pdf: coverage against eccedenza of every method, short and long clusters,
  with the curves of constant Jaccard index J = cop / (1 + ecc) (equation of Section 5.5);
- figures/era5_concordate_giudici.pdf: for each quantity that no score reads, the share of the non-concordant flags of
  every method around the long concordant clusters in which it is unusual, coloured by its ratio to the reference.
Order and colours of the methods from figures/stile.py.
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from genera_figure_g_d import TICKS, size  # fonts and sizes of every figure of the thesis
from stile import BLUE, COLOURS, GRID, LABELS, MUTED, ORDER, SHALLOW, TEXT, TEXT_COLOURS

HERE = pathlib.Path(__file__).parent
OVERLAP = HERE / "dati" / "era5_concordate_sovrapposizione.csv"
JUDGES = HERE / "dati" / "era5_concordate_giudici.csv"
HALO = [effects.withStroke(linewidth=2.2, foreground="white")]
SHORT_NAMES = {"Gn": r"$G_\mathrm{n}$", "change_3h": "variazione", "normal": "distanza", "pca": "PCA", "cblof": "CBLOF",
               "mcd": "MCD", "hbos": "HBOS", "copod": "COPOD", "iforest": "iForest"}


def overlap_figure():
    """Coverage on x, eccedenza on y: a method low on the curves of constant Jaccard misses part of the cluster (left) or
    flags much around it (up)."""
    table = pd.read_csv(OVERLAP).set_index(["cluster", "metodo"])
    figure, axes = plt.subplots(1, 2, figsize=size(0.62), constrained_layout=True, sharey=True)
    cover = np.linspace(0.55, 1.0, 50)
    # Every name where it does not touch another, in the units of the axes (x, y, alignment); a thin line joins it to
    # its point when it stands apart. The names are those of the tables.
    places = {
        "brevi": {"Gn": (0.665, 0.87, "right"), "change_3h": (0.79, 1.36, "left"), "normal": (0.60, -0.08, "left"),
                  "pca": (0.985, 0.18, "right"), "cblof": (0.80, 0.08, "left"), "mcd": (0.94, 0.77, "right"),
                  "hbos": (0.68, 0.37, "right"), "copod": (0.69, 0.15, "left"), "iforest": (0.79, 0.43, "left")},
        "lunghi": {"Gn": (0.69, 0.36, "right"), "change_3h": (0.69, 0.60, "left"), "normal": (0.60, 0.04, "left"),
                   "pca": (0.985, 0.06, "right"), "cblof": (0.60, 0.22, "left"), "mcd": (0.875, 0.40, "center"),
                   "hbos": (0.80, 0.46, "right"), "copod": (0.75, -0.08, "left"), "iforest": (0.985, 0.51, "right")}}
    for axis, kind, title in zip(axes, ("brevi", "lunghi"), ("cluster brevi", "cluster lunghi")):
        for level in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            excess = cover / level - 1
            keep = (excess >= -0.02) & (excess <= 1.5)
            axis.plot(cover[keep], excess[keep], color=GRID, linewidth=0.9, zorder=1)
            if keep.any():
                last = np.flatnonzero(keep)[-1]
                x, y = cover[last], excess[last]
                if y > 1.45:
                    axis.text(x, 1.47, f"J = {level:.1f}", fontsize=TICKS - 2, color=MUTED, ha="center", va="bottom")
                else:
                    axis.text(1.005, y, f"J = {level:.1f}", fontsize=TICKS - 2, color=MUTED, ha="left", va="center")
        for method in reversed(ORDER):
            cop, ecc = table.loc[(kind, method), ["copertura", "eccedenza"]]
            big = method == "Gn"
            axis.plot(cop, ecc, "o", markersize=7 if big else 5.5, color=COLOURS[method], markeredgecolor="white",
                      markeredgewidth=0.8, zorder=4 if method not in SHALLOW else 3)
            x, y, side = places[kind][method]
            far = abs(x - cop) > 0.03 or abs(y - ecc) > 0.05
            axis.annotate(LABELS[method], (cop, ecc), xytext=(x, y), textcoords="data", ha=side, va="center",
                          fontsize=TICKS - 1.5, color=TEXT_COLOURS[method], path_effects=HALO, zorder=5,
                          arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.5, shrinkA=1, shrinkB=3)
                          if far else None)
        axis.set_xlim(0.55, 1.0)
        axis.set_ylim(-0.13, 1.5)
        axis.set_xlabel("copertura del cluster concordato", fontsize=TICKS)
        axis.set_title(title, fontsize=TICKS, color=TEXT, pad=11)
        axis.tick_params(labelsize=TICKS, length=2)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    axes[0].set_ylabel("eccedenza attorno al cluster", fontsize=TICKS)
    figure.savefig(HERE / "era5_concordate_sovrapposizione.pdf")
    plt.close(figure)


JUDGE_COLUMNS = (("cape highest 5%", "CAPE"), ("i10fg highest 5%", "raffica"), ("cp highest 5%", "prec.\nconv."),
                 ("d2m highest 5%", "rugiada\nalta"), ("d2m lowest 5%", "rugiada\nbassa"), ("kx highest 5%", "indice\nK"),
                 ("mn2t lowest 5%", "minima\nbassa"), ("mx2t highest 5%", "massima\nalta"),
                 ("tcwv highest 5%", "vapore\ncolonna"))


def judges_figure():
    """Rows the nine methods and the reference, columns the quantities; the number is the share of values in which the
    quantity is unusual (by chance 5%), the colour its ratio to the reference: orange more often, blue less often."""
    table = pd.read_csv(JUDGES).set_index("metodo")
    columns = [key for key, _ in JUDGE_COLUMNS]
    rows = list(ORDER) + ["reference"]
    values = table.loc[rows, columns].to_numpy()
    ratio = values / table.loc["reference", columns].to_numpy()
    ramp = LinearSegmentedColormap.from_list("rapporto", ("#2a78d6", "#a9cbf3", "#f4f3f0", "#f6b393", "#d4541f"))
    norm = LogNorm(vmin=0.25, vmax=4)
    figure, axis = plt.subplots(figsize=size(0.62), constrained_layout=True)
    shown = np.where(np.arange(len(rows))[:, None] == len(rows) - 1, np.nan, ratio)  # the reference row uncoloured
    mesh = axis.imshow(np.clip(shown, 0.25, 4), cmap=ramp, norm=norm, aspect="auto")
    axis.add_patch(plt.Rectangle((-0.5, len(rows) - 1.5), len(columns), 1, facecolor="#f4f3f0", edgecolor="none"))
    for i, row in enumerate(rows):
        for j in range(len(columns)):
            dark = i < len(rows) - 1 and (ratio[i, j] >= 2.2 or ratio[i, j] <= 0.45)
            axis.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center", fontsize=TICKS - 1.5,
                      color="white" if dark else TEXT, fontweight="bold" if row == "Gn" else "normal")
    axis.axhline(len(rows) - 1.5, color=TEXT, linewidth=0.8)
    axis.axhline(0.5, color="white", linewidth=2.0)
    axis.set_xticks(range(len(columns)), [label for _, label in JUDGE_COLUMNS], fontsize=TICKS - 1)
    axis.xaxis.tick_top()
    labels = [LABELS[row] if row != "reference" else "Termine di confronto" for row in rows]
    axis.set_yticks(range(len(rows)), labels, fontsize=TICKS - 1)
    for tick, row in zip(axis.get_yticklabels(), rows):
        tick.set_color(TEXT_COLOURS.get(row, TEXT))
    axis.tick_params(length=0)
    for side in axis.spines.values():
        side.set_visible(False)
    bar = figure.colorbar(mesh, ax=axis, orientation="horizontal", shrink=0.7, pad=0.03, aspect=35,
                          ticks=[0.25, 0.5, 1, 2, 4])
    bar.ax.set_xticklabels(["1/4", "1/2", "1", "2", "4"], fontsize=TICKS - 1)
    bar.set_label("rapporto con il termine di confronto (sotto 1 meno spesso, sopra 1 più spesso)", fontsize=TICKS - 1)
    figure.savefig(HERE / "era5_concordate_giudici.pdf")
    plt.close(figure)


AROUND = HERE / "dati" / "era5_concordate_attorno.csv"
LEVELS = HERE / "dati" / "era5_concordate_livelli.csv"
REFERENCE_LABEL = "termine di confronto"


def around_figure():
    """Left: for the non-concordant flags around the short clusters, the jump in three hours before the moderate values
    (x) and the share of values with a variable beyond two standard deviations (y). Right: behind the short clusters,
    the largest standardised distance from the value to 24 hours after."""
    table = pd.read_csv(AROUND).set_index("metodo")
    figure, (left, right) = plt.subplots(1, 2, figsize=size(0.5), constrained_layout=True,
                                         gridspec_kw={"width_ratios": (1.1, 1)})
    # Names: one by one where a point stands apart; the four shallow models packed together get one name.
    places = {"Gn": (2.66, 74.2, "right"), "change_3h": (2.74, 62.5, "right"), "normal": (1.25, 101.5, "left"),
              "pca": (2.52, 96.2, "left"), "mcd": (2.34, 90.7, "left"), "reference": (1.37, 40.3, "left")}
    for method in list(reversed(ORDER)) + ["reference"]:
        x, y = table.loc[method, ["salto_moderati", "oltre2_brevi"]]
        colour = COLOURS.get(method, TEXT)
        marker = "s" if method == "reference" else "o"
        left.plot(x, y, marker, markersize=7 if method == "Gn" else 5.5, color=colour, markeredgecolor="white",
                  markeredgewidth=0.8, zorder=3 if method in SHALLOW else 4)
        if method not in places:
            continue
        tx, ty, side = places[method]
        name = REFERENCE_LABEL if method == "reference" else LABELS[method]
        left.text(tx, ty, name, ha=side, va="center", fontsize=TICKS - 1.5, color=TEXT_COLOURS.get(method, TEXT),
                  path_effects=HALO, zorder=5)
    left.text(1.62, 85.5, "CBLOF, HBOS, COPOD,\nIsolation Forest", ha="left", va="top", fontsize=TICKS - 1.5,
              color="#6f6d67", path_effects=HALO, zorder=5)
    left.set_xlim(1.2, 3.0)
    left.set_ylim(35, 105)
    left.set_xlabel("salto più grande in tre ore prima del valore\n(valori moderati, deviazioni standard)", fontsize=TICKS)
    left.set_ylabel("valori con una variabile oltre\ndue deviazioni standard (%)", fontsize=TICKS)

    hours = [0, 3, 6, 12, 24]
    for method in list(reversed(ORDER)) + ["reference"]:
        values = table.loc[method, [f"dietro_{h}h" for h in hours]].to_numpy()
        if method == "reference":
            right.plot(hours, values, color=TEXT, linewidth=1.2, linestyle="--", zorder=4)
        else:
            right.plot(hours, values, color=COLOURS[method], linewidth=2.2 if method == "Gn" else 1.4 if method not in
                       SHALLOW else 1.0, marker="o", markersize=3, zorder=4 if method not in SHALLOW else 2)
    handles = [plt.Line2D([], [], color=COLOURS["Gn"], linewidth=2.2, label=LABELS["Gn"]),
               plt.Line2D([], [], color=COLOURS["change_3h"], linewidth=1.4, label=LABELS["change_3h"]),
               plt.Line2D([], [], color=COLOURS["normal"], linewidth=1.4, label=LABELS["normal"]),
               plt.Line2D([], [], color=COLOURS["pca"], linewidth=1.0, label="sei modelli shallow"),
               plt.Line2D([], [], color=TEXT, linewidth=1.2, linestyle="--", label=REFERENCE_LABEL)]
    right.legend(handles=handles, loc="upper right", frameon=False, fontsize=TICKS - 1.5, handlelength=1.8)
    right.set_xticks(hours)
    right.set_xlim(-0.5, 24.5)
    right.set_xlabel("ore dopo il valore", fontsize=TICKS)
    right.set_ylabel("scarto standardizzato più grande\ndietro i cluster brevi\n(deviazioni standard)", fontsize=TICKS)
    for axis in (left, right):
        axis.tick_params(labelsize=TICKS, length=2)
        axis.grid(color=GRID, linewidth=0.5)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    figure.savefig(HERE / "era5_concordate_attorno.pdf")
    plt.close(figure)


def levels_figure():
    """The flags of G_n with the probability of exceedance at least the level against their control: share of the
    concordant clusters in which the level starts later or earlier, ends earlier or later, is further ahead or behind."""
    table = pd.read_csv(LEVELS)
    figure, axes = plt.subplots(1, 3, figsize=size(0.36), constrained_layout=True, sharex=True)
    panels = (("inizio_dopo", "inizio_prima", "comincia dopo", "comincia prima", "Inizio"),
              ("fine_prima", "fine_dopo", "finisce prima", "finisce dopo", "Fine"),
              ("moto_avanti", "moto_dietro", "più avanti", "più indietro", "Moto"))
    steps = {0.6: 0, 0.7: 1, 0.8: 2, 0.9: 3, 0.95: 4}  # the five levels evenly spaced: 0.9 and 0.95 apart
    for axis, (main, other, main_label, other_label, title) in zip(axes, panels):
        for kind, face in (("brevi", "white"), ("lunghi", BLUE)):
            part = table[table.cluster == kind]
            x = part.livello.round(2).map(steps)
            axis.plot(x, part[main], color=BLUE, linewidth=1.6, marker="o", markersize=4,
                      markerfacecolor=face, markeredgecolor=BLUE, zorder=3)
            axis.plot(x, part[other], color=MUTED, linewidth=1.0, linestyle="--", marker="o", markersize=3.5,
                      markerfacecolor=face if face == "white" else MUTED, markeredgecolor=MUTED, zorder=2)
        axis.text(0.03, 0.97, title, transform=axis.transAxes, fontsize=TICKS, color=TEXT, va="top")
        last = table[table.livello.round(2) == 0.95]
        axis.annotate(main_label, (4, last[main].max()), xytext=(0, 6), textcoords="offset points",
                      fontsize=TICKS - 2, color=BLUE, ha="right", va="bottom")
        axis.text(0.98, 0.03, f"tratteggio: {other_label}", transform=axis.transAxes, fontsize=TICKS - 2,
                  color=MUTED, ha="right", va="bottom")
        axis.set_xticks(range(5), ["0.6", "0.7", "0.8", "0.9", "0.95"])
        axis.set_xlim(-0.3, 4.3)
        axis.tick_params(labelsize=TICKS - 1, length=2)
        axis.grid(color=GRID, linewidth=0.5)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    axes[0].set_ylim(-6, 42)
    axes[1].set_ylim(-6, 56)
    axes[2].set_ylim(40, 61)
    axes[0].set_ylabel("cluster concordati (%)", fontsize=TICKS)
    axes[1].set_xlabel("livello di probabilità", fontsize=TICKS)
    handles = [plt.Line2D([], [], color=BLUE, marker="o", markerfacecolor="white", linestyle="none", label="cluster brevi"),
               plt.Line2D([], [], color=BLUE, marker="o", linestyle="none", label="cluster lunghi")]
    # Outside the panels, above them: inside it covered the curve of the long clusters.
    figure.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False, fontsize=TICKS - 1.5,
                  handletextpad=0.2, columnspacing=1.5)
    figure.savefig(HERE / "era5_concordate_livelli.pdf")
    plt.close(figure)


if __name__ == "__main__":
    overlap_figure()
    judges_figure()
    around_figure()
    levels_figure()
