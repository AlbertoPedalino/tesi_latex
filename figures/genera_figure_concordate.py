"""Generate the figures of Section 6.8: the methods around the concordant clusters.

Run from the thesis folder:
    python figures/genera_figure_concordate.py
which writes figures/era5_concordate_moto_a.pdf and _b.pdf from figures/dati/era5_concordate_moto.csv.

Data: the median along-track displacement (km, positive = ahead along the motion of the cluster) of the flags of the
overlapping cluster of each method, by class of the mean speed of the concordant cluster (cells of 0.5 degrees every
3 h), printed by scripts/check_era5_concordant_lead.py of the code repository ("median along-track error (km) by mean
speed of the cluster"), on outputs/concordate_5pc_regola_tesi with --kind analysed (short clusters: 8142 / 7646 / 7651 /
3043 clusters by class) and --kind long (long clusters: 916 / 670 / 508 / 48). Run optuna-27, test years 2005-2025,
thresholds at the 95th percentile of the validation years; logs of 12 November 2026.
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pannelli import save_panels

from genera_figure_eventi import (BLUE, GREEN, GREEN_TEXT, GRID, MUTED, ORANGE, SHALLOW_GREY, SHALLOW_KEYS, TEXT, TICKS,
                                  clean, end_labels, size, spaced)

HERE = pathlib.Path(__file__).parent
MOTION = HERE / "dati" / "era5_concordate_moto.csv"
CLASSES = ("0-0.5", "0.5-1", "1-2", ">=2")
CLASS_LABELS = ("0–0.5", "0.5–1", "1–2", "2 o più")


def motion_figure():
    """Left: on the short clusters G_n is a little behind, more in the fast clusters, the change in three hours more.
    Right: on the long clusters G_n is ahead along the motion, more than every baseline but in the few fastest."""
    table = pd.read_csv(MOTION)
    figure, panels = plt.subplots(1, 2, figsize=size(0.5), constrained_layout=True, sharey=False)
    spaced(figure, 0.2)
    x = range(len(CLASSES))
    for axis, kind in zip(panels, ("brevi", "lunghi")):
        part = table[table["kind"] == kind].set_index("detector")[list(CLASSES)]
        axis.axhline(0, color=TEXT, linewidth=0.8)
        shallow = part.loc[list(SHALLOW_KEYS)]
        axis.fill_between(x, shallow.min(), shallow.max(), color=SHALLOW_GREY, alpha=0.55, linewidth=0, zorder=1)
        for key, colour, width in (("change_3h", GREEN, 1.6), ("normal", ORANGE, 1.6), ("Gn", BLUE, 2.6)):
            axis.plot(x, part.loc[key], color=colour, linewidth=width, marker="o", markersize=4 if key == "Gn" else 3,
                      zorder=3 if key != "Gn" else 4)
        axis.set_xticks(list(x), CLASS_LABELS)
        axis.set_xlim(-0.2, len(CLASSES) - 0.8)
        axis.set_ylim(-40, 40)
        axis.set_yticks([-40, -20, 0, 20, 40])
        axis.set_xlabel("velocità del cluster (celle ogni tre ore)", fontsize=TICKS)
        axis.set_ylabel("spostamento lungo il moto (km)", fontsize=TICKS)
        axis.tick_params(labelsize=TICKS, length=0)
        axis.grid(axis="y", color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
        clean(axis, "top", "right", "left")
        axis.text(-0.15, 38, "davanti", fontsize=TICKS - 1, color=MUTED, va="top")
        axis.text(-0.15, -38, "dietro", fontsize=TICKS - 1, color=MUTED, va="bottom")
        last = part[CLASSES[-1]]
        end_labels(axis, ((last["Gn"], r"$G_\mathrm{n}$", BLUE),
                          (last["normal"], "distanza\ndal normale", ORANGE),
                          (last["change_3h"], "variazione\nin tre ore", GREEN_TEXT),
                          (shallow[CLASSES[-1]].mean(), "sei modelli\nshallow", "#8a8880")))
    save_panels(figure, [[panels[0]], [panels[1]]], HERE, "era5_concordate_moto")
    plt.close(figure)


if __name__ == "__main__":
    motion_figure()
