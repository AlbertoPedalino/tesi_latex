"""Generate the figures of the concordant reference: the along-motion displacement of every method and the
precision-recall curves of Appendix A.3.

Run from the thesis folder:
    python figures/genera_figure_concordate.py
which writes figures/era5_concordate_moto.pdf from figures/dati/era5_concordate_moto_r300.csv and
figures/era5_concordate_pr.pdf (Appendix A.3) from figures/dati/era5_concordate_pr.csv.

Motion data: for every method, the median over the matched concordant clusters of the along-motion displacement of the
flags of its matching cluster (km, positive = ahead along the motion of the concordant cluster), short and long
clusters, from the column along_km of lead_r300.csv and lead_r300_long.csv (scripts/check_era5_concordant_lead.py of
the code repository, zone of 300 km, cube at the 95th percentile; downloaded in Tesi/log_server/r300/outputs/
concordate_5pc_regola_tesi; the rows of the probability levels, Gn_p and Gn_rand, left out). Run optuna-27, test years
2005-2025, thresholds at the 95th percentile of the validation years.

Precision-recall data: precision and recall of every detector against the concordant cube as its own threshold goes from
the 80th to the 99.9th percentile of the validation years (scripts/check_era5_concordant_pr.py of the code repository,
commit 576547c, run of 13 November 2026; the cube fixed at the 95th percentile).
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from genera_figure_eventi import (BASELINES, BLUE, GREEN, GRID, MUTED, ORANGE, SHALLOW_GREY, SHALLOW_KEYS, TEXT, TICKS,
                                  clean, size)

HERE = pathlib.Path(__file__).parent
MOTION = HERE / "dati" / "era5_concordate_moto_r300.csv"
PR = HERE / "dati" / "era5_concordate_pr.csv"
from stile import COLOURS, ORDER, TEXT_COLOURS  # one order and one colour for every method, as in the tables
LABELS = {"Gn": r"$G_\mathrm{n}$", **BASELINES}


def motion_figure():
    """On the long clusters G_n is ahead along the motion, more than every baseline; on the short ones a little behind.
    One row per method, G_n on top; open dot = short clusters, filled dot = long clusters."""
    table = pd.read_csv(MOTION, comment="#").set_index(["kind", "detector"])["median_km"]
    figure, axis = plt.subplots(figsize=size(0.42), constrained_layout=True)
    ys = [len(ORDER)] + list(range(len(ORDER) - 1, 0, -1))  # G_n on top, a row of space below it
    axis.axvline(0, color=TEXT, linewidth=0.8, zorder=1)
    for y, key in zip(ys, ORDER):
        colour = COLOURS[key]
        short, long = table[("brevi", key)], table[("lunghi", key)]
        axis.plot([short, long], [y, y], color=colour, linewidth=1.0, alpha=0.5, zorder=2)
        axis.plot(short, y, "o", markersize=6, markerfacecolor="white", markeredgecolor=colour, markeredgewidth=1.4,
                  zorder=3)
        axis.plot(long, y, "o", markersize=6, color=colour, markeredgecolor="white", markeredgewidth=0.6, zorder=4)
    axis.set_yticks(ys, [LABELS[key] for key in ORDER])
    for label, key in zip(axis.get_yticklabels(), ORDER):
        label.set_color(TEXT_COLOURS[key])
    axis.set_ylim(0.4, len(ORDER) + 0.6)
    axis.set_xlim(-15, 20)
    axis.set_xticks([-15, -10, -5, 0, 5, 10, 15, 20])
    axis.set_xlabel("spostamento lungo il moto, mediana (km)", fontsize=TICKS)
    axis.tick_params(labelsize=TICKS, length=0)
    axis.grid(axis="x", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    clean(axis, "top", "right", "left")
    axis.text(19.5, 0.55, "davanti", fontsize=TICKS - 1, color=MUTED, ha="right", va="bottom")
    axis.text(-14.5, 0.55, "dietro", fontsize=TICKS - 1, color=MUTED, ha="left", va="bottom")
    handles = [plt.Line2D([], [], marker="o", linestyle="none", markersize=6, markerfacecolor="white",
                          markeredgecolor=MUTED, markeredgewidth=1.4, label="cluster brevi"),
               plt.Line2D([], [], marker="o", linestyle="none", markersize=6, color=MUTED, label="cluster lunghi")]
    axis.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 0.08), fontsize=TICKS - 1, frameon=False,
                handletextpad=0.3)
    figure.savefig(HERE / "era5_concordate_moto.pdf")
    plt.close(figure)


def pr_figure():
    """Precision against recall of every detector as its own threshold varies, against the concordant cube; the dot is
    the 95th percentile, the threshold of the cube. G_n in blue, the two simple baselines in their colours, the six
    shallow models in grey."""
    table = pd.read_csv(PR, comment="#")
    width, height = size(0.5)
    figure, axis = plt.subplots(figsize=(width / 2, height), constrained_layout=True)  # as one panel of before
    styles = [(key, SHALLOW_GREY, 1.0, 2) for key in SHALLOW_KEYS]
    styles += [("change_3h", GREEN, 1.6, 3), ("normal", ORANGE, 1.6, 3), ("Gn", BLUE, 2.6, 4)]
    for key, colour, width, order in styles:
        part = table[table["detector"] == key].sort_values("percentile")
        axis.plot(part["recall"], part["precision"], color=colour, linewidth=width, zorder=order)
        at = part[part["percentile"] == 95]
        axis.plot(at["recall"], at["precision"], "o", color=colour, markersize=5 if key == "Gn" else 4,
                  markeredgecolor="white", markeredgewidth=0.6, zorder=order + 1)
    axis.set_xlim(0, 1.02)
    axis.set_ylim(0, 1.02)
    axis.set_xticks([0, 0.25, 0.5, 0.75, 1])
    axis.set_yticks([0, 0.25, 0.5, 0.75, 1])
    axis.set_xlabel("recall", fontsize=TICKS)
    axis.set_ylabel("precision", fontsize=TICKS)
    axis.tick_params(labelsize=TICKS, length=0)
    axis.grid(color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    clean(axis, "top", "right")
    handles = [plt.Line2D([], [], color=BLUE, linewidth=2.6, label=r"$G_\mathrm{n}$"),
               plt.Line2D([], [], color=ORANGE, linewidth=1.6, label="distanza dal normale"),
               plt.Line2D([], [], color=GREEN, linewidth=1.6, label="variazione in tre ore"),
               plt.Line2D([], [], color=SHALLOW_GREY, linewidth=1.0, label="sei modelli shallow"),
               plt.Line2D([], [], color=MUTED, marker="o", linestyle="none", markersize=4, label="95-esimo percentile")]
    axis.legend(handles=handles, loc="lower left", fontsize=TICKS - 1, frameon=False, handlelength=1.6)
    figure.savefig(HERE / "era5_concordate_pr.pdf")
    plt.close(figure)


if __name__ == "__main__":
    motion_figure()
    pr_figure()
