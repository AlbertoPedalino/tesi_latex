"""Generate the figures of the concordant reference: the along-motion displacement of every method and the
precision-recall curves of Appendix A.3.

Run from the thesis folder:
    python figures/genera_figure_concordate.py
which writes figures/era5_concordate_moto.pdf from figures/dati/era5_concordate_moto_quartili.csv (written by
figures/prepara_dati_moto.py) and figures/era5_concordate_pr.pdf (Appendix A.3) from figures/dati/era5_concordate_pr.csv.

Motion data: for every method, the quartiles over the matched concordant clusters of the along-motion displacement of
the flags of its matching cluster (km, positive = ahead along the motion of the concordant cluster), short and long
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
from matplotlib.legend_handler import HandlerTuple

from genera_figure_eventi import (BASELINES, BLUE, GREEN, GRID, MUTED, ORANGE, SHALLOW_GREY, SHALLOW_KEYS, TEXT, TICKS,
                                  clean, size)

HERE = pathlib.Path(__file__).parent
MOTION = HERE / "dati" / "era5_concordate_moto_quartili.csv"  # written by figures/prepara_dati_moto.py
PR = HERE / "dati" / "era5_concordate_pr.csv"
MARKED = [80, 97, 99]  # percentiles marked with an open dot on every precision-recall curve, besides the 95th
from stile import COLOURS, ORDER, TEXT_COLOURS  # one order and one colour for every method, as in the tables
LABELS = {"Gn": r"$G_\mathrm{n}$", **BASELINES}


def motion_figure():
    """On the long clusters G_n is ahead along the motion, more than every baseline; on the short ones a little behind.
    One row per method, G_n on top. Long clusters: thick pale band from the 25th to the 75th percentile and filled dot
    on the median; short clusters, drawn over it: thin line and open dot. The two ranges overlap on the same row, so
    the distance between the two dots of a kind can be read across the rows."""
    frame = pd.read_csv(MOTION, comment="#").set_index(["kind", "detector"])
    figure, axis = plt.subplots(figsize=size(0.46), constrained_layout=True)
    ys = [len(ORDER)] + list(range(len(ORDER) - 1, 0, -1))  # G_n on top, a row of space below it
    axis.axvline(0, color=TEXT, linewidth=0.8, zorder=1)
    # One cell of the grid, 28-55 km (less in longitude than in latitude), between two dashed lines: drawn as lines and
    # not as a band, so that it does not mix with the bands of the long clusters.
    for edge in (28, 55):
        axis.axvline(edge, color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    axis.text(55, len(ORDER) + 1.75, "larghezza di una cella: 28–55 km", fontsize=TICKS - 1, color=MUTED, ha="right",
              va="top", bbox=dict(facecolor="white", edgecolor="none", pad=0.8), zorder=6)
    for y, key in zip(ys, ORDER):
        colour = COLOURS[key]
        long, short = frame.loc[("lunghi", key)], frame.loc[("brevi", key)]
        axis.plot([long["q25_km"], long["q75_km"]], [y, y], color=colour, linewidth=7, alpha=0.3,
                  solid_capstyle="butt", zorder=2)
        axis.plot([short["q25_km"], short["q75_km"]], [y, y], color=colour, linewidth=1.3, zorder=3)
        axis.plot(short["median_km"], y, "o", markersize=6, markerfacecolor="white", markeredgecolor=colour,
                  markeredgewidth=1.4, zorder=4)
        axis.plot(long["median_km"], y, "o", markersize=6, color=colour, markeredgecolor="white",
                  markeredgewidth=0.6, zorder=5)
    axis.set_yticks(ys, [LABELS[key] for key in ORDER])
    for label, key in zip(axis.get_yticklabels(), ORDER):
        label.set_color(TEXT_COLOURS[key])
    axis.set_ylim(0.3, len(ORDER) + 1.8)
    # G_n: the share of the concordant clusters in which it is ahead, short and long, above its two dots.
    top = len(ORDER)
    for kind, align in (("brevi", "right"), ("lunghi", "left")):
        row = frame.loc[(kind, "Gn")]
        axis.annotate(f"davanti nel {row['ahead_pct']:.0f}%", (row["median_km"], top), xytext=(0, 6),
                      textcoords="offset points", fontsize=TICKS - 1, color=TEXT_COLOURS["Gn"], ha=align,
                      va="bottom", bbox=dict(facecolor="white", edgecolor="none", pad=0.6), zorder=6)
    axis.set_xlim(-52, 60)
    axis.set_xticks([-40, -20, 0, 20, 40])
    axis.set_xlabel("spostamento lungo il moto (km)", fontsize=TICKS)
    axis.tick_params(labelsize=TICKS, length=0)
    axis.grid(axis="x", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    clean(axis, "top", "right", "left")
    axis.text(59, 0.35, "davanti", fontsize=TICKS - 1, color=MUTED, ha="right", va="bottom")
    axis.text(-51, 0.35, "dietro", fontsize=TICKS - 1, color=MUTED, ha="left", va="bottom")
    short_handle = plt.Line2D([], [], color=MUTED, linewidth=1.3, marker="o", markersize=6, markerfacecolor="white",
                              markeredgecolor=MUTED, markeredgewidth=1.4)
    long_handle = (plt.Line2D([], [], color=MUTED, linewidth=7, alpha=0.3, solid_capstyle="butt"),
                   plt.Line2D([], [], color=MUTED, marker="o", linestyle="none", markersize=6,
                              markeredgecolor="white", markeredgewidth=0.6))
    figure.legend([short_handle, long_handle], ["cluster brevi", "cluster lunghi"],
                  handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, loc="outside lower center", ncol=2,
                  fontsize=TICKS - 1, frameon=False, handletextpad=0.5, handlelength=2.2, columnspacing=2.0)
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
        others = part[part["percentile"].isin(MARKED)]  # open dots: the other marked percentiles
        axis.plot(others["recall"], others["precision"], "o", markersize=4 if key == "Gn" else 3,
                  markerfacecolor="white", markeredgecolor=colour, markeredgewidth=1.0, zorder=order + 1)
        if key == "Gn":  # the percentile written next to each marked point, on the curve of G_n only
            for _, row in part[part["percentile"].isin(MARKED + [95])].iterrows():
                low = row["percentile"] == 80  # at the right edge the label goes above, away from the other curves
                axis.annotate(f"{row['percentile']:g}", (row["recall"], row["precision"]),
                              xytext=(-4, 3) if low else (-3, -3), textcoords="offset points", ha="right",
                              va="bottom" if low else "top", fontsize=TICKS - 2, color=BLUE, zorder=order + 2)
    axis.set_xlim(0, 1.02)
    axis.set_ylim(0, 1.02)
    axis.set_xticks([0, 0.25, 0.5, 0.75, 1])
    axis.set_yticks([0.25, 0.5, 0.75, 1])  # one 0 at the origin, on the x axis
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
               plt.Line2D([], [], color=MUTED, marker="o", linestyle="none", markersize=4, label="95-esimo percentile"),
               plt.Line2D([], [], marker="o", linestyle="none", markersize=3.5, markerfacecolor="white",
                          markeredgecolor=MUTED, markeredgewidth=1.0, label="80-esimo, 97-esimo, 99-esimo")]
    axis.legend(handles=handles, loc="lower left", fontsize=TICKS - 1, frameon=False, handlelength=1.6)
    figure.savefig(HERE / "era5_concordate_pr.pdf")
    plt.close(figure)


if __name__ == "__main__":
    motion_figure()
    pr_figure()
