"""Generate the figures of the concordant reference: the along-motion displacement of every method and the
precision-recall curves of Appendix A.3.

Run from the thesis folder:
    python figures/genera_figure_concordate.py
which writes figures/era5_concordate_moto.pdf from figures/dati/era5_concordate_moto_medie.csv,
figures/era5_concordate_moto_distribuzione.pdf from figures/dati/era5_concordate_moto_distribuzione.csv (both written by
figures/prepara_dati_moto.py) and figures/era5_concordate_pr.pdf (Appendix A.3) from figures/dati/era5_concordate_pr.csv.

Motion data: for every method, the mean over the matched concordant clusters of the along-motion displacement of its
flags (km, positive = ahead along the mean motion of the concordant cluster), with its 95% bootstrap interval, and the
histogram of the displacement of G_n, three-hour change and MCD with the 20th and 80th percentiles, short and long
clusters (figures/prepara_dati_moto.py, from the column along_km of lead_r300.csv and lead_r300_long.csv of
scripts/check_era5_concordant_lead.py of the code repository, zone of 300 km, cube at the 95th percentile). Run
optuna-27, test years 2005-2025, thresholds at the 95th percentile of the validation years.

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
MOTION = HERE / "dati" / "era5_concordate_moto_medie.csv"  # written by figures/prepara_dati_moto.py
SPREAD = HERE / "dati" / "era5_concordate_moto_distribuzione.csv"  # idem
PR = HERE / "dati" / "era5_concordate_pr.csv"
MARKED = [80, 97, 99]  # percentiles marked with an open dot on every precision-recall curve, besides the 95th
from stile import COLOURS, ORDER, TEXT_COLOURS  # one order and one colour for every method, as in the tables
LABELS = {"Gn": r"$G_\mathrm{n}$", **BASELINES}


def motion_figure():
    """On the long clusters G_n is ahead along the motion, more than every baseline; on the short ones a little behind.
    One row per method, G_n on top: the mean over the clusters (open dot: short clusters; filled dot: long clusters)
    with its 95% bootstrap interval (thin line: short; pale band: long), as the mean along-track errors with their
    bootstrap intervals of Leonardo and Colle (2017, Fig. 4). On the right, the share of the clusters in which the
    method is ahead."""
    frame = pd.read_csv(MOTION).set_index(["kind", "detector"])
    figure, (axis, side) = plt.subplots(1, 2, figsize=size(0.46), constrained_layout=True, sharey=True,
                                        gridspec_kw={"width_ratios": [4.2, 1]})
    ys = [len(ORDER)] + list(range(len(ORDER) - 1, 0, -1))  # G_n on top
    axis.axvline(0, color=TEXT, linewidth=0.8, zorder=1)
    # One cell of the grid, 28-55 km (less in longitude than in latitude), between two dashed lines.
    for edge in (28, 55):
        axis.axvline(edge, color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    axis.text(55, len(ORDER) + 1.75, "larghezza di una cella: 28–55 km", fontsize=TICKS - 1, color=MUTED,
              ha="right", va="top", bbox=dict(facecolor="white", edgecolor="none", pad=0.8), zorder=6)
    for y, key in zip(ys, ORDER):
        colour = COLOURS[key]
        long, short = frame.loc[("lunghi", key)], frame.loc[("brevi", key)]
        axis.plot([long["ci_low_km"], long["ci_high_km"]], [y, y], color=colour, linewidth=7, alpha=0.3,
                  solid_capstyle="butt", zorder=2)
        axis.plot([short["ci_low_km"], short["ci_high_km"]], [y, y], color=colour, linewidth=1.3, zorder=3)
        axis.plot(short["mean_km"], y, "o", markersize=6, markerfacecolor="white", markeredgecolor=colour,
                  markeredgewidth=1.4, zorder=4)
        axis.plot(long["mean_km"], y, "o", markersize=6, color=colour, markeredgecolor="white", markeredgewidth=0.6,
                  zorder=5)
        for x, kind in ((0.28, "brevi"), (0.78, "lunghi")):
            side.text(x, y, f"{frame.loc[(kind, key), 'ahead_pct']:.0f}", ha="center", va="center", fontsize=TICKS,
                      color=TEXT_COLOURS[key], fontweight="bold" if key == "Gn" else "normal")
    axis.set_yticks(ys, [LABELS[key] for key in ORDER])
    for label, key in zip(axis.get_yticklabels(), ORDER):
        label.set_color(TEXT_COLOURS[key])
    axis.set_ylim(0.3, len(ORDER) + 1.8)
    axis.set_xlim(-22, 60)
    axis.set_xticks([-20, -10, 0, 10, 20, 30, 40, 50])
    axis.set_xlabel("spostamento lungo il moto (km)", fontsize=TICKS)
    axis.tick_params(labelsize=TICKS, length=0)
    axis.grid(axis="x", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    clean(axis, "top", "right", "left")
    axis.text(59, 0.35, "davanti", fontsize=TICKS - 1, color=MUTED, ha="right", va="bottom")
    axis.text(-21, 0.35, "dietro", fontsize=TICKS - 1, color=MUTED, ha="left", va="bottom")
    side.set_xlim(0, 1)
    side.text(0.53, len(ORDER) + 1.75, "davanti (%)", ha="center", va="top", fontsize=TICKS - 1, color=MUTED)
    side.text(0.28, len(ORDER) + 1.0, "brevi", ha="center", va="center", fontsize=TICKS - 1, color=MUTED)
    side.text(0.78, len(ORDER) + 1.0, "lunghi", ha="center", va="center", fontsize=TICKS - 1, color=MUTED)
    side.axis("off")
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


def spread_figure():
    """How the displacement varies from cluster to cluster: histogram of G_n and of the two baselines ahead, short and
    long clusters, bars of 10 km, with the mean and the 20th and 80th percentiles (Leonardo and Colle 2017, Fig. 3a;
    2020, Fig. 3). The clusters beyond +-150 km are not drawn; their share is written at the two ends."""
    table = pd.read_csv(SPREAD)
    keys = list(dict.fromkeys(table["detector"]))
    width, height = size(0.5)
    figure, axes = plt.subplots(len(keys), 2, figsize=(width, height * 1.15), sharex=True, sharey="col",
                                constrained_layout=True)
    for col, kind in enumerate(("brevi", "lunghi")):
        for row, key in enumerate(keys):
            axis = axes[row, col]
            part = table[(table["kind"] == kind) & (table["detector"] == key)]
            first = part.iloc[0]
            axis.bar(part["bin_from_km"] + 5, part["percent"], width=10, color=COLOURS[key], alpha=0.75)
            axis.axvline(0, color=TEXT, linewidth=0.8)
            for q in (first["q20_km"], first["q80_km"]):
                axis.axvline(q, color=TEXT_COLOURS[key], linewidth=1.0, linestyle=(0, (3, 2)))
            axis.axvline(first["mean_km"], color=TEXT_COLOURS[key], linewidth=1.8)
            axis.text(0.01, 0.97, f"← {first['below_pct']:.1f}%", transform=axis.transAxes, ha="left",
                      va="top", fontsize=TICKS - 2, color=MUTED)
            axis.text(0.99, 0.97, f"{first['above_pct']:.1f}% →", transform=axis.transAxes, ha="right",
                      va="top", fontsize=TICKS - 2, color=MUTED)
            if col == 0:
                axis.text(-0.3, 0.5, LABELS[key], transform=axis.transAxes, ha="right", va="center", fontsize=TICKS,
                          color=TEXT_COLOURS[key])
                axis.set_ylabel("cluster\nconcordati (%)", fontsize=TICKS - 2)
            axis.tick_params(labelsize=TICKS - 1, length=0)
            axis.grid(axis="y", color=GRID, linewidth=0.5)
            axis.set_axisbelow(True)
            clean(axis, "top", "right")
            if row == 0:
                axis.set_title(f"cluster {kind}", fontsize=TICKS)
    for axis in axes[-1]:
        axis.set_xlabel("spostamento lungo il moto (km)", fontsize=TICKS - 1)
        axis.set_xticks([-150, -100, -50, 0, 50, 100, 150])
        axis.set_xlim(-155, 155)
    handles = [plt.Line2D([], [], color=MUTED, linewidth=1.8),
               plt.Line2D([], [], color=MUTED, linewidth=1.0, linestyle=(0, (3, 2)))]
    figure.legend(handles, ["media", "20-esimo e 80-esimo percentile"], loc="outside lower center", ncol=2,
                  fontsize=TICKS - 1, frameon=False)
    figure.savefig(HERE / "era5_concordate_moto_distribuzione.pdf")
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
    spread_figure()
    pr_figure()
